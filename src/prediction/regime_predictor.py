"""Regime predictor for production use."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import joblib
from pathlib import Path

from ..optimization.chromosome import Chromosome
from ..optimization.calculator_factory import CalculatorFactory
from ..optimization.return_strategy import IReturnStrategy, DEFAULT_CLOSE_STRATEGY, get_strategy
from ..analysis.kmeans_regimes import KMeansRegimeClassifier
from ..analysis.probability_calculator import ProbabilityCalculator
from ..infrastructure.yahoo_data_source import YahooDataSource
from ..infrastructure.parquet_repository import ParquetRepository
from ..infrastructure.file_logger import NullLogger
from ..interfaces.logger import ILogger


@dataclass
class CurrentRegimeResult:
    """Result of current regime prediction."""
    date: datetime
    price: float
    cluster: int
    interpretation: str  # bull, bear, flat
    probability: float  # P(target) for this cluster
    probability_unconditional: float  # P(target) overall
    data_is_fresh: bool  # True if data is from today/last trading day
    features: Dict[str, float]  # Feature values for this observation
    days_since_update: int  # Days since last data update
    warning: Optional[str] = None  # Warning message if any


class RegimePredictor:
    """
    Predicts current market regime based on trained GA parameters.

    Encapsulates:
    - Data fetching and caching
    - Feature pipeline
    - K-Means classification
    - Probability calculation

    Usage:
        # Train and save
        predictor = RegimePredictor(chromosome, target_return=0.05, horizon=7)
        predictor.fit(df_historical)
        predictor.save("model.joblib")

        # Load and predict
        predictor = RegimePredictor.load("model.joblib")
        result = predictor.predict_current("BOVA11.SA")
    """

    def __init__(
        self,
        chromosome: Chromosome,
        target_return: float,
        horizon: int,
        ticker: Optional[str] = None,
        strategy: Optional[IReturnStrategy] = None,
        data_dir: str = "data",
        logger: Optional[ILogger] = None
    ):
        """
        Initialize predictor.

        Args:
            chromosome: Optimized chromosome from GA
            target_return: Target return for probability calculation
            horizon: Prediction horizon in days
            ticker: Ticker symbol (optional, can be set later)
            strategy: Return strategy (close or touch). Defaults to close.
            data_dir: Directory for data cache
            logger: Optional logger
        """
        self.chromosome = chromosome
        self.target_return = target_return
        self.horizon = horizon
        self.ticker = ticker
        self.strategy = strategy or DEFAULT_CLOSE_STRATEGY
        self.data_dir = data_dir
        self.logger = logger or NullLogger()

        # Create factory with strategy and get feature columns
        self.factory = CalculatorFactory(horizon=horizon, strategy=self.strategy)
        self.feature_cols = self.factory.get_feature_columns(chromosome)
        # Use strategy to get appropriate return column
        self.future_col = self.factory.get_return_column(target_return)

        # These will be set during fit()
        self.kmeans: Optional[KMeansRegimeClassifier] = None
        self.cluster_interpretations: Dict[int, str] = {}
        self.cluster_probabilities: Dict[int, float] = {}
        self.probability_unconditional: float = 0.0
        self._is_fitted = False

        # Data infrastructure
        self._data_source = YahooDataSource(calls_per_minute=5, logger=logger)
        self._repository = ParquetRepository(data_dir=data_dir, logger=logger)

    def fit(self, df: pd.DataFrame) -> 'RegimePredictor':
        """
        Fit the predictor on historical data.

        Args:
            df: Historical OHLCV data

        Returns:
            self for chaining
        """
        # Create and run pipeline
        pipeline = self.factory.create_pipeline(self.chromosome)
        df_processed = pipeline.run(df)

        # Fit K-Means
        self.kmeans = KMeansRegimeClassifier(
            n_clusters=self.chromosome.n_clusters,
            feature_columns=self.feature_cols,
            logger=self.logger
        )
        df_clustered = self.kmeans.fit_predict(df_processed)

        # Get cluster interpretations
        slope_col = f'slope_{self.chromosome.window_slope}'
        self.cluster_interpretations = self.kmeans.interpret_clusters(
            df_clustered, slope_col
        )

        # Calculate probabilities for each cluster
        prob_calc = ProbabilityCalculator(
            future_return_column=self.future_col,
            target_return=self.target_return,
            regime_column='cluster'
        )

        report = prob_calc.generate_report(df_clustered)
        self.probability_unconditional = report['raw_probability']

        for cluster_id, data in report['conditional_probabilities'].items():
            self.cluster_probabilities[int(float(cluster_id))] = data['probability']

        self._is_fitted = True
        self.logger.info(
            "Predictor fitted",
            n_clusters=self.chromosome.n_clusters,
            features=self.feature_cols
        )

        return self

    def predict_current(self, ticker: Optional[str] = None) -> CurrentRegimeResult:
        """
        Predict current regime for the ticker.

        Args:
            ticker: Ticker symbol (uses stored ticker if not provided)

        Returns:
            CurrentRegimeResult with prediction details
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError("No ticker provided")

        # Update data with cache intelligence
        df, data_info = self._get_updated_data(ticker)

        # Process through pipeline
        pipeline = self.factory.create_pipeline(self.chromosome)
        df_processed = pipeline.run(df)

        # Predict cluster for last row
        df_predicted = self.kmeans.predict(df_processed)

        # Get last row
        last_row = df_predicted.iloc[-1]
        cluster = int(last_row['cluster'])

        # Extract features
        features = {col: last_row[col] for col in self.feature_cols if col in last_row}

        return CurrentRegimeResult(
            date=last_row['date'] if 'date' in last_row else last_row.name,
            price=last_row['close'],
            cluster=cluster,
            interpretation=self.cluster_interpretations.get(cluster, 'unknown'),
            probability=self.cluster_probabilities.get(cluster, 0.0),
            probability_unconditional=self.probability_unconditional,
            data_is_fresh=data_info['is_fresh'],
            features=features,
            days_since_update=data_info['days_since_update'],
            warning=data_info.get('warning')
        )

    def _get_updated_data(self, ticker: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get data with intelligent caching.

        Returns:
            Tuple of (DataFrame, info dict with freshness details)
        """
        info = {
            'is_fresh': False,
            'days_since_update': 0,
            'warning': None
        }

        # Try to load from cache
        df = self._repository.load(ticker)

        if df is not None and len(df) > 0:
            last_date = df['date'].max()
            today = datetime.now().date()
            days_diff = (today - last_date.date()).days

            # Check if data is fresh (within 1 trading day)
            # Consider weekends: if today is Monday and last_date is Friday, it's fresh
            is_weekend = today.weekday() >= 5
            is_monday = today.weekday() == 0

            if days_diff == 0:
                # Data is from today
                info['is_fresh'] = True
                info['days_since_update'] = 0
                return df, info
            elif days_diff == 1 and not is_weekend:
                # Data is from yesterday (weekday)
                info['is_fresh'] = True
                info['days_since_update'] = 1
                return df, info
            elif is_monday and days_diff <= 3:
                # Monday and data is from Friday
                info['is_fresh'] = True
                info['days_since_update'] = days_diff
                return df, info

            # Data is stale, try to update
            info['days_since_update'] = days_diff

        # Try to fetch new data
        try:
            new_df = self._data_source.fetch_ohlcv(ticker)
            if new_df is not None and len(new_df) > 0:
                self._repository.save(new_df, ticker)
                info['is_fresh'] = True
                info['days_since_update'] = 0
                return new_df, info
        except Exception as e:
            # Fallback gracioso
            info['warning'] = f"Failed to update data: {str(e)}"
            self.logger.warning(
                "Data update failed, using cached data",
                error=str(e),
                ticker=ticker
            )

        # Return cached data if available (fallback)
        if df is not None:
            info['warning'] = info.get('warning') or f"Using cached data from {info['days_since_update']} days ago"
            return df, info

        raise RuntimeError(f"No data available for {ticker}")

    def save(self, path: str) -> None:
        """
        Save predictor to file.

        Args:
            path: File path (will save .joblib and .json files)
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted predictor")

        path = Path(path)
        base_path = path.with_suffix('')

        # Save model components with joblib
        model_data = {
            'kmeans_model': self.kmeans.kmeans if self.kmeans else None,
            'kmeans_scaler': self.kmeans.scaler if self.kmeans else None,
        }
        joblib.dump(model_data, f"{base_path}.joblib")

        # Save configuration as JSON
        config = {
            'chromosome': self.chromosome.to_dict(),
            'target_return': self.target_return,
            'horizon': self.horizon,
            'ticker': self.ticker,
            'strategy_name': self.strategy.name,
            'feature_cols': self.feature_cols,
            'future_col': self.future_col,
            'cluster_interpretations': self.cluster_interpretations,
            'cluster_probabilities': self.cluster_probabilities,
            'probability_unconditional': self.probability_unconditional,
        }

        with open(f"{base_path}.json", 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info("Predictor saved", path=str(path))

    @classmethod
    def load(cls, path: str, data_dir: str = "data", logger: Optional[ILogger] = None) -> 'RegimePredictor':
        """
        Load predictor from file.

        Args:
            path: File path (base path without extension)
            data_dir: Directory for data cache
            logger: Optional logger

        Returns:
            Loaded RegimePredictor
        """
        path = Path(path)
        base_path = path.with_suffix('')

        # Load configuration
        with open(f"{base_path}.json", 'r') as f:
            config = json.load(f)

        # Recreate chromosome
        chromosome = Chromosome.from_dict(config['chromosome'])

        # Recreate strategy
        strategy_name = config.get('strategy_name', 'close')  # Default for backward compat
        strategy = get_strategy(strategy_name)

        # Create predictor
        predictor = cls(
            chromosome=chromosome,
            target_return=config['target_return'],
            horizon=config['horizon'],
            ticker=config.get('ticker'),
            strategy=strategy,
            data_dir=data_dir,
            logger=logger
        )

        # Load model components
        model_data = joblib.load(f"{base_path}.joblib")

        # Recreate KMeans classifier
        predictor.kmeans = KMeansRegimeClassifier(
            n_clusters=chromosome.n_clusters,
            feature_columns=config['feature_cols'],
            logger=logger
        )
        predictor.kmeans.kmeans = model_data['kmeans_model']
        predictor.kmeans.scaler = model_data['kmeans_scaler']
        predictor.kmeans._is_fitted = True

        # Restore other attributes
        predictor.cluster_interpretations = {
            int(k): v for k, v in config['cluster_interpretations'].items()
        }
        predictor.cluster_probabilities = {
            int(k): v for k, v in config['cluster_probabilities'].items()
        }
        predictor.probability_unconditional = config['probability_unconditional']
        predictor._is_fitted = True

        return predictor

    def print_status(self, result: Optional[CurrentRegimeResult] = None, ticker: Optional[str] = None) -> None:
        """
        Print formatted status report.

        Args:
            result: Pre-computed result (optional)
            ticker: Ticker to predict (if result not provided)
        """
        if result is None:
            result = self.predict_current(ticker)

        print("=" * 60)
        print("REGIME ATUAL")
        print("=" * 60)
        print(f"\nTicker: {self.ticker or 'N/A'}")
        print(f"Data: {result.date.strftime('%Y-%m-%d') if hasattr(result.date, 'strftime') else result.date}")
        print(f"Preço: R$ {result.price:.2f}")

        if result.warning:
            print(f"\n⚠️  AVISO: {result.warning}")

        print(f"\n📊 Regime: Cluster {result.cluster} ({result.interpretation.upper()})")
        print(f"\n📈 Probabilidade de atingir {self.target_return:.1%} em {self.horizon} dias:")
        print(f"   Condicional (regime atual): {result.probability:.1%}")
        print(f"   Incondicional (média):      {result.probability_unconditional:.1%}")

        diff = result.probability - result.probability_unconditional
        if diff > 0:
            print(f"\n   ✅ {diff:.1%} pontos ACIMA da média")
        else:
            print(f"\n   ⚠️  {abs(diff):.1%} pontos ABAIXO da média")

        print(f"\n🔧 Features:")
        for name, value in result.features.items():
            print(f"   {name}: {value:.6f}")

        if not result.data_is_fresh:
            print(f"\n⏰ Dados de {result.days_since_update} dias atrás")
