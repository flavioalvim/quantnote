"""Strike grid predictor for options pricing."""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from ..optimization.strike_grid_optimizer import StrikeGridOptimizationResult, StrikeTarget
from ..optimization.return_strategy import get_strategy
from .regime_predictor import RegimePredictor, CurrentRegimeResult


@dataclass
class StrikeProbabilityRow:
    """Single row of the strike probability matrix."""
    strike: float
    target_return: float
    target_pct: str
    probability: float
    probability_unconditional: float
    delta: float
    cluster: int
    interpretation: str
    direction: str  # UP, DOWN, ATM
    data_is_fresh: bool


class StrikeGridPredictor:
    """
    Predicts probabilities across a grid of strike prices.

    Uses multiple RegimePredictor instances, one per strike.
    """

    def __init__(
        self,
        predictors: Dict[float, RegimePredictor],
        strike_targets: Dict[float, StrikeTarget],
        training_price: float,
        horizon: int,
        strategy_name: str = "close",
        logger: Optional[ILogger] = None
    ):
        """
        Initialize strike grid predictor.

        Args:
            predictors: Dict mapping strike -> RegimePredictor
            strike_targets: Dict mapping strike -> StrikeTarget
            training_price: Price used during training (for reference)
            horizon: Prediction horizon in days
            strategy_name: Strategy used ('close' or 'touch')
            logger: Optional logger
        """
        self.predictors = predictors
        self.strike_targets = strike_targets
        self.training_price = training_price
        self.horizon = horizon
        self.strategy_name = strategy_name
        self.logger = logger or NullLogger()
        self.strikes = sorted(predictors.keys())

    @classmethod
    def from_optimization(
        cls,
        optimization_result: StrikeGridOptimizationResult,
        df: pd.DataFrame,
        data_dir: str = "data",
        logger: Optional[ILogger] = None
    ) -> 'StrikeGridPredictor':
        """
        Create predictor from optimization results.

        Args:
            optimization_result: Result from StrikeGridOptimizer
            df: Historical data for fitting
            data_dir: Directory for data cache
            logger: Optional logger

        Returns:
            Fitted StrikeGridPredictor
        """
        predictors = {}

        # Get strategy from optimization result
        strategy = get_strategy(optimization_result.strategy_name)

        for strike, result in optimization_result.results.items():
            st = optimization_result.strike_targets[strike]

            # Create and fit predictor for this strike with the correct strategy
            predictor = RegimePredictor(
                chromosome=result.ga_result.best_chromosome,
                target_return=st.target_return,
                horizon=optimization_result.horizon,
                strategy=strategy,  # ← Pass the strategy!
                data_dir=data_dir,
                logger=logger
            )
            predictor.fit(df)
            predictors[strike] = predictor

        return cls(
            predictors=predictors,
            strike_targets=optimization_result.strike_targets,
            training_price=optimization_result.current_price,
            horizon=optimization_result.horizon,
            strategy_name=optimization_result.strategy_name,
            logger=logger
        )

    def predict_current(self, ticker: str) -> pd.DataFrame:
        """
        Predict probabilities for all strikes.

        Args:
            ticker: Ticker symbol

        Returns:
            DataFrame with strike probability matrix
        """
        rows = []

        for strike in self.strikes:
            predictor = self.predictors[strike]
            st = self.strike_targets[strike]
            result = predictor.predict_current(ticker)

            # Determine direction
            if st.target_return > 0.005:
                direction = "UP"
            elif st.target_return < -0.005:
                direction = "DOWN"
            else:
                direction = "ATM"

            rows.append(StrikeProbabilityRow(
                strike=strike,
                target_return=st.target_return,
                target_pct=st.target_pct,
                probability=result.probability,
                probability_unconditional=result.probability_unconditional,
                delta=result.probability - result.probability_unconditional,
                cluster=result.cluster,
                interpretation=result.interpretation,
                direction=direction,
                data_is_fresh=result.data_is_fresh
            ))

        df = pd.DataFrame([vars(r) for r in rows])
        return df

    def print_matrix(self, ticker: str, result: Optional[pd.DataFrame] = None) -> None:
        """
        Print formatted strike probability matrix.

        Args:
            ticker: Ticker symbol
            result: Pre-computed result (optional)
        """
        if result is None:
            result = self.predict_current(ticker)

        # Get current price from first predictor
        first_predictor = self.predictors[self.strikes[0]]
        current_result = first_predictor.predict_current(ticker)
        current_price = current_result.price
        current_date = current_result.date

        # Column name based on strategy
        prob_col_name = "P(tocar)" if self.strategy_name == "touch" else "P(fechar)"

        print("=" * 80)
        print(f"MATRIZ DE STRIKES - {prob_col_name.upper()}")
        print("=" * 80)
        print(f"Ticker: {ticker}")
        print(f"Estratégia: {self.strategy_name}")
        print(f"Horizonte: {self.horizon} dias")
        print(f"Data: {current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else current_date}")
        print(f"Preço Atual: R$ {current_price:.2f}")
        print(f"Preço de Treinamento: R$ {self.training_price:.2f}")
        print("=" * 80)

        # Find ATM strike (closest to current price)
        atm_strike = min(self.strikes, key=lambda x: abs(x - current_price))

        print(f"\n{'Strike':<10} {'Retorno':<10} {prob_col_name:<12} {'P(base)':<12} {'Delta':<10} {'Regime':<8}")
        print("-" * 62)

        for _, row in result.iterrows():
            delta_str = f"+{row['delta']:.1%}" if row['delta'] >= 0 else f"{row['delta']:.1%}"

            # Mark ATM
            atm_marker = " ← ATM" if row['strike'] == atm_strike else ""

            print(f"R$ {row['strike']:<7.2f} {row['target_pct']:<10} "
                  f"{row['probability']:.1%}        {row['probability_unconditional']:.1%}        "
                  f"{delta_str:<10} {row['interpretation']:<8}{atm_marker}")

        # Summary statistics
        print("\n" + "-" * 62)

        # Upside analysis
        upside = result[result['direction'] == 'UP']
        if len(upside) > 0:
            best_up = upside.loc[upside['delta'].idxmax()]
            print(f"\n📈 CALLS (upside):")
            print(f"   Melhor oportunidade: Strike R$ {best_up['strike']:.2f} "
                  f"({best_up['target_pct']}) - Delta {best_up['delta']:+.1%}")

        # Downside analysis
        downside = result[result['direction'] == 'DOWN']
        if len(downside) > 0:
            best_down = downside.loc[downside['delta'].idxmax()]
            print(f"\n📉 PUTS (downside):")
            print(f"   Melhor oportunidade: Strike R$ {best_down['strike']:.2f} "
                  f"({best_down['target_pct']}) - Delta {best_down['delta']:+.1%}")

        if not result['data_is_fresh'].all():
            print(f"\n⏰ Aviso: Alguns dados não estão atualizados")

    def get_probability_at_strike(self, ticker: str, strike: float) -> CurrentRegimeResult:
        """
        Get detailed prediction for a specific strike.

        Args:
            ticker: Ticker symbol
            strike: Strike price

        Returns:
            CurrentRegimeResult for that strike
        """
        if strike not in self.predictors:
            raise ValueError(f"Strike {strike} not in predictor. Available: {self.strikes}")

        return self.predictors[strike].predict_current(ticker)

    def save(self, path: str) -> None:
        """
        Save predictor to directory.

        Args:
            path: Base directory path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest = {
            'training_price': self.training_price,
            'horizon': self.horizon,
            'strategy_name': self.strategy_name,
            'strikes': self.strikes,
            'n_strikes': len(self.strikes),
            'strike_targets': {
                str(k): {'strike': v.strike, 'target_return': v.target_return}
                for k, v in self.strike_targets.items()
            }
        }
        with open(path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        # Save each predictor
        for strike, predictor in self.predictors.items():
            strike_str = f"{strike:.2f}".replace('.', '_')
            predictor_path = path / f"predictor_strike_{strike_str}"
            predictor.save(str(predictor_path))

        self.logger.info("StrikeGridPredictor saved", path=str(path))

    @classmethod
    def load(
        cls,
        path: str,
        data_dir: str = "data",
        logger: Optional[ILogger] = None
    ) -> 'StrikeGridPredictor':
        """
        Load predictor from directory.

        Args:
            path: Base directory path
            data_dir: Directory for data cache
            logger: Optional logger

        Returns:
            Loaded StrikeGridPredictor
        """
        path = Path(path)

        # Load manifest
        with open(path / 'manifest.json', 'r') as f:
            manifest = json.load(f)

        # Reconstruct strike_targets
        strike_targets = {}
        for k, v in manifest['strike_targets'].items():
            strike = float(k)
            strike_targets[strike] = StrikeTarget(
                strike=v['strike'],
                target_return=v['target_return']
            )

        # Load each predictor
        predictors = {}
        for strike in manifest['strikes']:
            strike_str = f"{strike:.2f}".replace('.', '_')
            predictor_path = path / f"predictor_strike_{strike_str}"
            predictors[strike] = RegimePredictor.load(
                str(predictor_path),
                data_dir=data_dir,
                logger=logger
            )

        return cls(
            predictors=predictors,
            strike_targets=strike_targets,
            training_price=manifest['training_price'],
            horizon=manifest['horizon'],
            strategy_name=manifest.get('strategy_name', 'close'),  # Default for backward compat
            logger=logger
        )
