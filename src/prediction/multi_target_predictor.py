"""Multi-target predictor for probability matrix generation."""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from ..optimization.multi_target_optimizer import MultiTargetOptimizationResult
from .regime_predictor import RegimePredictor, CurrentRegimeResult


@dataclass
class ProbabilityMatrixRow:
    """Single row of the probability matrix."""
    target: float
    target_pct: str
    probability: float
    probability_unconditional: float
    delta: float
    cluster: int
    interpretation: str
    data_is_fresh: bool


class MultiTargetPredictor:
    """
    Predicts probabilities across multiple target returns.

    Uses multiple RegimePredictor instances, one per target.
    """

    def __init__(
        self,
        predictors: Dict[float, RegimePredictor],
        horizon: int,
        logger: Optional[ILogger] = None
    ):
        """
        Initialize multi-target predictor.

        Args:
            predictors: Dict mapping target -> RegimePredictor
            horizon: Prediction horizon in days
            logger: Optional logger
        """
        self.predictors = predictors
        self.horizon = horizon
        self.logger = logger or NullLogger()
        self.targets = sorted(predictors.keys())

    @classmethod
    def from_optimization(
        cls,
        optimization_result: MultiTargetOptimizationResult,
        df: pd.DataFrame,
        data_dir: str = "data",
        logger: Optional[ILogger] = None
    ) -> 'MultiTargetPredictor':
        """
        Create predictor from optimization results.

        Args:
            optimization_result: Result from MultiTargetOptimizer
            df: Historical data for fitting
            data_dir: Directory for data cache
            logger: Optional logger

        Returns:
            Fitted MultiTargetPredictor
        """
        predictors = {}

        for target, result in optimization_result.results.items():
            # Create and fit predictor for this target
            predictor = RegimePredictor(
                chromosome=result.ga_result.best_chromosome,
                target_return=target,
                horizon=optimization_result.horizon,
                data_dir=data_dir,
                logger=logger
            )
            predictor.fit(df)
            predictors[target] = predictor

        return cls(
            predictors=predictors,
            horizon=optimization_result.horizon,
            logger=logger
        )

    def predict_current(self, ticker: str) -> pd.DataFrame:
        """
        Predict probabilities for all targets.

        Args:
            ticker: Ticker symbol

        Returns:
            DataFrame with probability matrix
        """
        rows = []

        for target in self.targets:
            predictor = self.predictors[target]
            result = predictor.predict_current(ticker)

            rows.append(ProbabilityMatrixRow(
                target=target,
                target_pct=f"{target*100:.1f}%",
                probability=result.probability,
                probability_unconditional=result.probability_unconditional,
                delta=result.probability - result.probability_unconditional,
                cluster=result.cluster,
                interpretation=result.interpretation,
                data_is_fresh=result.data_is_fresh
            ))

        df = pd.DataFrame([vars(r) for r in rows])
        return df

    def print_matrix(self, ticker: str, result: Optional[pd.DataFrame] = None) -> None:
        """
        Print formatted probability matrix.

        Args:
            ticker: Ticker symbol
            result: Pre-computed result (optional)
        """
        if result is None:
            result = self.predict_current(ticker)

        # Get current price from first predictor
        first_predictor = self.predictors[self.targets[0]]
        current_result = first_predictor.predict_current(ticker)
        current_price = current_result.price
        current_date = current_result.date

        print("=" * 70)
        print("MATRIZ DE PROBABILIDADES")
        print("=" * 70)
        print(f"Ticker: {ticker}")
        print(f"Horizonte: {self.horizon} dias")
        print(f"Data: {current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else current_date}")
        print(f"Preço Atual: R$ {current_price:.2f}")
        print("=" * 70)

        print(f"\n{'Target':<10} {'P(atual)':<12} {'P(base)':<12} {'Delta':<10} {'Regime':<10}")
        print("-" * 54)

        for _, row in result.iterrows():
            delta_str = f"+{row['delta']:.1%}" if row['delta'] >= 0 else f"{row['delta']:.1%}"
            print(f"{row['target_pct']:<10} {row['probability']:.1%}        "
                  f"{row['probability_unconditional']:.1%}        "
                  f"{delta_str:<10} {row['interpretation']:<10}")

        # Find best and worst
        best_idx = result['delta'].idxmax()
        worst_idx = result['delta'].idxmin()
        best = result.loc[best_idx]
        worst = result.loc[worst_idx]

        print()
        if best['delta'] > 0:
            print(f"✅ Melhor oportunidade: {best['target_pct']} ({best['delta']:+.1%} vs média)")
        if worst['delta'] < 0:
            print(f"⚠️  Pior: {worst['target_pct']} ({worst['delta']:+.1%} vs média)")

        if not result['data_is_fresh'].all():
            print(f"\n⏰ Aviso: Alguns dados não estão atualizados")

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
            'horizon': self.horizon,
            'targets': self.targets,
            'n_targets': len(self.targets)
        }
        with open(path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        # Save each predictor
        for target, predictor in self.predictors.items():
            target_str = f"{target:.4f}".replace('.', '_')
            predictor_path = path / f"predictor_{target_str}"
            predictor.save(str(predictor_path))

        self.logger.info("MultiTargetPredictor saved", path=str(path))

    @classmethod
    def load(
        cls,
        path: str,
        data_dir: str = "data",
        logger: Optional[ILogger] = None
    ) -> 'MultiTargetPredictor':
        """
        Load predictor from directory.

        Args:
            path: Base directory path
            data_dir: Directory for data cache
            logger: Optional logger

        Returns:
            Loaded MultiTargetPredictor
        """
        path = Path(path)

        # Load manifest
        with open(path / 'manifest.json', 'r') as f:
            manifest = json.load(f)

        # Load each predictor
        predictors = {}
        for target in manifest['targets']:
            target_str = f"{target:.4f}".replace('.', '_')
            predictor_path = path / f"predictor_{target_str}"
            predictors[target] = RegimePredictor.load(
                str(predictor_path),
                data_dir=data_dir,
                logger=logger
            )

        return cls(
            predictors=predictors,
            horizon=manifest['horizon'],
            logger=logger
        )
