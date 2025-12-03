"""Walk-forward validator to prevent overfitting."""
import pandas as pd
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from ..analysis.time_series_splitter import TimeSeriesSplitter, Split
from ..analysis.kmeans_regimes import KMeansRegimeClassifier
from ..analysis.probability_calculator import ProbabilityCalculator
from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from .chromosome import Chromosome
from .calculator_factory import CalculatorFactory
from .return_strategy import IReturnStrategy, DEFAULT_CLOSE_STRATEGY


@dataclass
class FoldResult:
    """Result for a single fold."""
    fold: int
    train_size: int
    test_size: int
    delta_p_train: float
    delta_p_test: float
    stability_score: float


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation result."""
    mean_delta_p_test: float
    std_delta_p_test: float
    mean_stability: float
    overfitting_ratio: float  # train_perf / test_perf
    fold_results: List[FoldResult]


class WalkForwardValidator:
    """
    Walk-forward validation to prevent overfitting.
    Tests parameter generalization across multiple time periods.

    Uses Strategy Pattern for return type selection (close vs touch).
    """

    def __init__(
        self,
        target_return: float,
        horizon: int,
        n_folds: int = 5,
        min_train_size: int = 252,
        strategy: Optional[IReturnStrategy] = None,
        logger: Optional[ILogger] = None
    ):
        """
        Initialize validator with fixed prediction parameters.

        Args:
            target_return: Target return to predict (e.g., 0.05 for 5%)
            horizon: Number of periods for future return
            n_folds: Number of walk-forward folds
            min_train_size: Minimum training set size
            strategy: Return strategy (close or touch). Defaults to close.
            logger: Optional logger
        """
        self.target_return = target_return
        self.horizon = horizon
        self.n_folds = n_folds
        self.min_train_size = min_train_size
        self.strategy = strategy or DEFAULT_CLOSE_STRATEGY
        self.logger = logger or NullLogger()
        self.splitter = TimeSeriesSplitter()
        self.factory = CalculatorFactory(horizon=horizon, strategy=self.strategy)

    def validate(
        self,
        df: pd.DataFrame,
        chromosome: Chromosome
    ) -> WalkForwardResult:
        """
        Perform walk-forward validation.

        Args:
            df: Base DataFrame with OHLCV data
            chromosome: Parameters to validate

        Returns:
            WalkForwardResult with aggregated metrics
        """
        fold_results = []

        for split in self.splitter.walk_forward_split(
            df, self.n_folds, self.min_train_size
        ):
            result = self._evaluate_fold(split, chromosome)
            fold_results.append(result)

            self.logger.debug(
                "Fold evaluated",
                fold=split.fold,
                delta_p_train=result.delta_p_train,
                delta_p_test=result.delta_p_test
            )

        # Aggregate results
        train_deltas = [r.delta_p_train for r in fold_results]
        test_deltas = [r.delta_p_test for r in fold_results]

        mean_train = np.mean(train_deltas)
        mean_test = np.mean(test_deltas)

        return WalkForwardResult(
            mean_delta_p_test=mean_test,
            std_delta_p_test=np.std(test_deltas),
            mean_stability=np.mean([r.stability_score for r in fold_results]),
            overfitting_ratio=mean_train / mean_test if mean_test > 0 else float('inf'),
            fold_results=fold_results
        )

    def _evaluate_fold(
        self,
        split: Split,
        chromosome: Chromosome
    ) -> FoldResult:
        """Evaluate a single fold."""
        # Create pipeline
        pipeline = self.factory.create_pipeline(chromosome)
        feature_cols = self.factory.get_feature_columns(chromosome)
        # Use strategy to get appropriate return column
        future_col = self.factory.get_return_column(self.target_return)

        # Process train data
        train_processed = pipeline.run(split.train)

        # Fit K-Means on train
        kmeans = KMeansRegimeClassifier(
            n_clusters=chromosome.n_clusters,
            feature_columns=feature_cols
        )
        train_with_clusters = kmeans.fit_predict(train_processed)

        # Calculate train metrics
        prob_calc = ProbabilityCalculator(
            future_return_column=future_col,
            target_return=self.target_return,
            regime_column='cluster'
        )

        try:
            train_cond = prob_calc.calculate_conditional_probabilities(train_with_clusters)
            train_sep = prob_calc.calculate_separation_metrics(train_cond)
            delta_p_train = train_sep.delta_p
        except Exception:
            delta_p_train = 0.0

        # Process test data
        test_processed = pipeline.run(split.test)

        # Apply trained K-Means to test (no refitting!)
        test_with_clusters = kmeans.predict(test_processed)

        # Calculate test metrics
        try:
            test_cond = prob_calc.calculate_conditional_probabilities(test_with_clusters)
            test_sep = prob_calc.calculate_separation_metrics(test_cond)
            delta_p_test = test_sep.delta_p
        except Exception:
            delta_p_test = 0.0

        # Calculate stability (regime changes)
        valid_clusters = train_with_clusters['cluster'].dropna()
        num_changes = (valid_clusters != valid_clusters.shift()).sum()
        stability = num_changes / len(valid_clusters) if len(valid_clusters) > 0 else 1.0

        return FoldResult(
            fold=split.fold,
            train_size=len(split.train),
            test_size=len(split.test),
            delta_p_train=delta_p_train,
            delta_p_test=delta_p_test,
            stability_score=stability
        )
