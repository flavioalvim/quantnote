"""Fitness evaluator for genetic algorithm."""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from .chromosome import Chromosome
from .calculator_factory import CalculatorFactory
from .walk_forward_validator import WalkForwardValidator, WalkForwardResult
from .return_strategy import IReturnStrategy, DEFAULT_CLOSE_STRATEGY


@dataclass
class FitnessResult:
    """Complete fitness evaluation result."""
    fitness: float
    delta_p: float
    delta_p_test: float
    stability_score: float
    overfitting_ratio: float
    walk_forward: Optional[WalkForwardResult]
    error: Optional[str] = None


class FitnessEvaluator:
    """
    Evaluates chromosome fitness using walk-forward validation.
    Prevents overfitting by measuring out-of-sample performance.
    Includes caching to avoid redundant evaluations.

    Uses Strategy Pattern for return type selection (close vs touch).
    """

    def __init__(
        self,
        df_base: pd.DataFrame,
        target_return: float,
        horizon: int,
        stability_penalty: float = 0.1,
        overfitting_penalty: float = 0.2,
        use_walk_forward: bool = True,
        n_folds: int = 5,
        strategy: Optional[IReturnStrategy] = None,
        logger: Optional[ILogger] = None,
        use_cache: bool = True
    ):
        """
        Initialize fitness evaluator with fixed prediction parameters.

        Args:
            df_base: Base DataFrame with OHLCV data
            target_return: Target return to predict (e.g., 0.05 for 5%)
            horizon: Number of periods for future return
            stability_penalty: Penalty for regime instability
            overfitting_penalty: Penalty for overfitting
            use_walk_forward: Use walk-forward validation
            n_folds: Number of validation folds
            strategy: Return strategy (close or touch). Defaults to close.
            logger: Optional logger
            use_cache: Enable evaluation caching
        """
        self.df_base = df_base
        self.target_return = target_return
        self.horizon = horizon
        self.stability_penalty = stability_penalty
        self.overfitting_penalty = overfitting_penalty
        self.use_walk_forward = use_walk_forward
        self.n_folds = n_folds
        self.strategy = strategy or DEFAULT_CLOSE_STRATEGY
        self.logger = logger or NullLogger()
        self.use_cache = use_cache

        # Cache for evaluated chromosomes
        self._cache: Dict[str, FitnessResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self.factory = CalculatorFactory(horizon=horizon, strategy=self.strategy)
        self.validator = WalkForwardValidator(
            target_return=target_return,
            horizon=horizon,
            n_folds=n_folds,
            strategy=self.strategy,
            logger=logger
        )

    def evaluate(self, chromosome: Chromosome) -> FitnessResult:
        """
        Evaluate a chromosome.

        The fitness function balances:
        - Regime separation (delta_p)
        - Out-of-sample performance (walk-forward)
        - Stability (penalize frequent regime changes)
        - Overfitting (penalize train >> test performance)
        """
        # Check cache first
        if self.use_cache:
            cache_key = chromosome.cache_key()
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            self._cache_misses += 1

        try:
            if self.use_walk_forward:
                result = self._evaluate_with_walk_forward(chromosome)
            else:
                result = self._evaluate_simple(chromosome)

            # Store in cache
            if self.use_cache:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(
                "Fitness evaluation failed",
                exception=e,
                chromosome=chromosome.to_dict()
            )
            return FitnessResult(
                fitness=-1.0,
                delta_p=0.0,
                delta_p_test=0.0,
                stability_score=1.0,
                overfitting_ratio=float('inf'),
                walk_forward=None,
                error=str(e)
            )

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._cache),
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _evaluate_with_walk_forward(self, chromosome: Chromosome) -> FitnessResult:
        """Evaluate using walk-forward validation."""
        wf_result = self.validator.validate(self.df_base, chromosome)

        # Primary metric: out-of-sample delta_p
        delta_p_test = wf_result.mean_delta_p_test

        # Penalties
        stability_penalty = self.stability_penalty * wf_result.mean_stability

        # Overfitting penalty (if train >> test)
        overfit_penalty = 0.0
        if wf_result.overfitting_ratio > 1.5:
            overfit_penalty = self.overfitting_penalty * (wf_result.overfitting_ratio - 1.0)

        # Consistency bonus (low std across folds)
        consistency_bonus = max(0, 0.1 - wf_result.std_delta_p_test)

        # Final fitness
        fitness = delta_p_test - stability_penalty - overfit_penalty + consistency_bonus

        self.logger.debug(
            "Fitness calculated",
            fitness=fitness,
            delta_p_test=delta_p_test,
            stability_penalty=stability_penalty,
            overfit_penalty=overfit_penalty,
            consistency_bonus=consistency_bonus
        )

        return FitnessResult(
            fitness=fitness,
            delta_p=delta_p_test,  # Use test as primary
            delta_p_test=delta_p_test,
            stability_score=wf_result.mean_stability,
            overfitting_ratio=wf_result.overfitting_ratio,
            walk_forward=wf_result
        )

    def _evaluate_simple(self, chromosome: Chromosome) -> FitnessResult:
        """Simple evaluation without walk-forward (faster but may overfit)."""
        from ..analysis.kmeans_regimes import KMeansRegimeClassifier
        from ..analysis.probability_calculator import ProbabilityCalculator

        pipeline = self.factory.create_pipeline(chromosome)
        feature_cols = self.factory.get_feature_columns(chromosome)
        # Use strategy to get appropriate return column
        future_col = self.factory.get_return_column(self.target_return)

        df_processed = pipeline.run(self.df_base)

        kmeans = KMeansRegimeClassifier(
            n_clusters=chromosome.n_clusters,
            feature_columns=feature_cols
        )
        df_with_clusters = kmeans.fit_predict(df_processed)

        prob_calc = ProbabilityCalculator(
            future_return_column=future_col,
            target_return=self.target_return,
            regime_column='cluster'
        )

        cond_probs = prob_calc.calculate_conditional_probabilities(df_with_clusters)
        separation = prob_calc.calculate_separation_metrics(cond_probs)

        # Stability
        valid_clusters = df_with_clusters['cluster'].dropna()
        num_changes = (valid_clusters != valid_clusters.shift()).sum()
        stability = num_changes / len(valid_clusters)

        fitness = separation.delta_p - self.stability_penalty * stability

        return FitnessResult(
            fitness=fitness,
            delta_p=separation.delta_p,
            delta_p_test=separation.delta_p,  # Same as train in simple mode
            stability_score=stability,
            overfitting_ratio=1.0,
            walk_forward=None
        )
