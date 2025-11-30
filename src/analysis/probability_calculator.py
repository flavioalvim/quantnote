"""Probability calculator for regime-conditioned returns."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from ..domain.value_objects import Probability


@dataclass
class ConditionalProbability:
    """Probability conditioned on regime."""
    regime: str
    probability: Probability
    count: int
    mean_return: float
    std_return: float


@dataclass
class SeparationMetrics:
    """Metrics measuring separation between regimes."""
    delta_p: float  # Max prob - min prob
    delta_mean: float  # Max mean return - min mean return
    max_probability: float
    min_probability: float
    information_ratio: float  # delta_mean / pooled_std


class ProbabilityCalculator:
    """Calculates raw and conditional probabilities."""

    def __init__(
        self,
        future_return_column: str,
        target_return: float,
        regime_column: str = 'regime'
    ):
        self.future_return_column = future_return_column
        self.target_return = target_return
        self.regime_column = regime_column

        # Convert target to log return
        self.log_target = np.log(1 + target_return)

    def _count_hits(self, returns: pd.Series) -> int:
        """
        Count hits based on target direction.

        For positive targets: count returns > target (appreciation)
        For negative targets: count returns < target (depreciation)
        """
        if self.log_target >= 0:
            return (returns > self.log_target).sum()
        else:
            return (returns < self.log_target).sum()

    def calculate_raw_probability(self, df: pd.DataFrame) -> Probability:
        """Calculate unconditional probability."""
        valid_returns = df[self.future_return_column].dropna()

        if len(valid_returns) == 0:
            raise ValueError("No valid returns to calculate probability")

        hits = self._count_hits(valid_returns)
        return Probability.from_frequency(hits, len(valid_returns))

    def calculate_conditional_probabilities(
        self,
        df: pd.DataFrame
    ) -> Dict[str, ConditionalProbability]:
        """Calculate probability for each regime."""
        results = {}

        valid_df = df.dropna(subset=[self.regime_column, self.future_return_column])

        for regime in valid_df[self.regime_column].unique():
            mask = valid_df[self.regime_column] == regime
            regime_returns = valid_df.loc[mask, self.future_return_column]

            hits = self._count_hits(regime_returns)
            total = len(regime_returns)

            results[str(regime)] = ConditionalProbability(
                regime=str(regime),
                probability=Probability.from_frequency(hits, total),
                count=total,
                mean_return=regime_returns.mean(),
                std_return=regime_returns.std()
            )

        return results

    def calculate_separation_metrics(
        self,
        conditional_probs: Dict[str, ConditionalProbability]
    ) -> SeparationMetrics:
        """Calculate metrics measuring regime separation."""
        probs = [cp.probability.value for cp in conditional_probs.values()]
        means = [cp.mean_return for cp in conditional_probs.values()]
        stds = [cp.std_return for cp in conditional_probs.values()]
        counts = [cp.count for cp in conditional_probs.values()]

        # Pooled standard deviation
        total_count = sum(counts)
        if total_count <= len(counts):
            pooled_std = np.mean(stds) if stds else 0
        else:
            pooled_var = sum(
                (n - 1) * s**2 for n, s in zip(counts, stds) if n > 1
            ) / (total_count - len(counts))
            pooled_std = np.sqrt(pooled_var)

        delta_mean = max(means) - min(means)
        info_ratio = delta_mean / pooled_std if pooled_std > 0 else 0

        return SeparationMetrics(
            delta_p=max(probs) - min(probs),
            delta_mean=delta_mean,
            max_probability=max(probs),
            min_probability=min(probs),
            information_ratio=info_ratio
        )

    def generate_report(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """Generate complete probability analysis report."""
        raw_prob = self.calculate_raw_probability(df)
        cond_probs = self.calculate_conditional_probabilities(df)
        separation = self.calculate_separation_metrics(cond_probs)

        return {
            'target_return': self.target_return,
            'log_target': self.log_target,
            'raw_probability': raw_prob.value,
            'raw_probability_pct': raw_prob.to_percent(),
            'conditional_probabilities': {
                regime: {
                    'probability': cp.probability.value,
                    'probability_pct': cp.probability.to_percent(),
                    'count': cp.count,
                    'mean_return': cp.mean_return,
                    'std_return': cp.std_return
                }
                for regime, cp in cond_probs.items()
            },
            'separation_metrics': {
                'delta_p': separation.delta_p,
                'delta_mean': separation.delta_mean,
                'max_probability': separation.max_probability,
                'min_probability': separation.min_probability,
                'information_ratio': separation.information_ratio
            }
        }
