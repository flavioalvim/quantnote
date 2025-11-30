"""Dual probability calculator for close and touch probabilities."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from ..domain.value_objects import Probability
from .probability_calculator import ProbabilityCalculator, ConditionalProbability, SeparationMetrics


@dataclass
class DualProbabilityResult:
    """Result containing both close and touch probabilities."""
    regime: str
    prob_close: float  # P(close >= target at t+H)
    prob_touch: float  # P(touch target at any point in [t+1, t+H])
    count: int
    mean_return_close: float
    mean_return_touch: float


@dataclass
class DualSeparationMetrics:
    """Separation metrics for both close and touch."""
    # Close metrics
    close_delta_p: float
    close_max_probability: float
    close_min_probability: float

    # Touch metrics
    touch_delta_p: float
    touch_max_probability: float
    touch_min_probability: float

    # Comparison
    touch_vs_close_ratio: float  # How much more likely is touch vs close?


class DualProbabilityCalculator:
    """
    Calculates both close and touch probabilities for comparison.

    Close: P(return at t+H >= target)
    Touch: P(max return in [t+1, t+H] >= target) for upside
           P(min return in [t+1, t+H] <= target) for downside
    """

    def __init__(
        self,
        close_return_column: str,
        touch_return_column: str,
        target_return: float,
        regime_column: str = 'regime'
    ):
        """
        Initialize dual calculator.

        Args:
            close_return_column: Column with future close return (e.g., 'log_return_future_7')
            touch_return_column: Column with touch return (e.g., 'log_return_touch_max_7')
            target_return: Target return as percentage (e.g., 0.05 for 5%)
            regime_column: Column containing regime labels
        """
        self.close_return_column = close_return_column
        self.touch_return_column = touch_return_column
        self.target_return = target_return
        self.regime_column = regime_column

        # Convert target to log return
        self.log_target = np.log(1 + target_return)

        # Internal calculators
        self._close_calc = ProbabilityCalculator(
            future_return_column=close_return_column,
            target_return=target_return,
            regime_column=regime_column
        )
        self._touch_calc = ProbabilityCalculator(
            future_return_column=touch_return_column,
            target_return=target_return,
            regime_column=regime_column
        )

    def calculate_raw_probabilities(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate unconditional close and touch probabilities."""
        close_prob = self._close_calc.calculate_raw_probability(df)
        touch_prob = self._touch_calc.calculate_raw_probability(df)

        return {
            'prob_close': close_prob.value,
            'prob_touch': touch_prob.value,
            'touch_vs_close_ratio': touch_prob.value / close_prob.value if close_prob.value > 0 else float('inf')
        }

    def calculate_conditional_probabilities(
        self,
        df: pd.DataFrame
    ) -> Dict[str, DualProbabilityResult]:
        """Calculate close and touch probabilities for each regime."""
        close_probs = self._close_calc.calculate_conditional_probabilities(df)
        touch_probs = self._touch_calc.calculate_conditional_probabilities(df)

        results = {}
        for regime in close_probs.keys():
            close_cp = close_probs[regime]
            touch_cp = touch_probs.get(regime)

            results[regime] = DualProbabilityResult(
                regime=regime,
                prob_close=close_cp.probability.value,
                prob_touch=touch_cp.probability.value if touch_cp else 0,
                count=close_cp.count,
                mean_return_close=close_cp.mean_return,
                mean_return_touch=touch_cp.mean_return if touch_cp else 0
            )

        return results

    def calculate_separation_metrics(
        self,
        dual_probs: Dict[str, DualProbabilityResult]
    ) -> DualSeparationMetrics:
        """Calculate separation metrics for both close and touch."""
        close_probs = [dp.prob_close for dp in dual_probs.values()]
        touch_probs = [dp.prob_touch for dp in dual_probs.values()]

        close_delta = max(close_probs) - min(close_probs)
        touch_delta = max(touch_probs) - min(touch_probs)

        # Average touch vs close ratio
        avg_touch = np.mean(touch_probs)
        avg_close = np.mean(close_probs)
        ratio = avg_touch / avg_close if avg_close > 0 else float('inf')

        return DualSeparationMetrics(
            close_delta_p=close_delta,
            close_max_probability=max(close_probs),
            close_min_probability=min(close_probs),
            touch_delta_p=touch_delta,
            touch_max_probability=max(touch_probs),
            touch_min_probability=min(touch_probs),
            touch_vs_close_ratio=ratio
        )

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate complete dual probability analysis report."""
        raw_probs = self.calculate_raw_probabilities(df)
        cond_probs = self.calculate_conditional_probabilities(df)
        separation = self.calculate_separation_metrics(cond_probs)

        return {
            'target_return': self.target_return,
            'target_return_pct': f"{self.target_return * 100:.1f}%",
            'log_target': self.log_target,

            # Raw probabilities
            'raw_probabilities': {
                'close': raw_probs['prob_close'],
                'close_pct': f"{raw_probs['prob_close'] * 100:.1f}%",
                'touch': raw_probs['prob_touch'],
                'touch_pct': f"{raw_probs['prob_touch'] * 100:.1f}%",
                'touch_vs_close_ratio': raw_probs['touch_vs_close_ratio']
            },

            # Conditional probabilities by regime
            'conditional_probabilities': {
                regime: {
                    'prob_close': dp.prob_close,
                    'prob_close_pct': f"{dp.prob_close * 100:.1f}%",
                    'prob_touch': dp.prob_touch,
                    'prob_touch_pct': f"{dp.prob_touch * 100:.1f}%",
                    'touch_vs_close_ratio': dp.prob_touch / dp.prob_close if dp.prob_close > 0 else float('inf'),
                    'count': dp.count
                }
                for regime, dp in cond_probs.items()
            },

            # Separation metrics
            'separation_metrics': {
                'close': {
                    'delta_p': separation.close_delta_p,
                    'max_probability': separation.close_max_probability,
                    'min_probability': separation.close_min_probability
                },
                'touch': {
                    'delta_p': separation.touch_delta_p,
                    'max_probability': separation.touch_max_probability,
                    'min_probability': separation.touch_min_probability
                },
                'touch_vs_close_ratio': separation.touch_vs_close_ratio
            }
        }

    def print_comparison(self, df: pd.DataFrame) -> None:
        """Print a formatted comparison of close vs touch probabilities."""
        report = self.generate_report(df)

        print(f"\n{'='*60}")
        print(f"Probabilidade de atingir {report['target_return_pct']} de retorno")
        print(f"{'='*60}")

        print(f"\n📊 Probabilidades Gerais (sem condicionamento):")
        print(f"   P(fechar ≥ alvo)  = {report['raw_probabilities']['close_pct']}")
        print(f"   P(tocar o alvo)   = {report['raw_probabilities']['touch_pct']}")
        print(f"   Ratio touch/close = {report['raw_probabilities']['touch_vs_close_ratio']:.2f}x")

        print(f"\n📈 Probabilidades por Regime:")
        print(f"   {'Regime':<10} {'P(fechar)':<12} {'P(tocar)':<12} {'Ratio':<10} {'N':<8}")
        print(f"   {'-'*52}")

        for regime, data in report['conditional_probabilities'].items():
            print(f"   {regime:<10} {data['prob_close_pct']:<12} {data['prob_touch_pct']:<12} "
                  f"{data['touch_vs_close_ratio']:.2f}x      {data['count']:<8}")

        print(f"\n📐 Métricas de Separação:")
        print(f"   Delta P (close): {report['separation_metrics']['close']['delta_p']:.3f}")
        print(f"   Delta P (touch): {report['separation_metrics']['touch']['delta_p']:.3f}")
