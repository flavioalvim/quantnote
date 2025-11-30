"""Strike grid optimizer for options pricing."""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np

from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from config.search_space import GAConfig
from .genetic_algorithm import GeneticAlgorithm, GAResult
from .chromosome import Chromosome
from .multi_target_optimizer import MultiTargetResult


@dataclass
class StrikeTarget:
    """Maps a strike price to its target return."""
    strike: float
    target_return: float  # (strike / current_price) - 1

    @property
    def target_pct(self) -> str:
        return f"{self.target_return*100:+.2f}%"


@dataclass
class StrikeGridOptimizationResult:
    """Complete result of strike grid optimization."""
    current_price: float
    horizon: int
    strike_targets: Dict[float, StrikeTarget]  # strike -> StrikeTarget
    results: Dict[float, MultiTargetResult]    # strike -> optimization result

    def get_best_chromosome(self, strike: float) -> Chromosome:
        """Get best chromosome for a specific strike."""
        return self.results[strike].ga_result.best_chromosome

    def get_target_return(self, strike: float) -> float:
        """Get target return for a strike."""
        return self.strike_targets[strike].target_return

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        rows = []
        for strike in sorted(self.results.keys()):
            result = self.results[strike]
            st = self.strike_targets[strike]
            best = result.ga_result.best_chromosome
            metrics = result.ga_result.best_metrics

            rows.append({
                'strike': strike,
                'target_return': st.target_return,
                'target_pct': st.target_pct,
                'fitness': result.ga_result.best_fitness,
                'delta_p_test': metrics.delta_p_test,
                'overfitting_ratio': metrics.overfitting_ratio,
                'window_slope': best.window_slope,
                'window_volatility': best.window_volatility,
                'n_clusters': best.n_clusters,
                'evaluations': result.ga_result.all_evaluations
            })
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Save results to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest = {
            'current_price': self.current_price,
            'horizon': self.horizon,
            'strikes': list(self.results.keys()),
            'n_strikes': len(self.results),
            'strike_targets': {
                str(k): {'strike': v.strike, 'target_return': v.target_return}
                for k, v in self.strike_targets.items()
            }
        }
        with open(path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        # Save each result
        for strike, result in self.results.items():
            strike_str = f"{strike:.2f}".replace('.', '_')
            result_path = path / f"strike_{strike_str}.json"
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'StrikeGridOptimizationResult':
        """Load results from directory."""
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

        # Load each result
        results = {}
        for strike in manifest['strikes']:
            strike_str = f"{strike:.2f}".replace('.', '_')
            result_path = path / f"strike_{strike_str}.json"

            with open(result_path, 'r') as f:
                data = json.load(f)

            chromosome = Chromosome.from_dict(data['best_chromosome'])

            from .fitness import FitnessResult
            metrics = FitnessResult(
                fitness=data['best_metrics']['fitness'],
                delta_p=data['best_metrics']['delta_p'],
                delta_p_test=data['best_metrics']['delta_p_test'],
                stability_score=data['best_metrics']['stability_score'],
                overfitting_ratio=data['best_metrics']['overfitting_ratio'],
                walk_forward=None,
                error=None
            )

            ga_result = GAResult(
                best_chromosome=chromosome,
                best_fitness=data['best_fitness'],
                best_metrics=metrics,
                history=[],
                all_evaluations=data['all_evaluations']
            )

            results[strike] = MultiTargetResult(
                target=data['target'],
                ga_result=ga_result
            )

        return cls(
            current_price=manifest['current_price'],
            horizon=manifest['horizon'],
            strike_targets=strike_targets,
            results=results
        )


class StrikeGridOptimizer:
    """
    Optimizes GA for a grid of strike prices.

    Converts strikes to target returns based on current price,
    then runs independent GAs for each strike.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        current_price: float,
        strikes: List[float],
        horizon: int,
        ga_config: GAConfig,
        logger: Optional[ILogger] = None
    ):
        """
        Initialize strike grid optimizer.

        Args:
            df: Historical OHLCV data
            current_price: Current asset price (for target calculation)
            strikes: List of strike prices (e.g., [100, 105, 110, ...])
            horizon: Prediction horizon in days
            ga_config: Base GA configuration
            logger: Optional logger
        """
        self.df = df
        self.current_price = current_price
        self.strikes = sorted(strikes)
        self.horizon = horizon
        self.ga_config = ga_config
        self.logger = logger or NullLogger()

        # Calculate target returns for each strike
        self.strike_targets: Dict[float, StrikeTarget] = {}
        for strike in self.strikes:
            target_return = (strike / current_price) - 1
            self.strike_targets[strike] = StrikeTarget(
                strike=strike,
                target_return=target_return
            )

        # Results storage
        self.results: Dict[float, MultiTargetResult] = {}

    def get_strike_target(self, strike: float) -> StrikeTarget:
        """Get target information for a strike."""
        return self.strike_targets[strike]

    def preview_targets(self) -> pd.DataFrame:
        """Preview strike-to-target mapping before running."""
        rows = []
        for strike in self.strikes:
            st = self.strike_targets[strike]
            rows.append({
                'strike': strike,
                'target_return': st.target_return,
                'target_pct': st.target_pct,
                'direction': 'UP' if st.target_return > 0 else 'DOWN' if st.target_return < 0 else 'ATM'
            })
        return pd.DataFrame(rows)

    def run(
        self,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        ga_progress_callback_factory: Optional[Callable[[int, float], 'LiveProgressCallback']] = None,
        parallel_ga: bool = True
    ) -> StrikeGridOptimizationResult:
        """
        Run optimization for all strikes.

        Args:
            verbose: Print progress
            progress_callback: Callback(completed, total, current_strike) for strike-level progress
            ga_progress_callback_factory: Factory function(strike_idx, strike) that returns a
                                          progress callback for each individual GA run.
                                          If None, no intra-GA progress is shown.
            parallel_ga: Use parallel evaluation within each GA (default True)

        Returns:
            StrikeGridOptimizationResult with all results
        """
        n_strikes = len(self.strikes)

        if verbose:
            print("=" * 70)
            print("STRIKE GRID OPTIMIZATION")
            print("=" * 70)
            print(f"Preço atual: R$ {self.current_price:.2f}")
            print(f"Horizonte: {self.horizon} dias")
            print(f"Strikes: {n_strikes} ({min(self.strikes):.2f} a {max(self.strikes):.2f})")
            print(f"GA gerações: {self.ga_config.generations}")
            print(f"População: {self.ga_config.population_size}")
            print("=" * 70)

            # Show target preview
            print("\nMapeamento Strike → Target:")
            print(f"{'Strike':<12} {'Retorno':<12} {'Direção':<10}")
            print("-" * 34)
            for strike in self.strikes[:5]:  # Show first 5
                st = self.strike_targets[strike]
                direction = 'UP' if st.target_return > 0 else 'DOWN' if st.target_return < 0 else 'ATM'
                print(f"R$ {strike:<9.2f} {st.target_pct:<12} {direction:<10}")
            if len(self.strikes) > 5:
                print(f"... ({len(self.strikes) - 5} mais)")
            print()

        # Run GA for each strike
        for i, strike in enumerate(self.strikes):
            st = self.strike_targets[strike]

            if verbose:
                print(f"\n{'='*70}")
                print(f"[{i+1}/{n_strikes}] Strike R$ {strike:.2f} (target {st.target_pct})")
                print(f"{'='*70}")

            # Create config for this target
            config = self.ga_config.model_copy(update={
                'target_return': st.target_return,
                'horizon': self.horizon
            })

            # Create GA progress callback if factory provided
            ga_callback = None
            if ga_progress_callback_factory is not None:
                ga_callback = ga_progress_callback_factory(i, strike)

            # Run GA
            ga = GeneticAlgorithm(
                self.df,
                config,
                logger=self.logger,
                progress_callback=ga_callback
            )
            result = ga.run(verbose=False, parallel=parallel_ga)

            # Finalize callback if it has finalize method
            if ga_callback is not None and hasattr(ga_callback, 'finalize'):
                ga_callback.finalize()

            self.results[strike] = MultiTargetResult(
                target=st.target_return,
                ga_result=result
            )

            if verbose:
                print(f"\n✅ Strike R$ {strike:.2f} concluído:")
                print(f"    Fitness: {result.best_fitness:.4f}")
                print(f"    Delta P (test): {result.best_metrics.delta_p_test:.4f}")
                print(f"    Parâmetros: slope={result.best_chromosome.window_slope}, "
                      f"clusters={result.best_chromosome.n_clusters}")

            if progress_callback:
                progress_callback(i + 1, n_strikes, strike)

        if verbose:
            print(f"\n{'='*70}")
            print(f"OTIMIZAÇÃO CONCLUÍDA - {n_strikes} strikes processados")
            print(f"{'='*70}")

        return StrikeGridOptimizationResult(
            current_price=self.current_price,
            horizon=self.horizon,
            strike_targets=self.strike_targets,
            results=self.results
        )

    def save(self, path: str) -> None:
        """Save current results."""
        if not self.results:
            raise RuntimeError("No results to save. Run optimization first.")

        result = StrikeGridOptimizationResult(
            current_price=self.current_price,
            horizon=self.horizon,
            strike_targets=self.strike_targets,
            results=self.results
        )
        result.save(path)
        self.logger.info("Results saved", path=path)
