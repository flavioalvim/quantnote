"""Multi-target optimizer for running GA across multiple target returns."""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from config.search_space import GAConfig, TargetGrid
from .genetic_algorithm import GeneticAlgorithm, GAResult
from .chromosome import Chromosome


@dataclass
class MultiTargetResult:
    """Result of multi-target optimization."""
    target: float
    ga_result: GAResult

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            'target': self.target,
            'best_chromosome': self.ga_result.best_chromosome.to_dict(),
            'best_fitness': self.ga_result.best_fitness,
            'best_metrics': {
                'fitness': self.ga_result.best_metrics.fitness,
                'delta_p': self.ga_result.best_metrics.delta_p,
                'delta_p_test': self.ga_result.best_metrics.delta_p_test,
                'stability_score': self.ga_result.best_metrics.stability_score,
                'overfitting_ratio': self.ga_result.best_metrics.overfitting_ratio,
            },
            'all_evaluations': self.ga_result.all_evaluations
        }


@dataclass
class MultiTargetOptimizationResult:
    """Complete result of multi-target optimization."""
    horizon: int
    results: Dict[float, MultiTargetResult]

    def get_best_chromosome(self, target: float) -> Chromosome:
        """Get best chromosome for a specific target."""
        return self.results[target].ga_result.best_chromosome

    def get_fitness(self, target: float) -> float:
        """Get best fitness for a specific target."""
        return self.results[target].ga_result.best_fitness

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        rows = []
        for target, result in sorted(self.results.items()):
            best = result.ga_result.best_chromosome
            metrics = result.ga_result.best_metrics
            rows.append({
                'target': target,
                'target_pct': f"{target*100:.1f}%",
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
            'horizon': self.horizon,
            'targets': list(self.results.keys()),
            'n_targets': len(self.results)
        }
        with open(path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        # Save each result
        for target, result in self.results.items():
            target_str = f"{target:.4f}".replace('.', '_')
            result_path = path / f"target_{target_str}.json"
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MultiTargetOptimizationResult':
        """Load results from directory."""
        path = Path(path)

        # Load manifest
        with open(path / 'manifest.json', 'r') as f:
            manifest = json.load(f)

        # Load each result
        results = {}
        for target in manifest['targets']:
            target_str = f"{target:.4f}".replace('.', '_')
            result_path = path / f"target_{target_str}.json"

            with open(result_path, 'r') as f:
                data = json.load(f)

            chromosome = Chromosome.from_dict(data['best_chromosome'])

            # Reconstruct GAResult (simplified - without full history)
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
                history=[],  # Not saved
                all_evaluations=data['all_evaluations']
            )

            results[target] = MultiTargetResult(target=target, ga_result=ga_result)

        return cls(horizon=manifest['horizon'], results=results)


def _run_single_ga(args) -> MultiTargetResult:
    """Worker function for parallel GA execution."""
    target, df, base_config_dict, verbose = args

    # Reconstruct config with specific target
    from config.search_space import GAConfig
    config = GAConfig(**base_config_dict)
    config = config.model_copy(update={'target_return': target})

    # Run GA
    ga = GeneticAlgorithm(df, config, logger=None)
    result = ga.run(verbose=verbose, parallel=False)  # No nested parallelism

    return MultiTargetResult(target=target, ga_result=result)


class MultiTargetOptimizer:
    """
    Optimizes GA for multiple target returns.

    Runs independent GAs for each target, either sequentially or in parallel.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        horizon: int,
        target_grid: TargetGrid,
        ga_config: GAConfig,
        logger: Optional[ILogger] = None
    ):
        """
        Initialize multi-target optimizer.

        Args:
            df: Historical OHLCV data
            horizon: Prediction horizon in days
            target_grid: Grid of target returns to optimize
            ga_config: Base GA configuration (target_return will be overridden)
            logger: Optional logger
        """
        self.df = df
        self.horizon = horizon
        self.target_grid = target_grid
        self.ga_config = ga_config
        self.logger = logger or NullLogger()

        # Get targets from grid
        self.targets = target_grid.to_array()

        # Results storage
        self.results: Dict[float, MultiTargetResult] = {}

    def run(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        ga_progress_callback_factory: Optional[Callable[[int, float], Any]] = None,
        parallel_ga: bool = True
    ) -> MultiTargetOptimizationResult:
        """
        Run optimization for all targets.

        Args:
            parallel: Run GAs in parallel (default False - each GA already uses parallelism)
            max_workers: Number of parallel workers (if parallel=True)
            verbose: Print progress
            progress_callback: Callback(completed, total, current_target) for target-level progress
            ga_progress_callback_factory: Factory function(target_idx, target) that returns a
                                          progress callback for each individual GA run.
                                          If None, no intra-GA progress is shown.
            parallel_ga: Use parallel evaluation within each GA (default True)

        Returns:
            MultiTargetOptimizationResult with all results
        """
        n_targets = len(self.targets)

        if verbose:
            print(f"=" * 60)
            print(f"MULTI-TARGET OPTIMIZATION")
            print(f"=" * 60)
            print(f"Horizon: {self.horizon} days")
            print(f"Targets: {n_targets} ({self.target_grid.target_min:.1%} to {self.target_grid.target_max:.1%})")
            print(f"GA generations: {self.ga_config.generations}")
            print(f"Population size: {self.ga_config.population_size}")
            print(f"=" * 60)

        if parallel:
            self._run_parallel(max_workers, verbose, progress_callback)
        else:
            self._run_sequential(verbose, progress_callback, ga_progress_callback_factory, parallel_ga)

        return MultiTargetOptimizationResult(
            horizon=self.horizon,
            results=self.results
        )

    def _run_sequential(
        self,
        verbose: bool,
        progress_callback: Optional[Callable],
        ga_progress_callback_factory: Optional[Callable] = None,
        parallel_ga: bool = True
    ) -> None:
        """Run GAs sequentially."""
        n_targets = len(self.targets)

        for i, target in enumerate(self.targets):
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{i+1}/{n_targets}] Optimizing for target {target:.2%}")
                print(f"{'='*60}")

            # Create config for this target
            config = self.ga_config.model_copy(update={
                'target_return': target,
                'horizon': self.horizon
            })

            # Create GA progress callback if factory provided
            ga_callback = None
            if ga_progress_callback_factory is not None:
                ga_callback = ga_progress_callback_factory(i, target)

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

            self.results[target] = MultiTargetResult(target=target, ga_result=result)

            if verbose:
                print(f"\n✅ Target {target:.2%} concluído:")
                print(f"    Fitness: {result.best_fitness:.4f}")
                print(f"    Delta P (test): {result.best_metrics.delta_p_test:.4f}")
                print(f"    Parâmetros: slope={result.best_chromosome.window_slope}, "
                      f"clusters={result.best_chromosome.n_clusters}")

            if progress_callback:
                progress_callback(i + 1, n_targets, target)

        if verbose:
            print(f"\n{'='*60}")
            print(f"OTIMIZAÇÃO CONCLUÍDA - {n_targets} targets processados")
            print(f"{'='*60}")

    def _run_parallel(
        self,
        max_workers: Optional[int],
        verbose: bool,
        progress_callback: Optional[Callable]
    ) -> None:
        """Run GAs in parallel."""
        n_workers = max_workers or max(1, mp.cpu_count() - 1)
        n_targets = len(self.targets)

        if verbose:
            print(f"Running {n_targets} GAs in parallel with {n_workers} workers...")

        # Prepare arguments
        base_config_dict = self.ga_config.model_dump()
        base_config_dict['horizon'] = self.horizon

        args_list = [
            (target, self.df, base_config_dict, False)
            for target in self.targets
        ]

        # Run in parallel
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_single_ga, args): args[0]
                for args in args_list
            }

            for future in as_completed(futures):
                result = future.result()
                self.results[result.target] = result
                completed += 1

                if verbose:
                    print(f"[{completed}/{n_targets}] Target {result.target:.2%} "
                          f"- Fitness: {result.ga_result.best_fitness:.4f}")

                if progress_callback:
                    progress_callback(completed, n_targets, result.target)

    def save(self, path: str) -> None:
        """Save current results."""
        if not self.results:
            raise RuntimeError("No results to save. Run optimization first.")

        result = MultiTargetOptimizationResult(
            horizon=self.horizon,
            results=self.results
        )
        result.save(path)
        self.logger.info("Results saved", path=path)
