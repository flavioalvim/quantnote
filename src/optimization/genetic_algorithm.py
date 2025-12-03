"""Genetic algorithm for parameter optimization."""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, asdict
import random
import sys
import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from config.search_space import GASearchSpace, GAConfig
from .chromosome import Chromosome
from .fitness import FitnessEvaluator, FitnessResult
from .progress_callback import ProgressInfo
from .return_strategy import IReturnStrategy, DEFAULT_CLOSE_STRATEGY


def _evaluate_chromosome_worker(args: Tuple) -> Tuple[int, FitnessResult]:
    """Worker function for parallel evaluation (must be top-level for pickling)."""
    idx, chrom_dict, df, target_return, horizon, stability_penalty, n_folds, strategy_name = args

    # Reconstruct chromosome from dict
    chrom = Chromosome(**chrom_dict)

    # Reconstruct strategy from name (strategies are not picklable directly)
    from .return_strategy import get_strategy
    strategy = get_strategy(strategy_name)

    # Create evaluator in worker process
    evaluator = FitnessEvaluator(
        df_base=df,
        target_return=target_return,
        horizon=horizon,
        stability_penalty=stability_penalty,
        use_walk_forward=True,
        n_folds=n_folds,
        strategy=strategy,
        logger=None
    )

    result = evaluator.evaluate(chrom)
    return idx, result


@dataclass
class GAResult:
    """Result of genetic algorithm optimization."""
    best_chromosome: Chromosome
    best_fitness: float
    best_metrics: FitnessResult
    history: List[Tuple[int, float, float]]  # (generation, best, mean)
    all_evaluations: int


@dataclass
class GACheckpoint:
    """Checkpoint for resuming genetic algorithm."""
    generation: int
    population: List[Chromosome]
    best_chromosome: Chromosome
    best_fitness: float
    best_metrics_dict: dict  # Simplified FitnessResult for serialization
    history: List[Tuple[int, float, float]]
    total_evaluations: int
    elapsed_seconds: float

    def to_dict(self) -> dict:
        """Convert checkpoint to dictionary for JSON serialization."""
        return {
            'generation': self.generation,
            'population': [c.to_dict() for c in self.population],
            'best_chromosome': self.best_chromosome.to_dict(),
            'best_fitness': self.best_fitness,
            'best_metrics_dict': self.best_metrics_dict,
            'history': self.history,
            'total_evaluations': self.total_evaluations,
            'elapsed_seconds': self.elapsed_seconds
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GACheckpoint':
        """Create checkpoint from dictionary."""
        return cls(
            generation=data['generation'],
            population=[Chromosome.from_dict(c) for c in data['population']],
            best_chromosome=Chromosome.from_dict(data['best_chromosome']),
            best_fitness=data['best_fitness'],
            best_metrics_dict=data['best_metrics_dict'],
            history=[tuple(h) for h in data['history']],
            total_evaluations=data['total_evaluations'],
            elapsed_seconds=data['elapsed_seconds']
        )

    def save(self, path: str):
        """Save checkpoint to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'GACheckpoint':
        """Load checkpoint from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class GeneticAlgorithm:
    """
    Genetic algorithm for parameter optimization.
    Uses walk-forward validation to prevent overfitting.
    Supports parallel evaluation of chromosomes for speedup.

    Uses Strategy Pattern for return type selection (close vs touch).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: GAConfig,
        strategy: Optional[IReturnStrategy] = None,
        logger: Optional[ILogger] = None,
        progress_callback: Optional[Callable[[int, float], None]] = None,
        n_workers: Optional[int] = None
    ):
        self.df = df
        self.config = config
        self.strategy = strategy or DEFAULT_CLOSE_STRATEGY
        self.logger = logger or NullLogger()
        self.progress_callback = progress_callback

        # Number of parallel workers (default: number of CPU cores - 1)
        self.n_workers = n_workers if n_workers is not None else max(1, mp.cpu_count() - 1)

        self.evaluator = FitnessEvaluator(
            df_base=df,
            target_return=config.target_return,
            horizon=config.horizon,
            stability_penalty=config.stability_penalty,
            use_walk_forward=True,
            n_folds=config.n_folds,
            strategy=self.strategy,
            logger=logger
        )

    def run(
        self,
        verbose: bool = True,
        parallel: bool = True,
        checkpoint: Optional[GACheckpoint] = None,
        auto_checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 10
    ) -> GAResult:
        """
        Execute the genetic algorithm.

        Args:
            verbose: Print progress messages
            parallel: Use parallel evaluation (default True for speedup)
            checkpoint: Resume from this checkpoint if provided
            auto_checkpoint_path: If provided, auto-save checkpoint to this path
            checkpoint_every: Save checkpoint every N generations
        """
        search_space = self.config.search_space

        # Resume from checkpoint or start fresh
        if checkpoint:
            population = checkpoint.population
            best_chromosome = checkpoint.best_chromosome
            best_fitness = checkpoint.best_fitness
            best_metrics = self._restore_metrics(checkpoint.best_metrics_dict)
            history = list(checkpoint.history)
            total_evaluations = checkpoint.total_evaluations
            start_gen = checkpoint.generation + 1
            start_time = time.time() - checkpoint.elapsed_seconds

            if verbose:
                print(f"Resuming from generation {start_gen} (fitness={best_fitness:.4f})")
        else:
            # Initialize population (pass ga_config for forced indicator flags)
            population = [
                Chromosome.random(search_space, ga_config=self.config)
                for _ in range(self.config.population_size)
            ]
            best_chromosome = None
            best_fitness = float('-inf')
            best_metrics = None
            history = []
            total_evaluations = 0
            start_gen = 0
            start_time = time.time()

        use_parallel = parallel and self.n_workers > 1

        if verbose and use_parallel and not checkpoint:
            print(f"Using {self.n_workers} parallel workers for fitness evaluation")

        # Store current state for checkpoint
        self._current_population = population
        self._current_best_chromosome = best_chromosome
        self._current_best_fitness = best_fitness
        self._current_best_metrics = best_metrics
        self._current_history = history
        self._current_total_evaluations = total_evaluations
        self._current_generation = start_gen
        self._start_time = start_time

        for gen in range(start_gen, self.config.generations):
            # Notify callback of generation start
            if self.progress_callback and hasattr(self.progress_callback, 'on_generation_start'):
                self.progress_callback.on_generation_start(gen, len(population))

            # Evaluate fitness (parallel or sequential)
            if use_parallel:
                scored = self._evaluate_population_parallel(population, gen)
            else:
                scored = self._evaluate_population_sequential(population, gen)

            # Notify callback of generation end
            if self.progress_callback and hasattr(self.progress_callback, 'on_generation_end'):
                self.progress_callback.on_generation_end()

            total_evaluations += len(population)

            # Sort by fitness (descending)
            scored.sort(key=lambda x: x[1].fitness, reverse=True)

            # Track best
            gen_best = scored[0]
            if gen_best[1].fitness > best_fitness:
                best_chromosome = gen_best[0]
                best_fitness = gen_best[1].fitness
                best_metrics = gen_best[1]

            # Calculate stats
            gen_fitnesses = [s[1].fitness for s in scored]
            gen_mean = np.mean(gen_fitnesses)
            history.append((gen, best_fitness, gen_mean))

            # Update current state for checkpoint
            self._current_population = population
            self._current_best_chromosome = best_chromosome
            self._current_best_fitness = best_fitness
            self._current_best_metrics = best_metrics
            self._current_history = history
            self._current_total_evaluations = total_evaluations
            self._current_generation = gen

            if verbose and gen % 10 == 0 and not self.progress_callback:
                self.logger.info(
                    f"Generation {gen}",
                    best_fitness=best_fitness,
                    gen_mean=gen_mean,
                    delta_p_test=best_metrics.delta_p_test if best_metrics else 0
                )
                print(f"Gen {gen}: Best={best_fitness:.4f}, Mean={gen_mean:.4f}")

            # Call progress callback with full info
            if self.progress_callback:
                progress_info = ProgressInfo(
                    generation=gen,
                    total_generations=self.config.generations,
                    best_fitness=best_fitness,
                    mean_fitness=gen_mean,
                    delta_p_test=best_metrics.delta_p_test if best_metrics else 0,
                    overfitting_ratio=best_metrics.overfitting_ratio if best_metrics else 0,
                    elapsed_seconds=time.time() - start_time,
                    history=history.copy()
                )
                self.progress_callback(progress_info)

            # Auto-checkpoint
            if auto_checkpoint_path and (gen + 1) % checkpoint_every == 0:
                self.save_checkpoint(auto_checkpoint_path)

            # Check for early stopping
            if self.config.early_stopping and self._should_stop_early(history):
                self.logger.info("Early stopping triggered")
                if verbose and not self.progress_callback:
                    print("Early stopping triggered")
                break

            # Selection and reproduction
            population = self._create_next_generation(scored, search_space)

        self.logger.info(
            "GA completed",
            best_fitness=best_fitness,
            total_evaluations=total_evaluations,
            generations=len(history)
        )

        return GAResult(
            best_chromosome=best_chromosome,
            best_fitness=best_fitness,
            best_metrics=best_metrics,
            history=history,
            all_evaluations=total_evaluations
        )

    def _create_next_generation(
        self,
        scored: List[Tuple[Chromosome, FitnessResult]],
        search_space: GASearchSpace
    ) -> List[Chromosome]:
        """Create next generation via selection, crossover, mutation."""
        new_population = []

        # Elitism: keep best individuals
        for i in range(self.config.elite_size):
            new_population.append(scored[i][0])

        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select(scored)
            parent2 = self._tournament_select(scored)

            # Crossover (pass ga_config for forced indicator flags)
            if random.random() < self.config.crossover_probability:
                child = Chromosome.crossover(parent1, parent2, ga_config=self.config)
            else:
                child = parent1

            # Mutation (pass ga_config for forced indicator flags)
            if random.random() < self.config.mutation_probability:
                child = child.mutate(search_space, ga_config=self.config)

            new_population.append(child)

        return new_population

    def _tournament_select(
        self,
        scored: List[Tuple[Chromosome, FitnessResult]],
        k: int = 3
    ) -> Chromosome:
        """Tournament selection."""
        tournament = random.sample(scored, min(k, len(scored)))
        winner = max(tournament, key=lambda x: x[1].fitness)
        return winner[0]

    def _should_stop_early(
        self,
        history: List[Tuple[int, float, float]]
    ) -> bool:
        """Check if should stop early due to no improvement."""
        patience = self.config.early_stopping_patience
        threshold = self.config.early_stopping_threshold

        if len(history) < patience:
            return False

        recent = [h[1] for h in history[-patience:]]
        improvement = max(recent) - min(recent)

        return improvement < threshold

    def _evaluate_population_sequential(
        self,
        population: List[Chromosome],
        generation: int = 0
    ) -> List[Tuple[Chromosome, FitnessResult]]:
        """Evaluate population sequentially (original behavior)."""
        scored = []
        for idx, chrom in enumerate(population):
            result = self.evaluator.evaluate(chrom)
            scored.append((chrom, result))

            # Notify progress
            if self.progress_callback and hasattr(self.progress_callback, 'on_chromosome_evaluated'):
                self.progress_callback.on_chromosome_evaluated(idx)

        return scored

    def _evaluate_population_parallel(
        self,
        population: List[Chromosome],
        generation: int = 0
    ) -> List[Tuple[Chromosome, FitnessResult]]:
        """Evaluate population in parallel using ProcessPoolExecutor."""
        # Prepare arguments for worker function
        # Note: strategy_name is used instead of strategy object for pickling
        args_list = [
            (
                idx,
                chrom.to_dict(),
                self.df,
                self.config.target_return,
                self.config.horizon,
                self.config.stability_penalty,
                self.config.n_folds,
                self.strategy.name  # Pass strategy name for reconstruction in worker
            )
            for idx, chrom in enumerate(population)
        ]

        # Evaluate in parallel
        results = [None] * len(population)
        completed_count = 0

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(_evaluate_chromosome_worker, args): args[0]
                for args in args_list
            }

            for future in as_completed(futures):
                idx, fitness_result = future.result()
                results[idx] = fitness_result

                # Notify progress
                if self.progress_callback and hasattr(self.progress_callback, 'on_chromosome_evaluated'):
                    self.progress_callback.on_chromosome_evaluated(completed_count)
                completed_count += 1

        # Combine with chromosomes
        scored = [(chrom, results[i]) for i, chrom in enumerate(population)]
        return scored

    def get_checkpoint(self) -> GACheckpoint:
        """Get current state as checkpoint (can be called during execution)."""
        if not hasattr(self, '_current_generation'):
            raise RuntimeError("No checkpoint available - GA hasn't started yet")

        metrics_dict = self._metrics_to_dict(self._current_best_metrics)

        return GACheckpoint(
            generation=self._current_generation,
            population=self._current_population.copy(),
            best_chromosome=self._current_best_chromosome,
            best_fitness=self._current_best_fitness,
            best_metrics_dict=metrics_dict,
            history=self._current_history.copy(),
            total_evaluations=self._current_total_evaluations,
            elapsed_seconds=time.time() - self._start_time
        )

    def save_checkpoint(self, path: str):
        """Save current state to file."""
        checkpoint = self.get_checkpoint()
        checkpoint.save(path)

    def _metrics_to_dict(self, metrics: Optional[FitnessResult]) -> dict:
        """Convert FitnessResult to serializable dict."""
        if metrics is None:
            return {}
        return {
            'fitness': metrics.fitness,
            'delta_p': metrics.delta_p,
            'delta_p_test': metrics.delta_p_test,
            'stability_score': metrics.stability_score,
            'overfitting_ratio': metrics.overfitting_ratio,
            'error': metrics.error
        }

    def _restore_metrics(self, data: dict) -> Optional[FitnessResult]:
        """Restore FitnessResult from dict."""
        if not data:
            return None
        return FitnessResult(
            fitness=data['fitness'],
            delta_p=data['delta_p'],
            delta_p_test=data['delta_p_test'],
            stability_score=data['stability_score'],
            overfitting_ratio=data['overfitting_ratio'],
            walk_forward=None,  # Not serialized
            error=data.get('error')
        )
