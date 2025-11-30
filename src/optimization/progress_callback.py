"""Progress callback for genetic algorithm with live visualization."""
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class ProgressInfo:
    """Information passed to progress callback."""
    generation: int
    total_generations: int
    best_fitness: float
    mean_fitness: float
    delta_p_test: float
    overfitting_ratio: float
    elapsed_seconds: float
    history: List[Tuple[int, float, float]]  # (gen, best, mean)


@dataclass
class ChromosomeProgressInfo:
    """Progress info for chromosome evaluation within a generation."""
    generation: int
    total_generations: int
    chromosome_idx: int
    total_chromosomes: int
    elapsed_seconds: float


class LiveProgressCallback:
    """
    Live progress callback with matplotlib visualization for Jupyter notebooks.

    Usage:
        callback = LiveProgressCallback(total_generations=50, update_every=1)
        ga = GeneticAlgorithm(df, config, progress_callback=callback)
        result = ga.run()
        callback.finalize()
    """

    def __init__(
        self,
        total_generations: int,
        population_size: int = 0,
        update_every: int = 1,
        figsize: tuple = (12, 4)
    ):
        """
        Initialize the callback.

        Args:
            total_generations: Total number of generations expected
            population_size: Size of population (for intra-generation progress)
            update_every: Update plot every N generations
            figsize: Figure size (width, height)
        """
        self.total_generations = total_generations
        self.population_size = population_size
        self.update_every = update_every
        self.figsize = figsize

        self.start_time: Optional[float] = None
        self.fig = None
        self.axes = None
        self._initialized = False
        self._tqdm_bar = None
        self._current_gen = 0
        self._gen_start_time: Optional[float] = None

        # Store history for plotting
        self.generations: List[int] = []
        self.best_fitnesses: List[float] = []
        self.mean_fitnesses: List[float] = []
        self.delta_ps: List[float] = []

    def _init_plot(self):
        """Initialize matplotlib figure with subplots."""
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display, clear_output

            self._plt = plt
            self._display = display
            self._clear_output = clear_output

            self.fig, self.axes = plt.subplots(1, 3, figsize=self.figsize)
            self.fig.suptitle('Genetic Algorithm Progress', fontsize=12, fontweight='bold')

            # Configure subplots
            self.axes[0].set_xlabel('Generation')
            self.axes[0].set_ylabel('Fitness')
            self.axes[0].set_title('Fitness Evolution')
            self.axes[0].grid(True, alpha=0.3)

            self.axes[1].set_xlabel('Generation')
            self.axes[1].set_ylabel('Delta P (test)')
            self.axes[1].set_title('Separation Quality')
            self.axes[1].grid(True, alpha=0.3)

            self.axes[2].axis('off')
            self.axes[2].set_title('Status')

            plt.tight_layout()
            self._initialized = True

        except ImportError:
            print("Warning: matplotlib or IPython not available. Using text-only progress.")
            self._initialized = False

    def __call__(self, info: ProgressInfo):
        """Called by GA at each generation."""
        if self.start_time is None:
            self.start_time = time.time() - info.elapsed_seconds

        # Store data
        self.generations.append(info.generation)
        self.best_fitnesses.append(info.best_fitness)
        self.mean_fitnesses.append(info.mean_fitness)
        self.delta_ps.append(info.delta_p_test)

        # Only update every N generations
        if info.generation % self.update_every != 0 and info.generation != info.total_generations - 1:
            return

        # Initialize plot on first call
        if not self._initialized:
            self._init_plot()

        if self._initialized:
            self._update_plot(info)
        else:
            self._print_progress(info)

    def _update_plot(self, info: ProgressInfo):
        """Update matplotlib plot."""
        self._clear_output(wait=True)

        # Clear axes
        for ax in self.axes[:2]:
            ax.clear()

        # Plot 1: Fitness evolution
        self.axes[0].plot(self.generations, self.best_fitnesses, 'b-',
                         label='Best', linewidth=2)
        self.axes[0].plot(self.generations, self.mean_fitnesses, 'g--',
                         label='Mean', alpha=0.7)
        self.axes[0].set_xlabel('Generation')
        self.axes[0].set_ylabel('Fitness')
        self.axes[0].set_title('Fitness Evolution')
        self.axes[0].legend(loc='lower right')
        self.axes[0].grid(True, alpha=0.3)

        # Plot 2: Delta P
        self.axes[1].plot(self.generations, self.delta_ps, 'r-', linewidth=2)
        self.axes[1].set_xlabel('Generation')
        self.axes[1].set_ylabel('Delta P (test)')
        self.axes[1].set_title('Separation Quality')
        self.axes[1].grid(True, alpha=0.3)

        # Plot 3: Status text
        self.axes[2].clear()
        self.axes[2].axis('off')

        # Calculate time estimates
        elapsed = info.elapsed_seconds
        progress = (info.generation + 1) / info.total_generations
        eta = (elapsed / progress - elapsed) if progress > 0 else 0

        status_text = (
            f"Generation: {info.generation + 1} / {info.total_generations}\n"
            f"Progress: {progress * 100:.1f}%\n"
            f"\n"
            f"Best Fitness: {info.best_fitness:.4f}\n"
            f"Mean Fitness: {info.mean_fitness:.4f}\n"
            f"Delta P (test): {info.delta_p_test:.4f}\n"
            f"Overfitting: {info.overfitting_ratio:.2f}\n"
            f"\n"
            f"Elapsed: {self._format_time(elapsed)}\n"
            f"ETA: {self._format_time(eta)}\n"
        )

        self.axes[2].text(0.1, 0.9, status_text, transform=self.axes[2].transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.axes[2].set_title('Status')

        self.fig.suptitle('Genetic Algorithm Progress', fontsize=12, fontweight='bold')
        self._plt.tight_layout()
        self._display(self.fig)

    def _print_progress(self, info: ProgressInfo):
        """Fallback text-only progress."""
        elapsed = info.elapsed_seconds
        progress = (info.generation + 1) / info.total_generations
        eta = (elapsed / progress - elapsed) if progress > 0 else 0

        print(f"Gen {info.generation + 1}/{info.total_generations} "
              f"({progress * 100:.0f}%) | "
              f"Best: {info.best_fitness:.4f} | "
              f"Delta P: {info.delta_p_test:.4f} | "
              f"ETA: {self._format_time(eta)}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def finalize(self):
        """Call after GA completes to keep the final plot visible."""
        self._close_tqdm()
        if self._initialized and self.fig is not None:
            self._plt.show()

    # --- Intra-generation progress methods ---

    def on_generation_start(self, generation: int, total_chromosomes: int):
        """Called when a new generation starts evaluation."""
        self._current_gen = generation
        self._gen_start_time = time.time()
        self._close_tqdm()
        self._text_progress_total = total_chromosomes
        self._text_progress_current = 0

        # Print initial status immediately
        print(f"Gen {generation + 1}/{self.total_generations}: Avaliando {total_chromosomes} cromossomos...", flush=True)

        # Try different tqdm imports
        tqdm_class = None
        try:
            # Try notebook version first (best for Jupyter)
            from tqdm.notebook import tqdm as tqdm_notebook
            tqdm_class = tqdm_notebook
        except ImportError:
            try:
                # Fallback to auto (detects environment)
                from tqdm.auto import tqdm as tqdm_auto
                tqdm_class = tqdm_auto
            except ImportError:
                try:
                    # Fallback to standard tqdm
                    from tqdm import tqdm as tqdm_std
                    tqdm_class = tqdm_std
                except ImportError:
                    tqdm_class = None

        if tqdm_class is not None:
            try:
                self._tqdm_bar = tqdm_class(
                    total=total_chromosomes,
                    desc=f"Gen {generation + 1}/{self.total_generations}",
                    leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )
            except Exception:
                self._tqdm_bar = None
        else:
            self._tqdm_bar = None

    def on_chromosome_evaluated(self, chromosome_idx: int):
        """Called after each chromosome is evaluated."""
        self._text_progress_current += 1

        if self._tqdm_bar is not None:
            try:
                self._tqdm_bar.update(1)
            except Exception:
                self._print_text_progress()
        else:
            self._print_text_progress()

    def _print_text_progress(self):
        """Print text-based progress."""
        pct = self._text_progress_current / self._text_progress_total * 100
        bar_width = 20
        filled = int(bar_width * self._text_progress_current / self._text_progress_total)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"\rGen {self._current_gen + 1}/{self.total_generations}: |{bar}| {self._text_progress_current}/{self._text_progress_total} ({pct:.0f}%)", end='', flush=True)

    def on_generation_end(self):
        """Called when generation evaluation is complete."""
        self._close_tqdm()
        # Clear the text progress line
        if self._tqdm_bar is None:
            print()  # New line after text progress

    def _close_tqdm(self):
        """Close tqdm bar if open."""
        if self._tqdm_bar is not None:
            try:
                self._tqdm_bar.close()
            except Exception:
                pass
            self._tqdm_bar = None


class SimpleProgressCallback:
    """Simple text-based progress callback."""

    def __init__(self, total_generations: int, print_every: int = 1):
        self.total_generations = total_generations
        self.print_every = print_every
        self.start_time: Optional[float] = None

    def __call__(self, info: ProgressInfo):
        if self.start_time is None:
            self.start_time = time.time() - info.elapsed_seconds

        if info.generation % self.print_every != 0:
            return

        elapsed = info.elapsed_seconds
        progress = (info.generation + 1) / info.total_generations
        eta = (elapsed / progress - elapsed) if progress > 0 else 0

        bar_width = 30
        filled = int(bar_width * progress)
        bar = '=' * filled + '>' + '.' * (bar_width - filled - 1)

        print(f"\r[{bar}] {progress * 100:5.1f}% | "
              f"Gen {info.generation + 1}/{info.total_generations} | "
              f"Fitness: {info.best_fitness:.4f} | "
              f"Delta P: {info.delta_p_test:.4f} | "
              f"ETA: {self._format_time(eta)}     ", end='', flush=True)

        if info.generation == info.total_generations - 1:
            print()  # New line at the end

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
