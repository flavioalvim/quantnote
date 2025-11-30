"""Chromosome representation for genetic algorithm."""
from dataclasses import dataclass
from typing import Tuple
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.search_space import GASearchSpace


@dataclass
class Chromosome:
    """Represents a set of parameters to be optimized (feature engineering only)."""

    window_slope: int
    window_volatility: int
    window_rolling_return: int
    n_clusters: int
    use_volatility: bool = True
    use_rolling_return: bool = True

    @classmethod
    def random(cls, search_space: GASearchSpace) -> 'Chromosome':
        """Create chromosome with random values within search space."""
        return cls(
            window_slope=random.randint(*search_space.window_slope),
            window_volatility=random.randint(*search_space.window_volatility),
            window_rolling_return=random.randint(*search_space.window_rolling_return),
            n_clusters=random.randint(*search_space.n_clusters),
            use_volatility=random.choice([True, False]),
            use_rolling_return=random.choice([True, False])
        )

    def mutate(
        self,
        search_space: GASearchSpace,
        mutation_rate: float = 0.1
    ) -> 'Chromosome':
        """Return mutated copy of chromosome."""
        def mutate_int(val: int, range_: Tuple[int, int]) -> int:
            if random.random() < mutation_rate:
                delta = random.randint(-5, 5)
                return max(range_[0], min(range_[1], val + delta))
            return val

        def mutate_bool(val: bool) -> bool:
            if random.random() < mutation_rate:
                return not val
            return val

        return Chromosome(
            window_slope=mutate_int(self.window_slope, search_space.window_slope),
            window_volatility=mutate_int(self.window_volatility, search_space.window_volatility),
            window_rolling_return=mutate_int(self.window_rolling_return, search_space.window_rolling_return),
            n_clusters=mutate_int(self.n_clusters, search_space.n_clusters),
            use_volatility=mutate_bool(self.use_volatility),
            use_rolling_return=mutate_bool(self.use_rolling_return)
        )

    @staticmethod
    def crossover(parent1: 'Chromosome', parent2: 'Chromosome') -> 'Chromosome':
        """Uniform crossover between two parents."""
        return Chromosome(
            window_slope=random.choice([parent1.window_slope, parent2.window_slope]),
            window_volatility=random.choice([parent1.window_volatility, parent2.window_volatility]),
            window_rolling_return=random.choice([parent1.window_rolling_return, parent2.window_rolling_return]),
            n_clusters=random.choice([parent1.n_clusters, parent2.n_clusters]),
            use_volatility=random.choice([parent1.use_volatility, parent2.use_volatility]),
            use_rolling_return=random.choice([parent1.use_rolling_return, parent2.use_rolling_return])
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'window_slope': self.window_slope,
            'window_volatility': self.window_volatility,
            'window_rolling_return': self.window_rolling_return,
            'n_clusters': self.n_clusters,
            'use_volatility': self.use_volatility,
            'use_rolling_return': self.use_rolling_return
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Chromosome':
        """Create chromosome from dictionary."""
        return cls(
            window_slope=data['window_slope'],
            window_volatility=data['window_volatility'],
            window_rolling_return=data['window_rolling_return'],
            n_clusters=data['n_clusters'],
            use_volatility=data.get('use_volatility', True),
            use_rolling_return=data.get('use_rolling_return', True)
        )

    def cache_key(self) -> str:
        """Generate a unique cache key for this chromosome."""
        return (
            f"{self.window_slope}_{self.window_volatility}_{self.window_rolling_return}_"
            f"{self.n_clusters}_{self.use_volatility}_{self.use_rolling_return}"
        )
