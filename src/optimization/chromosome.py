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
    ma_fast_period: int = 9
    ma_slow_period: int = 21
    use_volatility: bool = True
    use_rolling_return: bool = True
    use_ma_distance: bool = True

    @classmethod
    def random(cls, search_space: GASearchSpace) -> 'Chromosome':
        """Create chromosome with random values within search space."""
        # Ensure ma_fast < ma_slow
        ma_fast = random.randint(*search_space.ma_fast_period)
        ma_slow = random.randint(*search_space.ma_slow_period)
        if ma_fast >= ma_slow:
            ma_slow = ma_fast + 5

        return cls(
            window_slope=random.randint(*search_space.window_slope),
            window_volatility=random.randint(*search_space.window_volatility),
            window_rolling_return=random.randint(*search_space.window_rolling_return),
            n_clusters=random.randint(*search_space.n_clusters),
            ma_fast_period=ma_fast,
            ma_slow_period=ma_slow,
            use_volatility=random.choice([True, False]),
            use_rolling_return=random.choice([True, False]),
            use_ma_distance=random.choice([True, False])
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

        # Mutate MA periods ensuring fast < slow
        ma_fast = mutate_int(self.ma_fast_period, search_space.ma_fast_period)
        ma_slow = mutate_int(self.ma_slow_period, search_space.ma_slow_period)
        if ma_fast >= ma_slow:
            ma_slow = ma_fast + 5

        return Chromosome(
            window_slope=mutate_int(self.window_slope, search_space.window_slope),
            window_volatility=mutate_int(self.window_volatility, search_space.window_volatility),
            window_rolling_return=mutate_int(self.window_rolling_return, search_space.window_rolling_return),
            n_clusters=mutate_int(self.n_clusters, search_space.n_clusters),
            ma_fast_period=ma_fast,
            ma_slow_period=ma_slow,
            use_volatility=mutate_bool(self.use_volatility),
            use_rolling_return=mutate_bool(self.use_rolling_return),
            use_ma_distance=mutate_bool(self.use_ma_distance)
        )

    @staticmethod
    def crossover(parent1: 'Chromosome', parent2: 'Chromosome') -> 'Chromosome':
        """Uniform crossover between two parents."""
        # Crossover MA periods ensuring fast < slow
        ma_fast = random.choice([parent1.ma_fast_period, parent2.ma_fast_period])
        ma_slow = random.choice([parent1.ma_slow_period, parent2.ma_slow_period])
        if ma_fast >= ma_slow:
            ma_slow = ma_fast + 5

        return Chromosome(
            window_slope=random.choice([parent1.window_slope, parent2.window_slope]),
            window_volatility=random.choice([parent1.window_volatility, parent2.window_volatility]),
            window_rolling_return=random.choice([parent1.window_rolling_return, parent2.window_rolling_return]),
            n_clusters=random.choice([parent1.n_clusters, parent2.n_clusters]),
            ma_fast_period=ma_fast,
            ma_slow_period=ma_slow,
            use_volatility=random.choice([parent1.use_volatility, parent2.use_volatility]),
            use_rolling_return=random.choice([parent1.use_rolling_return, parent2.use_rolling_return]),
            use_ma_distance=random.choice([parent1.use_ma_distance, parent2.use_ma_distance])
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'window_slope': self.window_slope,
            'window_volatility': self.window_volatility,
            'window_rolling_return': self.window_rolling_return,
            'n_clusters': self.n_clusters,
            'ma_fast_period': self.ma_fast_period,
            'ma_slow_period': self.ma_slow_period,
            'use_volatility': self.use_volatility,
            'use_rolling_return': self.use_rolling_return,
            'use_ma_distance': self.use_ma_distance
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Chromosome':
        """Create chromosome from dictionary."""
        return cls(
            window_slope=data['window_slope'],
            window_volatility=data['window_volatility'],
            window_rolling_return=data['window_rolling_return'],
            n_clusters=data['n_clusters'],
            ma_fast_period=data.get('ma_fast_period', 9),
            ma_slow_period=data.get('ma_slow_period', 21),
            use_volatility=data.get('use_volatility', True),
            use_rolling_return=data.get('use_rolling_return', True),
            use_ma_distance=data.get('use_ma_distance', True)
        )

    def cache_key(self) -> str:
        """Generate a unique cache key for this chromosome."""
        return (
            f"{self.window_slope}_{self.window_volatility}_{self.window_rolling_return}_"
            f"{self.n_clusters}_{self.ma_fast_period}_{self.ma_slow_period}_"
            f"{self.use_volatility}_{self.use_rolling_return}_{self.use_ma_distance}"
        )
