"""Chromosome representation for genetic algorithm."""
from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.search_space import GASearchSpace

if TYPE_CHECKING:
    from config.search_space import GAConfig


@dataclass
class Chromosome:
    """Represents a set of parameters to be optimized (feature engineering only)."""

    window_slope: int
    window_volatility: int
    window_rolling_return: int
    window_trend_indicator: int
    trend_slope_multiplier: float
    n_clusters: int
    ma_fast_period: int = 9
    ma_slow_period: int = 21
    use_volatility: bool = True
    use_rolling_return: bool = True
    use_ma_distance: bool = True
    use_trend_indicator: bool = True

    # Default values for disabled indicators (midpoint of typical ranges)
    DEFAULT_WINDOW_VOLATILITY = 20
    DEFAULT_WINDOW_ROLLING_RETURN = 20
    DEFAULT_WINDOW_TREND_INDICATOR = 10
    DEFAULT_TREND_SLOPE_MULTIPLIER = 2.0
    DEFAULT_MA_FAST_PERIOD = 9
    DEFAULT_MA_SLOW_PERIOD = 21

    @classmethod
    def random(
        cls,
        search_space: GASearchSpace,
        ga_config: Optional['GAConfig'] = None
    ) -> 'Chromosome':
        """
        Create chromosome with random values within search space.

        Args:
            search_space: Search space for parameter ranges
            ga_config: Optional GA config with force_use_* flags.
                       If provided, respects the forced indicator settings.
                       When an indicator is disabled (force_use_*=False),
                       its parameters use fixed defaults instead of random values.
        """
        # Determine use_* flags based on ga_config forced settings
        def get_use_flag(force_value: Optional[bool]) -> bool:
            """Get use flag: forced value if set, else random."""
            if force_value is not None:
                return force_value
            return random.choice([True, False])

        if ga_config:
            use_volatility = get_use_flag(ga_config.force_use_volatility)
            use_rolling_return = get_use_flag(ga_config.force_use_rolling_return)
            use_ma_distance = get_use_flag(ga_config.force_use_ma_distance)
            use_trend_indicator = get_use_flag(ga_config.force_use_trend_indicator)
        else:
            use_volatility = random.choice([True, False])
            use_rolling_return = random.choice([True, False])
            use_ma_distance = random.choice([True, False])
            use_trend_indicator = random.choice([True, False])

        # Only randomize parameters for ACTIVE indicators
        # Disabled indicators use fixed defaults (saves genetic diversity & cache efficiency)

        # Volatility parameters
        if use_volatility:
            window_volatility = random.randint(*search_space.window_volatility)
        else:
            window_volatility = cls.DEFAULT_WINDOW_VOLATILITY

        # Rolling return parameters
        if use_rolling_return:
            window_rolling_return = random.randint(*search_space.window_rolling_return)
        else:
            window_rolling_return = cls.DEFAULT_WINDOW_ROLLING_RETURN

        # MA distance parameters
        if use_ma_distance:
            ma_fast = random.randint(*search_space.ma_fast_period)
            ma_slow = random.randint(*search_space.ma_slow_period)
            if ma_fast >= ma_slow:
                ma_slow = ma_fast + 5
        else:
            ma_fast = cls.DEFAULT_MA_FAST_PERIOD
            ma_slow = cls.DEFAULT_MA_SLOW_PERIOD

        # Trend indicator parameters
        if use_trend_indicator:
            window_trend_indicator = random.randint(*search_space.window_trend_indicator)
            slope_mult_min, slope_mult_max = search_space.trend_slope_multiplier
            trend_slope_mult = round(random.uniform(slope_mult_min, slope_mult_max), 1)
        else:
            window_trend_indicator = cls.DEFAULT_WINDOW_TREND_INDICATOR
            trend_slope_mult = cls.DEFAULT_TREND_SLOPE_MULTIPLIER

        return cls(
            window_slope=random.randint(*search_space.window_slope),
            window_volatility=window_volatility,
            window_rolling_return=window_rolling_return,
            window_trend_indicator=window_trend_indicator,
            trend_slope_multiplier=trend_slope_mult,
            n_clusters=random.randint(*search_space.n_clusters),
            ma_fast_period=ma_fast,
            ma_slow_period=ma_slow,
            use_volatility=use_volatility,
            use_rolling_return=use_rolling_return,
            use_ma_distance=use_ma_distance,
            use_trend_indicator=use_trend_indicator
        )

    def mutate(
        self,
        search_space: GASearchSpace,
        mutation_rate: float = 0.1,
        ga_config: Optional['GAConfig'] = None
    ) -> 'Chromosome':
        """
        Return mutated copy of chromosome.

        Args:
            search_space: Search space for parameter ranges
            mutation_rate: Probability of mutating each parameter
            ga_config: Optional GA config with force_use_* flags.
                       If provided, forced flags cannot be mutated, and
                       parameters of disabled indicators are not mutated.
        """
        def mutate_int(val: int, range_: Tuple[int, int]) -> int:
            if random.random() < mutation_rate:
                delta = random.randint(-5, 5)
                return max(range_[0], min(range_[1], val + delta))
            return val

        def mutate_float(val: float, range_: Tuple[float, float]) -> float:
            if random.random() < mutation_rate:
                delta = random.uniform(-0.5, 0.5)
                return round(max(range_[0], min(range_[1], val + delta)), 1)
            return val

        def mutate_bool(val: bool, force_value: Optional[bool] = None) -> bool:
            """Mutate bool, but respect forced value if set."""
            if force_value is not None:
                return force_value
            if random.random() < mutation_rate:
                return not val
            return val

        # Get forced values from ga_config
        force_vol = ga_config.force_use_volatility if ga_config else None
        force_roll = ga_config.force_use_rolling_return if ga_config else None
        force_ma = ga_config.force_use_ma_distance if ga_config else None
        force_trend = ga_config.force_use_trend_indicator if ga_config else None

        # Determine final use_* flags (after potential mutation)
        use_volatility = mutate_bool(self.use_volatility, force_vol)
        use_rolling_return = mutate_bool(self.use_rolling_return, force_roll)
        use_ma_distance = mutate_bool(self.use_ma_distance, force_ma)
        use_trend_indicator = mutate_bool(self.use_trend_indicator, force_trend)

        # Only mutate parameters for ACTIVE indicators
        # Disabled indicators keep fixed defaults

        # Volatility parameters
        if use_volatility:
            window_volatility = mutate_int(self.window_volatility, search_space.window_volatility)
        else:
            window_volatility = Chromosome.DEFAULT_WINDOW_VOLATILITY

        # Rolling return parameters
        if use_rolling_return:
            window_rolling_return = mutate_int(self.window_rolling_return, search_space.window_rolling_return)
        else:
            window_rolling_return = Chromosome.DEFAULT_WINDOW_ROLLING_RETURN

        # MA distance parameters
        if use_ma_distance:
            ma_fast = mutate_int(self.ma_fast_period, search_space.ma_fast_period)
            ma_slow = mutate_int(self.ma_slow_period, search_space.ma_slow_period)
            if ma_fast >= ma_slow:
                ma_slow = ma_fast + 5
        else:
            ma_fast = Chromosome.DEFAULT_MA_FAST_PERIOD
            ma_slow = Chromosome.DEFAULT_MA_SLOW_PERIOD

        # Trend indicator parameters
        if use_trend_indicator:
            window_trend_indicator = mutate_int(self.window_trend_indicator, search_space.window_trend_indicator)
            trend_slope_multiplier = mutate_float(self.trend_slope_multiplier, search_space.trend_slope_multiplier)
        else:
            window_trend_indicator = Chromosome.DEFAULT_WINDOW_TREND_INDICATOR
            trend_slope_multiplier = Chromosome.DEFAULT_TREND_SLOPE_MULTIPLIER

        return Chromosome(
            window_slope=mutate_int(self.window_slope, search_space.window_slope),
            window_volatility=window_volatility,
            window_rolling_return=window_rolling_return,
            window_trend_indicator=window_trend_indicator,
            trend_slope_multiplier=trend_slope_multiplier,
            n_clusters=mutate_int(self.n_clusters, search_space.n_clusters),
            ma_fast_period=ma_fast,
            ma_slow_period=ma_slow,
            use_volatility=use_volatility,
            use_rolling_return=use_rolling_return,
            use_ma_distance=use_ma_distance,
            use_trend_indicator=use_trend_indicator
        )

    @staticmethod
    def crossover(
        parent1: 'Chromosome',
        parent2: 'Chromosome',
        ga_config: Optional['GAConfig'] = None
    ) -> 'Chromosome':
        """
        Uniform crossover between two parents.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            ga_config: Optional GA config with force_use_* flags.
                       If provided, forced flags override crossover result, and
                       parameters of disabled indicators use fixed defaults.
        """
        # Get forced values from ga_config
        def get_use_flag(force_value: Optional[bool], val1: bool, val2: bool) -> bool:
            """Get use flag: forced value if set, else random choice from parents."""
            if force_value is not None:
                return force_value
            return random.choice([val1, val2])

        if ga_config:
            use_volatility = get_use_flag(
                ga_config.force_use_volatility,
                parent1.use_volatility, parent2.use_volatility
            )
            use_rolling_return = get_use_flag(
                ga_config.force_use_rolling_return,
                parent1.use_rolling_return, parent2.use_rolling_return
            )
            use_ma_distance = get_use_flag(
                ga_config.force_use_ma_distance,
                parent1.use_ma_distance, parent2.use_ma_distance
            )
            use_trend_indicator = get_use_flag(
                ga_config.force_use_trend_indicator,
                parent1.use_trend_indicator, parent2.use_trend_indicator
            )
        else:
            use_volatility = random.choice([parent1.use_volatility, parent2.use_volatility])
            use_rolling_return = random.choice([parent1.use_rolling_return, parent2.use_rolling_return])
            use_ma_distance = random.choice([parent1.use_ma_distance, parent2.use_ma_distance])
            use_trend_indicator = random.choice([parent1.use_trend_indicator, parent2.use_trend_indicator])

        # Only crossover parameters for ACTIVE indicators
        # Disabled indicators use fixed defaults

        # Volatility parameters
        if use_volatility:
            window_volatility = random.choice([parent1.window_volatility, parent2.window_volatility])
        else:
            window_volatility = Chromosome.DEFAULT_WINDOW_VOLATILITY

        # Rolling return parameters
        if use_rolling_return:
            window_rolling_return = random.choice([parent1.window_rolling_return, parent2.window_rolling_return])
        else:
            window_rolling_return = Chromosome.DEFAULT_WINDOW_ROLLING_RETURN

        # MA distance parameters
        if use_ma_distance:
            ma_fast = random.choice([parent1.ma_fast_period, parent2.ma_fast_period])
            ma_slow = random.choice([parent1.ma_slow_period, parent2.ma_slow_period])
            if ma_fast >= ma_slow:
                ma_slow = ma_fast + 5
        else:
            ma_fast = Chromosome.DEFAULT_MA_FAST_PERIOD
            ma_slow = Chromosome.DEFAULT_MA_SLOW_PERIOD

        # Trend indicator parameters
        if use_trend_indicator:
            window_trend_indicator = random.choice([parent1.window_trend_indicator, parent2.window_trend_indicator])
            trend_slope_multiplier = random.choice([parent1.trend_slope_multiplier, parent2.trend_slope_multiplier])
        else:
            window_trend_indicator = Chromosome.DEFAULT_WINDOW_TREND_INDICATOR
            trend_slope_multiplier = Chromosome.DEFAULT_TREND_SLOPE_MULTIPLIER

        return Chromosome(
            window_slope=random.choice([parent1.window_slope, parent2.window_slope]),
            window_volatility=window_volatility,
            window_rolling_return=window_rolling_return,
            window_trend_indicator=window_trend_indicator,
            trend_slope_multiplier=trend_slope_multiplier,
            n_clusters=random.choice([parent1.n_clusters, parent2.n_clusters]),
            ma_fast_period=ma_fast,
            ma_slow_period=ma_slow,
            use_volatility=use_volatility,
            use_rolling_return=use_rolling_return,
            use_ma_distance=use_ma_distance,
            use_trend_indicator=use_trend_indicator
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'window_slope': self.window_slope,
            'window_volatility': self.window_volatility,
            'window_rolling_return': self.window_rolling_return,
            'window_trend_indicator': self.window_trend_indicator,
            'trend_slope_multiplier': self.trend_slope_multiplier,
            'n_clusters': self.n_clusters,
            'ma_fast_period': self.ma_fast_period,
            'ma_slow_period': self.ma_slow_period,
            'use_volatility': self.use_volatility,
            'use_rolling_return': self.use_rolling_return,
            'use_ma_distance': self.use_ma_distance,
            'use_trend_indicator': self.use_trend_indicator
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Chromosome':
        """Create chromosome from dictionary."""
        return cls(
            window_slope=data['window_slope'],
            window_volatility=data['window_volatility'],
            window_rolling_return=data['window_rolling_return'],
            window_trend_indicator=data.get('window_trend_indicator', 10),
            trend_slope_multiplier=data.get('trend_slope_multiplier', 2.0),
            n_clusters=data['n_clusters'],
            ma_fast_period=data.get('ma_fast_period', 9),
            ma_slow_period=data.get('ma_slow_period', 21),
            use_volatility=data.get('use_volatility', True),
            use_rolling_return=data.get('use_rolling_return', True),
            use_ma_distance=data.get('use_ma_distance', True),
            use_trend_indicator=data.get('use_trend_indicator', True)
        )

    def cache_key(self) -> str:
        """Generate a unique cache key for this chromosome."""
        return (
            f"{self.window_slope}_{self.window_volatility}_{self.window_rolling_return}_"
            f"{self.window_trend_indicator}_{self.trend_slope_multiplier}_{self.n_clusters}_"
            f"{self.ma_fast_period}_{self.ma_slow_period}_"
            f"{self.use_volatility}_{self.use_rolling_return}_{self.use_ma_distance}_{self.use_trend_indicator}"
        )
