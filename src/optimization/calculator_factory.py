"""Factory for creating calculator pipelines from chromosomes."""
from typing import List, Optional, Union

from ..interfaces.column_calculator import IColumnCalculator
from ..calculators.log_price_calculator import LogPriceCalculator
from ..calculators.log_return_calculator import LogReturnCalculator
from ..calculators.future_return_calculator import FutureReturnCalculator
from ..calculators.future_touch_calculator import FutureTouchCalculatorVectorized
from ..calculators.volatility_calculator import VolatilityCalculator
from ..calculators.slope_calculator import SlopeCalculator
from ..calculators.ma_distance_calculator import MADistanceCalculator
from ..calculators.trend_indicator_calculator import TrendIndicatorCalculator
from ..calculators.pipeline import CalculatorPipeline
from .chromosome import Chromosome
from .return_strategy import (
    IReturnStrategy,
    CloseReturnStrategy,
    DEFAULT_CLOSE_STRATEGY
)


class CalculatorFactory:
    """
    Factory for creating calculator pipelines from chromosomes.
    Implements Dependency Inversion - GA depends on factory, not concrete calculators.

    Uses Strategy Pattern for return type selection (close vs touch).
    """

    def __init__(
        self,
        horizon: int = 7,
        strategy: Optional[IReturnStrategy] = None,
        include_touch: bool = False  # Deprecated: use strategy instead
    ):
        """
        Initialize factory with fixed prediction horizon.

        Args:
            horizon: Number of periods for future return calculation
            strategy: Return strategy (CloseReturnStrategy or TouchReturnStrategy).
                     If None, uses CloseReturnStrategy (default behavior).
            include_touch: DEPRECATED - use strategy parameter instead.
                          Kept for backward compatibility.
        """
        self.horizon = horizon

        # Strategy takes precedence over include_touch flag
        if strategy is not None:
            self.strategy = strategy
            self.include_touch = strategy.should_include_touch_calculator()
        else:
            self.strategy = DEFAULT_CLOSE_STRATEGY
            self.include_touch = include_touch

    def create_pipeline(self, chromosome: Chromosome) -> CalculatorPipeline:
        """Create a pipeline configured by the chromosome."""
        calculators: List[IColumnCalculator] = [
            LogPriceCalculator(),
            LogReturnCalculator(window=chromosome.window_rolling_return),
            FutureReturnCalculator(horizon=self.horizon),
        ]

        # Add touch calculator if requested
        if self.include_touch:
            calculators.append(
                FutureTouchCalculatorVectorized(horizon=self.horizon)
            )

        if chromosome.use_volatility:
            calculators.append(
                VolatilityCalculator(window=chromosome.window_volatility)
            )

        calculators.append(
            SlopeCalculator(window=chromosome.window_slope)
        )

        if chromosome.use_ma_distance:
            calculators.append(
                MADistanceCalculator(
                    fast_period=chromosome.ma_fast_period,
                    slow_period=chromosome.ma_slow_period
                )
            )

        if chromosome.use_trend_indicator:
            calculators.append(
                TrendIndicatorCalculator(
                    window=chromosome.window_trend_indicator,
                    slope_multiplier=chromosome.trend_slope_multiplier
                )
            )

        return CalculatorPipeline(calculators, auto_resolve=True)

    def get_feature_columns(self, chromosome: Chromosome) -> List[str]:
        """Get feature columns that will be produced for clustering."""
        features = [f'slope_{chromosome.window_slope}']

        if chromosome.use_volatility:
            features.append(f'volatility_{chromosome.window_volatility}')

        if chromosome.use_rolling_return:
            features.append(f'log_return_rolling_{chromosome.window_rolling_return}')

        if chromosome.use_ma_distance:
            features.append(f'ma_dist_{chromosome.ma_fast_period}_{chromosome.ma_slow_period}')

        if chromosome.use_trend_indicator:
            # Use trend_strength (abs) to avoid redundancy with slope
            features.append(f'trend_strength_norm_{chromosome.window_trend_indicator}')

        return features

    def get_return_column(self, target_return: float) -> str:
        """
        Get the appropriate return column using the configured strategy.

        Args:
            target_return: Target return (for direction detection in touch strategy)

        Returns:
            Column name for probability calculation
        """
        return self.strategy.get_return_column(self.horizon, target_return)

    def get_future_return_column(self) -> str:
        """Get the future return column name (close-to-close)."""
        return f'log_return_future_{self.horizon}'

    def get_future_touch_max_column(self) -> str:
        """Get the future touch max return column name (for upside targets)."""
        return f'log_return_touch_max_{self.horizon}'

    def get_future_touch_min_column(self) -> str:
        """Get the future touch min return column name (for downside targets)."""
        return f'log_return_touch_min_{self.horizon}'

    @property
    def strategy_name(self) -> str:
        """Get the name of the current strategy."""
        return self.strategy.name
