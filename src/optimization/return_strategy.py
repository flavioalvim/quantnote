"""Return type strategies for probability calculation.

Implements Strategy Pattern to handle different return types (close vs touch)
following SOLID principles - particularly Open/Closed Principle.
"""
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class IReturnStrategy(Protocol):
    """
    Strategy interface for return column selection.

    Defines how to:
    1. Configure the calculator pipeline (include touch calculator or not)
    2. Select the appropriate return column based on target direction
    """

    @property
    def name(self) -> str:
        """Strategy name for display/logging."""
        ...

    def should_include_touch_calculator(self) -> bool:
        """Whether to include FutureTouchCalculator in the pipeline."""
        ...

    def get_return_column(self, horizon: int, target_return: float) -> str:
        """
        Get the appropriate return column for probability calculation.

        Args:
            horizon: Prediction horizon in periods
            target_return: Target return (positive for upside, negative for downside)

        Returns:
            Column name to use for probability calculation
        """
        ...


class CloseReturnStrategy:
    """
    Strategy for close-based probability (P(close)).

    Calculates probability that price will CLOSE at or above target.
    Uses: log_return_future_{horizon}
    """

    @property
    def name(self) -> str:
        return "close"

    def should_include_touch_calculator(self) -> bool:
        return False

    def get_return_column(self, horizon: int, target_return: float) -> str:
        """Always use close-to-close return regardless of direction."""
        return f'log_return_future_{horizon}'


class TouchReturnStrategy:
    """
    Strategy for touch-based probability (P(touch)).

    Calculates probability that price will TOUCH target at any point.
    Uses:
    - log_return_touch_max_{horizon} for upside targets (calls)
    - log_return_touch_min_{horizon} for downside targets (puts)
    """

    @property
    def name(self) -> str:
        return "touch"

    def should_include_touch_calculator(self) -> bool:
        return True

    def get_return_column(self, horizon: int, target_return: float) -> str:
        """
        Select column based on target direction.

        - Upside (target >= 0): Use max high within horizon
        - Downside (target < 0): Use min low within horizon
        """
        if target_return >= 0:
            return f'log_return_touch_max_{horizon}'
        else:
            return f'log_return_touch_min_{horizon}'


# Default strategies for convenience
DEFAULT_CLOSE_STRATEGY = CloseReturnStrategy()
DEFAULT_TOUCH_STRATEGY = TouchReturnStrategy()


def get_strategy(strategy_name: str) -> IReturnStrategy:
    """
    Factory function to get strategy by name.

    Args:
        strategy_name: 'close' or 'touch'

    Returns:
        Corresponding strategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        'close': DEFAULT_CLOSE_STRATEGY,
        'touch': DEFAULT_TOUCH_STRATEGY,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy_name]
