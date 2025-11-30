"""Value Objects - Immutable domain objects with validation."""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from enum import Enum


class TrendDirection(Enum):
    """Market trend direction."""
    BULL = "bull"
    BEAR = "bear"
    FLAT = "flat"


class VolatilityLevel(Enum):
    """Volatility level."""
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class LogReturn:
    """
    Immutable value object for log-returns.
    Encapsulates conversion logic and validation.
    """
    value: float

    def __post_init__(self):
        if not isinstance(self.value, (int, float)) or np.isnan(self.value):
            raise ValueError(f"Invalid log return value: {self.value}")
        # Log returns typically between -100% and +100% in daily data
        if not -2.0 <= self.value <= 2.0:
            raise ValueError(f"Log return {self.value} outside expected range [-2, 2]")

    def to_percent(self) -> float:
        """Convert to percentage return."""
        return np.exp(self.value) - 1

    def to_basis_points(self) -> float:
        """Convert to basis points."""
        return self.to_percent() * 10000

    @classmethod
    def from_percent(cls, percent: float) -> 'LogReturn':
        """Create from percentage return."""
        if percent <= -1.0:
            raise ValueError("Percentage return must be > -100%")
        return cls(np.log(1 + percent))

    @classmethod
    def from_prices(cls, price_current: float, price_previous: float) -> 'LogReturn':
        """Create from two consecutive prices."""
        if price_current <= 0 or price_previous <= 0:
            raise ValueError("Prices must be positive")
        return cls(np.log(price_current / price_previous))

    def __add__(self, other: 'LogReturn') -> 'LogReturn':
        """Log returns are additive."""
        return LogReturn(self.value + other.value)

    def __neg__(self) -> 'LogReturn':
        return LogReturn(-self.value)


@dataclass(frozen=True)
class Price:
    """
    Immutable value object for prices.
    Ensures positive values.
    """
    value: float
    currency: str = "BRL"

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError(f"Price must be positive, got {self.value}")

    def log(self) -> float:
        """Return log of price."""
        return np.log(self.value)

    def apply_return(self, log_return: LogReturn) -> 'Price':
        """Apply a log return to get new price."""
        new_value = self.value * np.exp(log_return.value)
        return Price(new_value, self.currency)


@dataclass(frozen=True)
class Regime:
    """
    Immutable value object representing a market regime.
    Combines trend and volatility.
    """
    trend: TrendDirection
    volatility: VolatilityLevel

    @property
    def name(self) -> str:
        """Human-readable regime name."""
        return f"{self.trend.value}_{self.volatility.value}_vol"

    @property
    def is_favorable(self) -> bool:
        """Whether regime is typically favorable for long positions."""
        return self.trend == TrendDirection.BULL

    @classmethod
    def from_indicators(
        cls,
        slope: float,
        volatility: float,
        slope_threshold: float,
        volatility_threshold: float
    ) -> 'Regime':
        """Create regime from indicator values."""
        if slope > slope_threshold:
            trend = TrendDirection.BULL
        elif slope < -slope_threshold:
            trend = TrendDirection.BEAR
        else:
            trend = TrendDirection.FLAT

        vol_level = VolatilityLevel.HIGH if volatility > volatility_threshold else VolatilityLevel.LOW

        return cls(trend=trend, volatility=vol_level)

    @classmethod
    def all_regimes(cls) -> List['Regime']:
        """Return all possible regime combinations."""
        return [
            cls(trend, vol)
            for trend in TrendDirection
            for vol in VolatilityLevel
        ]


@dataclass(frozen=True)
class Probability:
    """
    Immutable value object for probabilities.
    Ensures value is between 0 and 1.
    """
    value: float

    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {self.value}")

    def to_percent(self) -> float:
        """Convert to percentage."""
        return self.value * 100

    def to_odds(self) -> float:
        """Convert to odds ratio."""
        if self.value == 1.0:
            return float('inf')
        return self.value / (1 - self.value)

    @classmethod
    def from_frequency(cls, hits: int, total: int) -> 'Probability':
        """Create from hit/total counts."""
        if total == 0:
            raise ValueError("Cannot compute probability with zero total")
        return cls(hits / total)
