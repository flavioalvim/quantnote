"""Return conversion utilities."""
import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


def log_return_from_prices(
    price_current: ArrayLike,
    price_previous: ArrayLike
) -> ArrayLike:
    """Calculate log-return between two prices."""
    return np.log(price_current / price_previous)


def log_to_percent(log_ret: ArrayLike) -> ArrayLike:
    """Convert log-return to percentage return."""
    return np.exp(log_ret) - 1


def percent_to_log(percent_ret: ArrayLike) -> ArrayLike:
    """Convert percentage return to log-return."""
    return np.log(1 + percent_ret)


def annualize_return(
    period_return: float,
    periods_per_year: int = 252
) -> float:
    """Annualize a periodic return."""
    return (1 + period_return) ** periods_per_year - 1


def annualize_volatility(
    period_volatility: float,
    periods_per_year: int = 252
) -> float:
    """Annualize periodic volatility."""
    return period_volatility * np.sqrt(periods_per_year)
