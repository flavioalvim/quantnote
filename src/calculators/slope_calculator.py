"""Slope calculator using linear regression with Numba acceleration."""
import pandas as pd
import numpy as np
from typing import Set

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ..interfaces.column_calculator import IColumnCalculator


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _linregress_slope(y: np.ndarray) -> float:
        """
        Calculate slope via linear regression using Numba.
        Optimized implementation without scipy overhead.
        """
        n = len(y)
        if n == 0:
            return np.nan

        # Check for NaN values
        for i in range(n):
            if np.isnan(y[i]):
                return np.nan

        # Calculate slope using least squares formula
        # slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        # For x = [0, 1, 2, ..., n-1], x_mean = (n-1)/2

        x_mean = (n - 1) / 2.0
        y_mean = 0.0
        for i in range(n):
            y_mean += y[i]
        y_mean /= n

        numerator = 0.0
        denominator = 0.0

        for i in range(n):
            x_diff = i - x_mean
            numerator += x_diff * (y[i] - y_mean)
            denominator += x_diff * x_diff

        if denominator == 0:
            return np.nan

        return numerator / denominator

    @jit(nopython=True, parallel=True, cache=True)
    def _rolling_slope_numba(values: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling slope using Numba with parallel processing."""
        n = len(values)
        result = np.empty(n)
        result[:] = np.nan

        for i in prange(window - 1, n):
            result[i] = _linregress_slope(values[i - window + 1:i + 1])

        return result

else:
    # Fallback without Numba
    def _linregress_slope(y: np.ndarray) -> float:
        """Calculate slope via linear regression (fallback without Numba)."""
        from scipy import stats
        if np.any(np.isnan(y)):
            return np.nan
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    def _rolling_slope_numba(values: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling slope (fallback without Numba)."""
        n = len(values)
        result = np.empty(n)
        result[:] = np.nan

        for i in range(window - 1, n):
            result[i] = _linregress_slope(values[i - window + 1:i + 1])

        return result


class SlopeCalculator(IColumnCalculator):
    """
    Calculates slope of log-price linear regression over rolling window.
    Uses Numba JIT compilation for ~10-50x speedup when available.
    """

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"slope_w{self._window}"

    @property
    def required_columns(self) -> Set[str]:
        return {'log_close'}

    @property
    def output_columns(self) -> Set[str]:
        return {f'slope_{self._window}'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        # Use Numba-accelerated rolling slope
        values = result['log_close'].values.astype(np.float64)
        result[f'slope_{self._window}'] = _rolling_slope_numba(values, self._window)

        return result

    @staticmethod
    def is_numba_available() -> bool:
        """Check if Numba acceleration is available."""
        return NUMBA_AVAILABLE
