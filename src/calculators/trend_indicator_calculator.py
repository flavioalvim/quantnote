"""Trend Indicator calculator based on Higher Lows and Lower Highs counts."""
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator


class TrendIndicatorCalculator(IColumnCalculator):
    """
    Calculates trend indicator based on consecutive higher lows and lower highs,
    filtered by the current trend direction.

    Logic:
    - Uses slope to determine trend direction (calculated on slope_window)
    - In uptrend (slope > 0): Count higher lows (bullish confirmation)
    - In downtrend (slope <= 0): Count lower highs (bearish confirmation)
    - Output is signed: positive for uptrend, negative for downtrend
    - The magnitude indicates trend strength (count of confirmations)

    Interpretation:
    - +5: Strong uptrend with 5 higher lows
    - -3: Moderate downtrend with 3 lower highs

    Output columns:
    - trend_indicator_{window}: Signed value (direction + strength)
    - trend_indicator_norm_{window}: Normalized to [-1, +1]
    - trend_strength_{window}: Absolute value (strength only, for clustering)
    - trend_strength_norm_{window}: Normalized to [0, 1]

    Note: For K-Means clustering, use trend_strength (abs) to avoid
    redundancy with slope, which already captures direction.
    """

    def __init__(self, window: int = 10, slope_multiplier: float = 2.0):
        """
        Initialize calculator.

        Args:
            window: Number of days to look back for counting pattern (default 10).
                    The actual comparisons will be window-1 since we compare
                    each day with its previous day.
            slope_multiplier: Multiplier for slope window (default 2.0).
                              slope_window = int(window * slope_multiplier)
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if slope_multiplier < 1.0:
            raise ValueError(f"slope_multiplier must be >= 1.0, got {slope_multiplier}")

        self._window = window
        self._slope_multiplier = slope_multiplier
        self._slope_window = int(window * slope_multiplier)

    @property
    def name(self) -> str:
        return f"trend_indicator_w{self._window}_m{self._slope_multiplier:.1f}"

    @property
    def required_columns(self) -> Set[str]:
        return {'high', 'low', 'close'}

    @property
    def output_columns(self) -> Set[str]:
        return {
            f'trend_indicator_{self._window}',
            f'trend_indicator_norm_{self._window}',
            f'trend_strength_{self._window}',
            f'trend_strength_norm_{self._window}'
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicator columns to DataFrame.

        Args:
            df: Input DataFrame with 'high', 'low', and 'close' columns

        Returns:
            New DataFrame with added columns:
            - trend_indicator_{window}: Signed count (+higher_lows or -lower_highs)
            - trend_indicator_norm_{window}: Normalized to [-1, +1]
        """
        self.validate_input(df)
        result = df.copy()

        window = self._window
        slope_window = self._slope_window
        comparisons = window - 1  # Number of day-to-day comparisons

        # Calculate slope for trend direction
        x = np.arange(slope_window)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        def calc_slope(y):
            if len(y) < slope_window or np.isnan(y).any():
                return np.nan
            y_mean = y.mean()
            return ((x - x_mean) * (y - y_mean)).sum() / x_var

        slope = result['close'].rolling(slope_window).apply(calc_slope, raw=True)
        is_uptrend = slope > 0

        # Day-to-day comparisons (1 if condition met, 0 otherwise)
        higher_low = (result['low'] > result['low'].shift(1)).astype(float)
        lower_high = (result['high'] < result['high'].shift(1)).astype(float)

        # Rolling count within window
        higher_lows_count = higher_low.rolling(comparisons).sum()
        lower_highs_count = lower_high.rolling(comparisons).sum()

        # Apply trend filter with sign
        # Uptrend: use +higher_lows_count
        # Downtrend: use -lower_highs_count
        trend_indicator = np.where(
            is_uptrend,
            higher_lows_count,
            -lower_highs_count
        )

        result[f'trend_indicator_{window}'] = trend_indicator

        # Normalized version: divide by number of comparisons to get [-1, +1]
        result[f'trend_indicator_norm_{window}'] = trend_indicator / comparisons

        # Absolute value versions for clustering (avoids redundancy with slope)
        result[f'trend_strength_{window}'] = np.abs(trend_indicator)
        result[f'trend_strength_norm_{window}'] = np.abs(trend_indicator) / comparisons

        return result
