"""Future Touch Return calculator - max/min returns within horizon window."""
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator


class FutureTouchCalculator(IColumnCalculator):
    """
    Calculates the maximum and minimum returns achievable within a horizon window.

    For each row t, calculates:
    - future_touch_max: max(high[t+1:t+horizon+1]) / close[t] - 1 (log return)
    - future_touch_min: min(low[t+1:t+horizon+1]) / close[t] - 1 (log return)

    This answers: "What's the probability of TOUCHING a target price within H periods?"
    as opposed to "What's the probability of CLOSING at a target price after H periods?"
    """

    def __init__(self, horizon: int = 7):
        self._horizon = horizon

    @property
    def name(self) -> str:
        return f"future_touch_h{self._horizon}"

    @property
    def required_columns(self) -> Set[str]:
        return {'close', 'high', 'low'}

    @property
    def output_columns(self) -> Set[str]:
        return {
            f'log_return_touch_max_{self._horizon}',
            f'log_return_touch_min_{self._horizon}'
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        # Shift high/low to look at future periods (t+1 onwards)
        high_shifted = result['high'].shift(-1)
        low_shifted = result['low'].shift(-1)

        # Rolling max/min over the horizon window (looking forward)
        # We need to use rolling with min_periods=1 to handle edge cases
        # Since we shifted by -1, rolling(horizon) will look at [t+1, t+horizon]
        future_high_max = high_shifted.rolling(
            window=self._horizon,
            min_periods=1
        ).max().shift(-(self._horizon - 1))

        future_low_min = low_shifted.rolling(
            window=self._horizon,
            min_periods=1
        ).min().shift(-(self._horizon - 1))

        # Alternative approach using iloc indexing for clarity
        # This is more explicit about what we're calculating
        future_high_max = pd.Series(index=result.index, dtype=float)
        future_low_min = pd.Series(index=result.index, dtype=float)

        for i in range(len(result) - self._horizon):
            # Window from t+1 to t+horizon (inclusive)
            window_high = result['high'].iloc[i + 1 : i + self._horizon + 1]
            window_low = result['low'].iloc[i + 1 : i + self._horizon + 1]

            future_high_max.iloc[i] = window_high.max()
            future_low_min.iloc[i] = window_low.min()

        # Calculate log returns relative to current close
        result[f'log_return_touch_max_{self._horizon}'] = np.log(
            future_high_max / result['close']
        )
        result[f'log_return_touch_min_{self._horizon}'] = np.log(
            future_low_min / result['close']
        )

        return result


class FutureTouchCalculatorVectorized(IColumnCalculator):
    """
    Vectorized version of FutureTouchCalculator for better performance.
    Uses numpy strides for efficient rolling window calculation.
    """

    def __init__(self, horizon: int = 7):
        self._horizon = horizon

    @property
    def name(self) -> str:
        return f"future_touch_vec_h{self._horizon}"

    @property
    def required_columns(self) -> Set[str]:
        return {'close', 'high', 'low'}

    @property
    def output_columns(self) -> Set[str]:
        return {
            f'log_return_touch_max_{self._horizon}',
            f'log_return_touch_min_{self._horizon}'
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()
        n = len(result)
        horizon = self._horizon

        # Pre-allocate arrays
        future_high_max = np.full(n, np.nan)
        future_low_min = np.full(n, np.nan)

        high_arr = result['high'].values
        low_arr = result['low'].values
        close_arr = result['close'].values

        # Calculate for each valid position
        for i in range(n - horizon):
            start = i + 1
            end = i + horizon + 1
            future_high_max[i] = np.max(high_arr[start:end])
            future_low_min[i] = np.min(low_arr[start:end])

        # Calculate log returns
        with np.errstate(divide='ignore', invalid='ignore'):
            result[f'log_return_touch_max_{self._horizon}'] = np.log(
                future_high_max / close_arr
            )
            result[f'log_return_touch_min_{self._horizon}'] = np.log(
                future_low_min / close_arr
            )

        return result
