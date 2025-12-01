"""Moving Average Distance calculator with min-max normalization."""
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator


class MADistanceCalculator(IColumnCalculator):
    """
    Calculates normalized distance between fast and slow moving averages.

    The distance is calculated as (fast_ma - slow_ma) and normalized to [-1, 1]
    using min-max normalization over the entire dataset.

    Positive values indicate bullish trend (fast > slow).
    Negative values indicate bearish trend (fast < slow).
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21, price_column: str = 'close'):
        """
        Initialize the MA Distance calculator.

        Args:
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            price_column: Column to use for MA calculation (default: 'close')
        """
        if fast_period >= slow_period:
            raise ValueError(f"fast_period ({fast_period}) must be less than slow_period ({slow_period})")

        self._fast_period = fast_period
        self._slow_period = slow_period
        self._price_column = price_column

    @property
    def name(self) -> str:
        return f"ma_distance_{self._fast_period}_{self._slow_period}"

    @property
    def required_columns(self) -> Set[str]:
        return {self._price_column}

    @property
    def output_columns(self) -> Set[str]:
        return {f'ma_dist_{self._fast_period}_{self._slow_period}'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate normalized MA distance.

        Returns:
            DataFrame with added column 'ma_dist_{fast}_{slow}' normalized to [-1, 1]
        """
        self.validate_input(df)
        result = df.copy()

        # Calculate moving averages
        prices = result[self._price_column]
        fast_ma = prices.rolling(window=self._fast_period).mean()
        slow_ma = prices.rolling(window=self._slow_period).mean()

        # Calculate raw distance
        distance = fast_ma - slow_ma

        # Min-max normalization to [-1, 1]
        col_name = f'ma_dist_{self._fast_period}_{self._slow_period}'
        result[col_name] = self._normalize_to_minus_one_one(distance)

        return result

    @staticmethod
    def _normalize_to_minus_one_one(series: pd.Series) -> pd.Series:
        """
        Normalize series to [-1, 1] using min-max normalization.

        Formula: 2 * (x - min) / (max - min) - 1
        """
        min_val = series.min()
        max_val = series.max()

        if max_val == min_val:
            # All values are the same, return 0
            return pd.Series(0.0, index=series.index)

        # Scale to [0, 1] then to [-1, 1]
        normalized = 2 * (series - min_val) / (max_val - min_val) - 1

        return normalized
