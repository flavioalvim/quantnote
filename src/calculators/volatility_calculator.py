"""Volatility calculator."""
import pandas as pd
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator


class VolatilityCalculator(IColumnCalculator):
    """Calculates rolling volatility (std of returns)."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"volatility_w{self._window}"

    @property
    def required_columns(self) -> Set[str]:
        return {'log_return'}

    @property
    def output_columns(self) -> Set[str]:
        return {f'volatility_{self._window}'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        result[f'volatility_{self._window}'] = (
            result['log_return'].rolling(window=self._window).std()
        )

        return result
