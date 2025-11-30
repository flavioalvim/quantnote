"""Log Return calculator."""
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator


class LogReturnCalculator(IColumnCalculator):
    """Calculates daily log return and rolling sum."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"log_return_w{self._window}"

    @property
    def required_columns(self) -> Set[str]:
        return {'close'}

    @property
    def output_columns(self) -> Set[str]:
        return {'log_return', f'log_return_rolling_{self._window}'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        # Daily log return
        result['log_return'] = np.log(result['close'] / result['close'].shift(1))

        # Rolling sum (= log of total return over window)
        result[f'log_return_rolling_{self._window}'] = (
            result['log_return'].rolling(window=self._window).sum()
        )

        return result
