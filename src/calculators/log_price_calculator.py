"""Log Price calculator."""
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator


class LogPriceCalculator(IColumnCalculator):
    """Calculates logarithm of closing price."""

    @property
    def name(self) -> str:
        return "log_price"

    @property
    def required_columns(self) -> Set[str]:
        return {'close'}

    @property
    def output_columns(self) -> Set[str]:
        return {'log_close'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()
        result['log_close'] = np.log(result['close'])
        return result
