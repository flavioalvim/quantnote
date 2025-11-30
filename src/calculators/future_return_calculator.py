"""Future Return calculator."""
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator


class FutureReturnCalculator(IColumnCalculator):
    """Calculates future log return over H periods."""

    def __init__(self, horizon: int = 3):
        self._horizon = horizon

    @property
    def name(self) -> str:
        return f"future_return_h{self._horizon}"

    @property
    def required_columns(self) -> Set[str]:
        return {'close'}

    @property
    def output_columns(self) -> Set[str]:
        return {f'log_return_future_{self._horizon}'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        # Future return: log(P_{t+H} / P_t)
        result[f'log_return_future_{self._horizon}'] = (
            np.log(result['close'].shift(-self._horizon) / result['close'])
        )

        return result
