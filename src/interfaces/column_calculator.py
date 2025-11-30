"""Column Calculator interface."""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Set


class IColumnCalculator(ABC):
    """Interface for column/indicator calculators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this calculator."""
        pass

    @property
    @abstractmethod
    def required_columns(self) -> Set[str]:
        """Columns required in input DataFrame."""
        pass

    @property
    @abstractmethod
    def output_columns(self) -> Set[str]:
        """Columns that will be added to DataFrame."""
        pass

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated columns to DataFrame.

        Args:
            df: Input DataFrame (not modified)

        Returns:
            New DataFrame with added columns

        Raises:
            CalculatorError: If calculation fails
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns."""
        missing = self.required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Calculator '{self.name}' missing columns: {missing}. "
                f"Available: {set(df.columns)}"
            )
