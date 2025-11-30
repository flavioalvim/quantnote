"""OHLCV data validator."""
import pandas as pd
from typing import Set

from ...interfaces.validator import IValidator, ValidationResult


class OHLCVValidator(IValidator):
    """Validates OHLCV data structure and values."""

    REQUIRED_COLUMNS: Set[str] = {'date', 'open', 'high', 'low', 'close', 'volume'}

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        # Check required columns
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(is_valid=False, errors=errors)

        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors)

        # Check for non-positive prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            non_positive = (df[col] <= 0).sum()
            if non_positive > 0:
                errors.append(f"Column '{col}' has {non_positive} non-positive values")

        # Check OHLC relationships
        invalid_high = (df['high'] < df['low']).sum()
        if invalid_high > 0:
            errors.append(f"{invalid_high} rows have high < low")

        invalid_open_high = (df['open'] > df['high']).sum()
        if invalid_open_high > 0:
            warnings.append(f"{invalid_open_high} rows have open > high")

        invalid_close_low = (df['close'] < df['low']).sum()
        if invalid_close_low > 0:
            warnings.append(f"{invalid_close_low} rows have close < low")

        # Check for NaN values
        for col in price_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                warnings.append(f"Column '{col}' has {nan_count} NaN values")

        # Check for duplicate dates
        if 'date' in df.columns:
            duplicates = df['date'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"{duplicates} duplicate dates found")

        # Check date ordering
        if 'date' in df.columns and len(df) > 1:
            if not df['date'].is_monotonic_increasing:
                warnings.append("Dates are not in ascending order")

        # Check for extreme price changes (potential data errors)
        if len(df) > 1:
            returns = df['close'].pct_change().abs()
            extreme_moves = (returns > 0.5).sum()  # >50% daily move
            if extreme_moves > 0:
                warnings.append(f"{extreme_moves} extreme price moves (>50%) detected")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
