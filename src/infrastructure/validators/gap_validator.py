"""Gap validator for time series data."""
import pandas as pd
from ...interfaces.validator import IValidator, ValidationResult


class GapValidator(IValidator):
    """Validates for data gaps in time series."""

    def __init__(self, max_gap_days: int = 5):
        """
        Args:
            max_gap_days: Maximum allowed gap between trading days
        """
        self.max_gap_days = max_gap_days

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        if 'date' not in df.columns or len(df) < 2:
            return ValidationResult(is_valid=True)

        # Calculate gaps between dates
        dates = pd.to_datetime(df['date'])
        gaps = dates.diff().dt.days

        # Find large gaps (excluding weekends which are typically 3 days)
        large_gaps = gaps[gaps > self.max_gap_days]

        if len(large_gaps) > 0:
            gap_info = [
                f"{dates.iloc[i].date()} ({gap} days)"
                for i, gap in large_gaps.items()
            ]
            if len(large_gaps) > 5:
                warnings.append(
                    f"{len(large_gaps)} gaps > {self.max_gap_days} days found. "
                    f"First 5: {gap_info[:5]}"
                )
            else:
                warnings.append(
                    f"Gaps > {self.max_gap_days} days found: {gap_info}"
                )

        return ValidationResult(
            is_valid=True,  # Gaps are warnings, not errors
            errors=errors,
            warnings=warnings
        )
