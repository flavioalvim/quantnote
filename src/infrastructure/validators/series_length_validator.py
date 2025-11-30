"""Series length validator."""
import pandas as pd
from ...interfaces.validator import IValidator, ValidationResult


class SeriesLengthValidator(IValidator):
    """Validates that series has minimum required length."""

    def __init__(self, min_length: int, max_window: int):
        """
        Args:
            min_length: Minimum required data points
            max_window: Maximum window size that will be used
        """
        self.min_length = min_length
        self.max_window = max_window

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        actual_length = len(df)

        if actual_length < self.min_length:
            errors.append(
                f"Series has {actual_length} points, minimum required is {self.min_length}"
            )

        if actual_length < self.max_window * 3:
            warnings.append(
                f"Series length ({actual_length}) is less than 3x max window ({self.max_window}). "
                "Results may be unreliable."
            )

        # Check for usable data after window application
        usable_points = actual_length - self.max_window
        if usable_points < 50:
            warnings.append(
                f"Only {usable_points} usable data points after applying windows"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
