"""Validator interface."""
from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge two validation results."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings
        )


class IValidator(ABC):
    """Interface for data validation."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with status and messages
        """
        pass


class CompositeValidator(IValidator):
    """Combines multiple validators."""

    def __init__(self, validators: List[IValidator]):
        self.validators = validators

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        for validator in self.validators:
            result = result.merge(validator.validate(df))
        return result
