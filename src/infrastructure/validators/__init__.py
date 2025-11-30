"""Data validators."""
from typing import List
from ...interfaces.validator import IValidator, CompositeValidator
from .ohlcv_validator import OHLCVValidator
from .series_length_validator import SeriesLengthValidator
from .gap_validator import GapValidator


def create_default_validator(
    min_length: int = 252,
    max_window: int = 60,
    max_gap_days: int = 5
) -> IValidator:
    """Create default validation pipeline."""
    return CompositeValidator([
        OHLCVValidator(),
        SeriesLengthValidator(min_length, max_window),
        GapValidator(max_gap_days)
    ])


__all__ = [
    'OHLCVValidator',
    'SeriesLengthValidator',
    'GapValidator',
    'create_default_validator'
]
