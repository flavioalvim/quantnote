"""Domain exceptions for QuantNote."""


class QuantNoteError(Exception):
    """Base exception for QuantNote."""
    pass


class DataSourceError(QuantNoteError):
    """Error fetching data from source."""
    pass


class ValidationError(QuantNoteError):
    """Data validation failed."""
    def __init__(self, errors: list):
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")


class CalculatorError(QuantNoteError):
    """Error in calculator execution."""
    pass


class PipelineError(QuantNoteError):
    """Error in pipeline execution."""
    pass


class ModelNotFoundError(QuantNoteError):
    """Requested model not found."""
    pass


class InsufficientDataError(QuantNoteError):
    """Not enough data for analysis."""
    pass
