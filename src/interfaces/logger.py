"""Logger interface."""
from abc import ABC, abstractmethod
from typing import Any, Optional
from enum import Enum


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class ILogger(ABC):
    """Interface for structured logging."""

    @abstractmethod
    def debug(self, message: str, **context: Any) -> None:
        pass

    @abstractmethod
    def info(self, message: str, **context: Any) -> None:
        pass

    @abstractmethod
    def warning(self, message: str, **context: Any) -> None:
        pass

    @abstractmethod
    def error(self, message: str, exception: Optional[Exception] = None, **context: Any) -> None:
        pass
