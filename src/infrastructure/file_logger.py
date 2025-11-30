"""File-based structured logger implementation."""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..interfaces.logger import ILogger


class FileLogger(ILogger):
    """Structured file logger with JSON formatting."""

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: int = logging.INFO
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            # File handler with JSON formatting
            log_file = self.log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

            # Console handler
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            self.logger.addHandler(console)

    def _format_message(self, level: str, message: str, **context: Any) -> str:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **context
        }
        return json.dumps(log_entry, default=str)

    def debug(self, message: str, **context: Any) -> None:
        self.logger.debug(self._format_message("DEBUG", message, **context))

    def info(self, message: str, **context: Any) -> None:
        self.logger.info(self._format_message("INFO", message, **context))

    def warning(self, message: str, **context: Any) -> None:
        self.logger.warning(self._format_message("WARNING", message, **context))

    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **context: Any
    ) -> None:
        if exception:
            context["exception"] = str(exception)
            context["exception_type"] = type(exception).__name__
        self.logger.error(self._format_message("ERROR", message, **context))


class NullLogger(ILogger):
    """No-op logger for testing."""

    def debug(self, message: str, **context: Any) -> None:
        pass

    def info(self, message: str, **context: Any) -> None:
        pass

    def warning(self, message: str, **context: Any) -> None:
        pass

    def error(self, message: str, exception: Optional[Exception] = None, **context: Any) -> None:
        pass
