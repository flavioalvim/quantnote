"""Repository interface."""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileMetadata:
    """Metadata for saved files."""
    filename: str
    ticker: str
    created_at: datetime
    row_count: int
    date_range: tuple


class IRepository(ABC):
    """Interface for data persistence."""

    @abstractmethod
    def save(self, df: pd.DataFrame, ticker: str) -> FileMetadata:
        """Save DataFrame with metadata."""
        pass

    @abstractmethod
    def load(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load most recent DataFrame for ticker."""
        pass

    @abstractmethod
    def load_by_filename(self, filename: str) -> pd.DataFrame:
        """Load specific file."""
        pass

    @abstractmethod
    def list_files(self, ticker: Optional[str] = None) -> List[FileMetadata]:
        """List available files."""
        pass

    @abstractmethod
    def delete(self, filename: str) -> bool:
        """Delete a file."""
        pass
