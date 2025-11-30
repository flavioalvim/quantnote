"""Parquet-based data repository."""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from ..interfaces.repository import IRepository, FileMetadata
from ..interfaces.logger import ILogger
from .file_logger import NullLogger


class ParquetRepository(IRepository):
    """Repository that persists DataFrames in Parquet format."""

    def __init__(
        self,
        data_dir: str = "data",
        logger: Optional[ILogger] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or NullLogger()

    def _generate_filename(self, ticker: str) -> str:
        """Generate filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_clean = ticker.replace(".", "_").replace("/", "_")
        return f"{ticker_clean}_{timestamp}.parquet"

    def _parse_filename(self, filename: str) -> tuple:
        """Parse ticker and timestamp from filename."""
        parts = filename.replace(".parquet", "").rsplit("_", 2)
        if len(parts) >= 3:
            ticker = "_".join(parts[:-2])
            date_str = parts[-2]
            time_str = parts[-1]
            try:
                created = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                return ticker, created
            except ValueError:
                return filename, None
        return filename, None

    def save(self, df: pd.DataFrame, ticker: str) -> FileMetadata:
        """Save DataFrame in Parquet format."""
        filename = self._generate_filename(ticker)
        filepath = self.data_dir / filename

        df.to_parquet(filepath, index=False)

        metadata = FileMetadata(
            filename=filename,
            ticker=ticker,
            created_at=datetime.now(),
            row_count=len(df),
            date_range=(
                df['date'].min() if 'date' in df.columns else None,
                df['date'].max() if 'date' in df.columns else None
            )
        )

        self.logger.info(
            "Data saved",
            filename=filename,
            rows=len(df)
        )

        return metadata

    def load(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load most recent DataFrame for ticker."""
        files = self.list_files(ticker)
        if not files:
            return None

        latest = files[-1]  # Sorted by date
        return self.load_by_filename(latest.filename)

    def load_by_filename(self, filename: str) -> pd.DataFrame:
        """Load specific file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        return pd.read_parquet(filepath)

    def list_files(self, ticker: Optional[str] = None) -> List[FileMetadata]:
        """List available files."""
        files = list(self.data_dir.glob("*.parquet"))

        result = []
        for f in files:
            parsed_ticker, created = self._parse_filename(f.name)

            if ticker:
                ticker_clean = ticker.replace(".", "_").replace("/", "_")
                if not f.name.startswith(ticker_clean):
                    continue

            # Read row count without loading full file
            try:
                df_meta = pd.read_parquet(f)
                row_count = len(df_meta)
                date_range = (df_meta['date'].min(), df_meta['date'].max()) if 'date' in df_meta.columns else (None, None)
            except Exception:
                row_count = 0
                date_range = (None, None)

            result.append(FileMetadata(
                filename=f.name,
                ticker=parsed_ticker,
                created_at=created or datetime.fromtimestamp(f.stat().st_mtime),
                row_count=row_count,
                date_range=date_range
            ))

        return sorted(result, key=lambda x: x.created_at)

    def delete(self, filename: str) -> bool:
        """Delete a file."""
        filepath = self.data_dir / filename
        if filepath.exists():
            filepath.unlink()
            self.logger.info("File deleted", filename=filename)
            return True
        return False
