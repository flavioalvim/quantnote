"""Data Source interface."""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from datetime import datetime


class IDataSource(ABC):
    """Interface for price data sources."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            ticker: Asset code (e.g., 'BOVA11.SA')
            start_date: Start date (None = maximum available)
            end_date: End date (None = today)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Prices already adjusted for splits and dividends.

        Raises:
            DataSourceError: If fetch fails
        """
        pass
