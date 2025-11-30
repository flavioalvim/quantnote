"""Yahoo Finance data source with rate limiting."""
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional
from functools import wraps
import time

from ..interfaces.data_source import IDataSource
from ..interfaces.logger import ILogger
from ..domain.exceptions import DataSourceError
from .file_logger import NullLogger


def rate_limit(calls: int, period: int):
    """Rate limiting decorator."""
    min_interval = period / calls
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator


class YahooDataSource(IDataSource):
    """Yahoo Finance data source with rate limiting."""

    def __init__(
        self,
        calls_per_minute: int = 5,
        logger: Optional[ILogger] = None
    ):
        self.calls_per_minute = calls_per_minute
        self.logger = logger or NullLogger()
        self._apply_rate_limit()

    def _apply_rate_limit(self):
        """Apply rate limiting to fetch method."""
        self._fetch_internal = rate_limit(
            self.calls_per_minute, 60
        )(self._fetch_internal)

    def _fetch_internal(self, ticker: str, **kwargs) -> pd.DataFrame:
        """Internal fetch method (rate limited)."""
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.history(**kwargs)

    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""

        self.logger.info(
            "Fetching data from Yahoo Finance",
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date)
        )

        try:
            if start_date is None:
                df = self._fetch_internal(ticker, period="max")
            else:
                df = self._fetch_internal(
                    ticker,
                    start=start_date,
                    end=end_date
                )

            if df.empty:
                raise DataSourceError(f"No data returned for ticker {ticker}")

            # Normalize column names
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            # Rename 'adj_close' if present
            if 'adj_close' in df.columns:
                df = df.drop(columns=['close'])
                df = df.rename(columns={'adj_close': 'close'})

            # Select only OHLCV columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in required_cols if c in df.columns]

            if len(available_cols) < len(required_cols):
                missing = set(required_cols) - set(available_cols)
                raise DataSourceError(f"Missing columns: {missing}")

            df = df[required_cols].copy()

            # Normalize date
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            self.logger.info(
                "Data fetched successfully",
                ticker=ticker,
                rows=len(df),
                date_range=f"{df['date'].min()} to {df['date'].max()}"
            )

            return df

        except Exception as e:
            self.logger.error(
                "Failed to fetch data",
                exception=e,
                ticker=ticker
            )
            raise DataSourceError(f"Failed to fetch {ticker}: {e}") from e
