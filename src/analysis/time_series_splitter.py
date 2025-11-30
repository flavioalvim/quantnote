"""Time series splitter for train/test and walk-forward validation."""
import pandas as pd
from typing import Tuple, Iterator
from dataclasses import dataclass


@dataclass
class Split:
    """A train/test split."""
    train: pd.DataFrame
    test: pd.DataFrame
    fold: int


class TimeSeriesSplitter:
    """
    Sequential time series splitter.
    Prevents lookahead bias by ensuring test data always comes after train.
    """

    def __init__(self, train_ratio: float = 0.7):
        """
        Args:
            train_ratio: Proportion of data for training (0.5 to 0.9)
        """
        if not 0.5 <= train_ratio <= 0.9:
            raise ValueError("train_ratio must be between 0.5 and 0.9")
        self.train_ratio = train_ratio

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple train/test split.

        Returns:
            (train_df, test_df)
        """
        split_idx = int(len(df) * self.train_ratio)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def walk_forward_split(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        min_train_size: int = 100
    ) -> Iterator[Split]:
        """
        Walk-forward cross-validation splits.

        Each fold uses all previous data for training and next segment for testing.

        Args:
            df: DataFrame to split
            n_folds: Number of folds
            min_train_size: Minimum training set size

        Yields:
            Split objects with train and test DataFrames
        """
        n = len(df)
        fold_size = (n - min_train_size) // n_folds

        if fold_size < 10:
            raise ValueError(
                f"Not enough data for {n_folds} folds. "
                f"Need at least {min_train_size + 10 * n_folds} rows."
            )

        for i in range(n_folds):
            train_end = min_train_size + i * fold_size
            test_end = train_end + fold_size

            if test_end > n:
                test_end = n

            yield Split(
                train=df.iloc[:train_end].copy(),
                test=df.iloc[train_end:test_end].copy(),
                fold=i
            )

    def expanding_window_split(
        self,
        df: pd.DataFrame,
        initial_train_size: int = 252,
        step_size: int = 21
    ) -> Iterator[Split]:
        """
        Expanding window splits (each train set grows).

        Args:
            df: DataFrame to split
            initial_train_size: Initial training set size
            step_size: Number of periods to add each iteration

        Yields:
            Split objects
        """
        n = len(df)
        fold = 0

        train_end = initial_train_size
        while train_end + step_size <= n:
            test_end = min(train_end + step_size, n)

            yield Split(
                train=df.iloc[:train_end].copy(),
                test=df.iloc[train_end:test_end].copy(),
                fold=fold
            )

            train_end = test_end
            fold += 1
