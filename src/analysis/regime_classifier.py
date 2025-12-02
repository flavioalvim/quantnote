"""Manual regime classifier using thresholds."""
import pandas as pd
from typing import Optional

from ..domain.value_objects import Regime


class SlopeOnlyClassifier:
    """Classifies regimes using only slope (bull/bear/flat)."""

    def __init__(
        self,
        slope_column: str,
        slope_threshold: Optional[float] = None
    ):
        """
        Initialize slope-only classifier.

        Args:
            slope_column: Name of the slope column
            slope_threshold: Threshold for bull/bear classification.
                           If None, uses 0.5 * std of slope.
        """
        self.slope_column = slope_column
        self.slope_threshold = slope_threshold
        self._slope_threshold: Optional[float] = None

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime column to DataFrame with bull/bear/flat classification."""
        result = df.copy()

        # Auto-calculate threshold if not provided
        slope_thresh = self.slope_threshold
        if slope_thresh is None:
            slope_thresh = result[self.slope_column].std() * 0.5

        # Store threshold for reference
        self._slope_threshold = slope_thresh

        # Classify based on slope only
        def classify_row(row):
            slope = row[self.slope_column]
            if pd.isna(slope):
                return None

            if slope > slope_thresh:
                return 'bull'
            elif slope < -slope_thresh:
                return 'bear'
            else:
                return 'flat'

        result['regime'] = result.apply(classify_row, axis=1)

        return result

    def get_thresholds(self) -> dict:
        """Return the thresholds used for classification."""
        return {
            'slope_threshold': self._slope_threshold
        }


class ManualRegimeClassifier:
    """Classifies regimes using manual thresholds."""

    def __init__(
        self,
        slope_column: str,
        volatility_column: str,
        slope_threshold: Optional[float] = None,
        volatility_threshold: Optional[float] = None
    ):
        self.slope_column = slope_column
        self.volatility_column = volatility_column
        self.slope_threshold = slope_threshold
        self.volatility_threshold = volatility_threshold
        self._slope_threshold: Optional[float] = None
        self._volatility_threshold: Optional[float] = None

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime column to DataFrame."""
        result = df.copy()

        # Auto-calculate thresholds if not provided
        slope_thresh = self.slope_threshold
        if slope_thresh is None:
            slope_thresh = result[self.slope_column].std() * 0.5

        vol_thresh = self.volatility_threshold
        if vol_thresh is None:
            vol_thresh = result[self.volatility_column].median()

        # Store thresholds for reference
        self._slope_threshold = slope_thresh
        self._volatility_threshold = vol_thresh

        # Classify each row
        def classify_row(row):
            if pd.isna(row[self.slope_column]) or pd.isna(row[self.volatility_column]):
                return None

            regime = Regime.from_indicators(
                slope=row[self.slope_column],
                volatility=row[self.volatility_column],
                slope_threshold=slope_thresh,
                volatility_threshold=vol_thresh
            )
            return regime.name

        result['regime'] = result.apply(classify_row, axis=1)

        return result

    def get_thresholds(self) -> dict:
        """Return the thresholds used for classification."""
        return {
            'slope_threshold': self._slope_threshold,
            'volatility_threshold': self._volatility_threshold
        }
