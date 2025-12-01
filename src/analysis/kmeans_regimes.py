"""K-Means regime classifier with persistence."""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..interfaces.model_store import IModelStore
from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger


@dataclass
class ClusterStatistics:
    """Statistics for a cluster."""
    cluster_id: int
    count: int
    percentage: float
    feature_means: Dict[str, float]
    future_return_mean: Optional[float]
    future_return_std: Optional[float]
    interpretation: Optional[str]


class KMeansRegimeClassifier:
    """K-Means clustering for regime classification with persistence."""

    def __init__(
        self,
        n_clusters: int = 3,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        logger: Optional[ILogger] = None
    ):
        self.n_clusters = n_clusters
        self.feature_columns = feature_columns
        self.random_state = random_state
        self.logger = logger or NullLogger()

        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self._is_fitted = False
        self._cluster_stats: List[ClusterStatistics] = []

    def fit(self, df: pd.DataFrame) -> 'KMeansRegimeClassifier':
        """Fit the K-Means model."""
        # Auto-detect feature columns if not specified
        if self.feature_columns is None:
            self.feature_columns = [
                c for c in df.columns
                if 'slope' in c or 'volatility' in c or 'log_return_rolling' in c or 'ma_dist' in c
            ]
            self.logger.info(
                "Auto-detected feature columns",
                columns=self.feature_columns
            )

        # Get valid data
        feature_data = df[self.feature_columns].dropna()

        if len(feature_data) < self.n_clusters * 10:
            raise ValueError(
                f"Not enough valid data points ({len(feature_data)}) "
                f"for {self.n_clusters} clusters"
            )

        # Fit scaler and kmeans
        X_scaled = self.scaler.fit_transform(feature_data)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            algorithm='lloyd'  # Use lloyd algorithm for better parallelization
        )
        self.kmeans.fit(X_scaled)

        self._is_fitted = True

        self.logger.info(
            "K-Means fitted",
            n_clusters=self.n_clusters,
            n_samples=len(feature_data)
        )

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict cluster labels for DataFrame."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        result = df.copy()

        # Get valid indices
        feature_data = result[self.feature_columns].dropna()
        valid_idx = feature_data.index

        # Predict
        X_scaled = self.scaler.transform(feature_data)
        clusters = self.kmeans.predict(X_scaled)

        # Add to result
        result['cluster'] = np.nan
        result.loc[valid_idx, 'cluster'] = clusters

        return result

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and predict in one step."""
        self.fit(df)
        return self.predict(df)

    def compute_statistics(
        self,
        df: pd.DataFrame,
        future_return_column: Optional[str] = None
    ) -> List[ClusterStatistics]:
        """Compute statistics for each cluster."""
        if 'cluster' not in df.columns:
            raise ValueError("DataFrame must have 'cluster' column")

        valid_df = df.dropna(subset=['cluster'])
        total = len(valid_df)

        stats = []
        for cluster_id in range(self.n_clusters):
            mask = valid_df['cluster'] == cluster_id
            cluster_data = valid_df[mask]

            # Feature means
            feature_means = {
                col: cluster_data[col].mean()
                for col in self.feature_columns
            }

            # Future return stats
            future_mean = None
            future_std = None
            if future_return_column and future_return_column in df.columns:
                future_rets = cluster_data[future_return_column].dropna()
                if len(future_rets) > 0:
                    future_mean = future_rets.mean()
                    future_std = future_rets.std()

            stats.append(ClusterStatistics(
                cluster_id=cluster_id,
                count=len(cluster_data),
                percentage=len(cluster_data) / total * 100 if total > 0 else 0,
                feature_means=feature_means,
                future_return_mean=future_mean,
                future_return_std=future_std,
                interpretation=None
            ))

        self._cluster_stats = stats
        return stats

    def interpret_clusters(
        self,
        df: pd.DataFrame,
        slope_column: str
    ) -> Dict[int, str]:
        """Interpret clusters as bull/bear/flat based on slope."""
        if 'cluster' not in df.columns:
            raise ValueError("DataFrame must have 'cluster' column")

        slope_std = df[slope_column].std()
        interpretations = {}

        for cluster_id in range(self.n_clusters):
            mask = df['cluster'] == cluster_id
            mean_slope = df.loc[mask, slope_column].mean()

            if pd.isna(mean_slope):
                interpretations[cluster_id] = 'unknown'
            elif mean_slope > slope_std * 0.3:
                interpretations[cluster_id] = 'bull'
            elif mean_slope < -slope_std * 0.3:
                interpretations[cluster_id] = 'bear'
            else:
                interpretations[cluster_id] = 'flat'

        # Update stats with interpretations
        for stat in self._cluster_stats:
            stat.interpretation = interpretations.get(stat.cluster_id)

        return interpretations

    def save(self, model_store: IModelStore, model_id: str) -> None:
        """Save model to store."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_clusters': self.n_clusters
        }

        model_store.save(
            model=model_data,
            model_id=model_id,
            model_type='kmeans_regime',
            parameters={
                'n_clusters': self.n_clusters,
                'feature_columns': self.feature_columns
            },
            metrics={}
        )

        self.logger.info("Model saved", model_id=model_id)

    @classmethod
    def load(
        cls,
        model_store: IModelStore,
        model_id: str,
        logger: Optional[ILogger] = None
    ) -> 'KMeansRegimeClassifier':
        """Load model from store."""
        model_data, metadata = model_store.load(model_id)

        instance = cls(
            n_clusters=model_data['n_clusters'],
            feature_columns=model_data['feature_columns'],
            logger=logger
        )
        instance.kmeans = model_data['kmeans']
        instance.scaler = model_data['scaler']
        instance._is_fitted = True

        return instance
