"""Regime analysis module."""
from .time_series_splitter import TimeSeriesSplitter, Split
from .regime_classifier import ManualRegimeClassifier
from .kmeans_regimes import KMeansRegimeClassifier, ClusterStatistics
from .probability_calculator import ProbabilityCalculator, ConditionalProbability, SeparationMetrics
from .dual_probability_calculator import (
    DualProbabilityCalculator,
    DualProbabilityResult,
    DualSeparationMetrics
)

__all__ = [
    'TimeSeriesSplitter',
    'Split',
    'ManualRegimeClassifier',
    'KMeansRegimeClassifier',
    'ClusterStatistics',
    'ProbabilityCalculator',
    'ConditionalProbability',
    'SeparationMetrics',
    'DualProbabilityCalculator',
    'DualProbabilityResult',
    'DualSeparationMetrics'
]
