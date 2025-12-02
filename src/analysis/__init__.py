"""Regime analysis module."""
from .time_series_splitter import TimeSeriesSplitter, Split
from .regime_classifier import ManualRegimeClassifier, SlopeOnlyClassifier
from .kmeans_regimes import KMeansRegimeClassifier, ClusterStatistics
from .probability_calculator import ProbabilityCalculator, ConditionalProbability, SeparationMetrics
from .dual_probability_calculator import (
    DualProbabilityCalculator,
    DualProbabilityResult,
    DualSeparationMetrics
)
from .cluster_explainer import (
    IClusterExplainer,
    DecisionTreeExplainer,
    RandomForestExplainer,
    CompositeExplainer,
    Condition,
    ClusterRule,
    ExplainerMetrics
)

__all__ = [
    'TimeSeriesSplitter',
    'Split',
    'ManualRegimeClassifier',
    'SlopeOnlyClassifier',
    'KMeansRegimeClassifier',
    'ClusterStatistics',
    'ProbabilityCalculator',
    'ConditionalProbability',
    'SeparationMetrics',
    'DualProbabilityCalculator',
    'DualProbabilityResult',
    'DualSeparationMetrics',
    # Cluster Explainers
    'IClusterExplainer',
    'DecisionTreeExplainer',
    'RandomForestExplainer',
    'CompositeExplainer',
    'Condition',
    'ClusterRule',
    'ExplainerMetrics'
]
