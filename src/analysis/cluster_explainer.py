"""Cluster explainers using Decision Tree and Random Forest."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


@dataclass
class Condition:
    """A single condition in a decision rule."""
    feature: str
    operator: str  # '<=', '>'
    threshold: float

    def __str__(self) -> str:
        return f"{self.feature} {self.operator} {self.threshold:.4f}"


@dataclass
class ClusterRule:
    """A rule that defines a cluster."""
    cluster_id: int
    conditions: List[Condition]
    samples: int
    accuracy: float

    def __str__(self) -> str:
        conditions_str = " AND ".join(str(c) for c in self.conditions)
        return f"IF {conditions_str} THEN cluster = {self.cluster_id} (n={self.samples}, acc={self.accuracy:.1%})"


@dataclass
class ExplainerMetrics:
    """Metrics from the explainer model."""
    accuracy: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    n_samples: int
    n_features: int
    n_clusters: int


class IClusterExplainer(ABC):
    """Interface for cluster explainers."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, cluster_column: str,
            feature_columns: Optional[List[str]] = None) -> 'IClusterExplainer':
        """
        Fit the explainer to predict clusters from features.

        Args:
            df: DataFrame with cluster labels and features
            cluster_column: Name of the cluster column
            feature_columns: List of feature columns (auto-detect if None)

        Returns:
            self
        """
        pass

    @abstractmethod
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances as {feature_name: importance}."""
        pass

    @abstractmethod
    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N most important features as [(name, importance), ...]."""
        pass

    @abstractmethod
    def get_metrics(self) -> ExplainerMetrics:
        """Get model metrics (accuracy, cv scores, etc.)."""
        pass

    @abstractmethod
    def explain_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """Get explanation for a specific cluster."""
        pass

    @abstractmethod
    def plot_importances(self, ax: Optional[plt.Axes] = None,
                         top_n: Optional[int] = None) -> plt.Figure:
        """Plot feature importances as horizontal bar chart."""
        pass


class DecisionTreeExplainer(IClusterExplainer):
    """
    Explains clusters using a Decision Tree classifier.

    Provides interpretable rules and tree visualization.
    """

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 20,
                 random_state: int = 42, cv_folds: int = 5):
        """
        Initialize Decision Tree explainer.

        Args:
            max_depth: Maximum depth of the tree
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds for accuracy estimation
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.cv_folds = cv_folds

        self._model: Optional[DecisionTreeClassifier] = None
        self._feature_columns: List[str] = []
        self._cluster_column: str = ''
        self._metrics: Optional[ExplainerMetrics] = None
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame, cluster_column: str,
            feature_columns: Optional[List[str]] = None) -> 'DecisionTreeExplainer':
        """Fit Decision Tree to predict clusters."""
        self._cluster_column = cluster_column

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            self._feature_columns = [
                c for c in df.columns
                if 'slope' in c or 'volatility' in c or 'log_return_rolling' in c
                or 'ma_dist' in c or 'trend_indicator' in c
            ]
        else:
            self._feature_columns = feature_columns

        # Prepare data
        valid_data = df[self._feature_columns + [cluster_column]].dropna()
        X = valid_data[self._feature_columns].values
        y = valid_data[cluster_column].values.astype(int)

        # Create and fit model
        self._model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self._model.fit(X, y)

        # Calculate metrics
        accuracy = self._model.score(X, y)
        cv_scores = cross_val_score(self._model, X, y, cv=self.cv_folds)

        self._metrics = ExplainerMetrics(
            accuracy=accuracy,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
            n_samples=len(y),
            n_features=len(self._feature_columns),
            n_clusters=len(np.unique(y))
        )

        self._is_fitted = True
        return self

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances."""
        self._check_fitted()
        importances = self._model.feature_importances_
        return dict(zip(self._feature_columns, importances))

    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        importances = self.get_feature_importances()
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def get_metrics(self) -> ExplainerMetrics:
        """Get model metrics."""
        self._check_fitted()
        return self._metrics

    def explain_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """Get explanation for a specific cluster."""
        self._check_fitted()

        # Get rules leading to this cluster
        rules = self.get_rules()
        cluster_rules = [r for r in rules if r.cluster_id == cluster_id]

        # Get feature importance contribution for this cluster
        # (simplified - uses global importance)
        importances = self.get_feature_importances()

        return {
            'cluster_id': cluster_id,
            'rules': cluster_rules,
            'feature_importances': importances,
            'n_rules': len(cluster_rules)
        }

    def plot_importances(self, ax: Optional[plt.Axes] = None,
                         top_n: Optional[int] = None) -> plt.Figure:
        """Plot feature importances."""
        self._check_fitted()

        importances = self.get_feature_importances()
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        if top_n:
            sorted_items = sorted_items[:top_n]

        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.4)))
        else:
            fig = ax.get_figure()

        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importances (Decision Tree, acc={self._metrics.accuracy:.1%})')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_tree(self, figsize: Tuple[int, int] = (20, 10),
                  fontsize: int = 10) -> plt.Figure:
        """
        Plot the decision tree.

        Args:
            figsize: Figure size (width, height)
            fontsize: Font size for labels

        Returns:
            matplotlib Figure
        """
        self._check_fitted()

        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            self._model,
            feature_names=self._feature_columns,
            class_names=[f'Cluster {i}' for i in range(self._metrics.n_clusters)],
            filled=True,
            rounded=True,
            fontsize=fontsize,
            ax=ax
        )
        ax.set_title(f'Decision Tree (depth={self.max_depth}, acc={self._metrics.accuracy:.1%})')

        return fig

    def get_rules(self) -> List[ClusterRule]:
        """
        Extract rules from the decision tree.

        Returns:
            List of ClusterRule objects
        """
        self._check_fitted()

        tree = self._model.tree_
        feature_names = self._feature_columns
        rules = []

        def recurse(node_id: int, conditions: List[Condition], depth: int = 0):
            if tree.feature[node_id] == -2:  # Leaf node
                # Get majority class and samples
                class_counts = tree.value[node_id][0]
                majority_class = int(np.argmax(class_counts))
                total_samples = int(np.sum(class_counts))
                accuracy = class_counts[majority_class] / total_samples if total_samples > 0 else 0

                if conditions:  # Only add if there are conditions
                    rules.append(ClusterRule(
                        cluster_id=majority_class,
                        conditions=conditions.copy(),
                        samples=total_samples,
                        accuracy=accuracy
                    ))
            else:
                feature_name = feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]

                # Left branch (<=)
                left_conditions = conditions + [
                    Condition(feature_name, '<=', threshold)
                ]
                recurse(tree.children_left[node_id], left_conditions, depth + 1)

                # Right branch (>)
                right_conditions = conditions + [
                    Condition(feature_name, '>', threshold)
                ]
                recurse(tree.children_right[node_id], right_conditions, depth + 1)

        recurse(0, [])
        return rules

    def get_rules_text(self) -> str:
        """
        Get rules as formatted text.

        Returns:
            Human-readable string with all rules
        """
        self._check_fitted()

        rules = self.get_rules()

        # Group by cluster
        by_cluster: Dict[int, List[ClusterRule]] = {}
        for rule in rules:
            if rule.cluster_id not in by_cluster:
                by_cluster[rule.cluster_id] = []
            by_cluster[rule.cluster_id].append(rule)

        lines = [
            f"=== Decision Tree Rules (acc={self._metrics.accuracy:.1%}) ===\n"
        ]

        for cluster_id in sorted(by_cluster.keys()):
            cluster_rules = by_cluster[cluster_id]
            lines.append(f"\nCluster {cluster_id}:")

            for i, rule in enumerate(cluster_rules, 1):
                conditions_str = " AND ".join(str(c) for c in rule.conditions)
                lines.append(f"  Rule {i}: IF {conditions_str}")
                lines.append(f"           THEN cluster = {cluster_id} (n={rule.samples}, acc={rule.accuracy:.1%})")

        return "\n".join(lines)

    def get_sklearn_tree_text(self) -> str:
        """Get the sklearn text representation of the tree."""
        self._check_fitted()
        return export_text(self._model, feature_names=self._feature_columns)

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError("Explainer not fitted. Call fit() first.")


class RandomForestExplainer(IClusterExplainer):
    """
    Explains clusters using a Random Forest classifier.

    More robust than Decision Tree but less interpretable.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_leaf: int = 20, random_state: int = 42,
                 cv_folds: int = 5, n_jobs: int = -1):
        """
        Initialize Random Forest explainer.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs

        self._model: Optional[RandomForestClassifier] = None
        self._feature_columns: List[str] = []
        self._cluster_column: str = ''
        self._metrics: Optional[ExplainerMetrics] = None
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame, cluster_column: str,
            feature_columns: Optional[List[str]] = None) -> 'RandomForestExplainer':
        """Fit Random Forest to predict clusters."""
        self._cluster_column = cluster_column

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            self._feature_columns = [
                c for c in df.columns
                if 'slope' in c or 'volatility' in c or 'log_return_rolling' in c
                or 'ma_dist' in c or 'trend_indicator' in c
            ]
        else:
            self._feature_columns = feature_columns

        # Prepare data
        valid_data = df[self._feature_columns + [cluster_column]].dropna()
        X = valid_data[self._feature_columns].values
        y = valid_data[cluster_column].values.astype(int)

        # Create and fit model
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self._model.fit(X, y)

        # Calculate metrics
        accuracy = self._model.score(X, y)
        cv_scores = cross_val_score(self._model, X, y, cv=self.cv_folds, n_jobs=self.n_jobs)

        self._metrics = ExplainerMetrics(
            accuracy=accuracy,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
            n_samples=len(y),
            n_features=len(self._feature_columns),
            n_clusters=len(np.unique(y))
        )

        self._is_fitted = True
        return self

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances."""
        self._check_fitted()
        importances = self._model.feature_importances_
        return dict(zip(self._feature_columns, importances))

    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        importances = self.get_feature_importances()
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def get_metrics(self) -> ExplainerMetrics:
        """Get model metrics."""
        self._check_fitted()
        return self._metrics

    def explain_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """Get explanation for a specific cluster."""
        self._check_fitted()

        # For Random Forest, we provide feature importances
        # and per-class probabilities
        importances = self.get_feature_importances()

        return {
            'cluster_id': cluster_id,
            'feature_importances': importances,
            'model_type': 'RandomForest',
            'note': 'Random Forest provides global importance, not cluster-specific rules'
        }

    def plot_importances(self, ax: Optional[plt.Axes] = None,
                         top_n: Optional[int] = None) -> plt.Figure:
        """Plot feature importances."""
        self._check_fitted()

        importances = self.get_feature_importances()
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        if top_n:
            sorted_items = sorted_items[:top_n]

        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.4)))
        else:
            fig = ax.get_figure()

        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='forestgreen', edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importances (Random Forest, cv_acc={self._metrics.cv_accuracy_mean:.1%})')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError("Explainer not fitted. Call fit() first.")


class CompositeExplainer(IClusterExplainer):
    """
    Combines Decision Tree and Random Forest explainers.

    Provides both interpretable rules (from Tree) and robust
    importance estimates (from Forest).
    """

    def __init__(self, tree_max_depth: int = 5, forest_n_estimators: int = 100,
                 forest_max_depth: int = 10, min_samples_leaf: int = 20,
                 random_state: int = 42, cv_folds: int = 5):
        """
        Initialize Composite explainer.

        Args:
            tree_max_depth: Max depth for Decision Tree
            forest_n_estimators: Number of trees in Random Forest
            forest_max_depth: Max depth for Random Forest trees
            min_samples_leaf: Minimum samples at leaf node
            random_state: Random seed
            cv_folds: Number of CV folds
        """
        self._tree_explainer = DecisionTreeExplainer(
            max_depth=tree_max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            cv_folds=cv_folds
        )
        self._forest_explainer = RandomForestExplainer(
            n_estimators=forest_n_estimators,
            max_depth=forest_max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            cv_folds=cv_folds
        )
        self._is_fitted: bool = False

    @property
    def tree(self) -> DecisionTreeExplainer:
        """Access the Decision Tree explainer."""
        return self._tree_explainer

    @property
    def forest(self) -> RandomForestExplainer:
        """Access the Random Forest explainer."""
        return self._forest_explainer

    def fit(self, df: pd.DataFrame, cluster_column: str,
            feature_columns: Optional[List[str]] = None) -> 'CompositeExplainer':
        """Fit both explainers."""
        self._tree_explainer.fit(df, cluster_column, feature_columns)
        self._forest_explainer.fit(df, cluster_column, feature_columns)
        self._is_fitted = True
        return self

    def get_feature_importances(self) -> Dict[str, float]:
        """Get averaged feature importances from both models."""
        self._check_fitted()

        tree_imp = self._tree_explainer.get_feature_importances()
        forest_imp = self._forest_explainer.get_feature_importances()

        # Average importances
        averaged = {}
        for feature in tree_imp:
            averaged[feature] = (tree_imp[feature] + forest_imp[feature]) / 2

        return averaged

    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N features by averaged importance."""
        importances = self.get_feature_importances()
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def get_metrics(self) -> ExplainerMetrics:
        """Get metrics (uses Random Forest as reference)."""
        self._check_fitted()
        return self._forest_explainer.get_metrics()

    def explain_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """Get combined explanation for a cluster."""
        self._check_fitted()

        tree_explanation = self._tree_explainer.explain_cluster(cluster_id)
        forest_explanation = self._forest_explainer.explain_cluster(cluster_id)

        return {
            'cluster_id': cluster_id,
            'tree': tree_explanation,
            'forest': forest_explanation,
            'averaged_importances': self.get_feature_importances()
        }

    def plot_importances(self, ax: Optional[plt.Axes] = None,
                         top_n: Optional[int] = None) -> plt.Figure:
        """Plot feature importances comparison."""
        self._check_fitted()

        tree_imp = self._tree_explainer.get_feature_importances()
        forest_imp = self._forest_explainer.get_feature_importances()
        avg_imp = self.get_feature_importances()

        # Sort by averaged importance
        sorted_features = sorted(avg_imp.keys(), key=lambda x: avg_imp[x], reverse=True)

        if top_n:
            sorted_features = sorted_features[:top_n]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, max(4, len(sorted_features) * 0.5)))
        else:
            fig = ax.get_figure()

        y_pos = np.arange(len(sorted_features))
        width = 0.35

        tree_values = [tree_imp[f] for f in sorted_features]
        forest_values = [forest_imp[f] for f in sorted_features]

        ax.barh(y_pos - width/2, tree_values, width, label='Decision Tree',
                color='steelblue', alpha=0.8)
        ax.barh(y_pos + width/2, forest_values, width, label='Random Forest',
                color='forestgreen', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances Comparison')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def get_rules(self) -> List[ClusterRule]:
        """Get rules from Decision Tree."""
        self._check_fitted()
        return self._tree_explainer.get_rules()

    def get_rules_text(self) -> str:
        """Get rules text from Decision Tree."""
        self._check_fitted()
        return self._tree_explainer.get_rules_text()

    def plot_tree(self, **kwargs) -> plt.Figure:
        """Plot Decision Tree."""
        self._check_fitted()
        return self._tree_explainer.plot_tree(**kwargs)

    def print_summary(self) -> None:
        """Print a comprehensive summary of both models."""
        self._check_fitted()

        tree_metrics = self._tree_explainer.get_metrics()
        forest_metrics = self._forest_explainer.get_metrics()

        print("=" * 60)
        print("CLUSTER EXPLAINER SUMMARY")
        print("=" * 60)

        print(f"\nDataset: {tree_metrics.n_samples} samples, "
              f"{tree_metrics.n_features} features, {tree_metrics.n_clusters} clusters")

        print(f"\n--- Decision Tree ---")
        print(f"  Train Accuracy: {tree_metrics.accuracy:.1%}")
        print(f"  CV Accuracy:    {tree_metrics.cv_accuracy_mean:.1%} (+/- {tree_metrics.cv_accuracy_std:.1%})")

        print(f"\n--- Random Forest ---")
        print(f"  Train Accuracy: {forest_metrics.accuracy:.1%}")
        print(f"  CV Accuracy:    {forest_metrics.cv_accuracy_mean:.1%} (+/- {forest_metrics.cv_accuracy_std:.1%})")

        print(f"\n--- Top Features (averaged) ---")
        for feature, importance in self.get_top_features(5):
            print(f"  {feature}: {importance:.3f}")

        print(f"\n--- Decision Tree Rules ---")
        print(self.get_rules_text())

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError("Explainer not fitted. Call fit() first.")
