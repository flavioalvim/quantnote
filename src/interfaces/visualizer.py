"""Visualizer interface."""
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.figure


class IVisualizer(ABC):
    """Interface for visualizations."""

    @abstractmethod
    def plot(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> matplotlib.figure.Figure:
        """
        Generate visualization.

        Args:
            df: DataFrame with data
            **kwargs: Visualization-specific parameters

        Returns:
            Matplotlib Figure
        """
        pass

    @abstractmethod
    def save(
        self,
        fig: matplotlib.figure.Figure,
        filepath: str,
        dpi: int = 150
    ) -> None:
        """Save figure to file."""
        pass
