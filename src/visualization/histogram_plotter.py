"""Histogram and price visualization plotters."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Optional

from ..interfaces.visualizer import IVisualizer


class HistogramPlotter(IVisualizer):
    """Plots return distribution histograms by regime."""

    def __init__(
        self,
        return_column: str,
        regime_column: str = 'regime',
        bins: int = 50,
        figsize: tuple = (12, 6)
    ):
        self.return_column = return_column
        self.regime_column = regime_column
        self.bins = bins
        self.figsize = figsize

    def plot(self, df: pd.DataFrame, **kwargs) -> matplotlib.figure.Figure:
        """Plot overall return histogram."""
        fig, ax = plt.subplots(figsize=self.figsize)

        data = df[self.return_column].dropna()

        ax.hist(data, bins=self.bins, edgecolor='black', alpha=0.7, density=True)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=data.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {data.mean():.4f}')

        ax.set_xlabel('Log Return')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Future Returns')
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_by_regime(
        self,
        df: pd.DataFrame,
        target_return: Optional[float] = None,
        **kwargs
    ) -> matplotlib.figure.Figure:
        """Plot histograms conditioned by regime."""
        regimes = df[self.regime_column].dropna().unique()
        n_regimes = len(regimes)

        # Calculate grid dimensions
        n_cols = 2
        n_rows = (n_regimes + 1) // 2

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(14, 4 * n_rows)
        )
        axes = axes.flatten() if n_regimes > 1 else [axes]

        # Color map for regimes
        colors = {
            'bull_high_vol': 'lightgreen',
            'bull_low_vol': 'darkgreen',
            'bear_high_vol': 'lightcoral',
            'bear_low_vol': 'darkred',
            'flat_high_vol': 'lightyellow',
            'flat_low_vol': 'gold'
        }

        for idx, regime in enumerate(sorted(regimes)):
            ax = axes[idx]
            mask = df[self.regime_column] == regime
            data = df.loc[mask, self.return_column].dropna()

            color = colors.get(str(regime), 'steelblue')
            ax.hist(data, bins=self.bins, edgecolor='black', alpha=0.7,
                    color=color, density=True)

            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax.axvline(x=data.mean(), color='black', linestyle='-', linewidth=2,
                       label=f'Mean: {data.mean():.4f}')

            if target_return:
                log_target = np.log(1 + target_return)
                ax.axvline(x=log_target, color='purple', linestyle=':', linewidth=2,
                           label=f'Target: {target_return:.1%}')
                prob = (data > log_target).mean()
                ax.text(0.95, 0.95, f'P(hit) = {prob:.1%}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='white'))

            ax.set_title(f'{regime}\n(n={len(data)})')
            ax.set_xlabel('Log Return')
            ax.legend(fontsize=8)

        # Hide unused axes
        for idx in range(len(regimes), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig

    def save(
        self,
        fig: matplotlib.figure.Figure,
        filepath: str,
        dpi: int = 150
    ) -> None:
        """Save figure to file."""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


class PriceRegimePlotter(IVisualizer):
    """Plots price series with regime coloring."""

    # Color palettes for interpreted clusters (bull/bear/flat)
    BULL_COLORS = ['#27ae60', '#2ecc71', '#58d68d', '#82e0aa', '#abebc6']  # greens
    BEAR_COLORS = ['#c0392b', '#e74c3c', '#ec7063', '#f1948a', '#f5b7b1']  # reds/pinks
    FLAT_COLORS = ['#f39c12', '#f1c40f', '#f4d03f', '#f7dc6f', '#fcf3cf']  # yellows

    def __init__(
        self,
        regime_column: str = 'regime',
        figsize: tuple = (14, 8),
        cluster_interpretations: Optional[dict] = None
    ):
        """
        Initialize plotter.

        Args:
            regime_column: Column name containing regime/cluster labels
            figsize: Figure size tuple
            cluster_interpretations: Dict mapping cluster_id -> 'bull'/'bear'/'flat'
                                     If provided, colors will be based on interpretation
        """
        self.regime_column = regime_column
        self.figsize = figsize
        self.cluster_interpretations = cluster_interpretations

    def plot(
        self,
        df: pd.DataFrame,
        cluster_interpretations: Optional[dict] = None,
        **kwargs
    ) -> matplotlib.figure.Figure:
        """
        Plot prices with regime background.

        Args:
            df: DataFrame with price data and regime column
            cluster_interpretations: Dict mapping cluster_id -> 'bull'/'bear'/'flat'
                                     Overrides instance-level interpretations
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize,
                                  gridspec_kw={'height_ratios': [3, 1]})

        ax_price = axes[0]
        ax_regime = axes[1]

        # Price plot
        ax_price.plot(df.index, df['close'], color='black', linewidth=0.5)

        # Use provided interpretations or instance-level
        interpretations = cluster_interpretations or self.cluster_interpretations

        # Named regime colors (for manual classification)
        named_regime_colors = {
            'bull_high_vol': '#2ecc71',
            'bull_low_vol': '#27ae60',
            'bear_high_vol': '#e74c3c',
            'bear_low_vol': '#c0392b',
            'flat_high_vol': '#f1c40f',
            'flat_low_vol': '#f39c12',
            'bull': '#2ecc71',
            'bear': '#e74c3c',
            'flat': '#f1c40f'
        }

        # Track used colors per interpretation for variety
        bull_idx, bear_idx, flat_idx = 0, 0, 0

        # Get unique regimes
        regimes = df[self.regime_column].dropna().unique()

        # Color background by regime
        for regime in sorted(regimes):
            mask = df[self.regime_column] == regime

            # Determine color and label
            if isinstance(regime, str):
                # String regime (manual classification)
                color = named_regime_colors.get(regime, '#95a5a6')
                label = regime
            else:
                # Numeric cluster - use interpretation if available
                cluster_id = int(regime)
                interp = None
                if interpretations:
                    interp = interpretations.get(cluster_id, 'flat')

                if interp == 'bull':
                    color = self.BULL_COLORS[bull_idx % len(self.BULL_COLORS)]
                    bull_idx += 1
                    label = f'Cluster {cluster_id} (bull)'
                elif interp == 'bear':
                    color = self.BEAR_COLORS[bear_idx % len(self.BEAR_COLORS)]
                    bear_idx += 1
                    label = f'Cluster {cluster_id} (bear)'
                else:
                    color = self.FLAT_COLORS[flat_idx % len(self.FLAT_COLORS)]
                    flat_idx += 1
                    label = f'Cluster {cluster_id} (flat)'

            if mask.any():
                ax_price.fill_between(
                    df.index, df['close'].min(), df['close'].max(),
                    where=mask, alpha=0.3, color=color, label=label
                )

        ax_price.set_ylabel('Price')
        ax_price.set_title('Price with Regime Background')
        ax_price.legend(loc='upper left', fontsize='small')

        # Regime timeline
        regimes = df[self.regime_column].dropna().unique()
        regime_to_num = {r: i for i, r in enumerate(sorted(regimes))}
        regime_nums = df[self.regime_column].map(regime_to_num)

        ax_regime.plot(df.index, regime_nums, linewidth=1)
        ax_regime.set_ylabel('Regime')
        ax_regime.set_yticks(list(regime_to_num.values()))

        # Format labels with interpretation if available
        labels = []
        for r in sorted(regimes):
            if isinstance(r, str):
                labels.append(r)
            else:
                cluster_id = int(r)
                if interpretations and cluster_id in interpretations:
                    labels.append(f'Cluster {cluster_id} ({interpretations[cluster_id]})')
                else:
                    labels.append(f'Cluster {cluster_id}')
        ax_regime.set_yticklabels(labels, fontsize=8)

        plt.tight_layout()
        return fig

    def save(
        self,
        fig: matplotlib.figure.Figure,
        filepath: str,
        dpi: int = 150
    ) -> None:
        """Save figure to file."""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
