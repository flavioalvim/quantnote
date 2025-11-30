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

    def __init__(
        self,
        regime_column: str = 'regime',
        figsize: tuple = (14, 8)
    ):
        self.regime_column = regime_column
        self.figsize = figsize

    def plot(self, df: pd.DataFrame, **kwargs) -> matplotlib.figure.Figure:
        """Plot prices with regime background."""
        fig, axes = plt.subplots(2, 1, figsize=self.figsize,
                                  gridspec_kw={'height_ratios': [3, 1]})

        ax_price = axes[0]
        ax_regime = axes[1]

        # Price plot
        ax_price.plot(df.index, df['close'], color='black', linewidth=0.5)

        # Regime colors
        regime_colors = {
            'bull_high_vol': 'lightgreen',
            'bull_low_vol': 'green',
            'bear_high_vol': 'lightcoral',
            'bear_low_vol': 'red',
            'flat_high_vol': 'lightyellow',
            'flat_low_vol': 'yellow',
            'bull': 'green',
            'bear': 'red',
            'flat': 'gray'
        }

        # Color background by regime
        for regime, color in regime_colors.items():
            mask = df[self.regime_column] == regime
            if mask.any():
                ax_price.fill_between(
                    df.index, df['close'].min(), df['close'].max(),
                    where=mask, alpha=0.3, color=color, label=regime
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
        ax_regime.set_yticklabels(list(regime_to_num.keys()), fontsize=8)

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
