"""Strike probability grid calculator for options analysis."""
import pandas as pd
import numpy as np
from typing import List, Union
from dataclasses import dataclass


@dataclass
class StrikeGridResult:
    """Result container for strike probability grid."""

    df: pd.DataFrame
    current_price: float
    current_cluster: int
    n_observations: int
    n_cluster_observations: int

    def to_records(self) -> List[tuple]:
        """Convert to list of tuples (strike, prob_naive, prob_calibrada)."""
        return [
            (row['strike'], row['prob_naive'], row['prob_calibrada'])
            for _, row in self.df.iterrows()
        ]

    def to_full_records(self) -> List[tuple]:
        """Convert to list of tuples with all columns."""
        return list(self.df.itertuples(index=False, name=None))


class StrikeProbabilityGrid:
    """
    Calculate probability grid for multiple strike prices.

    Given a current price and a list of strikes, calculates for each strike:
    - The required return to reach that strike
    - Naive (unconditional) probability of reaching/exceeding the strike
    - Calibrated (cluster-conditional) probability

    For strikes above current price: P(price >= strike)
    For strikes below current price: P(price <= strike)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        future_return_column: str,
        regime_column: str = 'cluster'
    ):
        """
        Initialize the grid calculator.

        Args:
            df: DataFrame with historical data including future returns and clusters
            future_return_column: Column name with log returns (e.g., 'log_return_future_7')
            regime_column: Column name with cluster/regime labels
        """
        self.df = df
        self.future_return_column = future_return_column
        self.regime_column = regime_column

        # Pre-extract valid returns for efficiency
        self._valid_df = df.dropna(subset=[future_return_column, regime_column])
        self._all_returns = self._valid_df[future_return_column]

    def calculate(
        self,
        current_price: float,
        strikes: List[float],
        current_cluster: Union[int, float]
    ) -> StrikeGridResult:
        """
        Calculate probability grid for given strikes.

        Args:
            current_price: Current asset price (e.g., 155.67)
            strikes: List of strike prices (e.g., [100, 101, ..., 170])
            current_cluster: Current regime/cluster identifier

        Returns:
            StrikeGridResult with DataFrame and metadata
        """
        # Get cluster-specific returns
        cluster_mask = self._valid_df[self.regime_column] == current_cluster
        cluster_returns = self._valid_df.loc[cluster_mask, self.future_return_column]

        n_total = len(self._all_returns)
        n_cluster = len(cluster_returns)

        if n_total == 0:
            raise ValueError("No valid returns in dataset")

        results = []

        for strike in strikes:
            # Calculate required return
            simple_return = (strike / current_price) - 1
            log_target = np.log(1 + simple_return)

            # Determine direction and count hits
            if strike >= current_price:
                # Need price to go UP: P(ret >= target)
                hits_naive = (self._all_returns >= log_target).sum()
                hits_calibrated = (cluster_returns >= log_target).sum() if n_cluster > 0 else 0
            else:
                # Need price to go DOWN: P(ret <= target)
                hits_naive = (self._all_returns <= log_target).sum()
                hits_calibrated = (cluster_returns <= log_target).sum() if n_cluster > 0 else 0

            prob_naive = hits_naive / n_total
            prob_calibrada = hits_calibrated / n_cluster if n_cluster > 0 else np.nan

            results.append({
                'strike': strike,
                'retorno_pct': simple_return,
                'prob_naive': prob_naive,
                'prob_calibrada': prob_calibrada
            })

        result_df = pd.DataFrame(results)

        return StrikeGridResult(
            df=result_df,
            current_price=current_price,
            current_cluster=current_cluster,
            n_observations=n_total,
            n_cluster_observations=n_cluster
        )

    def calculate_range(
        self,
        current_price: float,
        strike_min: float,
        strike_max: float,
        step: float,
        current_cluster: Union[int, float]
    ) -> StrikeGridResult:
        """
        Calculate probability grid for a range of strikes.

        Convenience method that generates strikes from min to max with given step.

        Args:
            current_price: Current asset price
            strike_min: Minimum strike price
            strike_max: Maximum strike price
            step: Step between strikes
            current_cluster: Current regime/cluster identifier

        Returns:
            StrikeGridResult with DataFrame and metadata
        """
        strikes = list(np.arange(strike_min, strike_max + step, step))
        return self.calculate(current_price, strikes, current_cluster)
