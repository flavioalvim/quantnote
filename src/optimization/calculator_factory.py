"""Factory for creating calculator pipelines from chromosomes."""
from typing import List

from ..interfaces.column_calculator import IColumnCalculator
from ..calculators.log_price_calculator import LogPriceCalculator
from ..calculators.log_return_calculator import LogReturnCalculator
from ..calculators.future_return_calculator import FutureReturnCalculator
from ..calculators.volatility_calculator import VolatilityCalculator
from ..calculators.slope_calculator import SlopeCalculator
from ..calculators.pipeline import CalculatorPipeline
from .chromosome import Chromosome


class CalculatorFactory:
    """
    Factory for creating calculator pipelines from chromosomes.
    Implements Dependency Inversion - GA depends on factory, not concrete calculators.
    """

    def __init__(self, horizon: int = 7):
        """
        Initialize factory with fixed prediction horizon.

        Args:
            horizon: Number of periods for future return calculation
        """
        self.horizon = horizon

    def create_pipeline(self, chromosome: Chromosome) -> CalculatorPipeline:
        """Create a pipeline configured by the chromosome."""
        calculators: List[IColumnCalculator] = [
            LogPriceCalculator(),
            LogReturnCalculator(window=chromosome.window_rolling_return),
            FutureReturnCalculator(horizon=self.horizon),
        ]

        if chromosome.use_volatility:
            calculators.append(
                VolatilityCalculator(window=chromosome.window_volatility)
            )

        calculators.append(
            SlopeCalculator(window=chromosome.window_slope)
        )

        return CalculatorPipeline(calculators, auto_resolve=True)

    def get_feature_columns(self, chromosome: Chromosome) -> List[str]:
        """Get feature columns that will be produced for clustering."""
        features = [f'slope_{chromosome.window_slope}']

        if chromosome.use_volatility:
            features.append(f'volatility_{chromosome.window_volatility}')

        if chromosome.use_rolling_return:
            features.append(f'log_return_rolling_{chromosome.window_rolling_return}')

        return features

    def get_future_return_column(self) -> str:
        """Get the future return column name."""
        return f'log_return_future_{self.horizon}'
