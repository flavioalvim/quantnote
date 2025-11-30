"""Calculator pipeline with automatic dependency resolution."""
import pandas as pd
from typing import List, Optional, Set

from ..interfaces.column_calculator import IColumnCalculator
from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from ..domain.exceptions import PipelineError
from .dependency_resolver import DependencyResolver


class CalculatorPipeline:
    """
    Orchestrates execution of calculators with automatic dependency resolution.
    """

    def __init__(
        self,
        calculators: List[IColumnCalculator],
        logger: Optional[ILogger] = None,
        auto_resolve: bool = True
    ):
        self.calculators = calculators
        self.logger = logger or NullLogger()
        self.auto_resolve = auto_resolve
        self._resolver = DependencyResolver()
        self._resolved_order: Optional[List[IColumnCalculator]] = None

    def run(
        self,
        df: pd.DataFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Execute all calculators in dependency order.

        Args:
            df: Input DataFrame
            validate: Whether to validate input columns

        Returns:
            DataFrame with all calculated columns
        """
        result = df.copy()

        # Resolve order if needed
        if self.auto_resolve and self._resolved_order is None:
            available = set(df.columns)
            self._resolved_order = self._resolver.resolve(
                self.calculators, available
            )
            self.logger.info(
                "Resolved calculator order",
                order=[c.name for c in self._resolved_order]
            )

        calculators_to_run = self._resolved_order or self.calculators

        # Execute each calculator
        for calc in calculators_to_run:
            self.logger.debug(
                f"Running calculator",
                calculator=calc.name,
                required=list(calc.required_columns),
                output=list(calc.output_columns)
            )

            try:
                if validate:
                    calc.validate_input(result)
                result = calc.calculate(result)
            except Exception as e:
                self.logger.error(
                    f"Calculator failed",
                    exception=e,
                    calculator=calc.name
                )
                raise PipelineError(f"Calculator '{calc.name}' failed: {e}") from e

        self.logger.info(
            "Pipeline completed",
            input_columns=len(df.columns),
            output_columns=len(result.columns),
            rows=len(result)
        )

        return result

    def get_all_output_columns(self) -> Set[str]:
        """Return all columns that will be produced."""
        columns = set()
        for calc in self.calculators:
            columns.update(calc.output_columns)
        return columns

    def get_execution_order(self) -> List[str]:
        """Return the resolved execution order."""
        if self._resolved_order:
            return [c.name for c in self._resolved_order]
        return [c.name for c in self.calculators]
