"""Dependency resolver for calculator pipeline."""
from typing import List, Set, Dict
from collections import defaultdict

from ..interfaces.column_calculator import IColumnCalculator
from ..domain.exceptions import PipelineError


class DependencyResolver:
    """
    Resolves calculator execution order via topological sort.
    Ensures calculators run in correct dependency order.
    """

    def resolve(
        self,
        calculators: List[IColumnCalculator],
        available_columns: Set[str]
    ) -> List[IColumnCalculator]:
        """
        Sort calculators by dependencies.

        Args:
            calculators: List of calculators to sort
            available_columns: Columns already available in DataFrame

        Returns:
            Sorted list of calculators

        Raises:
            PipelineError: If circular dependency or unresolvable dependency
        """
        # Build dependency graph
        calc_by_name = {c.name: c for c in calculators}
        all_outputs = available_columns.copy()

        # Map: output_column -> calculator that produces it
        producer: Dict[str, str] = {}
        for calc in calculators:
            for col in calc.output_columns:
                producer[col] = calc.name

        # Build adjacency list (calc -> calcs it depends on)
        dependencies: Dict[str, Set[str]] = defaultdict(set)
        for calc in calculators:
            for req_col in calc.required_columns:
                if req_col in producer:
                    dependencies[calc.name].add(producer[req_col])
                elif req_col not in available_columns:
                    raise PipelineError(
                        f"Calculator '{calc.name}' requires column '{req_col}' "
                        f"which is not available and no calculator produces it"
                    )

        # Topological sort (Kahn's algorithm)
        in_degree = {c.name: len(dependencies[c.name]) for c in calculators}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(calc_by_name[current])

            # Reduce in-degree for dependents
            for calc in calculators:
                if current in dependencies[calc.name]:
                    in_degree[calc.name] -= 1
                    if in_degree[calc.name] == 0:
                        queue.append(calc.name)

        if len(result) != len(calculators):
            remaining = set(c.name for c in calculators) - set(c.name for c in result)
            raise PipelineError(
                f"Circular dependency detected involving: {remaining}"
            )

        return result
