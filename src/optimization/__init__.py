"""Genetic algorithm optimization module."""
from .chromosome import Chromosome
from .calculator_factory import CalculatorFactory
from .walk_forward_validator import WalkForwardValidator, WalkForwardResult, FoldResult
from .fitness import FitnessEvaluator, FitnessResult
from .genetic_algorithm import GeneticAlgorithm, GAResult
from .multi_target_optimizer import (
    MultiTargetOptimizer,
    MultiTargetResult,
    MultiTargetOptimizationResult
)
from .strike_grid_optimizer import (
    StrikeGridOptimizer,
    StrikeTarget,
    StrikeGridOptimizationResult
)
from .return_strategy import (
    IReturnStrategy,
    CloseReturnStrategy,
    TouchReturnStrategy,
    DEFAULT_CLOSE_STRATEGY,
    DEFAULT_TOUCH_STRATEGY,
    get_strategy
)

__all__ = [
    'Chromosome',
    'CalculatorFactory',
    'WalkForwardValidator',
    'WalkForwardResult',
    'FoldResult',
    'FitnessEvaluator',
    'FitnessResult',
    'GeneticAlgorithm',
    'GAResult',
    'MultiTargetOptimizer',
    'MultiTargetResult',
    'MultiTargetOptimizationResult',
    'StrikeGridOptimizer',
    'StrikeTarget',
    'StrikeGridOptimizationResult',
    # Strategies
    'IReturnStrategy',
    'CloseReturnStrategy',
    'TouchReturnStrategy',
    'DEFAULT_CLOSE_STRATEGY',
    'DEFAULT_TOUCH_STRATEGY',
    'get_strategy'
]
