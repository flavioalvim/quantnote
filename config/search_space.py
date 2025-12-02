"""GA Search Space configuration."""
from pydantic import BaseModel, Field
from typing import Tuple, List
import numpy as np


class TargetGrid(BaseModel):
    """Configuration for multi-target optimization grid."""

    target_min: float = Field(default=0.01, ge=-0.5, le=0.5)
    target_max: float = Field(default=0.10, ge=-0.5, le=0.5)
    target_step: float = Field(default=0.01, ge=0.001, le=0.1)

    def to_array(self) -> List[float]:
        """Generate array of target values."""
        targets = np.arange(
            self.target_min,
            self.target_max + self.target_step / 2,
            self.target_step
        )
        return [round(t, 4) for t in targets]

    def __len__(self) -> int:
        return len(self.to_array())


class GASearchSpace(BaseModel):
    """Centralized search space for genetic algorithm."""

    window_slope: Tuple[int, int] = Field(default=(5, 60))
    window_volatility: Tuple[int, int] = Field(default=(5, 60))
    window_rolling_return: Tuple[int, int] = Field(default=(5, 60))
    window_trend_indicator: Tuple[int, int] = Field(default=(5, 30))
    trend_slope_multiplier: Tuple[float, float] = Field(default=(1.5, 3.0))
    n_clusters: Tuple[int, int] = Field(default=(2, 5))
    ma_fast_period: Tuple[int, int] = Field(default=(5, 20))
    ma_slow_period: Tuple[int, int] = Field(default=(20, 60))


class GAConfig(BaseModel):
    """Genetic algorithm configuration."""

    # Fixed prediction parameters (not optimized)
    target_return: float = Field(default=0.05, ge=-0.5, le=0.5)
    horizon: int = Field(default=7, ge=1, le=60)

    # GA parameters
    population_size: int = Field(default=50, ge=10, le=500)
    generations: int = Field(default=100, ge=10, le=100000)
    crossover_probability: float = Field(default=0.7, ge=0.0, le=1.0)
    mutation_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    elite_size: int = Field(default=5, ge=1, le=20)
    stability_penalty: float = Field(default=0.1, ge=0.0, le=1.0)

    # Walk-forward validation
    n_folds: int = Field(default=5, ge=2, le=10)
    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)

    # Early stopping
    early_stopping: bool = Field(default=True)
    early_stopping_patience: int = Field(default=20, ge=5, le=1000)
    early_stopping_threshold: float = Field(default=0.001, ge=0.0, le=0.1)

    search_space: GASearchSpace = GASearchSpace()