"""Central configuration for QuantNote with Pydantic validation."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AnalysisConfig(BaseModel):
    """Validated configuration for regime analysis."""

    # Future horizon
    future_return_periods: int = Field(default=3, ge=1, le=30)

    # Analysis windows
    window_rolling_return: int = Field(default=20, ge=5, le=100)
    window_slope: int = Field(default=20, ge=5, le=100)
    window_volatility: int = Field(default=20, ge=5, le=100)
    window_trend_indicator: int = Field(default=10, ge=2, le=50)
    trend_slope_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)

    # Moving average distance
    ma_fast_period: int = Field(default=9, ge=2, le=50)
    ma_slow_period: int = Field(default=21, ge=5, le=200)

    # Clustering
    n_clusters: int = Field(default=6, ge=2, le=10)

    # Targets
    target_return: float = Field(default=0.05, ge=0.001, le=0.50)

    # Optional manual thresholds
    slope_threshold: Optional[float] = None
    volatility_threshold: Optional[float] = None

    # Validation
    min_data_points: int = Field(default=252, ge=50)  # 1 year minimum

    @field_validator('window_slope', 'window_volatility', 'window_rolling_return')
    @classmethod
    def window_less_than_min_data(cls, v, info):
        # Note: In Pydantic v2, we can't access other fields easily in validators
        # This validation is simplified
        if v >= 126:  # Half of default min_data_points
            raise ValueError(f"Window {v} too large")
        return v

    @field_validator('ma_slow_period')
    @classmethod
    def slow_greater_than_fast(cls, v, info):
        fast = info.data.get('ma_fast_period', 9)
        if v <= fast:
            raise ValueError(f"ma_slow_period ({v}) must be greater than ma_fast_period ({fast})")
        return v


class InfrastructureConfig(BaseModel):
    """Infrastructure settings."""

    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    log_level: LogLevel = LogLevel.INFO

    # Rate limiting for Yahoo Finance
    yahoo_calls_per_minute: int = Field(default=5, ge=1, le=60)


class Config(BaseModel):
    """Root configuration."""

    analysis: AnalysisConfig = AnalysisConfig()
    infrastructure: InfrastructureConfig = InfrastructureConfig()
