# Technical Specification - QuantNote

## Quantitative System for Regime-Conditioned Return Probabilities

**Version 2.0** - With SOLID improvements, validation, and walk-forward optimization

---

## 1. Overview

### 1.1 Objective

Build a Python system that:
- Estimates probabilities of an asset reaching a target return within a time horizon
- Adjusts probabilities based on market regime (trend/volatility)
- Optimizes windows, parameters, and regimes using K-Means and genetic algorithms
- Validates results with walk-forward optimization to prevent overfitting

### 1.2 SOLID Principles Applied

| Principle | Application |
|-----------|-------------|
| **S** - Single Responsibility | Each class has one responsibility (load data, calculate indicators, persist, etc.) |
| **O** - Open/Closed | Classes open for extension via interfaces, closed for modification |
| **L** - Liskov Substitution | Concrete implementations replaceable by their base interfaces |
| **I** - Interface Segregation | Small, specific interfaces for each functionality |
| **D** - Dependency Inversion | Depend on abstractions via Factory Pattern and DI |

### 1.3 Key Improvements Over v1.0

- **Data Validation Layer**: Robust input validation before processing
- **Rich Domain Layer**: Value Objects encapsulating business rules
- **Automatic Dependency Resolution**: Pipeline resolves calculator order via topological sort
- **Walk-Forward Validation**: Prevents overfitting in genetic algorithm
- **Model Persistence**: K-Means models can be saved and loaded
- **Structured Logging**: Observable pipeline for debugging
- **Rate Limiting**: Protection against API blocking
- **Comprehensive Testing**: Unit and integration test strategies

---

## 2. Architecture

### 2.1 Directory Structure

```
quantnote/
├── .venv/                              # Python virtual environment
├── data/                               # Persisted data (parquet)
├── models/                             # Saved ML models (joblib)
├── logs/                               # Log files
├── config/
│   ├── __init__.py
│   ├── settings.py                     # Central configuration
│   └── search_space.py                 # GA search space config
├── src/
│   ├── __init__.py
│   ├── domain/                         # Entities and business rules
│   │   ├── __init__.py
│   │   ├── value_objects.py            # LogReturn, Price, Regime
│   │   └── exceptions.py               # Domain exceptions
│   ├── interfaces/                     # Abstract contracts
│   │   ├── __init__.py
│   │   ├── data_source.py
│   │   ├── repository.py
│   │   ├── column_calculator.py
│   │   ├── validator.py
│   │   ├── visualizer.py
│   │   ├── logger.py
│   │   └── model_store.py
│   ├── infrastructure/                 # Concrete implementations
│   │   ├── __init__.py
│   │   ├── yahoo_data_source.py
│   │   ├── parquet_repository.py
│   │   ├── joblib_model_store.py
│   │   ├── file_logger.py
│   │   └── validators/
│   │       ├── __init__.py
│   │       ├── ohlcv_validator.py
│   │       └── series_length_validator.py
│   ├── calculators/                    # Column calculators
│   │   ├── __init__.py
│   │   ├── log_price_calculator.py
│   │   ├── log_return_calculator.py
│   │   ├── future_return_calculator.py
│   │   ├── volatility_calculator.py
│   │   ├── slope_calculator.py
│   │   ├── pipeline.py
│   │   └── dependency_resolver.py
│   ├── analysis/                       # Regime analysis
│   │   ├── __init__.py
│   │   ├── regime_classifier.py
│   │   ├── kmeans_regimes.py
│   │   ├── probability_calculator.py
│   │   └── time_series_splitter.py
│   ├── optimization/                   # Genetic algorithm
│   │   ├── __init__.py
│   │   ├── chromosome.py
│   │   ├── fitness.py
│   │   ├── genetic_algorithm.py
│   │   ├── walk_forward_validator.py
│   │   └── calculator_factory.py
│   ├── visualization/                  # Plots and charts
│   │   ├── __init__.py
│   │   └── histogram_plotter.py
│   └── utils/                          # Utilities
│       ├── __init__.py
│       └── return_converter.py
├── notebooks/
│   └── quantnote_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_calculators.py
│   │   ├── test_validators.py
│   │   ├── test_value_objects.py
│   │   └── test_probability.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_ga_optimization.py
│   └── fixtures/
│       └── sample_data.py
├── requirements.txt
└── ESPECIFICACAO_TECNICA.md
```

### 2.2 Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            INTERFACES                                    │
├───────────────┬───────────────┬───────────────┬─────────────────────────┤
│ IDataSource   │ IRepository   │ IValidator    │ IColumnCalculator       │
│ ILogger       │ IModelStore   │ IVisualizer   │                         │
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬─────────────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌───────────────┐┌──────────────┐┌──────────────┐┌─────────────────────────┐
│YahooDataSource││ParquetRepo   ││OHLCVValidator││ LogPriceCalculator      │
│(rate-limited) ││              ││SeriesLenValid││ LogReturnCalculator     │
└───────────────┘└──────────────┘└──────────────┘│ FutureReturnCalculator  │
                                                 │ VolatilityCalculator    │
                                                 │ SlopeCalculator         │
                                                 └───────────┬─────────────┘
                                                             │
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │    DependencyResolver        │
                                              │    (topological sort)        │
                                              └──────────────┬───────────────┘
                                                             │
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │    CalculatorPipeline        │
                                              └──────────────┬───────────────┘
                                                             │
                       ┌─────────────────────────────────────┼─────────────────────────────────────┐
                       ▼                                     ▼                                     ▼
            ┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
            │ManualRegimeClass │                  │KMeansRegimeClass │                  │CalculatorFactory │
            └──────────────────┘                  │(with persistence)│                  └────────┬─────────┘
                                                  └──────────────────┘                           │
                                                                                                 ▼
                                                                                  ┌──────────────────────────────┐
                                                                                  │    GeneticAlgorithm          │
                                                                                  │  + WalkForwardValidator      │
                                                                                  └──────────────────────────────┘
```

---

## 3. Configuration

### 3.1 requirements.txt

```
# Core
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0

# Data Sources
yfinance>=0.2.30
ratelimit>=2.2.1

# Machine Learning
scikit-learn>=1.3.0
scipy>=1.11.0
deap>=1.4.0
joblib>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Validation
pydantic>=2.0.0

# Development
jupyter>=1.0.0
ipykernel>=6.25.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

### 3.2 Central Configuration (config/settings.py)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Tuple
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

    # Clustering
    n_clusters: int = Field(default=3, ge=2, le=10)

    # Targets
    target_return: float = Field(default=0.05, ge=0.001, le=0.50)

    # Optional manual thresholds
    slope_threshold: Optional[float] = None
    volatility_threshold: Optional[float] = None

    # Validation
    min_data_points: int = Field(default=252, ge=50)  # 1 year minimum

    @validator('window_slope', 'window_volatility', 'window_rolling_return')
    def window_less_than_min_data(cls, v, values):
        min_points = values.get('min_data_points', 252)
        if v >= min_points // 2:
            raise ValueError(f"Window {v} too large for min_data_points {min_points}")
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
```

### 3.3 GA Search Space (config/search_space.py)

```python
from pydantic import BaseModel, Field
from typing import Tuple

class GASearchSpace(BaseModel):
    """Centralized search space for genetic algorithm."""

    window_slope: Tuple[int, int] = Field(default=(5, 60))
    window_volatility: Tuple[int, int] = Field(default=(5, 60))
    window_rolling_return: Tuple[int, int] = Field(default=(5, 60))
    horizon: Tuple[int, int] = Field(default=(1, 10))
    target_return: Tuple[float, float] = Field(default=(0.02, 0.10))
    n_clusters: Tuple[int, int] = Field(default=(2, 5))

class GAConfig(BaseModel):
    """Genetic algorithm configuration."""

    population_size: int = Field(default=50, ge=10, le=500)
    generations: int = Field(default=100, ge=10, le=1000)
    crossover_probability: float = Field(default=0.7, ge=0.0, le=1.0)
    mutation_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    elite_size: int = Field(default=5, ge=1, le=20)
    stability_penalty: float = Field(default=0.1, ge=0.0, le=1.0)

    # Walk-forward validation
    n_folds: int = Field(default=5, ge=2, le=10)
    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)

    search_space: GASearchSpace = GASearchSpace()
```

---

## 4. Interfaces (Contracts)

### 4.1 Logger Interface (interfaces/logger.py)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum

class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40

class ILogger(ABC):
    """Interface for structured logging."""

    @abstractmethod
    def debug(self, message: str, **context: Any) -> None:
        pass

    @abstractmethod
    def info(self, message: str, **context: Any) -> None:
        pass

    @abstractmethod
    def warning(self, message: str, **context: Any) -> None:
        pass

    @abstractmethod
    def error(self, message: str, exception: Optional[Exception] = None, **context: Any) -> None:
        pass
```

### 4.2 Data Source Interface (interfaces/data_source.py)

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from datetime import datetime

class IDataSource(ABC):
    """Interface for price data sources."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            ticker: Asset code (e.g., 'BOVA11.SA')
            start_date: Start date (None = maximum available)
            end_date: End date (None = today)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Prices already adjusted for splits and dividends.

        Raises:
            DataSourceError: If fetch fails
        """
        pass
```

### 4.3 Repository Interface (interfaces/repository.py)

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FileMetadata:
    """Metadata for saved files."""
    filename: str
    ticker: str
    created_at: datetime
    row_count: int
    date_range: tuple

class IRepository(ABC):
    """Interface for data persistence."""

    @abstractmethod
    def save(self, df: pd.DataFrame, ticker: str) -> FileMetadata:
        """Save DataFrame with metadata."""
        pass

    @abstractmethod
    def load(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load most recent DataFrame for ticker."""
        pass

    @abstractmethod
    def load_by_filename(self, filename: str) -> pd.DataFrame:
        """Load specific file."""
        pass

    @abstractmethod
    def list_files(self, ticker: Optional[str] = None) -> List[FileMetadata]:
        """List available files."""
        pass

    @abstractmethod
    def delete(self, filename: str) -> bool:
        """Delete a file."""
        pass
```

### 4.4 Validator Interface (interfaces/validator.py)

```python
from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge two validation results."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings
        )

class IValidator(ABC):
    """Interface for data validation."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with status and messages
        """
        pass

class CompositeValidator(IValidator):
    """Combines multiple validators."""

    def __init__(self, validators: List[IValidator]):
        self.validators = validators

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        for validator in self.validators:
            result = result.merge(validator.validate(df))
        return result
```

### 4.5 Column Calculator Interface (interfaces/column_calculator.py)

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Set

class IColumnCalculator(ABC):
    """Interface for column/indicator calculators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this calculator."""
        pass

    @property
    @abstractmethod
    def required_columns(self) -> Set[str]:
        """Columns required in input DataFrame."""
        pass

    @property
    @abstractmethod
    def output_columns(self) -> Set[str]:
        """Columns that will be added to DataFrame."""
        pass

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated columns to DataFrame.

        Args:
            df: Input DataFrame (not modified)

        Returns:
            New DataFrame with added columns

        Raises:
            CalculatorError: If calculation fails
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns."""
        missing = self.required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Calculator '{self.name}' missing columns: {missing}. "
                f"Available: {set(df.columns)}"
            )
```

### 4.6 Model Store Interface (interfaces/model_store.py)

```python
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelMetadata:
    """Metadata for saved models."""
    model_id: str
    model_type: str
    created_at: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]

class IModelStore(ABC):
    """Interface for ML model persistence."""

    @abstractmethod
    def save(
        self,
        model: Any,
        model_id: str,
        model_type: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> ModelMetadata:
        """Save model with metadata."""
        pass

    @abstractmethod
    def load(self, model_id: str) -> tuple:
        """Load model and metadata."""
        pass

    @abstractmethod
    def list_models(self, model_type: Optional[str] = None) -> list:
        """List available models."""
        pass

    @abstractmethod
    def delete(self, model_id: str) -> bool:
        """Delete a model."""
        pass
```

### 4.7 Visualizer Interface (interfaces/visualizer.py)

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any
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
```

---

## 5. Domain Layer (Value Objects and Exceptions)

### 5.1 Domain Exceptions (domain/exceptions.py)

```python
class QuantNoteError(Exception):
    """Base exception for QuantNote."""
    pass

class DataSourceError(QuantNoteError):
    """Error fetching data from source."""
    pass

class ValidationError(QuantNoteError):
    """Data validation failed."""
    def __init__(self, errors: list):
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")

class CalculatorError(QuantNoteError):
    """Error in calculator execution."""
    pass

class PipelineError(QuantNoteError):
    """Error in pipeline execution."""
    pass

class ModelNotFoundError(QuantNoteError):
    """Requested model not found."""
    pass

class InsufficientDataError(QuantNoteError):
    """Not enough data for analysis."""
    pass
```

### 5.2 Value Objects (domain/value_objects.py)

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
from enum import Enum

class TrendDirection(Enum):
    """Market trend direction."""
    BULL = "bull"
    BEAR = "bear"
    FLAT = "flat"

class VolatilityLevel(Enum):
    """Volatility level."""
    HIGH = "high"
    LOW = "low"

@dataclass(frozen=True)
class LogReturn:
    """
    Immutable value object for log-returns.
    Encapsulates conversion logic and validation.
    """
    value: float

    def __post_init__(self):
        if not isinstance(self.value, (int, float)) or np.isnan(self.value):
            raise ValueError(f"Invalid log return value: {self.value}")
        # Log returns typically between -100% and +100% in daily data
        if not -2.0 <= self.value <= 2.0:
            raise ValueError(f"Log return {self.value} outside expected range [-2, 2]")

    def to_percent(self) -> float:
        """Convert to percentage return."""
        return np.exp(self.value) - 1

    def to_basis_points(self) -> float:
        """Convert to basis points."""
        return self.to_percent() * 10000

    @classmethod
    def from_percent(cls, percent: float) -> 'LogReturn':
        """Create from percentage return."""
        if percent <= -1.0:
            raise ValueError("Percentage return must be > -100%")
        return cls(np.log(1 + percent))

    @classmethod
    def from_prices(cls, price_current: float, price_previous: float) -> 'LogReturn':
        """Create from two consecutive prices."""
        if price_current <= 0 or price_previous <= 0:
            raise ValueError("Prices must be positive")
        return cls(np.log(price_current / price_previous))

    def __add__(self, other: 'LogReturn') -> 'LogReturn':
        """Log returns are additive."""
        return LogReturn(self.value + other.value)

    def __neg__(self) -> 'LogReturn':
        return LogReturn(-self.value)

@dataclass(frozen=True)
class Price:
    """
    Immutable value object for prices.
    Ensures positive values.
    """
    value: float
    currency: str = "BRL"

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError(f"Price must be positive, got {self.value}")

    def log(self) -> float:
        """Return log of price."""
        return np.log(self.value)

    def apply_return(self, log_return: LogReturn) -> 'Price':
        """Apply a log return to get new price."""
        new_value = self.value * np.exp(log_return.value)
        return Price(new_value, self.currency)

@dataclass(frozen=True)
class Regime:
    """
    Immutable value object representing a market regime.
    Combines trend and volatility.
    """
    trend: TrendDirection
    volatility: VolatilityLevel

    @property
    def name(self) -> str:
        """Human-readable regime name."""
        return f"{self.trend.value}_{self.volatility.value}_vol"

    @property
    def is_favorable(self) -> bool:
        """Whether regime is typically favorable for long positions."""
        return self.trend == TrendDirection.BULL

    @classmethod
    def from_indicators(
        cls,
        slope: float,
        volatility: float,
        slope_threshold: float,
        volatility_threshold: float
    ) -> 'Regime':
        """Create regime from indicator values."""
        if slope > slope_threshold:
            trend = TrendDirection.BULL
        elif slope < -slope_threshold:
            trend = TrendDirection.BEAR
        else:
            trend = TrendDirection.FLAT

        vol_level = VolatilityLevel.HIGH if volatility > volatility_threshold else VolatilityLevel.LOW

        return cls(trend=trend, volatility=vol_level)

    @classmethod
    def all_regimes(cls) -> list:
        """Return all possible regime combinations."""
        return [
            cls(trend, vol)
            for trend in TrendDirection
            for vol in VolatilityLevel
        ]

@dataclass(frozen=True)
class Probability:
    """
    Immutable value object for probabilities.
    Ensures value is between 0 and 1.
    """
    value: float

    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {self.value}")

    def to_percent(self) -> float:
        """Convert to percentage."""
        return self.value * 100

    def to_odds(self) -> float:
        """Convert to odds ratio."""
        if self.value == 1.0:
            return float('inf')
        return self.value / (1 - self.value)

    @classmethod
    def from_frequency(cls, hits: int, total: int) -> 'Probability':
        """Create from hit/total counts."""
        if total == 0:
            raise ValueError("Cannot compute probability with zero total")
        return cls(hits / total)
```

---

## 6. Data Validation

### 6.1 OHLCV Validator (infrastructure/validators/ohlcv_validator.py)

```python
import pandas as pd
import numpy as np
from typing import Set

from ...interfaces.validator import IValidator, ValidationResult

class OHLCVValidator(IValidator):
    """Validates OHLCV data structure and values."""

    REQUIRED_COLUMNS: Set[str] = {'date', 'open', 'high', 'low', 'close', 'volume'}

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        # Check required columns
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(is_valid=False, errors=errors)

        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors)

        # Check for non-positive prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            non_positive = (df[col] <= 0).sum()
            if non_positive > 0:
                errors.append(f"Column '{col}' has {non_positive} non-positive values")

        # Check OHLC relationships
        invalid_high = (df['high'] < df['low']).sum()
        if invalid_high > 0:
            errors.append(f"{invalid_high} rows have high < low")

        invalid_open_high = (df['open'] > df['high']).sum()
        if invalid_open_high > 0:
            warnings.append(f"{invalid_open_high} rows have open > high")

        invalid_close_low = (df['close'] < df['low']).sum()
        if invalid_close_low > 0:
            warnings.append(f"{invalid_close_low} rows have close < low")

        # Check for NaN values
        for col in price_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                warnings.append(f"Column '{col}' has {nan_count} NaN values")

        # Check for duplicate dates
        if 'date' in df.columns:
            duplicates = df['date'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"{duplicates} duplicate dates found")

        # Check date ordering
        if 'date' in df.columns and len(df) > 1:
            if not df['date'].is_monotonic_increasing:
                warnings.append("Dates are not in ascending order")

        # Check for extreme price changes (potential data errors)
        if len(df) > 1:
            returns = df['close'].pct_change().abs()
            extreme_moves = (returns > 0.5).sum()  # >50% daily move
            if extreme_moves > 0:
                warnings.append(f"{extreme_moves} extreme price moves (>50%) detected")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

### 6.2 Series Length Validator (infrastructure/validators/series_length_validator.py)

```python
import pandas as pd
from ...interfaces.validator import IValidator, ValidationResult

class SeriesLengthValidator(IValidator):
    """Validates that series has minimum required length."""

    def __init__(self, min_length: int, max_window: int):
        """
        Args:
            min_length: Minimum required data points
            max_window: Maximum window size that will be used
        """
        self.min_length = min_length
        self.max_window = max_window

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        actual_length = len(df)

        if actual_length < self.min_length:
            errors.append(
                f"Series has {actual_length} points, minimum required is {self.min_length}"
            )

        if actual_length < self.max_window * 3:
            warnings.append(
                f"Series length ({actual_length}) is less than 3x max window ({self.max_window}). "
                "Results may be unreliable."
            )

        # Check for usable data after window application
        usable_points = actual_length - self.max_window
        if usable_points < 50:
            warnings.append(
                f"Only {usable_points} usable data points after applying windows"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

### 6.3 Gap Validator (infrastructure/validators/gap_validator.py)

```python
import pandas as pd
import numpy as np
from ...interfaces.validator import IValidator, ValidationResult

class GapValidator(IValidator):
    """Validates for data gaps in time series."""

    def __init__(self, max_gap_days: int = 5):
        """
        Args:
            max_gap_days: Maximum allowed gap between trading days
        """
        self.max_gap_days = max_gap_days

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        if 'date' not in df.columns or len(df) < 2:
            return ValidationResult(is_valid=True)

        # Calculate gaps between dates
        dates = pd.to_datetime(df['date'])
        gaps = dates.diff().dt.days

        # Find large gaps (excluding weekends which are typically 3 days)
        large_gaps = gaps[gaps > self.max_gap_days]

        if len(large_gaps) > 0:
            gap_info = [
                f"{dates.iloc[i].date()} ({gap} days)"
                for i, gap in large_gaps.items()
            ]
            if len(large_gaps) > 5:
                warnings.append(
                    f"{len(large_gaps)} gaps > {self.max_gap_days} days found. "
                    f"First 5: {gap_info[:5]}"
                )
            else:
                warnings.append(
                    f"Gaps > {self.max_gap_days} days found: {gap_info}"
                )

        return ValidationResult(
            is_valid=True,  # Gaps are warnings, not errors
            errors=errors,
            warnings=warnings
        )
```

### 6.4 Validation Pipeline (infrastructure/validators/__init__.py)

```python
from typing import List
from ...interfaces.validator import IValidator, CompositeValidator, ValidationResult
from .ohlcv_validator import OHLCVValidator
from .series_length_validator import SeriesLengthValidator
from .gap_validator import GapValidator

def create_default_validator(
    min_length: int = 252,
    max_window: int = 60,
    max_gap_days: int = 5
) -> IValidator:
    """Create default validation pipeline."""
    return CompositeValidator([
        OHLCVValidator(),
        SeriesLengthValidator(min_length, max_window),
        GapValidator(max_gap_days)
    ])
```

---

## 7. Infrastructure Implementations

### 7.1 Structured Logger (infrastructure/file_logger.py)

```python
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..interfaces.logger import ILogger

class FileLogger(ILogger):
    """Structured file logger with JSON formatting."""

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: int = logging.INFO
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # File handler with JSON formatting
        log_file = self.log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        self.logger.addHandler(console)

    def _format_message(self, level: str, message: str, **context: Any) -> str:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **context
        }
        return json.dumps(log_entry)

    def debug(self, message: str, **context: Any) -> None:
        self.logger.debug(self._format_message("DEBUG", message, **context))

    def info(self, message: str, **context: Any) -> None:
        self.logger.info(self._format_message("INFO", message, **context))

    def warning(self, message: str, **context: Any) -> None:
        self.logger.warning(self._format_message("WARNING", message, **context))

    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **context: Any
    ) -> None:
        if exception:
            context["exception"] = str(exception)
            context["exception_type"] = type(exception).__name__
        self.logger.error(self._format_message("ERROR", message, **context))

class NullLogger(ILogger):
    """No-op logger for testing."""

    def debug(self, message: str, **context: Any) -> None:
        pass

    def info(self, message: str, **context: Any) -> None:
        pass

    def warning(self, message: str, **context: Any) -> None:
        pass

    def error(self, message: str, exception: Optional[Exception] = None, **context: Any) -> None:
        pass
```

### 7.2 Yahoo Data Source with Rate Limiting (infrastructure/yahoo_data_source.py)

```python
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional
from functools import wraps
import time

from ..interfaces.data_source import IDataSource
from ..interfaces.logger import ILogger
from ..domain.exceptions import DataSourceError
from .file_logger import NullLogger

def rate_limit(calls: int, period: int):
    """Rate limiting decorator."""
    min_interval = period / calls
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

class YahooDataSource(IDataSource):
    """Yahoo Finance data source with rate limiting."""

    def __init__(
        self,
        calls_per_minute: int = 5,
        logger: Optional[ILogger] = None
    ):
        self.calls_per_minute = calls_per_minute
        self.logger = logger or NullLogger()
        self._apply_rate_limit()

    def _apply_rate_limit(self):
        """Apply rate limiting to fetch method."""
        self._fetch_internal = rate_limit(
            self.calls_per_minute, 60
        )(self._fetch_internal)

    def _fetch_internal(self, ticker: str, **kwargs) -> pd.DataFrame:
        """Internal fetch method (rate limited)."""
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.history(**kwargs)

    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""

        self.logger.info(
            "Fetching data from Yahoo Finance",
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date)
        )

        try:
            if start_date is None:
                df = self._fetch_internal(ticker, period="max")
            else:
                df = self._fetch_internal(
                    ticker,
                    start=start_date,
                    end=end_date
                )

            if df.empty:
                raise DataSourceError(f"No data returned for ticker {ticker}")

            # Normalize column names
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            # Rename 'adj_close' if present
            if 'adj_close' in df.columns:
                df = df.drop(columns=['close'])
                df = df.rename(columns={'adj_close': 'close'})

            # Select only OHLCV columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in required_cols if c in df.columns]

            if len(available_cols) < len(required_cols):
                missing = set(required_cols) - set(available_cols)
                raise DataSourceError(f"Missing columns: {missing}")

            df = df[required_cols].copy()

            # Normalize date
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            self.logger.info(
                "Data fetched successfully",
                ticker=ticker,
                rows=len(df),
                date_range=f"{df['date'].min()} to {df['date'].max()}"
            )

            return df

        except Exception as e:
            self.logger.error(
                "Failed to fetch data",
                exception=e,
                ticker=ticker
            )
            raise DataSourceError(f"Failed to fetch {ticker}: {e}") from e
```

### 7.3 Parquet Repository (infrastructure/parquet_repository.py)

```python
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from ..interfaces.repository import IRepository, FileMetadata
from ..interfaces.logger import ILogger
from .file_logger import NullLogger

class ParquetRepository(IRepository):
    """Repository that persists DataFrames in Parquet format."""

    def __init__(
        self,
        data_dir: str = "data",
        logger: Optional[ILogger] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or NullLogger()

    def _generate_filename(self, ticker: str) -> str:
        """Generate filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_clean = ticker.replace(".", "_").replace("/", "_")
        return f"{ticker_clean}_{timestamp}.parquet"

    def _parse_filename(self, filename: str) -> tuple:
        """Parse ticker and timestamp from filename."""
        parts = filename.replace(".parquet", "").rsplit("_", 2)
        if len(parts) >= 3:
            ticker = "_".join(parts[:-2])
            date_str = parts[-2]
            time_str = parts[-1]
            created = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return ticker, created
        return filename, None

    def save(self, df: pd.DataFrame, ticker: str) -> FileMetadata:
        """Save DataFrame in Parquet format."""
        filename = self._generate_filename(ticker)
        filepath = self.data_dir / filename

        df.to_parquet(filepath, index=False)

        metadata = FileMetadata(
            filename=filename,
            ticker=ticker,
            created_at=datetime.now(),
            row_count=len(df),
            date_range=(
                df['date'].min() if 'date' in df.columns else None,
                df['date'].max() if 'date' in df.columns else None
            )
        )

        self.logger.info(
            "Data saved",
            filename=filename,
            rows=len(df)
        )

        return metadata

    def load(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load most recent DataFrame for ticker."""
        files = self.list_files(ticker)
        if not files:
            return None

        latest = files[-1]  # Sorted by date
        return self.load_by_filename(latest.filename)

    def load_by_filename(self, filename: str) -> pd.DataFrame:
        """Load specific file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        return pd.read_parquet(filepath)

    def list_files(self, ticker: Optional[str] = None) -> List[FileMetadata]:
        """List available files."""
        files = list(self.data_dir.glob("*.parquet"))

        result = []
        for f in files:
            parsed_ticker, created = self._parse_filename(f.name)

            if ticker:
                ticker_clean = ticker.replace(".", "_").replace("/", "_")
                if not f.name.startswith(ticker_clean):
                    continue

            # Read row count without loading full file
            try:
                df_meta = pd.read_parquet(f, columns=['date'] if 'date' else [])
                row_count = len(df_meta)
                date_range = (df_meta['date'].min(), df_meta['date'].max()) if 'date' in df_meta.columns else (None, None)
            except:
                row_count = 0
                date_range = (None, None)

            result.append(FileMetadata(
                filename=f.name,
                ticker=parsed_ticker,
                created_at=created or datetime.fromtimestamp(f.stat().st_mtime),
                row_count=row_count,
                date_range=date_range
            ))

        return sorted(result, key=lambda x: x.created_at)

    def delete(self, filename: str) -> bool:
        """Delete a file."""
        filepath = self.data_dir / filename
        if filepath.exists():
            filepath.unlink()
            self.logger.info("File deleted", filename=filename)
            return True
        return False
```

### 7.4 Model Store (infrastructure/joblib_model_store.py)

```python
import joblib
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple

from ..interfaces.model_store import IModelStore, ModelMetadata
from ..interfaces.logger import ILogger
from ..domain.exceptions import ModelNotFoundError
from .file_logger import NullLogger

class JoblibModelStore(IModelStore):
    """Model persistence using joblib."""

    def __init__(
        self,
        models_dir: str = "models",
        logger: Optional[ILogger] = None
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or NullLogger()

    def _get_filepath(self, model_id: str) -> Path:
        return self.models_dir / f"{model_id}.joblib"

    def save(
        self,
        model: Any,
        model_id: str,
        model_type: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> ModelMetadata:
        """Save model with metadata."""

        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            created_at=datetime.now(),
            parameters=parameters,
            metrics=metrics
        )

        data = {
            'model': model,
            'metadata': metadata
        }

        filepath = self._get_filepath(model_id)
        joblib.dump(data, filepath)

        self.logger.info(
            "Model saved",
            model_id=model_id,
            model_type=model_type,
            metrics=metrics
        )

        return metadata

    def load(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata."""
        filepath = self._get_filepath(model_id)

        if not filepath.exists():
            raise ModelNotFoundError(f"Model not found: {model_id}")

        data = joblib.load(filepath)

        self.logger.info("Model loaded", model_id=model_id)

        return data['model'], data['metadata']

    def list_models(self, model_type: Optional[str] = None) -> List[ModelMetadata]:
        """List available models."""
        result = []

        for filepath in self.models_dir.glob("*.joblib"):
            try:
                data = joblib.load(filepath)
                metadata = data['metadata']

                if model_type and metadata.model_type != model_type:
                    continue

                result.append(metadata)
            except Exception:
                continue

        return sorted(result, key=lambda x: x.created_at, reverse=True)

    def delete(self, model_id: str) -> bool:
        """Delete a model."""
        filepath = self._get_filepath(model_id)
        if filepath.exists():
            filepath.unlink()
            self.logger.info("Model deleted", model_id=model_id)
            return True
        return False
```

---

## 8. Column Calculators

### 8.1 Log Price Calculator (calculators/log_price_calculator.py)

```python
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator

class LogPriceCalculator(IColumnCalculator):
    """Calculates logarithm of closing price."""

    @property
    def name(self) -> str:
        return "log_price"

    @property
    def required_columns(self) -> Set[str]:
        return {'close'}

    @property
    def output_columns(self) -> Set[str]:
        return {'log_close'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()
        result['log_close'] = np.log(result['close'])
        return result
```

### 8.2 Log Return Calculator (calculators/log_return_calculator.py)

```python
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator

class LogReturnCalculator(IColumnCalculator):
    """Calculates daily log return and rolling sum."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"log_return_w{self._window}"

    @property
    def required_columns(self) -> Set[str]:
        return {'close'}

    @property
    def output_columns(self) -> Set[str]:
        return {'log_return', f'log_return_rolling_{self._window}'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        # Daily log return
        result['log_return'] = np.log(result['close'] / result['close'].shift(1))

        # Rolling sum (= log of total return over window)
        result[f'log_return_rolling_{self._window}'] = (
            result['log_return'].rolling(window=self._window).sum()
        )

        return result
```

### 8.3 Future Return Calculator (calculators/future_return_calculator.py)

```python
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator

class FutureReturnCalculator(IColumnCalculator):
    """Calculates future log return over H periods."""

    def __init__(self, horizon: int = 3):
        self._horizon = horizon

    @property
    def name(self) -> str:
        return f"future_return_h{self._horizon}"

    @property
    def required_columns(self) -> Set[str]:
        return {'close'}

    @property
    def output_columns(self) -> Set[str]:
        return {f'log_return_future_{self._horizon}'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        # Future return: log(P_{t+H} / P_t)
        result[f'log_return_future_{self._horizon}'] = (
            np.log(result['close'].shift(-self._horizon) / result['close'])
        )

        return result
```

### 8.4 Volatility Calculator (calculators/volatility_calculator.py)

```python
import pandas as pd
import numpy as np
from typing import Set

from ..interfaces.column_calculator import IColumnCalculator

class VolatilityCalculator(IColumnCalculator):
    """Calculates rolling volatility (std of returns)."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"volatility_w{self._window}"

    @property
    def required_columns(self) -> Set[str]:
        return {'log_return'}

    @property
    def output_columns(self) -> Set[str]:
        return {f'volatility_{self._window}'}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        result[f'volatility_{self._window}'] = (
            result['log_return'].rolling(window=self._window).std()
        )

        return result
```

### 8.5 Slope Calculator (calculators/slope_calculator.py)

```python
import pandas as pd
import numpy as np
from scipy import stats
from typing import Set
from functools import lru_cache

from ..interfaces.column_calculator import IColumnCalculator

class SlopeCalculator(IColumnCalculator):
    """Calculates slope of log-price linear regression over rolling window."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"slope_w{self._window}"

    @property
    def required_columns(self) -> Set[str]:
        return {'log_close'}

    @property
    def output_columns(self) -> Set[str]:
        return {f'slope_{self._window}'}

    def _calculate_slope(self, values: np.ndarray) -> float:
        """Calculate slope via linear regression."""
        if np.any(np.isnan(values)):
            return np.nan

        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        result = df.copy()

        result[f'slope_{self._window}'] = (
            result['log_close']
            .rolling(window=self._window)
            .apply(self._calculate_slope, raw=True)
        )

        return result
```

### 8.6 Return Converter Utility (utils/return_converter.py)

```python
import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]

def log_return_from_prices(
    price_current: ArrayLike,
    price_previous: ArrayLike
) -> ArrayLike:
    """Calculate log-return between two prices."""
    return np.log(price_current / price_previous)

def log_to_percent(log_ret: ArrayLike) -> ArrayLike:
    """Convert log-return to percentage return."""
    return np.exp(log_ret) - 1

def percent_to_log(percent_ret: ArrayLike) -> ArrayLike:
    """Convert percentage return to log-return."""
    return np.log(1 + percent_ret)

def annualize_return(
    period_return: float,
    periods_per_year: int = 252
) -> float:
    """Annualize a periodic return."""
    return (1 + period_return) ** periods_per_year - 1

def annualize_volatility(
    period_volatility: float,
    periods_per_year: int = 252
) -> float:
    """Annualize periodic volatility."""
    return period_volatility * np.sqrt(periods_per_year)
```

---

## 9. Pipeline with Dependency Resolution

### 9.1 Dependency Resolver (calculators/dependency_resolver.py)

```python
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
```

### 9.2 Calculator Pipeline (calculators/pipeline.py)

```python
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
```

---

## 10. Regime Analysis

### 10.1 Time Series Splitter (analysis/time_series_splitter.py)

```python
import pandas as pd
from typing import Tuple, List, Iterator
from dataclasses import dataclass

@dataclass
class Split:
    """A train/test split."""
    train: pd.DataFrame
    test: pd.DataFrame
    fold: int

class TimeSeriesSplitter:
    """
    Sequential time series splitter.
    Prevents lookahead bias by ensuring test data always comes after train.
    """

    def __init__(self, train_ratio: float = 0.7):
        """
        Args:
            train_ratio: Proportion of data for training (0.5 to 0.9)
        """
        if not 0.5 <= train_ratio <= 0.9:
            raise ValueError("train_ratio must be between 0.5 and 0.9")
        self.train_ratio = train_ratio

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple train/test split.

        Returns:
            (train_df, test_df)
        """
        split_idx = int(len(df) * self.train_ratio)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def walk_forward_split(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        min_train_size: int = 100
    ) -> Iterator[Split]:
        """
        Walk-forward cross-validation splits.

        Each fold uses all previous data for training and next segment for testing.

        Args:
            df: DataFrame to split
            n_folds: Number of folds
            min_train_size: Minimum training set size

        Yields:
            Split objects with train and test DataFrames
        """
        n = len(df)
        fold_size = (n - min_train_size) // n_folds

        if fold_size < 10:
            raise ValueError(
                f"Not enough data for {n_folds} folds. "
                f"Need at least {min_train_size + 10 * n_folds} rows."
            )

        for i in range(n_folds):
            train_end = min_train_size + i * fold_size
            test_end = train_end + fold_size

            if test_end > n:
                test_end = n

            yield Split(
                train=df.iloc[:train_end].copy(),
                test=df.iloc[train_end:test_end].copy(),
                fold=i
            )

    def expanding_window_split(
        self,
        df: pd.DataFrame,
        initial_train_size: int = 252,
        step_size: int = 21
    ) -> Iterator[Split]:
        """
        Expanding window splits (each train set grows).

        Args:
            df: DataFrame to split
            initial_train_size: Initial training set size
            step_size: Number of periods to add each iteration

        Yields:
            Split objects
        """
        n = len(df)
        fold = 0

        train_end = initial_train_size
        while train_end + step_size <= n:
            test_end = min(train_end + step_size, n)

            yield Split(
                train=df.iloc[:train_end].copy(),
                test=df.iloc[train_end:test_end].copy(),
                fold=fold
            )

            train_end = test_end
            fold += 1
```

### 10.2 Manual Regime Classifier (analysis/regime_classifier.py)

```python
import pandas as pd
import numpy as np
from typing import Optional

from ..domain.value_objects import Regime, TrendDirection, VolatilityLevel

class ManualRegimeClassifier:
    """Classifies regimes using manual thresholds."""

    def __init__(
        self,
        slope_column: str,
        volatility_column: str,
        slope_threshold: Optional[float] = None,
        volatility_threshold: Optional[float] = None
    ):
        self.slope_column = slope_column
        self.volatility_column = volatility_column
        self.slope_threshold = slope_threshold
        self.volatility_threshold = volatility_threshold

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime column to DataFrame."""
        result = df.copy()

        # Auto-calculate thresholds if not provided
        slope_thresh = self.slope_threshold
        if slope_thresh is None:
            slope_thresh = result[self.slope_column].std() * 0.5

        vol_thresh = self.volatility_threshold
        if vol_thresh is None:
            vol_thresh = result[self.volatility_column].median()

        # Store thresholds for reference
        self._slope_threshold = slope_thresh
        self._volatility_threshold = vol_thresh

        # Classify each row
        def classify_row(row):
            if pd.isna(row[self.slope_column]) or pd.isna(row[self.volatility_column]):
                return None

            regime = Regime.from_indicators(
                slope=row[self.slope_column],
                volatility=row[self.volatility_column],
                slope_threshold=slope_thresh,
                volatility_threshold=vol_thresh
            )
            return regime.name

        result['regime'] = result.apply(classify_row, axis=1)

        return result

    def get_thresholds(self) -> dict:
        """Return the thresholds used for classification."""
        return {
            'slope_threshold': self._slope_threshold,
            'volatility_threshold': self._volatility_threshold
        }
```

### 10.3 K-Means Regime Classifier with Persistence (analysis/kmeans_regimes.py)

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from ..interfaces.model_store import IModelStore
from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger

@dataclass
class ClusterStatistics:
    """Statistics for a cluster."""
    cluster_id: int
    count: int
    percentage: float
    feature_means: Dict[str, float]
    future_return_mean: Optional[float]
    future_return_std: Optional[float]
    interpretation: Optional[str]

class KMeansRegimeClassifier:
    """K-Means clustering for regime classification with persistence."""

    def __init__(
        self,
        n_clusters: int = 3,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        logger: Optional[ILogger] = None
    ):
        self.n_clusters = n_clusters
        self.feature_columns = feature_columns
        self.random_state = random_state
        self.logger = logger or NullLogger()

        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self._is_fitted = False
        self._cluster_stats: List[ClusterStatistics] = []

    def fit(self, df: pd.DataFrame) -> 'KMeansRegimeClassifier':
        """Fit the K-Means model."""
        # Auto-detect feature columns if not specified
        if self.feature_columns is None:
            self.feature_columns = [
                c for c in df.columns
                if 'slope' in c or 'volatility' in c or 'log_return_rolling' in c
            ]
            self.logger.info(
                "Auto-detected feature columns",
                columns=self.feature_columns
            )

        # Get valid data
        feature_data = df[self.feature_columns].dropna()

        if len(feature_data) < self.n_clusters * 10:
            raise ValueError(
                f"Not enough valid data points ({len(feature_data)}) "
                f"for {self.n_clusters} clusters"
            )

        # Fit scaler and kmeans
        X_scaled = self.scaler.fit_transform(feature_data)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(X_scaled)

        self._is_fitted = True

        self.logger.info(
            "K-Means fitted",
            n_clusters=self.n_clusters,
            n_samples=len(feature_data)
        )

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict cluster labels for DataFrame."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        result = df.copy()

        # Get valid indices
        feature_data = result[self.feature_columns].dropna()
        valid_idx = feature_data.index

        # Predict
        X_scaled = self.scaler.transform(feature_data)
        clusters = self.kmeans.predict(X_scaled)

        # Add to result
        result['cluster'] = np.nan
        result.loc[valid_idx, 'cluster'] = clusters

        return result

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and predict in one step."""
        self.fit(df)
        return self.predict(df)

    def compute_statistics(
        self,
        df: pd.DataFrame,
        future_return_column: Optional[str] = None
    ) -> List[ClusterStatistics]:
        """Compute statistics for each cluster."""
        if 'cluster' not in df.columns:
            raise ValueError("DataFrame must have 'cluster' column")

        valid_df = df.dropna(subset=['cluster'])
        total = len(valid_df)

        stats = []
        for cluster_id in range(self.n_clusters):
            mask = valid_df['cluster'] == cluster_id
            cluster_data = valid_df[mask]

            # Feature means
            feature_means = {
                col: cluster_data[col].mean()
                for col in self.feature_columns
            }

            # Future return stats
            future_mean = None
            future_std = None
            if future_return_column and future_return_column in df.columns:
                future_rets = cluster_data[future_return_column].dropna()
                if len(future_rets) > 0:
                    future_mean = future_rets.mean()
                    future_std = future_rets.std()

            stats.append(ClusterStatistics(
                cluster_id=cluster_id,
                count=len(cluster_data),
                percentage=len(cluster_data) / total * 100,
                feature_means=feature_means,
                future_return_mean=future_mean,
                future_return_std=future_std,
                interpretation=None
            ))

        self._cluster_stats = stats
        return stats

    def interpret_clusters(
        self,
        df: pd.DataFrame,
        slope_column: str
    ) -> Dict[int, str]:
        """Interpret clusters as bull/bear/flat based on slope."""
        if 'cluster' not in df.columns:
            raise ValueError("DataFrame must have 'cluster' column")

        slope_std = df[slope_column].std()
        interpretations = {}

        for cluster_id in range(self.n_clusters):
            mask = df['cluster'] == cluster_id
            mean_slope = df.loc[mask, slope_column].mean()

            if mean_slope > slope_std * 0.3:
                interpretations[cluster_id] = 'bull'
            elif mean_slope < -slope_std * 0.3:
                interpretations[cluster_id] = 'bear'
            else:
                interpretations[cluster_id] = 'flat'

        # Update stats with interpretations
        for stat in self._cluster_stats:
            stat.interpretation = interpretations.get(stat.cluster_id)

        return interpretations

    def save(self, model_store: IModelStore, model_id: str) -> None:
        """Save model to store."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_clusters': self.n_clusters
        }

        model_store.save(
            model=model_data,
            model_id=model_id,
            model_type='kmeans_regime',
            parameters={
                'n_clusters': self.n_clusters,
                'feature_columns': self.feature_columns
            },
            metrics={}
        )

        self.logger.info("Model saved", model_id=model_id)

    @classmethod
    def load(
        cls,
        model_store: IModelStore,
        model_id: str,
        logger: Optional[ILogger] = None
    ) -> 'KMeansRegimeClassifier':
        """Load model from store."""
        model_data, metadata = model_store.load(model_id)

        instance = cls(
            n_clusters=model_data['n_clusters'],
            feature_columns=model_data['feature_columns'],
            logger=logger
        )
        instance.kmeans = model_data['kmeans']
        instance.scaler = model_data['scaler']
        instance._is_fitted = True

        return instance
```

### 10.4 Probability Calculator (analysis/probability_calculator.py)

```python
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from ..domain.value_objects import Probability

@dataclass
class ConditionalProbability:
    """Probability conditioned on regime."""
    regime: str
    probability: Probability
    count: int
    mean_return: float
    std_return: float

@dataclass
class SeparationMetrics:
    """Metrics measuring separation between regimes."""
    delta_p: float  # Max prob - min prob
    delta_mean: float  # Max mean return - min mean return
    max_probability: float
    min_probability: float
    information_ratio: float  # delta_mean / pooled_std

class ProbabilityCalculator:
    """Calculates raw and conditional probabilities."""

    def __init__(
        self,
        future_return_column: str,
        target_return: float,
        regime_column: str = 'regime'
    ):
        self.future_return_column = future_return_column
        self.target_return = target_return
        self.regime_column = regime_column

        # Convert target to log return
        self.log_target = np.log(1 + target_return)

    def calculate_raw_probability(self, df: pd.DataFrame) -> Probability:
        """Calculate unconditional probability."""
        valid_returns = df[self.future_return_column].dropna()

        if len(valid_returns) == 0:
            raise ValueError("No valid returns to calculate probability")

        hits = (valid_returns > self.log_target).sum()
        return Probability.from_frequency(hits, len(valid_returns))

    def calculate_conditional_probabilities(
        self,
        df: pd.DataFrame
    ) -> Dict[str, ConditionalProbability]:
        """Calculate probability for each regime."""
        results = {}

        valid_df = df.dropna(subset=[self.regime_column, self.future_return_column])

        for regime in valid_df[self.regime_column].unique():
            mask = valid_df[self.regime_column] == regime
            regime_returns = valid_df.loc[mask, self.future_return_column]

            hits = (regime_returns > self.log_target).sum()
            total = len(regime_returns)

            results[regime] = ConditionalProbability(
                regime=regime,
                probability=Probability.from_frequency(hits, total),
                count=total,
                mean_return=regime_returns.mean(),
                std_return=regime_returns.std()
            )

        return results

    def calculate_separation_metrics(
        self,
        conditional_probs: Dict[str, ConditionalProbability]
    ) -> SeparationMetrics:
        """Calculate metrics measuring regime separation."""
        probs = [cp.probability.value for cp in conditional_probs.values()]
        means = [cp.mean_return for cp in conditional_probs.values()]
        stds = [cp.std_return for cp in conditional_probs.values()]
        counts = [cp.count for cp in conditional_probs.values()]

        # Pooled standard deviation
        total_count = sum(counts)
        pooled_var = sum(
            (n - 1) * s**2 for n, s in zip(counts, stds)
        ) / (total_count - len(counts))
        pooled_std = np.sqrt(pooled_var)

        delta_mean = max(means) - min(means)
        info_ratio = delta_mean / pooled_std if pooled_std > 0 else 0

        return SeparationMetrics(
            delta_p=max(probs) - min(probs),
            delta_mean=delta_mean,
            max_probability=max(probs),
            min_probability=min(probs),
            information_ratio=info_ratio
        )

    def generate_report(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """Generate complete probability analysis report."""
        raw_prob = self.calculate_raw_probability(df)
        cond_probs = self.calculate_conditional_probabilities(df)
        separation = self.calculate_separation_metrics(cond_probs)

        return {
            'target_return': self.target_return,
            'log_target': self.log_target,
            'raw_probability': raw_prob.value,
            'raw_probability_pct': raw_prob.to_percent(),
            'conditional_probabilities': {
                regime: {
                    'probability': cp.probability.value,
                    'probability_pct': cp.probability.to_percent(),
                    'count': cp.count,
                    'mean_return': cp.mean_return,
                    'std_return': cp.std_return
                }
                for regime, cp in cond_probs.items()
            },
            'separation_metrics': {
                'delta_p': separation.delta_p,
                'delta_mean': separation.delta_mean,
                'max_probability': separation.max_probability,
                'min_probability': separation.min_probability,
                'information_ratio': separation.information_ratio
            }
        }
```

---

## 11. Genetic Algorithm Optimization

### 11.1 Chromosome (optimization/chromosome.py)

```python
from dataclasses import dataclass
from typing import Tuple, Optional
import random

from ..config.search_space import GASearchSpace

@dataclass
class Chromosome:
    """Represents a set of parameters to be optimized."""

    window_slope: int
    window_volatility: int
    window_rolling_return: int
    horizon: int
    target_return: float
    n_clusters: int
    use_volatility: bool = True
    use_rolling_return: bool = True

    @classmethod
    def random(cls, search_space: GASearchSpace) -> 'Chromosome':
        """Create chromosome with random values within search space."""
        return cls(
            window_slope=random.randint(*search_space.window_slope),
            window_volatility=random.randint(*search_space.window_volatility),
            window_rolling_return=random.randint(*search_space.window_rolling_return),
            horizon=random.randint(*search_space.horizon),
            target_return=random.uniform(*search_space.target_return),
            n_clusters=random.randint(*search_space.n_clusters),
            use_volatility=random.choice([True, False]),
            use_rolling_return=random.choice([True, False])
        )

    def mutate(
        self,
        search_space: GASearchSpace,
        mutation_rate: float = 0.1
    ) -> 'Chromosome':
        """Return mutated copy of chromosome."""
        def mutate_int(val: int, range_: Tuple[int, int]) -> int:
            if random.random() < mutation_rate:
                delta = random.randint(-5, 5)
                return max(range_[0], min(range_[1], val + delta))
            return val

        def mutate_float(val: float, range_: Tuple[float, float]) -> float:
            if random.random() < mutation_rate:
                delta = random.uniform(-0.01, 0.01)
                return max(range_[0], min(range_[1], val + delta))
            return val

        def mutate_bool(val: bool) -> bool:
            if random.random() < mutation_rate:
                return not val
            return val

        return Chromosome(
            window_slope=mutate_int(self.window_slope, search_space.window_slope),
            window_volatility=mutate_int(self.window_volatility, search_space.window_volatility),
            window_rolling_return=mutate_int(self.window_rolling_return, search_space.window_rolling_return),
            horizon=mutate_int(self.horizon, search_space.horizon),
            target_return=mutate_float(self.target_return, search_space.target_return),
            n_clusters=mutate_int(self.n_clusters, search_space.n_clusters),
            use_volatility=mutate_bool(self.use_volatility),
            use_rolling_return=mutate_bool(self.use_rolling_return)
        )

    @staticmethod
    def crossover(parent1: 'Chromosome', parent2: 'Chromosome') -> 'Chromosome':
        """Uniform crossover between two parents."""
        return Chromosome(
            window_slope=random.choice([parent1.window_slope, parent2.window_slope]),
            window_volatility=random.choice([parent1.window_volatility, parent2.window_volatility]),
            window_rolling_return=random.choice([parent1.window_rolling_return, parent2.window_rolling_return]),
            horizon=random.choice([parent1.horizon, parent2.horizon]),
            target_return=random.choice([parent1.target_return, parent2.target_return]),
            n_clusters=random.choice([parent1.n_clusters, parent2.n_clusters]),
            use_volatility=random.choice([parent1.use_volatility, parent2.use_volatility]),
            use_rolling_return=random.choice([parent1.use_rolling_return, parent2.use_rolling_return])
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'window_slope': self.window_slope,
            'window_volatility': self.window_volatility,
            'window_rolling_return': self.window_rolling_return,
            'horizon': self.horizon,
            'target_return': self.target_return,
            'n_clusters': self.n_clusters,
            'use_volatility': self.use_volatility,
            'use_rolling_return': self.use_rolling_return
        }
```

### 11.2 Calculator Factory (optimization/calculator_factory.py)

```python
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

    def create_pipeline(self, chromosome: Chromosome) -> CalculatorPipeline:
        """Create a pipeline configured by the chromosome."""
        calculators: List[IColumnCalculator] = [
            LogPriceCalculator(),
            LogReturnCalculator(window=chromosome.window_rolling_return),
            FutureReturnCalculator(horizon=chromosome.horizon),
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

    def get_future_return_column(self, chromosome: Chromosome) -> str:
        """Get the future return column name."""
        return f'log_return_future_{chromosome.horizon}'
```

### 11.3 Walk-Forward Validator (optimization/walk_forward_validator.py)

```python
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..analysis.time_series_splitter import TimeSeriesSplitter, Split
from ..analysis.kmeans_regimes import KMeansRegimeClassifier
from ..analysis.probability_calculator import ProbabilityCalculator
from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from .chromosome import Chromosome
from .calculator_factory import CalculatorFactory

@dataclass
class FoldResult:
    """Result for a single fold."""
    fold: int
    train_size: int
    test_size: int
    delta_p_train: float
    delta_p_test: float
    stability_score: float

@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation result."""
    mean_delta_p_test: float
    std_delta_p_test: float
    mean_stability: float
    overfitting_ratio: float  # train_perf / test_perf
    fold_results: List[FoldResult]

class WalkForwardValidator:
    """
    Walk-forward validation to prevent overfitting.
    Tests parameter generalization across multiple time periods.
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_train_size: int = 252,
        logger: Optional[ILogger] = None
    ):
        self.n_folds = n_folds
        self.min_train_size = min_train_size
        self.logger = logger or NullLogger()
        self.splitter = TimeSeriesSplitter()
        self.factory = CalculatorFactory()

    def validate(
        self,
        df: pd.DataFrame,
        chromosome: Chromosome
    ) -> WalkForwardResult:
        """
        Perform walk-forward validation.

        Args:
            df: Base DataFrame with OHLCV data
            chromosome: Parameters to validate

        Returns:
            WalkForwardResult with aggregated metrics
        """
        fold_results = []

        for split in self.splitter.walk_forward_split(
            df, self.n_folds, self.min_train_size
        ):
            result = self._evaluate_fold(split, chromosome)
            fold_results.append(result)

            self.logger.debug(
                "Fold evaluated",
                fold=split.fold,
                delta_p_train=result.delta_p_train,
                delta_p_test=result.delta_p_test
            )

        # Aggregate results
        train_deltas = [r.delta_p_train for r in fold_results]
        test_deltas = [r.delta_p_test for r in fold_results]

        mean_train = np.mean(train_deltas)
        mean_test = np.mean(test_deltas)

        return WalkForwardResult(
            mean_delta_p_test=mean_test,
            std_delta_p_test=np.std(test_deltas),
            mean_stability=np.mean([r.stability_score for r in fold_results]),
            overfitting_ratio=mean_train / mean_test if mean_test > 0 else float('inf'),
            fold_results=fold_results
        )

    def _evaluate_fold(
        self,
        split: Split,
        chromosome: Chromosome
    ) -> FoldResult:
        """Evaluate a single fold."""
        # Create pipeline
        pipeline = self.factory.create_pipeline(chromosome)
        feature_cols = self.factory.get_feature_columns(chromosome)
        future_col = self.factory.get_future_return_column(chromosome)

        # Process train data
        train_processed = pipeline.run(split.train)

        # Fit K-Means on train
        kmeans = KMeansRegimeClassifier(
            n_clusters=chromosome.n_clusters,
            feature_columns=feature_cols
        )
        train_with_clusters = kmeans.fit_predict(train_processed)

        # Calculate train metrics
        prob_calc = ProbabilityCalculator(
            future_return_column=future_col,
            target_return=chromosome.target_return,
            regime_column='cluster'
        )

        try:
            train_cond = prob_calc.calculate_conditional_probabilities(train_with_clusters)
            train_sep = prob_calc.calculate_separation_metrics(train_cond)
            delta_p_train = train_sep.delta_p
        except:
            delta_p_train = 0.0

        # Process test data
        test_processed = pipeline.run(split.test)

        # Apply trained K-Means to test (no refitting!)
        test_with_clusters = kmeans.predict(test_processed)

        # Calculate test metrics
        try:
            test_cond = prob_calc.calculate_conditional_probabilities(test_with_clusters)
            test_sep = prob_calc.calculate_separation_metrics(test_cond)
            delta_p_test = test_sep.delta_p
        except:
            delta_p_test = 0.0

        # Calculate stability (regime changes)
        valid_clusters = train_with_clusters['cluster'].dropna()
        num_changes = (valid_clusters != valid_clusters.shift()).sum()
        stability = num_changes / len(valid_clusters) if len(valid_clusters) > 0 else 1.0

        return FoldResult(
            fold=split.fold,
            train_size=len(split.train),
            test_size=len(split.test),
            delta_p_train=delta_p_train,
            delta_p_test=delta_p_test,
            stability_score=stability
        )
```

### 11.4 Fitness Evaluator (optimization/fitness.py)

```python
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from .chromosome import Chromosome
from .calculator_factory import CalculatorFactory
from .walk_forward_validator import WalkForwardValidator, WalkForwardResult

@dataclass
class FitnessResult:
    """Complete fitness evaluation result."""
    fitness: float
    delta_p: float
    delta_p_test: float
    stability_score: float
    overfitting_ratio: float
    walk_forward: Optional[WalkForwardResult]
    error: Optional[str] = None

class FitnessEvaluator:
    """
    Evaluates chromosome fitness using walk-forward validation.
    Prevents overfitting by measuring out-of-sample performance.
    """

    def __init__(
        self,
        df_base: pd.DataFrame,
        stability_penalty: float = 0.1,
        overfitting_penalty: float = 0.2,
        use_walk_forward: bool = True,
        n_folds: int = 5,
        logger: Optional[ILogger] = None
    ):
        self.df_base = df_base
        self.stability_penalty = stability_penalty
        self.overfitting_penalty = overfitting_penalty
        self.use_walk_forward = use_walk_forward
        self.n_folds = n_folds
        self.logger = logger or NullLogger()

        self.factory = CalculatorFactory()
        self.validator = WalkForwardValidator(
            n_folds=n_folds,
            logger=logger
        )

    def evaluate(self, chromosome: Chromosome) -> FitnessResult:
        """
        Evaluate a chromosome.

        The fitness function balances:
        - Regime separation (delta_p)
        - Out-of-sample performance (walk-forward)
        - Stability (penalize frequent regime changes)
        - Overfitting (penalize train >> test performance)
        """
        try:
            if self.use_walk_forward:
                return self._evaluate_with_walk_forward(chromosome)
            else:
                return self._evaluate_simple(chromosome)

        except Exception as e:
            self.logger.error(
                "Fitness evaluation failed",
                exception=e,
                chromosome=chromosome.to_dict()
            )
            return FitnessResult(
                fitness=-1.0,
                delta_p=0.0,
                delta_p_test=0.0,
                stability_score=1.0,
                overfitting_ratio=float('inf'),
                walk_forward=None,
                error=str(e)
            )

    def _evaluate_with_walk_forward(self, chromosome: Chromosome) -> FitnessResult:
        """Evaluate using walk-forward validation."""
        wf_result = self.validator.validate(self.df_base, chromosome)

        # Primary metric: out-of-sample delta_p
        delta_p_test = wf_result.mean_delta_p_test

        # Penalties
        stability_penalty = self.stability_penalty * wf_result.mean_stability

        # Overfitting penalty (if train >> test)
        overfit_penalty = 0.0
        if wf_result.overfitting_ratio > 1.5:
            overfit_penalty = self.overfitting_penalty * (wf_result.overfitting_ratio - 1.0)

        # Consistency bonus (low std across folds)
        consistency_bonus = max(0, 0.1 - wf_result.std_delta_p_test)

        # Final fitness
        fitness = delta_p_test - stability_penalty - overfit_penalty + consistency_bonus

        self.logger.debug(
            "Fitness calculated",
            fitness=fitness,
            delta_p_test=delta_p_test,
            stability_penalty=stability_penalty,
            overfit_penalty=overfit_penalty,
            consistency_bonus=consistency_bonus
        )

        return FitnessResult(
            fitness=fitness,
            delta_p=delta_p_test,  # Use test as primary
            delta_p_test=delta_p_test,
            stability_score=wf_result.mean_stability,
            overfitting_ratio=wf_result.overfitting_ratio,
            walk_forward=wf_result
        )

    def _evaluate_simple(self, chromosome: Chromosome) -> FitnessResult:
        """Simple evaluation without walk-forward (faster but may overfit)."""
        from ..analysis.kmeans_regimes import KMeansRegimeClassifier
        from ..analysis.probability_calculator import ProbabilityCalculator

        pipeline = self.factory.create_pipeline(chromosome)
        feature_cols = self.factory.get_feature_columns(chromosome)
        future_col = self.factory.get_future_return_column(chromosome)

        df_processed = pipeline.run(self.df_base)

        kmeans = KMeansRegimeClassifier(
            n_clusters=chromosome.n_clusters,
            feature_columns=feature_cols
        )
        df_with_clusters = kmeans.fit_predict(df_processed)

        prob_calc = ProbabilityCalculator(
            future_return_column=future_col,
            target_return=chromosome.target_return,
            regime_column='cluster'
        )

        cond_probs = prob_calc.calculate_conditional_probabilities(df_with_clusters)
        separation = prob_calc.calculate_separation_metrics(cond_probs)

        # Stability
        valid_clusters = df_with_clusters['cluster'].dropna()
        num_changes = (valid_clusters != valid_clusters.shift()).sum()
        stability = num_changes / len(valid_clusters)

        fitness = separation.delta_p - self.stability_penalty * stability

        return FitnessResult(
            fitness=fitness,
            delta_p=separation.delta_p,
            delta_p_test=separation.delta_p,  # Same as train in simple mode
            stability_score=stability,
            overfitting_ratio=1.0,
            walk_forward=None
        )
```

### 11.5 Genetic Algorithm (optimization/genetic_algorithm.py)

```python
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import random

from ..interfaces.logger import ILogger
from ..infrastructure.file_logger import NullLogger
from ..config.search_space import GASearchSpace, GAConfig
from .chromosome import Chromosome
from .fitness import FitnessEvaluator, FitnessResult

@dataclass
class GAResult:
    """Result of genetic algorithm optimization."""
    best_chromosome: Chromosome
    best_fitness: float
    best_metrics: FitnessResult
    history: List[Tuple[int, float, float]]  # (generation, best, mean)
    all_evaluations: int

class GeneticAlgorithm:
    """
    Genetic algorithm for parameter optimization.
    Uses walk-forward validation to prevent overfitting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: GAConfig,
        logger: Optional[ILogger] = None,
        progress_callback: Optional[Callable[[int, float], None]] = None
    ):
        self.df = df
        self.config = config
        self.logger = logger or NullLogger()
        self.progress_callback = progress_callback

        self.evaluator = FitnessEvaluator(
            df_base=df,
            stability_penalty=config.stability_penalty,
            use_walk_forward=True,
            n_folds=config.n_folds,
            logger=logger
        )

    def run(self, verbose: bool = True) -> GAResult:
        """Execute the genetic algorithm."""
        search_space = self.config.search_space

        # Initialize population
        population = [
            Chromosome.random(search_space)
            for _ in range(self.config.population_size)
        ]

        best_chromosome = None
        best_fitness = float('-inf')
        best_metrics = None
        history = []
        total_evaluations = 0

        for gen in range(self.config.generations):
            # Evaluate fitness
            scored: List[Tuple[Chromosome, FitnessResult]] = []
            for chrom in population:
                result = self.evaluator.evaluate(chrom)
                scored.append((chrom, result))
                total_evaluations += 1

            # Sort by fitness (descending)
            scored.sort(key=lambda x: x[1].fitness, reverse=True)

            # Track best
            gen_best = scored[0]
            if gen_best[1].fitness > best_fitness:
                best_chromosome = gen_best[0]
                best_fitness = gen_best[1].fitness
                best_metrics = gen_best[1]

            # Calculate stats
            gen_fitnesses = [s[1].fitness for s in scored]
            gen_mean = np.mean(gen_fitnesses)
            history.append((gen, best_fitness, gen_mean))

            if verbose and gen % 10 == 0:
                self.logger.info(
                    f"Generation {gen}",
                    best_fitness=best_fitness,
                    gen_mean=gen_mean,
                    delta_p_test=best_metrics.delta_p_test if best_metrics else 0
                )

            if self.progress_callback:
                self.progress_callback(gen, best_fitness)

            # Check for early stopping
            if self._should_stop_early(history):
                self.logger.info("Early stopping triggered")
                break

            # Selection and reproduction
            population = self._create_next_generation(scored, search_space)

        self.logger.info(
            "GA completed",
            best_fitness=best_fitness,
            total_evaluations=total_evaluations,
            generations=len(history)
        )

        return GAResult(
            best_chromosome=best_chromosome,
            best_fitness=best_fitness,
            best_metrics=best_metrics,
            history=history,
            all_evaluations=total_evaluations
        )

    def _create_next_generation(
        self,
        scored: List[Tuple[Chromosome, FitnessResult]],
        search_space: GASearchSpace
    ) -> List[Chromosome]:
        """Create next generation via selection, crossover, mutation."""
        new_population = []

        # Elitism: keep best individuals
        for i in range(self.config.elite_size):
            new_population.append(scored[i][0])

        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select(scored)
            parent2 = self._tournament_select(scored)

            # Crossover
            if random.random() < self.config.crossover_probability:
                child = Chromosome.crossover(parent1, parent2)
            else:
                child = parent1

            # Mutation
            if random.random() < self.config.mutation_probability:
                child = child.mutate(search_space)

            new_population.append(child)

        return new_population

    def _tournament_select(
        self,
        scored: List[Tuple[Chromosome, FitnessResult]],
        k: int = 3
    ) -> Chromosome:
        """Tournament selection."""
        tournament = random.sample(scored, min(k, len(scored)))
        winner = max(tournament, key=lambda x: x[1].fitness)
        return winner[0]

    def _should_stop_early(
        self,
        history: List[Tuple[int, float, float]],
        patience: int = 20
    ) -> bool:
        """Check if should stop early due to no improvement."""
        if len(history) < patience:
            return False

        recent = [h[1] for h in history[-patience:]]
        improvement = max(recent) - min(recent)

        return improvement < 0.001  # No significant improvement
```

---

## 12. Visualization

### 12.1 Histogram Plotter (visualization/histogram_plotter.py)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Optional, List

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

            color = colors.get(regime, 'steelblue')
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
```

---

## 13. Testing Strategy

### 13.1 Unit Tests - Calculators (tests/unit/test_calculators.py)

```python
import pytest
import pandas as pd
import numpy as np

from src.calculators.log_price_calculator import LogPriceCalculator
from src.calculators.log_return_calculator import LogReturnCalculator
from src.calculators.future_return_calculator import FutureReturnCalculator
from src.calculators.volatility_calculator import VolatilityCalculator
from src.calculators.slope_calculator import SlopeCalculator

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))

    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n),
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })

class TestLogPriceCalculator:
    def test_returns_correct_columns(self, sample_df):
        calc = LogPriceCalculator()
        result = calc.calculate(sample_df)
        assert 'log_close' in result.columns

    def test_log_values_correct(self, sample_df):
        calc = LogPriceCalculator()
        result = calc.calculate(sample_df)
        expected = np.log(sample_df['close'])
        pd.testing.assert_series_equal(
            result['log_close'],
            expected,
            check_names=False
        )

    def test_validates_missing_columns(self):
        calc = LogPriceCalculator()
        df = pd.DataFrame({'other': [1, 2, 3]})
        with pytest.raises(ValueError, match="missing columns"):
            calc.calculate(df)

class TestLogReturnCalculator:
    def test_returns_correct_columns(self, sample_df):
        calc = LogReturnCalculator(window=5)
        result = calc.calculate(sample_df)
        assert 'log_return' in result.columns
        assert 'log_return_rolling_5' in result.columns

    def test_log_return_values_correct(self, sample_df):
        calc = LogReturnCalculator(window=5)
        result = calc.calculate(sample_df)

        # Check first valid log return
        expected = np.log(sample_df['close'].iloc[1] / sample_df['close'].iloc[0])
        assert abs(result['log_return'].iloc[1] - expected) < 1e-10

    def test_rolling_sum_correct(self, sample_df):
        calc = LogReturnCalculator(window=3)
        result = calc.calculate(sample_df)

        # Rolling sum of log returns = total log return
        idx = 5
        expected = result['log_return'].iloc[idx-2:idx+1].sum()
        assert abs(result['log_return_rolling_3'].iloc[idx] - expected) < 1e-10

class TestFutureReturnCalculator:
    def test_future_return_correct(self, sample_df):
        horizon = 3
        calc = FutureReturnCalculator(horizon=horizon)
        result = calc.calculate(sample_df)

        col = f'log_return_future_{horizon}'
        assert col in result.columns

        # Check calculation
        idx = 10
        expected = np.log(sample_df['close'].iloc[idx + horizon] / sample_df['close'].iloc[idx])
        assert abs(result[col].iloc[idx] - expected) < 1e-10

    def test_last_rows_are_nan(self, sample_df):
        horizon = 3
        calc = FutureReturnCalculator(horizon=horizon)
        result = calc.calculate(sample_df)

        col = f'log_return_future_{horizon}'
        assert result[col].iloc[-horizon:].isna().all()

class TestSlopeCalculator:
    def test_returns_correct_columns(self, sample_df):
        # First need log_close
        log_calc = LogPriceCalculator()
        df = log_calc.calculate(sample_df)

        slope_calc = SlopeCalculator(window=10)
        result = slope_calc.calculate(df)

        assert 'slope_10' in result.columns

    def test_positive_trend_positive_slope(self):
        # Create clearly trending data
        prices = np.linspace(100, 150, 50)
        df = pd.DataFrame({
            'close': prices,
            'log_close': np.log(prices)
        })

        calc = SlopeCalculator(window=10)
        result = calc.calculate(df)

        # All slopes should be positive
        valid_slopes = result['slope_10'].dropna()
        assert (valid_slopes > 0).all()
```

### 13.2 Unit Tests - Value Objects (tests/unit/test_value_objects.py)

```python
import pytest
import numpy as np

from src.domain.value_objects import LogReturn, Price, Regime, Probability
from src.domain.value_objects import TrendDirection, VolatilityLevel

class TestLogReturn:
    def test_creation_valid(self):
        lr = LogReturn(0.05)
        assert lr.value == 0.05

    def test_creation_invalid_range(self):
        with pytest.raises(ValueError):
            LogReturn(3.0)  # Too large

    def test_to_percent(self):
        lr = LogReturn(0.05)
        pct = lr.to_percent()
        assert abs(pct - (np.exp(0.05) - 1)) < 1e-10

    def test_from_percent(self):
        lr = LogReturn.from_percent(0.05)  # 5%
        assert abs(lr.value - np.log(1.05)) < 1e-10

    def test_from_prices(self):
        lr = LogReturn.from_prices(110, 100)
        assert abs(lr.value - np.log(1.1)) < 1e-10

    def test_addition(self):
        lr1 = LogReturn(0.03)
        lr2 = LogReturn(0.02)
        result = lr1 + lr2
        assert abs(result.value - 0.05) < 1e-10

class TestPrice:
    def test_creation_valid(self):
        p = Price(100.0)
        assert p.value == 100.0

    def test_creation_invalid(self):
        with pytest.raises(ValueError):
            Price(-10.0)

    def test_apply_return(self):
        p = Price(100.0)
        lr = LogReturn(np.log(1.1))  # 10% return
        new_p = p.apply_return(lr)
        assert abs(new_p.value - 110.0) < 1e-10

class TestRegime:
    def test_from_indicators_bull(self):
        regime = Regime.from_indicators(
            slope=0.02,
            volatility=0.015,
            slope_threshold=0.01,
            volatility_threshold=0.02
        )
        assert regime.trend == TrendDirection.BULL
        assert regime.volatility == VolatilityLevel.LOW

    def test_from_indicators_bear_high_vol(self):
        regime = Regime.from_indicators(
            slope=-0.02,
            volatility=0.03,
            slope_threshold=0.01,
            volatility_threshold=0.02
        )
        assert regime.trend == TrendDirection.BEAR
        assert regime.volatility == VolatilityLevel.HIGH

    def test_all_regimes(self):
        all_r = Regime.all_regimes()
        assert len(all_r) == 6  # 3 trends x 2 volatilities

class TestProbability:
    def test_creation_valid(self):
        p = Probability(0.5)
        assert p.value == 0.5

    def test_creation_invalid(self):
        with pytest.raises(ValueError):
            Probability(1.5)

    def test_from_frequency(self):
        p = Probability.from_frequency(25, 100)
        assert p.value == 0.25

    def test_to_odds(self):
        p = Probability(0.25)
        odds = p.to_odds()
        assert abs(odds - (0.25 / 0.75)) < 1e-10
```

### 13.3 Unit Tests - Validators (tests/unit/test_validators.py)

```python
import pytest
import pandas as pd
import numpy as np

from src.infrastructure.validators.ohlcv_validator import OHLCVValidator
from src.infrastructure.validators.series_length_validator import SeriesLengthValidator
from src.interfaces.validator import CompositeValidator

@pytest.fixture
def valid_df():
    n = 100
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n),
        'open': np.random.uniform(99, 101, n),
        'high': np.random.uniform(101, 103, n),
        'low': np.random.uniform(97, 99, n),
        'close': np.random.uniform(99, 101, n),
        'volume': np.random.randint(1000000, 5000000, n)
    })

class TestOHLCVValidator:
    def test_valid_data(self, valid_df):
        validator = OHLCVValidator()
        result = validator.validate(valid_df)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_columns(self):
        df = pd.DataFrame({'close': [100, 101, 102]})
        validator = OHLCVValidator()
        result = validator.validate(df)
        assert not result.is_valid
        assert any('Missing' in e for e in result.errors)

    def test_negative_prices(self, valid_df):
        valid_df.loc[5, 'close'] = -10
        validator = OHLCVValidator()
        result = validator.validate(valid_df)
        assert not result.is_valid

    def test_high_less_than_low(self, valid_df):
        valid_df.loc[5, 'high'] = 95
        valid_df.loc[5, 'low'] = 100
        validator = OHLCVValidator()
        result = validator.validate(valid_df)
        assert not result.is_valid

class TestSeriesLengthValidator:
    def test_sufficient_length(self, valid_df):
        validator = SeriesLengthValidator(min_length=50, max_window=20)
        result = validator.validate(valid_df)
        assert result.is_valid

    def test_insufficient_length(self):
        df = pd.DataFrame({'close': [100, 101, 102]})
        validator = SeriesLengthValidator(min_length=50, max_window=20)
        result = validator.validate(df)
        assert not result.is_valid

class TestCompositeValidator:
    def test_all_pass(self, valid_df):
        validator = CompositeValidator([
            OHLCVValidator(),
            SeriesLengthValidator(min_length=50, max_window=20)
        ])
        result = validator.validate(valid_df)
        assert result.is_valid

    def test_one_fails(self):
        df = pd.DataFrame({'close': [100, 101, 102]})
        validator = CompositeValidator([
            OHLCVValidator(),
            SeriesLengthValidator(min_length=50, max_window=20)
        ])
        result = validator.validate(df)
        assert not result.is_valid
```

### 13.4 Integration Tests (tests/integration/test_pipeline.py)

```python
import pytest
import pandas as pd
import numpy as np

from src.calculators.pipeline import CalculatorPipeline
from src.calculators.log_price_calculator import LogPriceCalculator
from src.calculators.log_return_calculator import LogReturnCalculator
from src.calculators.future_return_calculator import FutureReturnCalculator
from src.calculators.volatility_calculator import VolatilityCalculator
from src.calculators.slope_calculator import SlopeCalculator
from src.analysis.kmeans_regimes import KMeansRegimeClassifier
from src.analysis.probability_calculator import ProbabilityCalculator

@pytest.fixture
def market_data():
    """Generate realistic market data."""
    np.random.seed(42)
    n = 500

    # Generate price with regime changes
    returns = np.zeros(n)
    regime = 0
    for i in range(n):
        if np.random.random() < 0.02:  # Regime change
            regime = np.random.choice([0, 1, 2])

        if regime == 0:  # Bull
            returns[i] = np.random.normal(0.001, 0.01)
        elif regime == 1:  # Bear
            returns[i] = np.random.normal(-0.001, 0.015)
        else:  # Flat
            returns[i] = np.random.normal(0, 0.008)

    prices = 100 * np.cumprod(1 + returns)

    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n),
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })

class TestFullPipeline:
    def test_pipeline_runs_successfully(self, market_data):
        pipeline = CalculatorPipeline([
            LogPriceCalculator(),
            LogReturnCalculator(window=20),
            FutureReturnCalculator(horizon=3),
            VolatilityCalculator(window=20),
            SlopeCalculator(window=20)
        ])

        result = pipeline.run(market_data)

        # Check all columns exist
        assert 'log_close' in result.columns
        assert 'log_return' in result.columns
        assert 'log_return_rolling_20' in result.columns
        assert 'log_return_future_3' in result.columns
        assert 'volatility_20' in result.columns
        assert 'slope_20' in result.columns

    def test_kmeans_produces_clusters(self, market_data):
        # Run pipeline
        pipeline = CalculatorPipeline([
            LogPriceCalculator(),
            LogReturnCalculator(window=20),
            VolatilityCalculator(window=20),
            SlopeCalculator(window=20)
        ])
        df = pipeline.run(market_data)

        # Cluster
        kmeans = KMeansRegimeClassifier(n_clusters=3)
        result = kmeans.fit_predict(df)

        assert 'cluster' in result.columns
        assert result['cluster'].dropna().nunique() == 3

    def test_probability_calculation(self, market_data):
        # Full pipeline
        pipeline = CalculatorPipeline([
            LogPriceCalculator(),
            LogReturnCalculator(window=20),
            FutureReturnCalculator(horizon=3),
            VolatilityCalculator(window=20),
            SlopeCalculator(window=20)
        ])
        df = pipeline.run(market_data)

        kmeans = KMeansRegimeClassifier(n_clusters=3)
        df = kmeans.fit_predict(df)

        prob_calc = ProbabilityCalculator(
            future_return_column='log_return_future_3',
            target_return=0.02,
            regime_column='cluster'
        )

        raw_prob = prob_calc.calculate_raw_probability(df)
        cond_probs = prob_calc.calculate_conditional_probabilities(df)

        assert 0 <= raw_prob.value <= 1
        assert len(cond_probs) == 3
```

### 13.5 Test Fixtures (tests/fixtures/sample_data.py)

```python
import pandas as pd
import numpy as np

def create_trending_data(n: int = 100, direction: str = 'up') -> pd.DataFrame:
    """Create data with clear trend."""
    if direction == 'up':
        trend = np.linspace(0, 0.5, n)
    else:
        trend = np.linspace(0, -0.3, n)

    noise = np.random.normal(0, 0.01, n)
    log_prices = 4.6 + trend + np.cumsum(noise)  # Start at ~100
    prices = np.exp(log_prices)

    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n),
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })

def create_volatile_data(n: int = 100, vol_level: float = 0.03) -> pd.DataFrame:
    """Create high volatility data."""
    returns = np.random.normal(0, vol_level, n)
    prices = 100 * np.cumprod(1 + returns)

    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n),
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })

def create_regime_switching_data(n: int = 500) -> pd.DataFrame:
    """Create data with regime switches."""
    returns = []
    regimes = []

    current_regime = 0
    for i in range(n):
        if np.random.random() < 0.03:
            current_regime = np.random.choice([0, 1, 2])

        regimes.append(current_regime)

        if current_regime == 0:  # Bull low vol
            returns.append(np.random.normal(0.001, 0.008))
        elif current_regime == 1:  # Bear high vol
            returns.append(np.random.normal(-0.001, 0.02))
        else:  # Flat
            returns.append(np.random.normal(0, 0.01))

    prices = 100 * np.cumprod(1 + np.array(returns))

    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n),
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })
    df['true_regime'] = regimes

    return df
```

---

## 14. Jupyter Notebook Example

### 14.1 Main Notebook Structure (notebooks/quantnote_analysis.ipynb)

```python
# Cell 1: Imports and Setup
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config.settings import AnalysisConfig, Config
from config.search_space import GAConfig, GASearchSpace

from src.infrastructure.yahoo_data_source import YahooDataSource
from src.infrastructure.parquet_repository import ParquetRepository
from src.infrastructure.file_logger import FileLogger
from src.infrastructure.validators import create_default_validator

from src.calculators.pipeline import CalculatorPipeline
from src.calculators.log_price_calculator import LogPriceCalculator
from src.calculators.log_return_calculator import LogReturnCalculator
from src.calculators.future_return_calculator import FutureReturnCalculator
from src.calculators.volatility_calculator import VolatilityCalculator
from src.calculators.slope_calculator import SlopeCalculator

from src.analysis.regime_classifier import ManualRegimeClassifier
from src.analysis.kmeans_regimes import KMeansRegimeClassifier
from src.analysis.probability_calculator import ProbabilityCalculator

from src.visualization.histogram_plotter import HistogramPlotter, PriceRegimePlotter
from src.optimization.genetic_algorithm import GeneticAlgorithm

%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')

# Cell 2: Configuration
config = Config()
config.analysis.future_return_periods = 3
config.analysis.window_slope = 20
config.analysis.window_volatility = 20
config.analysis.n_clusters = 3
config.analysis.target_return = 0.05

print("Configuration loaded:")
print(config.analysis)

# Cell 3: Load Data
logger = FileLogger("quantnote")
data_source = YahooDataSource(logger=logger)
repository = ParquetRepository(logger=logger)

ticker = "BOVA11.SA"

# Try to load from cache first
df = repository.load(ticker)
if df is None:
    print(f"Fetching {ticker} from Yahoo Finance...")
    df = data_source.fetch_ohlcv(ticker)
    repository.save(df, ticker)
else:
    print(f"Loaded {ticker} from cache")

print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Cell 4: Validate Data
validator = create_default_validator(
    min_length=config.analysis.min_data_points,
    max_window=max(config.analysis.window_slope, config.analysis.window_volatility)
)

validation = validator.validate(df)
print(f"Validation: {'PASSED' if validation.is_valid else 'FAILED'}")
if validation.errors:
    print(f"Errors: {validation.errors}")
if validation.warnings:
    print(f"Warnings: {validation.warnings}")

# Cell 5: Run Calculator Pipeline
pipeline = CalculatorPipeline([
    LogPriceCalculator(),
    LogReturnCalculator(window=config.analysis.window_rolling_return),
    FutureReturnCalculator(horizon=config.analysis.future_return_periods),
    VolatilityCalculator(window=config.analysis.window_volatility),
    SlopeCalculator(window=config.analysis.window_slope)
], logger=logger)

df_analysis = pipeline.run(df)
print(f"Columns after pipeline: {list(df_analysis.columns)}")

# Cell 6: Manual Regime Classification
slope_col = f'slope_{config.analysis.window_slope}'
vol_col = f'volatility_{config.analysis.window_volatility}'

manual_classifier = ManualRegimeClassifier(
    slope_column=slope_col,
    volatility_column=vol_col
)

df_manual = manual_classifier.classify(df_analysis)
print(f"Regimes: {df_manual['regime'].value_counts()}")
print(f"Thresholds: {manual_classifier.get_thresholds()}")

# Cell 7: K-Means Clustering
kmeans = KMeansRegimeClassifier(
    n_clusters=config.analysis.n_clusters,
    logger=logger
)

df_kmeans = kmeans.fit_predict(df_analysis)

future_col = f'log_return_future_{config.analysis.future_return_periods}'
stats = kmeans.compute_statistics(df_kmeans, future_col)
interpretations = kmeans.interpret_clusters(df_kmeans, slope_col)

print("\nCluster Statistics:")
for stat in stats:
    print(f"  Cluster {stat.cluster_id} ({interpretations.get(stat.cluster_id, '?')}): "
          f"n={stat.count}, mean_return={stat.future_return_mean:.4f}")

# Cell 8: Calculate Probabilities
prob_calc = ProbabilityCalculator(
    future_return_column=future_col,
    target_return=config.analysis.target_return,
    regime_column='regime'
)

report = prob_calc.generate_report(df_manual)

print(f"\n=== Probability Report ===")
print(f"Target Return: {report['target_return']:.1%}")
print(f"Raw Probability: {report['raw_probability_pct']:.2f}%")
print(f"\nConditional Probabilities:")
for regime, data in report['conditional_probabilities'].items():
    print(f"  {regime}: {data['probability_pct']:.2f}% (n={data['count']})")
print(f"\nSeparation Metrics:")
print(f"  Delta P: {report['separation_metrics']['delta_p']:.4f}")
print(f"  Information Ratio: {report['separation_metrics']['information_ratio']:.4f}")

# Cell 9: Visualizations
hist_plotter = HistogramPlotter(
    return_column=future_col,
    regime_column='regime'
)

fig1 = hist_plotter.plot(df_manual)
plt.show()

fig2 = hist_plotter.plot_by_regime(df_manual, target_return=config.analysis.target_return)
plt.show()

price_plotter = PriceRegimePlotter(regime_column='regime')
fig3 = price_plotter.plot(df_manual)
plt.show()

# Cell 10: Genetic Algorithm Optimization
ga_config = GAConfig(
    population_size=30,
    generations=50,
    n_folds=3,
    stability_penalty=0.1
)

print("Starting Genetic Algorithm optimization...")
print(f"Population: {ga_config.population_size}, Generations: {ga_config.generations}")

ga = GeneticAlgorithm(df, ga_config, logger=logger)
result = ga.run(verbose=True)

print(f"\n=== Best Parameters Found ===")
best = result.best_chromosome
print(f"  Window Slope: {best.window_slope}")
print(f"  Window Volatility: {best.window_volatility}")
print(f"  Horizon: {best.horizon}")
print(f"  Target Return: {best.target_return:.2%}")
print(f"  N Clusters: {best.n_clusters}")
print(f"  Use Volatility: {best.use_volatility}")
print(f"  Use Rolling Return: {best.use_rolling_return}")
print(f"\nBest Fitness: {result.best_fitness:.4f}")
print(f"Delta P (test): {result.best_metrics.delta_p_test:.4f}")
print(f"Overfitting Ratio: {result.best_metrics.overfitting_ratio:.2f}")

# Cell 11: Apply Best Parameters
best_pipeline = CalculatorPipeline([
    LogPriceCalculator(),
    LogReturnCalculator(window=best.window_rolling_return),
    FutureReturnCalculator(horizon=best.horizon),
    VolatilityCalculator(window=best.window_volatility),
    SlopeCalculator(window=best.window_slope)
])

df_best = best_pipeline.run(df)

best_kmeans = KMeansRegimeClassifier(n_clusters=best.n_clusters)
df_best = best_kmeans.fit_predict(df_best)

best_prob = ProbabilityCalculator(
    future_return_column=f'log_return_future_{best.horizon}',
    target_return=best.target_return,
    regime_column='cluster'
)

best_report = best_prob.generate_report(df_best)
print("\n=== Optimized Probability Report ===")
for regime, data in best_report['conditional_probabilities'].items():
    interp = best_kmeans.interpret_clusters(df_best, f'slope_{best.window_slope}')
    label = interp.get(int(float(regime)), '?')
    print(f"  Cluster {regime} ({label}): {data['probability_pct']:.2f}%")
```

---

## 15. Implementation Phases

### Phase 1: Foundation (Steps 1-3)
- [ ] Create virtual environment and `requirements.txt`
- [ ] Create directory structure
- [ ] Implement `config/settings.py` and `config/search_space.py`
- [ ] Implement domain exceptions and value objects
- [ ] Implement all interfaces

### Phase 2: Infrastructure (Steps 4-6)
- [ ] Implement `FileLogger` and `NullLogger`
- [ ] Implement `YahooDataSource` with rate limiting
- [ ] Implement `ParquetRepository`
- [ ] Implement `JoblibModelStore`
- [ ] Implement all validators

### Phase 3: Calculators (Steps 7-9)
- [ ] Implement `LogPriceCalculator`
- [ ] Implement `LogReturnCalculator`
- [ ] Implement `FutureReturnCalculator`
- [ ] Implement `VolatilityCalculator`
- [ ] Implement `SlopeCalculator`
- [ ] Implement `DependencyResolver`
- [ ] Implement `CalculatorPipeline`

### Phase 4: Analysis (Steps 10-12)
- [ ] Implement `TimeSeriesSplitter`
- [ ] Implement `ManualRegimeClassifier`
- [ ] Implement `KMeansRegimeClassifier` with persistence
- [ ] Implement `ProbabilityCalculator`

### Phase 5: Optimization (Steps 13-15)
- [ ] Implement `Chromosome`
- [ ] Implement `CalculatorFactory`
- [ ] Implement `WalkForwardValidator`
- [ ] Implement `FitnessEvaluator`
- [ ] Implement `GeneticAlgorithm`

### Phase 6: Visualization & Testing (Steps 16-18)
- [ ] Implement `HistogramPlotter`
- [ ] Implement `PriceRegimePlotter`
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Create main notebook

---

## 16. Final Notes

### 16.1 Key Design Decisions

1. **Walk-Forward Validation**: Prevents overfitting by testing on unseen data
2. **Dependency Injection**: All major components receive dependencies via constructor
3. **Topological Sort**: Pipeline automatically resolves calculator order
4. **Model Persistence**: K-Means models can be saved/loaded for production use
5. **Value Objects**: Immutable domain objects with validation

### 16.2 Performance Considerations

- Use `numba` to accelerate slope calculations for large datasets
- Consider `multiprocessing` for parallel GA evaluation
- Cache intermediate calculations that don't change

### 16.3 Extension Points

- New data sources: Implement `IDataSource`
- New indicators: Implement `IColumnCalculator`
- New clustering methods: Create classes analogous to `KMeansRegimeClassifier`
- New visualizations: Implement `IVisualizer`

---

**Document Version**: 2.0
**Last Updated**: 2024
**Author**: QuantNote Team