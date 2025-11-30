"""Interfaces (Abstract contracts) for QuantNote."""
from .logger import ILogger, LogLevel
from .data_source import IDataSource
from .repository import IRepository, FileMetadata
from .validator import IValidator, ValidationResult, CompositeValidator
from .column_calculator import IColumnCalculator
from .model_store import IModelStore, ModelMetadata
from .visualizer import IVisualizer

__all__ = [
    'ILogger',
    'LogLevel',
    'IDataSource',
    'IRepository',
    'FileMetadata',
    'IValidator',
    'ValidationResult',
    'CompositeValidator',
    'IColumnCalculator',
    'IModelStore',
    'ModelMetadata',
    'IVisualizer'
]
