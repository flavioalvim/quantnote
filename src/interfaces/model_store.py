"""Model Store interface."""
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Tuple
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
    def load(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata."""
        pass

    @abstractmethod
    def list_models(self, model_type: Optional[str] = None) -> List[ModelMetadata]:
        """List available models."""
        pass

    @abstractmethod
    def delete(self, model_id: str) -> bool:
        """Delete a model."""
        pass
