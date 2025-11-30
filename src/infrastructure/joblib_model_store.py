"""Joblib-based model store for ML model persistence."""
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
