"""Base model configuration and lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pioneer.core.exceptions import ModelError


class ModelConfig(BaseModel):
    """Declarative model configuration."""

    name: str
    version: str = "0.1.0"
    model_type: str
    path: Path | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelArtifact(BaseModel):
    """Serialized model artifact metadata."""

    config: ModelConfig
    checkpoint_path: Path
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, float] = Field(default_factory=dict)


class PioneerModel(ABC):
    """Abstract base for trainable and deployable models."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self, path: Path | None = None) -> None:
        """Load model weights from disk."""

    @abstractmethod
    def save(self, path: Path) -> ModelArtifact:
        """Persist model weights and return artifact metadata."""

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Run forward pass / inference."""

    def validate_loaded(self) -> None:
        raise ModelError("Model not loaded; call load() first")
