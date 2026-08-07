"""Structured exception hierarchy for predictable error handling."""

from __future__ import annotations

from typing import Any


class PioneerError(Exception):
    """Base exception for all Pioneer ML errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(PioneerError):
    """Invalid or missing configuration."""


class ValidationError(PioneerError):
    """Input or schema validation failure."""


class ModelError(PioneerError):
    """Model loading, inference, or serialization failure."""


class DataError(PioneerError):
    """Dataset or data pipeline failure."""


class TrainingError(PioneerError):
    """Training loop or checkpoint failure."""


class InferenceError(PioneerError):
    """Serving or batch inference failure."""


class AgentError(PioneerError):
    """Agent orchestration or tool execution failure."""


class RAGError(PioneerError):
    """Retrieval-augmented generation pipeline failure."""


class ExternalServiceError(PioneerError):
    """Upstream API or service failure."""


class RiskError(PioneerError):
    """Risk assessment or scenario analysis failure."""


class IngestionError(PioneerError):
    """External information ingestion failure."""
