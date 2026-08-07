"""OpenTelemetry and Prometheus instrumentation hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pioneer.core.config import get_settings
from pioneer.core.logging import get_logger

logger = get_logger(__name__)

_metrics: dict[str, float] = {}


@dataclass
class TelemetryConfig:
    service_name: str = "pioneer-ml"
    enable_tracing: bool = False
    enable_metrics: bool = True
    otlp_endpoint: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)


def init_telemetry(config: TelemetryConfig | None = None) -> None:
    """Initialize observability backends when optional deps are available."""
    settings = get_settings()
    config = config or TelemetryConfig(
        enable_tracing=settings.enable_telemetry,
        otlp_endpoint=settings.otel_endpoint,
    )

    if config.enable_tracing and config.otlp_endpoint:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider

            resource = Resource.create({"service.name": config.service_name, **config.attributes})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            logger.info("telemetry_initialized", tracing=True)
        except ImportError:
            logger.warning("telemetry_skipped", reason="opentelemetry not installed")

    if config.enable_metrics:
        logger.info("telemetry_initialized", metrics=True)


def record_metric(name: str, value: float, *, tags: dict[str, Any] | None = None) -> None:
    """Record a metric value (in-memory fallback when Prometheus is unavailable)."""
    key = name if tags is None else f"{name}:{sorted(tags.items())}"
    _metrics[key] = value
    logger.debug("metric_recorded", name=name, value=value, tags=tags or {})


def get_metrics_snapshot() -> dict[str, float]:
    """Return current in-memory metrics (for testing and debugging)."""
    return dict(_metrics)
