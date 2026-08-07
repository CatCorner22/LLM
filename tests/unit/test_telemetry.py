"""Unit tests for observability."""

import pytest

from pioneer.observability.telemetry import (
    TelemetryConfig,
    get_metrics_snapshot,
    init_telemetry,
    record_metric,
)


@pytest.mark.unit
def test_record_metric() -> None:
    record_metric("latency_ms", 12.5, tags={"endpoint": "predict"})
    snapshot = get_metrics_snapshot()
    assert any(value == 12.5 for value in snapshot.values())


@pytest.mark.unit
def test_init_telemetry_without_otel() -> None:
    init_telemetry(TelemetryConfig(enable_tracing=False, enable_metrics=True))
    record_metric("requests_total", 1.0)
