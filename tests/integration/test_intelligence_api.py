"""Integration tests for intelligence API."""

import pytest


@pytest.mark.integration
def test_intelligence_risk_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pioneer.serving.app import create_app

    client = TestClient(create_app())
    response = client.post("/v1/intelligence/risk", json={"use_sample": True})
    assert response.status_code == 200
    payload = response.json()
    assert "overall_score" in payload
    assert payload["overall_score"] > 0


@pytest.mark.integration
def test_intelligence_benchmark_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pioneer.serving.app import create_app

    client = TestClient(create_app())
    response = client.get("/v1/intelligence/benchmark")
    assert response.status_code == 200
    assert "disruptive_advantages" in response.json()
