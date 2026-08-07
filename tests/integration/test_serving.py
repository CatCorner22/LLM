"""Integration tests for serving API."""

import pytest


@pytest.mark.integration
def test_health_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pioneer.serving.app import create_app

    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "version" in payload


@pytest.mark.integration
def test_predict_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pioneer.serving.app import create_app

    client = TestClient(create_app())
    response = client.post("/v1/predict", json={"input": "hello"})
    assert response.status_code == 200
    assert response.json()["output"] == "echo: hello"
