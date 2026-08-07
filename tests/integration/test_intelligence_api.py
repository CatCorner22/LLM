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


@pytest.mark.integration
def test_intelligence_knowledge_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pioneer.serving.app import create_app

    client = TestClient(create_app())
    response = client.get("/v1/intelligence/knowledge?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert "store" in payload
    assert "recent_chemicals" in payload
    assert "recent_health" in payload
    assert "recent_governance" in payload


@pytest.mark.integration
def test_intelligence_relationship_map_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pioneer.serving.app import create_app

    client = TestClient(create_app())
    response = client.get("/v1/intelligence/relationship-map")
    assert response.status_code == 200
    payload = response.json()
    assert "nodes" in payload
    assert "edges" in payload
    assert "focus_areas" in payload
    focus = payload["focus_areas"]
    assert "employee_conduct" in focus or "segregation_of_duties" in focus
    assert "acquisition_assessments" in payload
    assert payload["acquisition_assessments"]
    assert "knowledge_acquisition" in focus
