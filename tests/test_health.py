"""Smoke test for the /health endpoint."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_returns_ok():
    """GET /health should return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.2.0"
