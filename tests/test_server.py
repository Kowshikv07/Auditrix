from fastapi.testclient import TestClient

from openenv_compliance_audit.server import app


def test_state_before_reset_returns_400() -> None:
    client = TestClient(app)

    response = client.get("/state")

    assert response.status_code == 400
    body = response.json()
    assert "Call reset() before state()." in body["detail"]


def test_reset_post_without_body_is_accepted() -> None:
    client = TestClient(app)

    response = client.post("/reset")

    assert response.status_code == 200
    body = response.json()
    assert "observation" in body
    assert "task_id" in body["observation"]


def test_reset_post_with_empty_body_is_accepted() -> None:
    client = TestClient(app)

    response = client.post("/reset", json={})

    assert response.status_code == 200
    body = response.json()
    assert "observation" in body
    assert "task_id" in body["observation"]
