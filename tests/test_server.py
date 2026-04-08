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


def test_reset_with_seed_returns_meta() -> None:
    client = TestClient(app)

    response = client.post("/reset", json={"task_id": "easy_basic_audit", "seed": 123})

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["seed"] == 123
    assert body["meta"]["task_id"] == "easy_basic_audit"


def test_tasks_includes_action_schema_and_score_range() -> None:
    client = TestClient(app)

    response = client.get("/tasks")

    assert response.status_code == 200
    body = response.json()
    assert body["benchmark"] == "openenv_compliance_audit"
    assert body["score_range"] == [0.0, 1.0]
    assert "action_schema" in body
    assert "baseline" in body["endpoints"]


def test_baseline_endpoint_returns_task_scores() -> None:
    client = TestClient(app)

    response = client.get("/baseline")

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "rule_based"
    assert body["score_range"] == [0.0, 1.0]
    assert len(body["tasks"]) > 0
    for task_item in body["tasks"]:
        assert "task_id" in task_item
        assert "score" in task_item
        assert 0.0 <= float(task_item["score"]) <= 1.0
