from fastapi.testclient import TestClient

from pymicroglia.ui.jobs import JobManager
from pymicroglia.ui.server import create_app


def test_live_action_describe_and_validate():
    client = TestClient(create_app())
    action = client.get("/api/actions/rhythm_screen")
    assert action.status_code == 200
    assert any(row["name"] == "run" for row in action.json()["params"])
    assert client.post("/api/validate", json={"action": "rhythm_screen", "params": {}}).json()["ok"]
    assert client.get("/api/actions/not_an_action").status_code == 404
    assert client.post("/api/jobs", json={"action": "not_an_action"}).status_code == 422


def test_recorded_job_result_without_leaking_raw_result():
    manager = JobManager(lambda action, **kwargs: {"recorded": True, "result": object(),
                          "record": {"run_id": "saved-1", "outputs": []}})
    client = TestClient(create_app(jobs=manager))
    started = client.post("/api/jobs", json={"action": "rhythm_screen", "params": {"run": "example"}})
    assert started.status_code == 202
    identifier = started.json()["id"]
    manager.pool.shutdown(wait=True)
    done = client.get(f"/api/jobs/{identifier}").json()
    assert done["state"] == "completed"
    assert done["record"]["run_id"] == "saved-1"
    assert "result" not in done


def test_live_pipeline_schema_exposes_accepted_refinement_controls():
    client = TestClient(create_app())
    response = client.get("/api/pipelines/auto_microglia")
    assert response.status_code == 200, response.text
    params = {row["name"]: row for row in response.json()["params"]}
    assert params["tracked_outline_refinement"]["default"] is True
    assert params["tracked_refinement_method"]["default"] == "single_frame"
    assert "tracked_refinement_options" in params
    assert client.get("/api/pipelines/not_a_pipeline").status_code == 404
    invalid = client.post("/api/pipelines/validate", json={"action": "rhythm_discovery", "params": {}})
    assert invalid.status_code == 200 and invalid.json()["ok"] is False
