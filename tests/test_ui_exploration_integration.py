from fastapi.testclient import TestClient
from pymicroglia.ui.server import create_app
from test_ui_exploration_data import saved_run


def test_exploration_endpoints_are_read_only(saved_run):
    client = TestClient(create_app())
    assert len(client.get("/api/exploration/catalogue").json()["views"]) == 9
    data = client.get("/api/exploration", params={"run": str(saved_run)}).json()
    assert data["identities"] == [19, 26]
    assert client.get("/api/exploration/cell", params={"run": str(saved_run), "stem": "one", "identity": 19}).status_code == 200
    assert client.get("/api/exploration/values", params={"run": str(saved_run), "stem": "one", "frame": 3}).status_code == 200
    grid = client.get("/api/exploration/traces", params={"run": str(saved_run), "stem": "one", "metric": "integrated_density"})
    assert grid.status_code == 200 and grid.json()["total_cells"] == 2
    assert grid.json()["cells"][0]["identity"] == 19
    assert client.get("/api/exploration/traces", params={"run": str(saved_run), "stem": "one", "metric": "missing"}).status_code == 422
    assert client.get("/api/jobs").json() == []
    assert client.get("/assets/exploration.html").status_code == 200
    assert client.get("/api/exploration", params={"run": str(saved_run), "stem": "nope"}).status_code == 422
