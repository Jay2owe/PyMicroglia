from pathlib import Path

from fastapi.testclient import TestClient

from pymicroglia.ui.results import ArtifactIndex
from pymicroglia.ui.server import create_app


def test_selected_folder_only_exposes_shallow_results(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "figure.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    (root / "deep").mkdir()
    (root / "deep" / "nested").mkdir()
    (root / "deep" / "nested" / "secret.svg").write_text("secret", encoding="utf-8")
    (tmp_path / "outside.svg").write_text("outside", encoding="utf-8")
    index = ArtifactIndex()
    selected = index.select_folder(root)
    assert [row["name"] for row in selected["artifacts"]] == ["figure.svg"]
    assert index.get(selected["artifacts"][0]["token"])[0] == root / "figure.svg"


def test_artifact_url_is_token_limited(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "preview.svg").write_text("<svg/>", encoding="utf-8")
    client = TestClient(create_app())
    selected = client.post("/api/results/select", json={"folder": str(root)})
    assert selected.status_code == 200
    token = selected.json()["artifacts"][0]["token"]
    assert client.get(f"/api/artifacts/{token}").status_code == 200
    assert client.get("/api/artifacts/outside.svg").status_code == 404
