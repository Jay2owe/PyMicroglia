"""Automatic dashboard plots use bounded saved values without a rhythm refit."""

import csv
import json

from fastapi.testclient import TestClient

from pymicroglia.ui.quicklook import MAX_POINTS, load_quicklook
from pymicroglia.ui.server import create_app


def _run(tmp_path):
    run = tmp_path / "measured"
    table = run / "measure" / "sample_01"
    table.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"movies": [{"stem": "sample_01"}]}), encoding="utf-8")
    with (table / "cell_frame.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stem", "identity", "frame_index", "hours",
                                                    "integrated_density", "area_px"])
        writer.writeheader()
        for cell in (19, 26):
            for frame in range(1000):
                writer.writerow({"stem": "sample_01", "identity": cell, "frame_index": frame,
                                 "hours": frame / 2, "integrated_density": cell + frame,
                                 "area_px": 30 if cell == 19 else 45})
    with (table / "rhythms.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stem", "identity", "metric", "rhythm_status",
                                                    "best_period_hours"])
        writer.writeheader()
        writer.writerow({"stem": "sample_01", "identity": 19, "metric": "integrated_density",
                         "rhythm_status": "rhythmic", "best_period_hours": 24.2})
    return run


def test_quicklook_reads_saved_measurements_and_periods_without_refitting(tmp_path):
    run = _run(tmp_path)
    shown = load_quicklook(run)
    assert shown["total_cells"] == shown["displayed_cells"] == 2
    assert "integrated_density" in shown["pages"]["rhythms"]
    assert len(shown["traces"][0]["hours"]) <= MAX_POINTS
    assert shown["traces"][0]["values"]["integrated_density"][0] == 19
    assert shown["rhythms"][0]["period_hours"] == 24.2
    assert not list(run.rglob("*.png"))


def test_quicklook_http_rejects_an_unrecorded_folder(tmp_path):
    app = create_app()
    client = TestClient(app)
    assert client.get("/api/quicklook", params={"run": str(tmp_path)}).status_code == 422
    run = _run(tmp_path)
    response = client.get("/api/quicklook", params={"run": str(run)})
    assert response.status_code == 200
    assert response.json()["total_cells"] == 2
    assert client.get("/assets/quicklook.js").status_code == 200
    app.state.jobs.close()
