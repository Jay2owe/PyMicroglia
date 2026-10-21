import base64
import csv
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from pymicroglia.ui.exploration_filters import filter_catalogue, preview_filters
from pymicroglia.ui.exploration_context import load_tissue_map
from test_ui_exploration_context import context_run
from test_cell_filters import rule


@pytest.fixture
def cohort_run(tmp_path):
    tmp_path = tmp_path / "cohort"
    folder = tmp_path / "measure/one"
    folder.mkdir(parents=True)
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "signal_mean", "centroid_x", "centroid_y"])
        for identity, indices in [(19, [0, 1, 5, 7]), (26, range(8)), (42, [2, 3, 4])]:
            for f in indices:
                writer.writerow([identity, f, 10 + f*.5, "nan" if f == 3 else f*identity, identity, 0])
    with (folder / "lifecycle_events.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "event", "event_status", "partner_identity"])
        writer.writerow([19, 7, "absorbed", "candidate", 26])
        writer.writerow([42, 2, "born", "candidate", 26])
    (tmp_path / "manifest.json").write_text(json.dumps({"settings": {"microns_per_pixel": 2}, "movies": [{"stem": "one", "spec": {}}]}))
    return tmp_path


def test_presence_uses_full_frames_and_distinguishes_internal_gaps(cohort_run):
    expressions = [rule("longest_gap_frames", 3, "gte"), rule("missing_pct", 50, "gte"), rule("internal_missing_pct", 0, "eq")]
    results = [preview_filters(cohort_run, "one", e, mode="keep") for e in expressions]
    assert [r["identities"] for r in results] == [[19], [19, 42], [26, 42]]
    result = preview_filters(cohort_run, "one", rule("longest_gap_hours", 1.5, "eq"), mode="keep")
    assert result["identities"] == [19] and result["frame_count"] == 8 and result["exact"]


def test_imported_decisions_have_explicit_unknown_evidence_and_replay(cohort_run, tmp_path):
    from pymicroglia.cell_display import import_decisions, resolve
    decisions = import_decisions("identity,passed\n19,true\n26,false\n", "one")
    result = preview_filters(cohort_run, "one", rule("review_passed"), mode="keep", unknown="exclude", decisions=decisions)
    assert result["identities"] == [19] and result["unknown_count"] == 1
    assert resolve(result, [19, 26, 42])[0] == {26: (170, 170, 170), 42: (170, 170, 170)}
    script = tmp_path / "replay.py"
    script.write_text(result["equivalent_script"], encoding="utf-8")
    replay = subprocess.run([sys.executable, str(script)], text=True, capture_output=True, check=True)
    assert json.loads(replay.stdout)["rows"] == result["rows"]


def test_saved_rhythm_verdicts_are_metric_specific_and_untested_stays_unknown(cohort_run, monkeypatch):
    from pymicroglia.ui import exploration_results
    monkeypatch.setattr(exploration_results, "saved_results", lambda *args: {"rhythms": {"rows": [
        {"identity": 19, "metric": "signal_mean", "status": "rhythmic", "p_value": .01},
        {"identity": 26, "metric": "signal_mean", "status": "not rhythmic", "p_value": .2}]}})
    expression = rule("rhythmic", False, metric="signal_mean")
    result = preview_filters(cohort_run, "one", expression)
    assert result["identities"] == [19, 42] and result["unknown_count"] == 1
    assert result["rows"][-1]["evidence"][0]["match"] is None
    assert preview_filters(cohort_run, "one", rule("rhythm_tested", False, metric="signal_mean"), mode="keep")["identities"] == [42]
    assert preview_filters(cohort_run, "one", rule("rhythm_p", .05, "lt", metric="signal_mean"), mode="keep", unknown="exclude")["identities"] == [19]


def test_events_include_both_participants_and_missing_sources_are_unknown(cohort_run):
    assert preview_filters(cohort_run, "one", rule("has_merge"), mode="keep")["identities"] == [19, 26]
    assert preview_filters(cohort_run, "one", rule("has_split"), mode="keep")["identities"] == [26, 42]
    assert preview_filters(cohort_run, "one", rule("has_birth"), mode="keep")["identities"] == [42]
    (cohort_run / "measure/one/lifecycle_events.csv").unlink()
    assert preview_filters(cohort_run, "one", rule("has_merge"))["unknown_count"] == 3


def test_native_contact_separation_is_not_a_biological_division(cohort_run):
    path = cohort_run / "measure/one/history_merge_split_events.csv"
    path.write_text("identity,event_type,candidate_identities\n19,contact_separate,26|42\n")
    assert preview_filters(cohort_run, "one", rule("has_contact_separation"), mode="keep")["identities"] == [19, 26, 42]
    assert preview_filters(cohort_run, "one", rule("has_merge"), mode="keep")["identities"] == [19, 26]  # absorption only


def test_distances_use_exact_same_frame_full_population_and_calibration(cohort_run):
    # Native centroids stay in pixels: 19 and 26 differ by 7 pixels = 14 µm.
    expression = {"op": "AND", "children": [rule("identity", 26), rule("min_distance_um", 15, "lt")]}
    result = preview_filters(cohort_run, "one", expression, mode="keep")
    assert result["identities"] == [26]
    assert result["rows"][1]["evidence"][1]["actual"] == 14
    assert preview_filters(cohort_run, "one", rule("min_distance_px", 7, "eq"), mode="keep")["identities"] == [19, 26]
    assert preview_filters(cohort_run, "one", rule("metric_missing_pct", 50, "gt", metric="signal_mean"), mode="keep")["identities"] == [42]


def test_mask_centroids_and_missing_calibration(context_run):
    assert preview_filters(context_run, "one", rule("min_distance_px", 7, "lt"))["identities"] == []
    result = preview_filters(context_run, "one", rule("min_distance_um", 7, "lt"))
    assert result["unknown_count"] == 2 and result["identities"] == [19, 26]


def test_preview_replay_and_native_filtered_map_do_not_write_inputs(cohort_run, context_run, tmp_path):
    before = {p: p.read_bytes() for root in [cohort_run, context_run] for p in root.rglob("*") if p.is_file()}
    result = preview_filters(cohort_run, "one", rule("longest_gap_frames", 2, "gt"))
    script = tmp_path / "replay.py"
    script.write_text(result["equivalent_script"], encoding="utf-8")
    replay = subprocess.run([sys.executable, str(script)], text=True, capture_output=True, check=True)
    assert json.loads(replay.stdout)["identities"] == result["identities"]
    payload = load_tissue_map(context_run, "one", "unique_cells", identities="19")
    image = np.asarray(Image.open(BytesIO(base64.b64decode(payload["image"].split(",", 1)[1]))))
    assert image[2, 2, 3] == 255 and image[6, 7, 3] == 0
    assert "Filtered cohort: 1 cells" in payload["scope"]
    assert all(p.read_bytes() == data for p, data in before.items())


def test_catalogue_validation_and_api(cohort_run):
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    client = TestClient(create_app())
    catalogue = filter_catalogue(cohort_run, "one")
    assert len(catalogue["fields"]) >= 30 and all(f["help"] and f["comparators"] for f in catalogue["fields"])
    assert client.get("/api/exploration/filter-options", params={"run": str(cohort_run), "stem": "one"}).status_code == 200
    request = {"run": str(cohort_run), "stem": "one", "expression": rule("longest_gap_frames", 3, "gte")}
    assert client.post("/api/exploration/filter-preview", json=request).json()["identities"] == [26, 42]
    request["expression"] = rule("rhythmic", metric="imaginary")
    assert client.post("/api/exploration/filter-preview", json=request).status_code == 422
    assert not client.app.state.jobs.list()
