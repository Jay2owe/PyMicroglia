"""Trace review: variants from saved analyses, human-only decisions and a bound handoff."""
import base64
import csv
import hashlib
import json
from io import BytesIO

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

from pymicroglia.ui import trace_review as tr
from pymicroglia.ui.exploration_rhythms import run_rhythm_analysis
from pymicroglia.ui.server import create_app

QUICK = {"fit_method": "lomb", "significance_method": "lomb", "fft_component_test": False,
         "detrend": "none", "metrics": ["integrated_density"]}


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    folder = root / "measure/slice"
    folder.mkdir(parents=True)
    frames = np.arange(97)
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "integrated_density", "area"])
        for identity, phase in ((3, 4), (8, 11)):
            signal = 100 + 20 * np.cos(2 * np.pi * (frames / 2 - phase) / 24)
            area = 50 + 5 * np.cos(2 * np.pi * frames / 40)
            for frame in frames:
                writer.writerow([identity, frame, frame / 2, signal[frame], area[frame]])
    raw = np.full((97, 40, 40), 500, np.uint16)
    labels = np.zeros_like(raw)
    labels[:, 10:16, 10:16] = 3
    labels[40, 10:16, 10:16] = 0  # Cell 3 is missing in one frame.
    spec = {}
    for role, pixels in (("raw", raw), ("labels", labels)):
        path = root / f"{role}.tif"
        tifffile.imwrite(path, pixels, photometric="minisblack")
        spec[role] = str(path)
    (root / "manifest.json").write_text(json.dumps({"movies": [{"stem": "slice", "spec": spec, "display": {"contrast": [0, 1000]}}]}))
    return root


def test_variants_grid_decisions_and_bound_handoff(run):
    whole = run_rhythm_analysis(run, "slice", {**QUICK, "label": "whole"}, workers=1)
    window = run_rhythm_analysis(run, "slice", {**QUICK, "window_start_hours": 12, "window_end_hours": 36}, workers=1)
    assert window["resolved_options"]["window_end_hours"] == 36
    home, review = tr.review_for_run(run)
    listing = tr.list_variants(home)
    assert len(listing["variants"]) == 2
    by_id = {v["recordings"][listing["recordings"][0]["key"]]["analysis_id"]: v for v in listing["variants"]}
    assert "whole" in by_id[whole["analysis_id"]]["description"]
    assert "window 12-36 h" in by_id[window["analysis_id"]]["description"]
    assert all(r["fresh"] for v in listing["variants"] for r in v["recordings"].values())

    keys = [by_id[whole["analysis_id"]]["key"], by_id[window["analysis_id"]]["key"]]
    # Different time windows never share an axis: view them one at a time.
    with pytest.raises(ValueError, match="different time windows"):
        tr.load_grid(home, keys)
    grid = tr.load_grid(home, keys[:1])
    assert [c["identity"] for c in grid["cells"]] == [3, 8]
    cell = grid["cells"][0]
    sid = grid["series"][0]["id"]
    assert grid["series"][0]["metric"] == "integrated_density"
    assert cell["stats"][sid]["period"] == pytest.approx(24, abs=1.5)
    assert len(cell["traces"][sid]["hours"]) == 97
    windowed = tr.load_grid(home, keys[1:])
    hours = windowed["cells"][0]["traces"][windowed["series"][0]["id"]]["hours"]
    assert min(hours) >= 12 and max(hours) <= 36

    # Agents only suggest; the human mark stays unreviewed until accepted.
    tr.suggest(home, [{"cell": cell["key"], "status": "exclude", "reason": "spike at 20 h"}], agent="codex-test")
    decisions = tr.load_decisions(home)
    assert cell["key"] not in decisions["cells"] and decisions["suggestions"][cell["key"]]["agent"] == "codex-test"
    saved = tr.save_decisions(home, [{"cell": cell["key"], "status": "include", "star": True}], revision=decisions["revision"],
                              preferred_variant=keys[0])
    with pytest.raises(ValueError, match="changed elsewhere"):
        tr.save_decisions(home, [{"cell": cell["key"], "status": "exclude"}], revision=decisions["revision"])
    tr.save_decisions(home, [{"cell": grid["cells"][1]["key"], "status": "exclude", "note": "bad mask"}], revision=saved["revision"])
    assert (home / "decisions_log.jsonl").read_text().count("\n") == 3

    record = tr.handoff(home, note="use starred")
    assert record["counts"] == {"unreviewed": 0, "include": 1, "exclude": 1, "representative": 1}
    assert record["variant"]["key"] == keys[0] and record["variant"]["metric"] == "integrated_density"
    rows = list(csv.DictReader((home / "included_cells.csv").open()))
    assert [(r["identity"], r["status"], r["representative"]) for r in rows] == [("3", "include", "True"), ("8", "exclude", "False")]
    assert tr.approved_cells(home) == [(str(run.resolve()), "slice", 3)]

    # Re-measuring after the handoff must not be silently reused downstream.
    table = run / "measure/slice/cell_frame.csv"
    table.write_text(table.read_text() + "\n")
    assert tr.read_handoff(home)["stale_recordings"] == [listing["recordings"][0]["key"]]
    with pytest.raises(ValueError, match="changed after the handoff"):
        tr.approved_cells(home)
    assert all(v["stale"] == 1 for v in tr.list_variants(home)["variants"])


def test_same_window_fits_and_metrics_overlay(run):
    lomb = run_rhythm_analysis(run, "slice", {**QUICK, "metrics": ["integrated_density", "area"]}, workers=1)
    linear = run_rhythm_analysis(run, "slice", {**QUICK, "detrend": "robust_linear"}, workers=1)
    home, _ = tr.review_for_run(run)
    variants = tr.list_variants(home)["variants"]
    assert len(variants) == 2
    grid = tr.load_grid(home, [v["key"] for v in variants], ["integrated_density", "area"])
    pairs = sorted((s["variant"] == variants[0]["key"], s["metric"]) for s in grid["series"])
    # The robust-linear variant only analysed one metric, so it adds one series.
    assert len(grid["series"]) == 3 and ("area" in {m for _, m in pairs})
    cell = grid["cells"][0]
    assert set(cell["traces"]) == {s["id"] for s in grid["series"]}
    assert lomb["analysis_id"] != linear["analysis_id"]


def test_window_and_label_are_validated(run):
    from pymicroglia.ui.exploration_rhythms import validate_request
    with pytest.raises(ValueError, match="end after"):
        validate_request(run, "slice", {**QUICK, "window_start_hours": 10, "window_end_hours": 5})
    with pytest.raises(ValueError, match="label"):
        validate_request(run, "slice", {**QUICK, "label": ""})
    with pytest.raises(ValueError, match="No saved observations"):
        run_rhythm_analysis(run, "slice", {**QUICK, "window_start_hours": 500}, workers=1)


def test_legacy_figure_folder_imports_once_with_provenance(run, tmp_path):
    legacy = tmp_path / "all_cell_trace_grid_slice"
    legacy.mkdir()
    cell_frame = run / "measure/slice/cell_frame.csv"
    with (legacy / "statistics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "metric", "observations", "rhythm_status", "period_hours", "p_value", "q_value"])
        writer.writerow([3, "corrected_mean", 97, "rhythmic", 23.9, 0.001, 0.002])
    with (legacy / "figure_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "metric", "frame_index", "hours", "raw_value", "processed_value", "value"])
        for frame in range(5):
            writer.writerow([3, "corrected_mean", frame, frame / 2, 10 + frame, frame, frame / 4])
    (legacy / "der_settings.csv").write_text('settings_json\n"{""detrend"": ""robust_linear"", ""identities"": [3]}"\n')
    sha = hashlib.sha256(cell_frame.read_bytes()).hexdigest()
    (legacy / "sources.csv").write_text(
        "original_path,copied_path,file_name,sha256\n"
        f"{cell_frame},src_cell_frame.csv,cell_frame.csv,{sha}\n"
        f"{run / 'manifest.json'},src_manifest.json,manifest.json,x\n")
    home = tmp_path / "review"
    first = tr.import_legacy(home, legacy)
    assert tr.import_legacy(home, legacy)["analysis_id"] == first["analysis_id"]
    listing = tr.list_variants(home)
    assert len(listing["variants"]) == 1
    variant = listing["variants"][0]
    assert variant["descriptor"]["detrend"] == "robust_linear" and "all_cell_trace_grid_slice" in variant["description"]
    analysis = next(iter(variant["recordings"].values()))
    assert analysis["origin"] == "imported" and analysis["fresh"] and analysis["imported_from"] == str(legacy.resolve())
    grid = tr.load_grid(home, [variant["key"]])
    sid = grid["series"][0]["id"]
    assert grid["series"][0]["metric"] == "corrected_mean"
    assert grid["cells"][0]["stats"][sid]["period"] == pytest.approx(23.9)
    assert grid["cells"][0]["traces"][sid]["value"] == [0, .25, .5, .75, 1]


def test_crop_follows_cell_with_fixed_size_and_outline(run):
    shown = tr.cell_crop(run, "slice", 3, 0, size=32)
    assert shown["available"] and shown["present"]
    pixels = np.array(Image.open(BytesIO(base64.b64decode(shown["image"].split(",")[1]))))
    assert pixels.shape == (32, 32, 3)
    assert (pixels == (64, 224, 208)).all(-1).sum() == 20  # 6x6 square's perimeter
    missing = tr.cell_crop(run, "slice", 3, 40, size=32)
    assert missing["available"] and not missing["present"]
    assert not tr.cell_crop(run, "slice", 99, 0)["available"]


def test_server_routes_page_and_decisions(run):
    run_rhythm_analysis(run, "slice", QUICK, workers=1)
    with TestClient(create_app()) as client:
        assert "trace-review.js" in client.get("/review").text
        created = client.post("/api/trace-review/create", json={"run": str(run)}).json()
        home = created["home"]
        listing = client.get("/api/trace-review", params={"home": home}).json()
        key = listing["variants"][0]["key"]
        grid = client.get("/api/trace-review/grid", params={"home": home, "variants": key}).json()
        cell = grid["cells"][0]["key"]
        saved = client.put("/api/trace-review/decisions", json={"home": home, "revision": 0, "changes": [{"cell": cell, "status": "exclude"}]})
        assert saved.status_code == 200 and saved.json()["cells"][cell]["status"] == "exclude"
        stale = client.put("/api/trace-review/decisions", json={"home": home, "revision": 0, "changes": [{"cell": cell, "status": "include"}]})
        assert stale.status_code == 422
        assert client.post("/api/trace-review/handoff", json={"home": home, "variant": key}).json()["counts"]["exclude"] == 1
        crop = client.get("/api/trace-review/crop", params={"run": str(run), "stem": "slice", "identity": 3, "frame": 1}).json()
        assert crop["present"]


def test_registered_actions_drive_the_review_without_human_marks(run, tmp_path):
    import inspect
    from pymicroglia import catalogue, run_action
    from pymicroglia.registry import NEEDS_A_CLAIM, pending, resolve
    names = ["trace_review_create", "trace_review_variant", "trace_review_import", "trace_review_suggest", "trace_review_handoff"]
    for name in names:
        entry = next(item for item in catalogue.actions() if item["name"] == name)
        function = resolve(entry["method"])
        assert name not in pending() and set(entry["params"]) == set(inspect.signature(function).parameters)
    assert "trace_review_variant" in NEEDS_A_CLAIM
    created = run_action("trace_review_create", source=str(run), output_roots=[tmp_path])
    home = created["home"]
    made = run_action("trace_review_variant", source=home, rhythm_options={**QUICK, "label": "agent window", "window_start_hours": 6},
                      fit_workers=1, output_roots=[tmp_path], claim="compare a 6 h onward window")
    assert [a["stem"] for a in made["analyses"]] == ["slice"]
    again = run_action("trace_review_variant", source=home, rhythm_options={**QUICK, "label": "agent window", "window_start_hours": 6},
                       fit_workers=1, output_roots=[tmp_path], claim="repeat")
    assert again["analyses"][0]["reused"] and again["analyses"][0]["analysis_id"] == made["analyses"][0]["analysis_id"]
    cell = tr.load_grid(home, [tr.list_variants(home)["variants"][0]["key"]])["cells"][0]["key"]
    run_action("trace_review_suggest", source=home, suggestions=[{"cell": cell, "status": "include", "reason": "clean"}],
               agent="test-agent", output_roots=[tmp_path])
    assert tr.load_decisions(home)["cells"] == {}
    with pytest.raises(ValueError, match="No handoff"):
        run_action("trace_review_handoff", source=home, output_roots=[tmp_path])
