"""Dropped stacks or folders become one full automated run, with labels."""

import json

import numpy as np
import pytest
import tifffile

from auto_organotypic import local_inputs, recording_labels
from pymicroglia.pipelines._auto_microglia_support import _movie_labels
from pymicroglia.ui import intake


def _stack(path, channels=1, interval=1800.):
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (4, channels, 12, 14) if channels > 1 else (4, 12, 14)
    metadata = {"axes": "TCYX" if channels > 1 else "TYX"}
    if interval:
        metadata.update(finterval=interval, tunit="s")
    tifffile.imwrite(path, np.zeros(shape, np.uint16), imagej=True, metadata=metadata)
    return path


def test_a_dropped_folder_of_stacks_is_ready_to_run(tmp_path):
    _stack(tmp_path / "A1.tif"); _stack(tmp_path / "B1.tif")
    (tmp_path / "notes.txt").write_text("not a stack")
    found = intake.inspect([str(tmp_path)])
    assert found["mode"] == "local"
    assert [row["key"] for row in found["recordings"]] == ["A1.tif", "B1.tif"]
    assert all(row["valid"] and row["selected"] for row in found["recordings"])
    # The signal is filled in for a one-channel stack; the outline channel never is.
    assert found["recordings"][0]["roles"] == {"outline": None, "signal": 1,
                                               "brightfield": None, "structural": None}


def test_a_recording_too_short_for_cell_traces_says_so_but_still_runs(tmp_path):
    found = intake.inspect([str(_stack(tmp_path / "A1.tif"))])["recordings"][0]
    assert found["valid"] and found["selected"]
    assert "will stop at the cells stage" in found["issues"][-1]


@pytest.mark.parametrize("frames, warned", [(49, True), (193, True), (300, False)])
def test_the_length_rule_is_the_cells_stage_rule(frames, warned):
    # 24 h and 4 days both stopped at cells on the workstation (EmptyWindow and
    # the 24 h baseline); six days fits 72 h settling plus the 48 h baseline.
    row = intake._too_short({"valid": True, "frames": frames, "interval_s": 1800., "issues": []})
    assert bool(row["issues"]) is warned
    if warned:
        assert "need about 121 h" in row["issues"][0]


def test_an_unreadable_recording_gets_no_length_note():
    row = {"valid": False, "frames": None, "interval_s": None, "issues": ["bad"]}
    assert intake._too_short(row)["issues"] == ["bad"]


def test_a_dropped_file_selects_only_itself_and_keeps_its_labels(tmp_path):
    first, second = _stack(tmp_path / "A1.tif"), _stack(tmp_path / "B1.tif")
    recording_labels.save(tmp_path, {first: {"condition": "HCQ", "name": "Slice A"}})
    found = intake.inspect([str(first)])
    assert [row["key"] for row in found["recordings"]] == ["A1.tif"]
    assert found["recordings"][0]["labels"]["condition"] == "HCQ"
    assert found["recordings"][0]["name"] == "Slice A"
    assert second.exists()


def test_a_stack_with_several_channels_waits_for_its_signal_to_be_chosen(tmp_path):
    path = _stack(tmp_path / "two.tif", channels=2)
    row = intake._local_row(tmp_path, path, {})
    assert not row["valid"] and row["roles"]["signal"] is None and row["roles"]["outline"] is None
    assert row["issues"] == ["Choose the signal channel"]
    assert intake._only_channel([{"index": 1, "name": "BF"}, {"index": 2, "name": "Biolum"}]) is None
    assert intake._only_channel([{"index": 1, "name": "Biolum"}]) == {"signal": 1}
    chosen = intake.run_params(str(tmp_path), "local", [{**row, "roles": {"signal": 2}}])
    written = (tmp_path / intake.HOME).rglob("*.json")
    assert chosen["folder"] and any('"signal": 2' in one.read_text(encoding="utf-8") for one in written)


def test_mixed_or_scattered_drops_are_refused_plainly(tmp_path):
    _stack(tmp_path / "a" / "x.tif"); _stack(tmp_path / "b" / "y.tif")
    with pytest.raises(ValueError, match="one folder"):
        intake.inspect([str(tmp_path / "a" / "x.tif"), str(tmp_path / "b" / "y.tif")])
    with pytest.raises(ValueError, match="No TIFF stacks"):
        intake.inspect([str(tmp_path)])
    with pytest.raises(ValueError, match="does not exist"):
        intake.inspect([str(tmp_path / "missing.tif")])


def test_a_per_image_instrument_export_is_refused_with_what_to_do(tmp_path):
    files = []
    for minute in ("00h00m", "00h30m"):
        plane = tmp_path / f"VID95_B4_2_00d{minute}.tif"
        plane.write_bytes(b"")
        files.append({"path": str(plane), "vessel_id": 95, "well": "B4", "layout": "separate",
                      "axes": "YX", "channels": [{"index": 1, "name": "Green"}],
                      "scan_times": ["2026-07-20T15:00:00"]})
    (tmp_path / "pyincucyte-manifest.json").write_text(json.dumps({
        "manifest_version": 2, "generated_by": "pyincucyte 0.3.1", "layout": "separate",
        "axes": "YX", "files": files}), encoding="utf-8")
    with pytest.raises(ValueError, match="2 single images.*time_channel_stack"):
        intake.inspect([str(tmp_path)])


def test_run_params_write_the_input_manifest_away_from_the_recordings(tmp_path):
    _stack(tmp_path / "A1.tif"); _stack(tmp_path / "B1.tif", interval=None)
    found = intake.inspect([str(tmp_path)])
    rows = found["recordings"]
    assert not rows[1]["valid"]
    rows[1]["interval_s"] = 600.
    rows[0]["name"] = "Named"
    params = intake.run_params(found["folder"], found["mode"], rows)
    home = tmp_path / intake.HOME
    assert params == {"folder": str(home)}
    written = json.loads((home / local_inputs.MANIFEST_NAME).read_text())["files"]
    assert [one["name"] for one in written] == ["Named", "B1"]
    assert written[1]["interval_s"] == {"value": 600., "source": "confirmed by user"}
    assert not (tmp_path / local_inputs.MANIFEST_NAME).exists()


def test_nothing_ticked_is_not_a_run(tmp_path):
    _stack(tmp_path / "A1.tif")
    found = intake.inspect([str(tmp_path)])
    for row in found["recordings"]:
        row["selected"] = False
    with pytest.raises(ValueError, match="at least one"):
        intake.run_params(found["folder"], found["mode"], found["recordings"])


def test_labels_saved_from_the_page_reach_the_measurement_columns(tmp_path):
    _stack(tmp_path / "A1.tif")
    intake.save_labels(str(tmp_path), {"A1.tif": {"condition": "KO", "sample_id": "M2",
                                                  "treatment": "LPS", "name": "First"}})
    labels = recording_labels.by_stem(tmp_path / intake.HOME, ["A1_registered_translation"])
    assert _movie_labels(labels["A1_registered_translation"]) == {
        "condition": "KO", "subject": "M2", "notes": "name: First; treatment: LPS"}


def test_finished_runs_are_listed_with_their_measured_recordings(tmp_path):
    run = tmp_path / intake.HOME / "auto_microglia" / "first"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"outputs": {"tracked_measurement": {
        "A1_registered_translation": {"folder": str(run / "tracked_measurement" / "A1" / "m")}}}}))
    [found] = intake.results(tmp_path)
    assert found["label"] == "first"
    assert found["recordings"][0]["stem"] == "A1_registered_translation"


def test_server_routes_inspect_save_and_queue(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from pymicroglia.ui.jobs import JobManager
    from pymicroglia.ui.server import create_app
    from pymicroglia.pipelines import auto_microglia

    calls = []
    monkeypatch.setattr(auto_microglia, "run", lambda **kw: calls.append(kw) or {"folder": "x"})
    monkeypatch.setenv("PYMICROGLIA_AI", "0")
    _stack(tmp_path / "A1.tif")
    manager = JobManager(runner=lambda *a, **k: {})
    client = TestClient(create_app(jobs=manager))
    found = client.post("/api/intake/inspect", json={"paths": [str(tmp_path)]}).json()
    assert client.post("/api/intake/labels", json={"folder": found["folder"], "labels": {
        "A1.tif": {"condition": "WT"}}}).status_code == 200
    job = client.post("/api/intake/run", json={"folder": found["folder"], "mode": found["mode"],
                                               "recordings": found["recordings"]}).json()
    manager.pool.shutdown(wait=True)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["state"] == "completed", done.get("error")
    assert calls[0]["folder"] == str(tmp_path / intake.HOME) and callable(calls[0]["on_progress"])
    assert client.post("/api/intake/inspect", json={"paths": [str(tmp_path / "nope")]}).status_code == 422


def test_a_started_run_no_longer_says_queued(tmp_path, monkeypatch):
    """The chain reports stages as they finish; until the first does, the page says starting."""
    from pymicroglia.ui.jobs import JobManager
    from pymicroglia.pipelines import auto_microglia

    manager = JobManager(runner=lambda *a, **k: {})
    seen = []
    monkeypatch.setattr(auto_microglia, "run", lambda **kw: seen.append(
        [(row["state"], row["progress"]["stage"]) for row in manager.jobs.values()]) or {"folder": "x"})
    monkeypatch.setattr("pymicroglia.ui.intake.run_params", lambda *a: {"folder": str(tmp_path)})
    monkeypatch.setattr("pymicroglia.ui.intake.results", lambda folder: [])
    manager.submit_intake(str(tmp_path), "local", [])
    manager.pool.shutdown(wait=True)
    assert seen == [[("running", "starting")]]


def test_the_page_offers_switches_with_the_pipelines_own_defaults(tmp_path):
    import inspect as signature
    from pymicroglia.pipelines import auto_microglia
    _stack(tmp_path / "A1.tif")
    offered = intake.inspect([str(tmp_path)])["options"]["switches"]
    defaults = signature.signature(auto_microglia.run).parameters
    assert [row["key"] for row in offered] == [key for key, *_ in intake.SWITCHES]
    assert all(row["default"] == defaults[row["key"]].default for row in offered)
    assert all(row["needs"] in (None, "motion") for row in offered)


def test_run_choices_become_pipeline_keywords_and_nothing_else(tmp_path):
    _stack(tmp_path / "A1.tif")
    found = intake.inspect([str(tmp_path)])
    rows = found["recordings"]
    plain = intake.run_params(found["folder"], found["mode"], rows)
    assert set(plain) == {"folder"}, "No choices means the pipeline's defaults, not a copy of them"
    chosen = intake.run_params(found["folder"], found["mode"], rows,
                               {"outline_channel": 2, "tracked_video": False})
    assert chosen["outline"] is True and chosen["channel"] == 2 and chosen["tracked_video"] is False
    unchosen = intake.run_params(found["folder"], found["mode"], rows, {"outline_channel": None})
    assert "outline" not in unchosen and "channel" not in unchosen
    for bad in ({"force": True}, {"motion": "yes"}, {"outline_channel": 0}, {"outline_channel": True}):
        with pytest.raises(ValueError):
            intake.run_params(found["folder"], found["mode"], rows, bad)
