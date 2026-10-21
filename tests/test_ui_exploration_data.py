import csv
import json
import pytest
from pymicroglia.ui.exploration_data import load_exploration, load_cell, load_frame_values, load_trace_grid


@pytest.fixture
def saved_run(tmp_path):
    (tmp_path / ".auto-organotypic").mkdir()
    movies = []
    for stem in ("one", "two"):
        folder = tmp_path / "measure" / stem
        folder.mkdir(parents=True)
        with (folder / "observations-hash.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["identity", "frame_index", "source_imagej_frame", "hours", "integrated_density", "area_px"])
            for identity in (19, 26):
                for frame in range(12):
                    if identity == 19 and frame == 3:
                        continue
                    w.writerow([identity, frame, frame + 6, (frame + 5) / 2, "nan" if frame == 5 else frame * identity, 20])
        movies.append({"stem": stem, "spec": {"source_frame_offset": 5}, "tables": {"cell_frame": {"path": "observations-hash.csv", "folder": stem}}})
    (tmp_path / ".auto-organotypic/manifest.json").write_text(json.dumps({"movies": movies}))
    return tmp_path


def test_exact_observations_and_gaps(saved_run):
    before = sorted(p.relative_to(saved_run) for p in saved_run.rglob("*"))
    data = load_exploration(saved_run)
    assert data["stem"] == "one"
    assert len(data["recordings"]) == 2
    from pymicroglia.visualisation.panels import colour
    assert data["trace_style"] == {"rhythmic": colour("circadian_red"), "other": colour("nan_text")}
    cell = load_cell(saved_run, "one", 19)["cell"]
    assert not cell["observed"][3]
    assert cell["values"]["integrated_density"][3] is None
    assert cell["values"]["integrated_density"][5] is None
    assert data["frames"][3] == {"frame_index": 3, "source_frame_index": 8, "hours": 4}
    assert load_exploration(saved_run, "two")["stem"] == "two"
    assert [c["identity"] for c in load_frame_values(saved_run, "one", 3)["cells"]] == [26]
    assert sorted(p.relative_to(saved_run) for p in saved_run.rglob("*")) == before


def test_deferred_overview_preserves_complete_measurements(saved_run):
    from pymicroglia.ui.exploration_data import load_overview
    complete = load_exploration(saved_run, "one")
    compact = load_exploration(saved_run, "one", overview=False)
    assert compact["preview"].pop("pending")
    assert load_overview(saved_run, "one")["cells"] == complete["cells"]
    compact["cells"] = complete["cells"]
    assert compact == complete


def test_sparse_cells_convert_only_observed_samples(monkeypatch):
    from pymicroglia.ui import exploration_data as module
    observations = {2: {"a": "7", "b": "nan"}, 9999: {"a": "0", "b": "-3"}}
    original_number = module.number
    calls = []
    monkeypatch.setattr(module, "number", lambda value: calls.append(value) or original_number(value))
    cell = module._cell(19, observations, range(10000), [{"key": "a"}, {"key": "b"}], {})
    assert len(calls) == 4
    assert sum(cell["observed"]) == 2
    assert cell["values"]["a"][2] == 7
    assert cell["values"]["a"][9999] == 0
    assert cell["values"]["b"][2] is None
    assert cell["values"]["b"][9999] == -3
    assert all(value is None for value in cell["values"]["a"][3:9999])


def test_budget_and_alignment(saved_run, monkeypatch):
    import pymicroglia.ui.exploration_data as module
    monkeypatch.setattr(module, "MAX_TABLE_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        load_exploration(saved_run)


def test_declared_paths_cannot_escape_run(saved_run):
    p = saved_run / ".auto-organotypic/manifest.json"
    data = json.loads(p.read_text())
    data["movies"][0]["tables"]["cell_frame"]["path"] = "../../../../outside.csv"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside"):
        load_exploration(saved_run)


def test_trace_grid_uses_exact_observations_and_keeps_missingness(saved_run):
    before = {str(p): p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    result = load_trace_grid(saved_run, "one", "integrated_density")
    assert result["exact"] and result["next_offset"] is None
    assert [c["identity"] for c in result["cells"]] == [19, 26]
    cell = result["cells"][0]
    assert 3 not in cell["indices"]  # no invented observation in the gap
    assert cell["values"][cell["indices"].index(5)] is None
    assert cell["values"][cell["indices"].index(4)] == 4 * 19
    assert before == {str(p): p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="not available"):
        load_trace_grid(saved_run, "one", "nonexistent")
    with pytest.raises(ValueError, match="nonnegative"):
        load_trace_grid(saved_run, "one", "area_px", offset=-1)


def test_trace_grid_pages_all_cells_and_keeps_frames_beyond_overview(tmp_path):
    folder = tmp_path / "measure/many"
    folder.mkdir(parents=True)
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "integrated_density"])
        for identity in range(1, 261):
            for frame in (0, 1, 600):
                writer.writerow([identity, frame, frame / 2, frame + identity])
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [{"stem": "many"}]}))
    cells, offset = [], 0
    while offset is not None:
        result = load_trace_grid(tmp_path, "many", "integrated_density", offset)
        assert len(result["cells"]) <= 128
        cells.extend(result["cells"])
        offset = result["next_offset"]
    assert result["total_cells"] == 260
    assert [c["identity"] for c in cells] == list(range(1, 261))
    assert cells[-1]["indices"] == [0, 1, 600]
    assert cells[-1]["values"] == [260, 261, 860]


def test_multiple_trace_measurements_share_exact_frames_without_writing(saved_run):
    before = {str(p): p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    result = load_trace_grid(saved_run, "one", "integrated_density",
                             metrics=["area_px", "integrated_density", "area_px"])
    assert result["metrics"] == ["integrated_density", "area_px"]
    first = result["cells"][0]
    assert 3 not in first["indices"]
    assert first["values"] == first["values_by_metric"]["integrated_density"]
    assert first["values_by_metric"]["area_px"] == [20] * len(first["indices"])
    assert first["values"][first["indices"].index(5)] is None
    assert before == {str(p): p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="not available"):
        load_trace_grid(saved_run, "one", "integrated_density", metrics=["invalid"])


def test_multiple_metric_endpoint_keeps_single_metric_compatibility(saved_run):
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    with TestClient(create_app()) as client:
        reply = client.get("/api/exploration/traces", params={
            "run": str(saved_run), "stem": "one", "metric": "integrated_density",
            "metrics": "area_px,integrated_density", "offset": 1, "limit": 1})
        assert reply.status_code == 200
        body = reply.json()
        assert body["cells"][0]["identity"] == 26
        assert body["metrics"] == ["integrated_density", "area_px"]
        assert body["cells"][0]["values_by_metric"]["area_px"] == [20] * 12
        bad = client.get("/api/exploration/traces", params={
            "run": str(saved_run), "stem": "one", "metric": "area_px", "metrics": "unknown"})
        assert bad.status_code == 422
