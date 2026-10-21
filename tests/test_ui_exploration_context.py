"""Native-map and observation-history contracts for the linked interface."""
import base64
import csv
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest
import tifffile

from pymicroglia.figure_tables.territory_values import distinct_owner_count
from pymicroglia.ui.exploration_context import load_tissue_map, load_track_timeline


@pytest.fixture
def context_run(tmp_path):
    folder = tmp_path / "measure/one"
    folder.mkdir(parents=True)
    labels = np.zeros((6, 12, 16), dtype=np.uint16)
    labels[[0, 1, 4, 5], 2:4, 2:4] = 19
    labels[[2, 3], 2:4, 2:4] = 26
    labels[:, 6:8, 7:9] = 26
    tifffile.imwrite(tmp_path / "labels.tif", labels, photometric="minisblack")
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "integrated_density", "speed"])
        for identity in (19, 26):
            for frame in range(6):
                if identity == 19 and frame in (2, 3):
                    continue
                writer.writerow([identity, frame, 10 + frame, 100 + frame, identity / 10])
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [{"stem": "one", "spec": {"labels": "labels.tif"}}]}))
    return tmp_path


def test_gantt_preserves_identities_gaps_and_unclassified_endpoints(context_run):
    result = load_track_timeline(context_run, "one")
    cells = {r["identity"]: r for r in result["cells"]}
    assert set(cells) == {19, 26}
    assert cells[19]["segments"] == [[0, 1], [4, 5]]
    assert cells[19]["start_hours"] == 10 and cells[19]["end_hours"] == 15
    assert cells[19]["start_event"] == "first_observed"
    assert not result["saved_events"]
    assert "missing observations" in result["meaning"]
    assert cells[19]["gaps"][0]["reason"] == "no_recorded_cause"
    assert cells[19]["gaps"][0]["start_frame"] == 2
    assert cells[19]["gaps"][0]["end_frame"] == 3
    assert [r["identity"] for r in load_track_timeline(context_run, "one", "identity")["cells"]] == [19, 26]
    with pytest.raises(ValueError, match="order"):
        load_track_timeline(context_run, "one", "guess")


def test_tissue_maps_use_native_masks_and_saved_tests_without_writes(context_run, monkeypatch):
    before = {str(p): p.read_bytes() for p in context_run.rglob("*") if p.is_file()}
    from pymicroglia.ui import exploration_results
    monkeypatch.setattr(exploration_results, "saved_results", lambda *args: {"rhythms": {
        "metrics": ["integrated_density"], "rows": [
            {"identity": 19, "metric": "integrated_density", "status": "rhythmic", "period_hours": 24},
            {"identity": 26, "metric": "integrated_density", "status": "not rhythmic", "period_hours": 20}]}})
    results = {layer: load_tissue_map(context_run, "one", layer) for layer in
               ("first_coverage", "cumulative_occupancy", "unique_cells", "speed", "significant_period", "splitting_events")}
    assert results["first_coverage"]["limits"] == [0, 5]
    assert results["unique_cells"]["limits"] == [1, 2]
    assert results["speed"]["message"] == ""
    assert results["significant_period"]["message"] == ""
    assert results["significant_period"]["limits"] == [2, 48]
    image = np.asarray(Image.open(BytesIO(base64.b64decode(results["significant_period"]["image"].split(",", 1)[1]))))
    assert image[2, 2, 3] == 255  # significant cell's occupied territory
    assert image[6, 7, 3] == 0  # non-significant cell is not a period-map contributor
    assert "No saved" in results["splitting_events"]["message"]
    assert before == {str(p): p.read_bytes() for p in context_run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Unknown"):
        load_tissue_map(context_run, "one", "invented")


def test_distinct_owners_counts_revisits_once(context_run):
    counts = distinct_owner_count(tifffile.imread(context_run / "labels.tif"))
    assert counts[2, 2] == 2
    assert counts[6, 7] == 1
    assert counts[0, 0] == 0
    with pytest.raises(ValueError, match="nonnegative integer"):
        distinct_owner_count(np.array([[[-1]]]))


def test_tissue_clock_includes_unoccupied_trailing_frames(context_run):
    path = context_run / "measure/one/cell_frame.csv"
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(row for row in rows if row[1] != "5")
    labels = tifffile.imread(context_run / "labels.tif")
    labels[-1] = 0
    tifffile.imwrite(context_run / "labels.tif", labels, photometric="minisblack")
    with (context_run / "measure/one/frame_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "hours"])
        writer.writerows((index, 10 + index) for index in range(6))
    assert load_tissue_map(context_run, "one")["limits"] == [0, 5]


def test_reporter_only_run_names_missing_speed(context_run):
    path = context_run / "measure/one/cell_frame.csv"
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(row[:-1] for row in rows)
    result = load_tissue_map(context_run, "one", "speed")
    assert result["message"] == "No saved cell speeds."


def _save_table(folder, name, rows):
    with (folder / (name + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_gantt_keeps_every_reason_and_ignores_repaired_gaps(context_run):
    folder = context_run / "measure/one"
    rows = [
        {"identity": 19, "frame_index": 0, "mechanism": "distant_alias_risk", "still_missing_in_accepted_labels": True},
        {"identity": 19, "frame_index": 2, "mechanism": "same_host_merge_hiding", "still_missing_in_accepted_labels": True},
        {"identity": 19, "frame_index": 2, "mechanism": "long_unproven_gap", "still_missing_in_accepted_labels": False},
        {"identity": 19, "frame_index": 3, "mechanism": "local_raw_supported_dropout", "still_missing_in_accepted_labels": True},
        {"identity": 19, "frame_index": 3, "mechanism": "blue_red_motion_link_candidate", "still_missing_in_accepted_labels": True},
        {"identity": 999, "frame_index": 2, "mechanism": "distant_alias_risk", "still_missing_in_accepted_labels": True}]
    _save_table(folder, "history_gap_frames", rows)
    _save_table(folder, "history_residency", [{"identity": 19, "frame_index": 2, "host_identity": 26, "held_px": 5}])
    _save_table(folder, "history_lifespans", [{"identity": 19, "last_source_frame": 6,
                                             "silent_nonborder_ending": True, "termination_mechanism": "registered_resident"}])
    result = load_track_timeline(context_run, "one")
    cell = next(c for c in result["cells"] if c["identity"] == 19)
    assert len(cell["gaps"]) == 2  # both reasons within one continuous gap survive
    assert cell["gaps"][0]["reason"] == "same_host_merge_hiding"
    assert cell["gaps"][0]["hosts"] == [26]
    assert cell["gaps"][1]["reason"] == "multiple_recorded_causes"
    assert set(cell["gaps"][1]["mechanisms"]) == {"local_raw_supported_dropout", "blue_red_motion_link_candidate"}
    assert "distant_alias_risk" not in result["gap_legend"]
    assert "long_unproven_gap" not in result["gap_legend"]
    assert cell["termination_current"]


def test_declared_history_maps_source_frames_and_pins_attachment(context_run):
    import hashlib
    from pymicroglia.ui.exploration_history import attach_tracking_history
    from pymicroglia.tracking.contract import DecisionTables

    manifest = context_run / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["movies"][0]["spec"]["source_frame_offset"] = 2
    manifest.write_text(json.dumps(saved))
    source = context_run / "decisions"
    source.mkdir()
    rows = [{"identity": 19, "gap_start_source_frame": 5, "gap_end_source_frame": 6,
             "mechanism": "local_raw_supported_dropout"}]
    _save_table(source, "gap_evidence", rows)
    declaration = DecisionTables(source, (("gap_evidence", "gap_evidence.csv", ()),)).as_dict()
    labels = context_run / "labels.tif"
    original = (context_run / "measure/one/cell_frame.csv").read_bytes()
    result = attach_tracking_history(context_run, "one", labels, declaration,
                                     expected_tracking_sha256=hashlib.sha256(labels.read_bytes()).hexdigest())
    assert result["tables"]["history_gap_frames"] == 2
    timeline = load_track_timeline(context_run, "one")
    gap = next(c for c in timeline["cells"] if c["identity"] == 19)["gaps"][0]
    assert [gap["start_frame"], gap["end_frame"]] == [2, 3]
    assert gap["reason"] == "local_raw_supported_dropout"
    assert (context_run / "measure/one/cell_frame.csv").read_bytes() == original
    with pytest.raises(ValueError, match="checksum"):
        attach_tracking_history(context_run, "one", labels, declaration, expected_tracking_sha256="wrong")
    # A new alignment cannot silently reuse an old explanation at the wrong time.
    saved["movies"][0]["spec"]["source_frame_offset"] = 1
    manifest.write_text(json.dumps(saved))
    changed = load_track_timeline(context_run, "one")
    assert "do not match" in changed["history_message"]
    assert next(c for c in changed["cells"] if c["identity"] == 19)["gaps"][0]["reason"] == "no_recorded_cause"
