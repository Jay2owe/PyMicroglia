"""Cell rows reuse Auto-Organotypic's best-cycle and shared-time grid."""

import numpy as np
import pandas as pd
import tifffile

from pymicroglia.visualisation.cell_image_grid import cell_image_grid
from pymicroglia.visualisation.cell_tiles import cell_tiles


def _recording(tmp_path):
    frames = 192
    raw = np.full((frames, 1, 30, 40), 200, np.uint16)
    labels = np.zeros((frames, 30, 40), np.uint16)
    times = np.arange(frames) * .5
    for index, hour in enumerate(times):
        first_gain = 4 if 24 <= hour < 48 else 1
        second_gain = 4 if 48 <= hour < 72 else 1
        raw[index, 0, 5:10, 5:10] = int(900 + 90 * first_gain * np.cos(2 * np.pi * hour / 24))
        raw[index, 0, 17:22, 25:30] = int(900 + 90 * second_gain * np.cos(2 * np.pi * hour / 24))
        labels[index, 5:10, 5:10] = 1
        labels[index, 17:22, 25:30] = 2
        if index < 10:
            labels[index, 5:8, 30:33] = 3  # kept for images; too short for a cycle
    labels[12, 5:10, 5:10] = 0  # a short trace gap, with the real frame retained
    raw_path, labels_path = tmp_path / "photons.tif", tmp_path / "images_labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True,
                     metadata={"axes": "TCYX", "finterval": 1800,
                               "tunit": "s", "mode": "composite"})
    tifffile.imwrite(labels_path, labels, imagej=True,
                     metadata={"axes": "TYX"})
    return raw_path, labels_path


def test_own_cycles_keep_every_image_identity_and_placeholder(tmp_path):
    raw, labels = _recording(tmp_path)
    report = cell_image_grid(raw, labels, output_dir=tmp_path / "out",
                             output_name="cells", channels=1, lut="grays",
                             display_range=(0, 1600), soft_range="hard")
    assert report["rows"] == 3
    assert report["cell_grid"]["centre_method"] == "intensity_weighted"
    assert report["time_scale"] == "ct"
    assert [one["key"] for one in report["tile_sources"]] == ["1", "2", "3"]
    selections = report["cell_grid"]["cycle_selection"]
    assert selections[0]["status"] == selections[1]["status"] == "selected"
    assert selections[0]["chosen_cycle"] != selections[1]["chosen_cycle"]
    assert selections[0]["interpolated_trace_frames"] == [12]
    assert selections[2]["status"] == "cycle unavailable"
    assert any(one.get("unavailable_reason") == "cycle unavailable"
               for one in report["tiles"] if one["source"] == "tile:3")


def test_cycle_only_display_omits_blank_rows_and_keeps_selection_record(tmp_path):
    raw, labels = _recording(tmp_path)
    report = cell_image_grid(raw, labels, output_dir=tmp_path / "out",
                             output_name="cycle_cells", channels=1,
                             lut="grays", display_range=(0, 1600),
                             exclude_unavailable_cycles=True)
    assert report["rows"] == 2
    assert report["well_label_width"] == 0
    assert report["well_label_position"] == "top-left"
    assert report["cell_grid"]["cell_identities"] == ["1", "2"]
    assert report["cell_grid"]["all_cell_identities"] == ["1", "2", "3"]
    assert report["cell_grid"]["excluded_cycle_identities"] == ["3"]
    assert report["cell_grid"]["frame_crop"] is False
    assert report["cell_grid"]["clamp_to_frame"] is True
    assert all(one.get("unavailable_reason") != "cycle unavailable"
               for one in report["tiles"])


def test_intensity_weighted_centre_places_bright_mask_region_in_middle(tmp_path):
    raw = np.full((3, 1, 16, 20), 10, np.uint16)
    raw[:, 0, 6, 9] = 100
    labels = np.zeros((3, 16, 20), np.uint16)
    labels[:, 5:8, 5:10] = 1
    raw_path, labels_path = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True, metadata={"axes": "TCYX"})
    tifffile.imwrite(labels_path, labels, imagej=True, metadata={"axes": "TYX"})
    common = dict(frame_interval_h=1, crop_size_px=(9, 9))
    mask = cell_tiles(raw_path, labels_path, centre_method="mask", **common)[0]
    bright = cell_tiles(raw_path, labels_path,
                        centre_method="intensity_weighted", **common)[0]
    with mask.open_series() as view:
        assert view.frame(1, 0)[4, 6] == 100
    with bright.open_series() as view:
        assert view.frame(1, 0)[4, 4] == 100
    assert bright.trace[1][1] == mask.trace[1][1]
    assert bright.provenance["centre_weighting"] == "raw photons above within-mask minimum"


def test_shared_recording_and_event_time_use_same_source_hours(tmp_path):
    raw, labels = _recording(tmp_path)
    common = dict(output_dir=tmp_path / "out", channels=1, lut="grays",
                  display_range=(0, 1600), soft_range="hard", moments=3,
                  between=(20, 30))
    shared = cell_image_grid(raw, labels, output_name="shared",
                             shared_time="recording", **common)
    assert shared["time_scale"] == "elapsed"
    assert len(set(one["window"] for one in shared["tiles"] if one["column"] == 0)) == 1
    event = cell_image_grid(raw, labels, output_name="event",
                            shared_time="event", event_hour=24,
                            between=(-4, 6), **{key: value for key, value in common.items()
                                               if key != "between"})
    assert event["time_scale"] == "elapsed"
    assert event["timestamp_format"] == "Event {total_hours:+.1f} h"
    assert event["cell_grid"]["event_hour"] == 24


def test_long_gap_disqualifies_affected_cycle(tmp_path):
    raw, labels = _recording(tmp_path)
    values = tifffile.imread(labels)
    values[55:71, 5:10, 5:10] = 0
    tifffile.imwrite(labels, values, imagej=True, metadata={"axes": "TYX"})
    report = cell_image_grid(raw, labels, output_dir=tmp_path / "out",
                             output_name="gap", channels=1, lut="grays",
                             display_range=(0, 1600), soft_range="hard")
    selection = report["cell_grid"]["cycle_selection"][0]
    assert selection["status"] == "selected"
    assert set(range(55, 71)).issubset(selection["ineligible_trace_frames"])
    first, last = selection["selected_trace_frames"]
    assert last < 55 or first > 70


def test_frame_filters_remove_cells_before_sizing_the_grid(tmp_path):
    raw, labels = _recording(tmp_path)
    report = cell_image_grid(
        raw, labels, output_dir=tmp_path / "out", output_name="quality",
        shared_time="recording", moments=2, max_gap_frames=0,
        max_missing_frames=0, channels=1, lut="grays",
        display_range=(0, 1600), soft_range="hard")
    assert report["cell_grid"]["cell_identities"] == ["2"]
    selection = report["cell_grid"]["selection"]
    assert selection["selected_identities"] == [2]
    reasons = {row["identity"]: row["selection_reason"]
               for row in selection["cells"]}
    assert "internal_gap_frames_over_limit" in reasons[1]
    assert "missing_frames_over_limit" in reasons[3]


def test_period_recipe_reaches_the_shared_test_and_selects_supported_cells(
        tmp_path, monkeypatch):
    from pymicroglia.figure_tables import all_cell_traces

    raw, labels = _recording(tmp_path)
    def evidence(_frame, _metrics, identities, resolved, _view, _normal,
                 _config, **options):
        assert resolved["method"] == "lomb"
        assert resolved["significance_method"] == "lomb"
        assert resolved["min_cycles"] == 1.0
        assert options["fft_component_test"] is False
        return pd.DataFrame(), pd.DataFrame([
            {"identity": identity, "period_hours": 24.0, "p_value": .01,
             "q_value": .02, "significance_status": "ok",
             "rhythm_status": "rhythmic" if identity != 2 else "not rhythmic",
             "supported_period": identity != 3}
            for identity in identities])
    monkeypatch.setattr(all_cell_traces, "trace_data", evidence)
    recipe = {"fft_component_test": False, "fit_method": "lomb",
              "significance_method": "lomb", "detrend": "none",
              "multiple_testing": "bh", "min_cycles": 1.0,
              "period_config": {}}
    report = cell_image_grid(
        raw, labels, output_dir=tmp_path / "out", output_name="period",
        shared_time="recording", moments=2,
        significant_period_only=True, period_recipe=recipe,
        channels=1, lut="grays", display_range=(0, 1600),
        soft_range="hard")
    assert report["cell_grid"]["cell_identities"] == ["1"]
    selected = report["cell_grid"]["selection"]
    assert selected["period_recipe"]["fit_method"] == "lomb"
    assert selected["excluded_identities"] == [2, 3]
