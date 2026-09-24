"""Cell rows reuse Auto-Organotypic's best-cycle and shared-time grid."""

import numpy as np
import tifffile

from pymicroglia.visualisation.cell_image_grid import cell_image_grid


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
    assert report["time_scale"] == "ct"
    assert [one["key"] for one in report["tile_sources"]] == ["1", "2", "3"]
    selections = report["cell_grid"]["cycle_selection"]
    assert selections[0]["status"] == selections[1]["status"] == "selected"
    assert selections[0]["chosen_cycle"] != selections[1]["chosen_cycle"]
    assert selections[0]["interpolated_trace_frames"] == [12]
    assert selections[2]["status"] == "cycle unavailable"
    assert any(one.get("unavailable_reason") == "cycle unavailable"
               for one in report["tiles"] if one["source"] == "tile:3")


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
