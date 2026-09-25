"""All selected tracked cells play on the original photon-frame clock."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from auto_organotypic.video import encode
from pymicroglia.visualisation.cell_tiles import (
    _hold_small_movements, cell_tiles)
from pymicroglia.visualisation.cell_video_grid import cell_video_grid

pytestmark = pytest.mark.skipif(not encode.available(), reason="ffmpeg unavailable")


def test_full_recording_keeps_all_cells_and_marks_missing(tmp_path, monkeypatch):
    raw = np.stack([np.full((1, 30, 75), 200 + 100 * t, np.uint16)
                    for t in range(8)])
    labels = np.zeros((6, 30, 75), np.uint16)
    for index in range(6):
        if index != 3:
            labels[index, 8:13, 3 + index:8 + index] = 1
        labels[index, 8:13, 30:35] = 2
        labels[index, 8:13, 60:65] = 3  # retained for videos independently of analysis
    raw_path, labels_path = tmp_path / "photons.tif", tmp_path / "videos_labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True,
                     metadata={"axes": "TCYX", "finterval": 1800,
                               "tunit": "s", "mode": "composite"})
    tifffile.imwrite(labels_path, labels, imagej=True,
                     metadata={"axes": "TYX"})
    captured = []
    original = encode.write_video

    def capture(target, frames, **options):
        def watched():
            for frame in frames:
                captured.append(np.asarray(frame).copy())
                yield frame
        return original(target, watched(), **options)

    monkeypatch.setattr(encode, "write_video", capture)
    report = cell_video_grid(
        raw_path, labels_path, source_frame_offset=2,
        output_dir=tmp_path / "out", output_name="cells", fps=12,
        channels=1, lut="grays", display_range=(0, 1100),
        soft_range="hard", tile_label="none", columns=3, gap_px=0)
    assert Path(report["output"]).exists()
    assert len(captured) == report["frames"] == 8
    assert [one["tile_key"] for one in report["tiles"]] == ["1", "2", "3"]
    assert report["tiles"][0]["unavailable_frames"][5] == "tracked mask absent"
    assert report["tiles"][0]["unavailable_frames"][0] == "tracked mask absent"
    assert report["on_length"] == report["on_cadence"] == "raise"
    assert report["cell_grid"]["clock"] == "original source frame order"
    assert report["cell_grid"]["outline"] is True
    assert report["cell_grid"]["missing_centre"] == "interpolate"
    assert report["cell_grid"]["centre_smoothing_frames"] == 5
    assert report["cell_grid"]["centre_deadband_fraction"] == 0.05
    assert report["cell_grid"]["clamp_to_frame"] is True
    assert all(not tile["tile_provenance"]["frame_crop"] and
               tile["tile_provenance"]["clamp_to_frame"]
               for tile in report["tiles"])
    assert report["well_label_position"] == "top-left"
    width = report["image_width"]
    # Cell 2 remains observed; its central photon pixel brightens on every
    # original frame from the first labelled source frame onward.
    levels = [int(frame[width // 2, width + width // 2, 0])
              for frame in captured[2:]]
    assert levels == sorted(levels) and len(set(levels)) == len(levels)
    cyan = np.array([0, 255, 255], np.uint8)
    assert np.any(np.all(captured[2][:width, :width] == cyan, axis=2))
    assert not np.any(np.all(captured[5][:width, :width] == cyan, axis=2))


def test_missing_mask_keeps_new_photon_frames_and_interpolates_crop(tmp_path):
    raw = np.zeros((5, 1, 20, 20), np.uint16)
    labels = np.zeros((5, 20, 20), np.uint16)
    for frame in range(5):
        raw[frame, 0, 10, 4 + frame] = 100 + frame
    labels[0, 10, 4] = 1
    labels[4, 10, 8] = 1
    raw_path, labels_path = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True, metadata={"axes": "TCYX"})
    tifffile.imwrite(labels_path, labels, imagej=True,
                     metadata={"axes": "TYX"})

    moving = cell_tiles(raw_path, labels_path, frame_interval_h=1,
                        missing_centre="interpolate", outline=True)[0]
    held = cell_tiles(raw_path, labels_path, frame_interval_h=1,
                      missing_centre="hold", outline=False)[0]
    with moving.open_series() as view:
        assert view.frame(2, 0)[1, 1] == 102
        assert view.frame(3, 0)[1, 1] == 103
    with held.open_series() as view:
        assert view.frame(2, 0)[1, 1] == 0
    canvas = np.zeros((3, 3, 3), np.uint8)
    assert np.any(moving.frame_overlay(canvas, 0))
    assert not np.any(moving.frame_overlay(canvas, 2))
    assert held.frame_overlay is None


def test_smoothing_steadies_crop_but_keeps_observed_mask_and_trace(tmp_path):
    raw = np.zeros((7, 1, 20, 30), np.uint16)
    raw[:, 0, 10, 10] = 100
    labels = np.zeros((7, 20, 30), np.uint16)
    for frame, x in enumerate((10, 10, 10, 15, 10, 10, 10)):
        labels[frame, 10, x] = 1
    raw_path, labels_path = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True, metadata={"axes": "TCYX"})
    tifffile.imwrite(labels_path, labels, imagej=True, metadata={"axes": "TYX"})
    common = dict(frame_interval_h=1, crop_basis="own_cell",
                  crop_size_px=(15, 15), outline=True)
    exact = cell_tiles(raw_path, labels_path, centre_smoothing_frames=0,
                       **common)[0]
    steady = cell_tiles(raw_path, labels_path, centre_smoothing_frames=5,
                        **common)[0]
    with exact.open_series() as view:
        assert view.frame(3, 0)[7, 2] == 100
    with steady.open_series() as view:
        assert view.frame(3, 0)[7, 7] == 100
    assert exact.trace[1][3] == steady.trace[1][3] == 0
    canvas = np.zeros((15, 15, 3), np.uint8)
    # The outlying observed mask remains at its real source position.
    assert np.any(steady.frame_overlay(canvas, 3)[:, 11:, :])
    assert steady.provenance["centre_smoothing_frames"] == 5


def test_crop_deadband_holds_small_shifts_then_follows_excess():
    candidate = np.array([[4., 10.], [4., 11.], [4., 11.9],
                          [4., 12.1], [4., 14.1]])
    held = _hold_small_movements(candidate, 2.0)
    np.testing.assert_allclose(held[:, 1], [10., 10., 10., 10.1, 12.1])
    np.testing.assert_array_equal(held[:, 0], 4.)
    np.testing.assert_array_equal(_hold_small_movements(candidate, 0),
                                  candidate)


@pytest.mark.parametrize("window", [-1, 2, True])
def test_bad_centre_smoothing_window_is_rejected(tmp_path, window):
    with pytest.raises(ValueError, match="centre_smoothing_frames"):
        cell_tiles(tmp_path / "raw.tif", np.zeros((1, 2, 2), np.uint8),
                   centre_smoothing_frames=window)


@pytest.mark.parametrize("threshold", [-1, float("nan"), 1.1, True])
def test_bad_centre_deadband_is_rejected(tmp_path, threshold):
    with pytest.raises(ValueError, match="centre_deadband_fraction"):
        cell_tiles(tmp_path / "raw.tif", np.zeros((1, 2, 2), np.uint8),
                   centre_deadband_fraction=threshold)


def test_relative_deadband_scales_with_crop_width(tmp_path):
    raw = np.zeros((4, 1, 20, 30), np.uint16)
    labels = np.zeros((4, 20, 30), np.uint16)
    for frame, x in enumerate((10, 11, 12, 13)):
        raw[frame, 0, 10, 10] = 100
        labels[frame, 10, x] = 1
    raw_path, labels_path = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True, metadata={"axes": "TCYX"})
    tifffile.imwrite(labels_path, labels, imagej=True, metadata={"axes": "TYX"})
    tile = cell_tiles(raw_path, labels_path, frame_interval_h=1,
                      crop_size_px=(20, 20), centre_deadband_fraction=.1)[0]
    assert tile.provenance["centre_deadband_px"] == 2.0
    with tile.open_series() as view:
        assert view.frame(0, 0)[10, 10] == 100
        assert view.frame(1, 0)[10, 10] == 100
        assert view.frame(2, 0)[10, 10] == 100
        assert view.frame(3, 0)[10, 9] == 100


def test_video_grid_frame_gap_limit_is_optional(tmp_path):
    raw = np.full((6, 1, 18, 30), 100, np.uint16)
    labels = np.zeros((6, 18, 30), np.uint16)
    labels[:, 4:7, 4:7] = 1
    labels[:, 10:13, 22:25] = 2
    labels[3, 4:7, 4:7] = 0
    raw_path, labels_path = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True,
                     metadata={"axes": "TCYX", "finterval": 1800, "tunit": "s"})
    tifffile.imwrite(labels_path, labels, imagej=True,
                     metadata={"axes": "TYX"})
    report = cell_video_grid(
        raw_path, labels_path, output_dir=tmp_path / "out",
        output_name="quality.mp4", max_gap_frames=0, fps=12,
        channels=1, lut="grays", display_range=(0, 200),
        soft_range="hard", tile_label="none")
    assert report["frames"] == 6
    assert report["cell_grid"]["cell_identities"] == ["2"]
    assert report["cell_grid"]["selection"]["excluded_identities"] == [1]
