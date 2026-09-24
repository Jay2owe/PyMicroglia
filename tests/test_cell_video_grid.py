"""All selected tracked cells play on the original photon-frame clock."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from auto_organotypic.video import encode
from pymicroglia.visualisation.cell_tiles import cell_tiles
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
