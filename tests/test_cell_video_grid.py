"""All selected tracked cells play on the original photon-frame clock."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from auto_organotypic.video import encode
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
    width = report["image_width"]
    # Cell 2 remains observed; its central photon pixel brightens on every
    # original frame from the first labelled source frame onward.
    levels = [int(frame[width // 2, width + width // 2, 0])
              for frame in captured[2:]]
    assert levels == sorted(levels) and len(set(levels)) == len(levels)
