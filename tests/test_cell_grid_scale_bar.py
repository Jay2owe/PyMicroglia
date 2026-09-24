"""Tracked-cell grids pass physical calibration to their shared renderer."""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from auto_organotypic.video import encode
from pymicroglia.visualisation.cell_image_grid import cell_image_grid
from pymicroglia.visualisation.cell_video_grid import cell_video_grid


def test_cell_still_bar_defaults_on_and_can_be_hidden(tmp_path):
    raw = tmp_path / "photons.tif"
    labels = tmp_path / "labels.tif"
    photons = np.full((4, 1, 100, 100), 100, np.uint16)
    marked = np.zeros((4, 100, 100), np.uint16)
    marked[:, 40:60, 40:60] = 1
    tifffile.imwrite(raw, photons, imagej=True, metadata={"axes": "TCYX"})
    tifffile.imwrite(labels, marked, imagej=True, metadata={"axes": "TYX"})
    kwargs = {"shared_time": "recording", "when": 1, "frame_interval_h": 1,
              "crop_rectangle_px": (91, 91), "um_per_px": 2.0,
              "show_outline": False, "tile_label": "none", "channels": 1,
              "lut": "grays", "display_range": (0, 200)}
    on = cell_image_grid(raw, labels, output_dir=tmp_path / "on", **kwargs)
    off = cell_image_grid(raw, labels, output_dir=tmp_path / "off",
                          scale_bar=False, **kwargs)
    assert on["scale_bars"][0]["length_px"] == 25
    assert off["scale_bars"][0]["status"] == "disabled"
    assert not np.array_equal(np.asarray(Image.open(on["output"])),
                              np.asarray(Image.open(off["output"])))


def test_cell_video_bar_uses_same_calibration(tmp_path):
    if not encode.available():
        return
    raw = tmp_path / "photons.tif"
    labels = tmp_path / "labels.tif"
    photons = np.full((4, 1, 100, 100), 100, np.uint16)
    marked = np.zeros((4, 100, 100), np.uint16)
    marked[:, 40:60, 40:60] = 1
    tifffile.imwrite(raw, photons, imagej=True, metadata={"axes": "TCYX"})
    tifffile.imwrite(labels, marked, imagej=True, metadata={"axes": "TYX"})
    report = cell_video_grid(raw, labels, output_dir=tmp_path / "video",
                             frame_interval_h=1, crop_rectangle_px=(91, 91),
                             um_per_px=2.0, show_outline=False, fps=4,
                             tile_label="none", channels=1, lut="grays",
                             display_range=(0, 200))
    assert Path(report["output"]).exists()
    assert report["scale_bars"][0]["length_px"] == 25
