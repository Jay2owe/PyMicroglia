"""Tracked-cell grids pass physical calibration to their shared renderer."""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from auto_organotypic.video import encode
from pymicroglia.visualisation.cell_image_grid import cell_image_grid
from pymicroglia.visualisation.cell_video_grid import cell_video_grid
from pymicroglia.visualisation.cell_tiles import cell_tiles


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


def test_own_cell_crops_fill_equal_grid_slots_without_thickening_outline(tmp_path):
    raw = tmp_path / "photons.tif"
    labels = tmp_path / "labels.tif"
    photons = np.full((4, 1, 100, 100), 100, np.uint16)
    marked = np.zeros((4, 100, 100), np.uint16)
    marked[:, 20:28, 20:28] = 1
    marked[:, 55:79, 55:79] = 2
    tifffile.imwrite(raw, photons, imagej=True, metadata={"axes": "TCYX"})
    tifffile.imwrite(labels, marked, imagej=True, metadata={"axes": "TYX"})
    tiles = cell_tiles(raw, labels, frame_interval_h=1,
                       crop_basis="own_cell", fill_tile=True,
                       um_per_px=2.0, outline=True,
                       outline_colour="yellow", outline_width_px=1)
    small, large = tiles
    assert small.provenance["crop_size_px"][0] < large.provenance["crop_size_px"][0]
    assert small.provenance["display_size_px"] == large.provenance["display_size_px"]
    with small.open_series() as view:
        frame = view.frame(0, 0)
        assert frame.shape == tuple(reversed(small.provenance["display_size_px"]))
        assert np.isfinite(frame).all() and np.min(frame) == 100
        assert view.meta.um_per_px < 2.0
        plain = np.zeros((*frame.shape, 3), np.uint8)
        outlined = small.frame_overlay(plain, 0)
        yellow = np.all(outlined == (255, 255, 0), axis=2)
        assert yellow.any()
        assert not np.all(yellow[15:18, 15:18])
    report = cell_image_grid(
        raw, labels, output_dir=tmp_path / "images",
        crop_basis="own_cell", shared_time="recording", when=1,
        frame_interval_h=1, um_per_px=2.0, channels=1,
        tile_label="none", lut="grays", display_range=(0, 200))
    assert report["cell_grid"]["fill_tile"] is True
    assert report["scale_bars"][0]["um_per_px"] < report["scale_bars"][1]["um_per_px"]
    if encode.available():
        movie = cell_video_grid(
            raw, labels, output_dir=tmp_path / "videos",
            crop_basis="own_cell", tile_size_px=128, frame_interval_h=1,
            um_per_px=2.0, fps=4, channels=1, tile_label="none",
            lut="grays", display_range=(0, 200))
        assert movie["cell_grid"]["fill_tile"] is True
        assert movie["cell_grid"]["tile_size_px"] == 128
        assert all(one["provenance"]["display_size_px"] == [128, 128]
                   for one in movie["tile_sources"])
        assert movie["scale_bars"][0]["um_per_px"] < movie["scale_bars"][1]["um_per_px"]
