"""A104 is a cached display branch, separate from original-photon traces."""

import json

import numpy as np
import tifffile

from pymicroglia.visualisation.cell_display import prepare_cell_display
from pymicroglia.visualisation.cell_image_grid import cell_image_grid
from pymicroglia.visualisation.cell_tiles import cell_tiles


def test_a104_display_uses_full_frames_and_keeps_original_trace(tmp_path):
    raw = np.full((16, 20, 20), 0.8, np.float32)
    raw[:, 7:11, 7:11] += 0.25 * np.sin(np.arange(16)[:, None, None])
    labels = np.zeros((16, 20, 20), np.uint16)
    labels[:, 7:11, 7:11] = 1
    raw_path = tmp_path / "photons.ome.tif"
    tifffile.imwrite(raw_path, raw, ome=True, metadata={"axes": "TYX"})

    filtered, display_range, report = prepare_cell_display(
        raw_path, {"method": "a104", "counts_gain": 388,
                   "counts_offset": 2039})
    assert filtered.is_file()
    assert "DISPLAY_ONLY" in filtered.name
    assert display_range[0] < display_range[1]
    assert report["display_only"] is True
    assert report["recipe"]["counts_gain"] == 388
    assert json.loads(filtered.with_suffix("").with_suffix(".json").read_text())["output"] == str(filtered)
    again, _, _ = prepare_cell_display(
        raw_path, {"method": "a104", "counts_gain": 388,
                   "counts_offset": 2039})
    assert again == filtered

    tiles = cell_tiles(raw_path, labels, display_raw=filtered,
                       frame_interval_h=0.5, outline=True,
                       mask_style="fill", outline_colour="red")
    assert tiles[0].trace[1][0] == float(raw[0, 7:11, 7:11].mean())
    with tiles[0].open_series() as view:
        assert view.frame(0, 0)[2, 2] != raw[0, 7, 7]
    image = np.full((tiles[0].provenance["crop_size_px"][1],
                     tiles[0].provenance["crop_size_px"][0], 3), 100, np.uint8)
    painted = tiles[0].frame_overlay(image, 0)
    assert np.any(painted != image)
    assert np.any(painted == image)

    result = cell_image_grid(
        raw_path, labels, output_dir=tmp_path / "images",
        output_name="a104_fill.png", frame_interval_h=0.5,
        shared_time="recording", moments=2, between=(0, 4),
        channels=1, mask_style="fill", outline_colour="red",
        display_filter={"method": "a104", "counts_gain": 388,
                        "counts_offset": 2039})
    assert result["cell_grid"]["mask_style"] == "fill"
    assert result["cell_grid"]["display_filter"]["output"] == str(filtered)
    assert result["output"]
