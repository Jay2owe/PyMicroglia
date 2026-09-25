"""Still crops fit each observed mask without losing scale calibration."""

import numpy as np
import pytest
import tifffile

from pymicroglia.visualisation.cell_tiles import cell_tiles


def test_own_frame_crop_fills_shared_slot_and_tracks_physical_size(tmp_path):
    raw = np.full((2, 1, 40, 40), 100, np.uint16)
    labels = np.zeros((2, 40, 40), np.uint16)
    labels[0, 0:3, 0:3] = 1
    labels[1, 8:17, 8:17] = 1
    raw_path, label_path = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True, metadata={"axes": "TCYX"})
    tifffile.imwrite(label_path, labels, imagej=True,
                     metadata={"axes": "TYX"})

    tile = cell_tiles(raw_path, label_path, crop_basis="own_cell",
                      crop="tight", fill_tile=True, frame_crop=True,
                      um_per_px=2.0)[0]
    with tile.open_series() as opened:
        first, second = opened.frame(0, 0), opened.frame(1, 0)
    assert first.shape == second.shape == (11, 11)
    assert np.isfinite(first).all()  # crop at the source edge shifts inward
    assert tile.frame_um_per_px(0) == pytest.approx(2.0 * 5 / 11)
    assert tile.frame_um_per_px(1) == pytest.approx(2.0)
    assert tile.provenance["frame_crop"] is True


def test_frame_crop_requires_a_resized_own_cell_slot(tmp_path):
    raw_path = tmp_path / "raw.tif"
    label_path = tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, np.ones((2, 1, 20, 20), np.uint16),
                     imagej=True, metadata={"axes": "TCYX"})
    labels = np.zeros((2, 20, 20), np.uint16)
    labels[:, 5:8, 5:8] = 1
    tifffile.imwrite(label_path, labels, imagej=True,
                     metadata={"axes": "TYX"})
    with pytest.raises(ValueError, match="frame_crop needs"):
        cell_tiles(raw_path, label_path, frame_crop=True)
