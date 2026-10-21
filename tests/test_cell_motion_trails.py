"""Fading motion history is a display choice over fixed identity masks."""

import numpy as np
import pytest
import tifffile

from pymicroglia.visualisation.cell_tiles import cell_tiles, _paint_motion_trail
from pymicroglia.visualisation.cell_video_grid import cell_video_grid


def test_trail_fades_toward_older_positions_and_can_be_disabled(tmp_path, monkeypatch):
    raw = np.full((5, 24, 24), 50, np.uint16)
    labels = np.zeros_like(raw)
    for frame in range(5):
        labels[frame, 11:13, 4 + 3 * frame:6 + 3 * frame] = 1
    source = tmp_path / "raw.ome.tif"
    tifffile.imwrite(source, raw, ome=True, metadata={"axes": "TYX"})

    seen = {}

    def render(tiles, **options):
        seen["tiles"] = tiles
        return {"output": str(tmp_path / "grid.mp4"), "frames": 5}

    monkeypatch.setattr("pymicroglia.visualisation.cell_video_grid.video_grid.stack_to_video_grid", render)
    result = cell_video_grid(source, labels, frame_interval_h=.5,
                             outline_colour="yellow", show_outline=False,
                             crop_size_px=(22, 14), fill_tile=False,
                             scale_bar=False)
    assert result["cell_grid"]["show_motion_trails"] is False
    assert seen["tiles"][0].frame_overlay is None
    result = cell_video_grid(source, labels, frame_interval_h=.5,
                             outline_colour="yellow", show_outline=False,
                             crop_size_px=(22, 14), fill_tile=False,
                             scale_bar=False, show_motion_trails=True)
    assert result["cell_grid"]["show_motion_trails"] is True
    assert result["cell_grid"]["trail_frames"] == 4
    tile = seen["tiles"][0]
    background = np.full((14, 22, 3), 50, np.uint8)
    painted = tile.frame_overlay(background, 4)
    assert np.any(painted != background)
    assert not np.array_equal(painted, background)

    shorter = cell_tiles(source, labels, frame_interval_h=.5,
                         crop_size_px=(22, 14), outline=False,
                         show_motion_trails=True, trail_frames=1,
                         outline_colour="yellow", pixel_scale_bar=False,
                         clamp_to_frame=True, missing_centre="interpolate",
                         centre_smoothing_frames=5,
                         centre_deadband_fraction=.05)[0]
    assert not np.array_equal(shorter.frame_overlay(background, 4), painted)

    no_trail = cell_tiles(source, labels, frame_interval_h=.5,
                          crop_size_px=(22, 14), outline=False,
                          show_motion_trails=False, pixel_scale_bar=False)[0]
    assert no_trail.frame_overlay is None
    with pytest.raises(ValueError, match="positive whole number"):
        cell_tiles(source, labels, trail_frames=0)


def test_trail_does_not_bridge_missing_identity_and_old_end_is_dim(tmp_path):
    base = np.full((12, 24, 3), 40, np.uint8)
    positions = [(6., 18.), (6., 14.), (6., 10.), (6., 6.)]
    image = _paint_motion_trail(
        base, positions, centre=(6, 12), frame_size=(24, 12),
        display_size=(24, 12), source_shape=(12, 24), clamp=False,
        colour=(255, 255, 0), opacity=.8, trail_frames=3)
    assert image[6, 15, 0] > image[6, 7, 0] > 40

    raw = np.full((5, 18, 18), 40, np.uint16)
    labels = np.zeros_like(raw)
    labels[0, 8:10, 3:5] = 1
    labels[1, 8:10, 5:7] = 1
    labels[3, 8:10, 9:11] = 1
    labels[4, 8:10, 11:13] = 1
    source = tmp_path / "raw.ome.tif"
    tifffile.imwrite(source, raw, ome=True, metadata={"axes": "TYX"})
    tile = cell_tiles(source, labels, frame_interval_h=.5,
                      crop_size_px=(18, 18), show_motion_trails=True,
                      outline=False)[0]
    frame = np.full((18, 18, 3), 40, np.uint8)
    assert np.array_equal(tile.frame_overlay(frame, 3), frame)
    assert not np.array_equal(tile.frame_overlay(frame, 4), frame)
