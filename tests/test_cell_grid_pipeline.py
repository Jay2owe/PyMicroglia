"""Pipeline helpers write destination-specific cell grids and link the files."""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from auto_organotypic.video import encode
from pymicroglia.pipelines._auto_microglia_support import (
    _cell_grids, _validate_cell_grid_options,
)


def _inputs(tmp_path):
    stem = "well_a"
    frames = 60
    raw = np.full((frames, 1, 20, 32), 200, np.uint16)
    labels = np.zeros((frames, 20, 32), np.uint16)
    for index in range(frames):
        raw[index, 0, 4:9, 3:8] = int(850 + 220 * np.cos(2 * np.pi * index / 48))
        raw[index, 0, 12:17, 24:29] = 700 + index
        labels[index, 4:9, 3:8] = 1
        labels[index, 12:17, 24:29] = 2
    images = labels.copy()
    images[images == 2] = 0
    raw_path = tmp_path / "photons.tif"
    all_path = tmp_path / "videos_labels.tif"
    images_path = tmp_path / "images_labels.tif"
    tifffile.imwrite(raw_path, raw, imagej=True,
                     metadata={"axes": "TCYX", "finterval": 1800,
                               "tunit": "s", "mode": "composite"})
    tifffile.imwrite(all_path, labels, imagej=True,
                     metadata={"axes": "TYX"})
    tifffile.imwrite(images_path, images, imagej=True,
                     metadata={"axes": "TYX"})
    document = {"prepared": {stem: {"measurement_raw": str(raw_path),
                                     "frame_interval_min": 30}}}
    handoff_path = tmp_path / "motion_inputs.json"
    handoff_path.write_text(json.dumps(document), encoding="utf-8")
    handoff = {"outputs": {"motion_inputs": str(handoff_path)}}
    tracking = {stem: {"source_frame_offset": 0}}
    eligibility = {stem: {"views": {
        "images": {"labels": str(images_path)},
        "videos": {"labels": str(all_path)},
    }}}
    return tracking, handoff, eligibility


@pytest.mark.skipif(not encode.available(), reason="ffmpeg unavailable")
def test_real_grid_outputs_use_separate_eligibility_views(tmp_path):
    tracking, handoff, eligibility = _inputs(tmp_path)
    run = tmp_path / "run"
    images = _cell_grids(
        "images", tracking, handoff, eligibility, run,
        options={"shared_time": "recording", "moments": 3,
                 "display_range": (0, 1200), "lut": "grays",
                 "soft_range": "hard"})
    videos = _cell_grids(
        "videos", tracking, handoff, eligibility, run,
        options={"fps": 12, "display_range": (0, 1200),
                 "lut": "grays", "soft_range": "hard",
                 "tile_label": "none"})
    image, video = images["well_a"], videos["well_a"]
    assert Path(image["output"]).exists() and Path(video["output"]).exists()
    assert image["cell_identities"] == ["1"]
    assert video["cell_identities"] == ["1", "2"]
    assert "/visual/images/" in image["output"].replace("\\", "/")
    assert "/visual/videos/" in video["output"].replace("\\", "/")
    assert image["display_only"] and video["display_only"]
    assert image["eligibility_labels"] != video["eligibility_labels"]


def test_visual_options_are_checked_before_the_chain_runs():
    with pytest.raises(ValueError, match="unknown setting"):
        _validate_cell_grid_options({"display_rnage": "auto"}, None)
    with pytest.raises(ValueError, match="pipeline-owned"):
        _validate_cell_grid_options(None, {"source_frame_offset": 2})
    _validate_cell_grid_options(
        {"crop_basis": "own_cell", "display_options": {"lut": "grays"}},
        {"fps": 12})


def test_both_direct_actions_are_discoverable():
    from pymicroglia.knowledge import describe

    image = describe("cell_image_grid")
    video = describe("cell_video_grid")
    assert image["pending"] is video["pending"] is False
    assert image["display_only"] is video["display_only"] is True
    assert {row["name"] for row in image["params"]} >= {
        "raw", "labels", "crop_basis", "crop_rectangle_px",
        "shared_time", "display_options"}
