"""Accepted identity outlines are configurable display-only consumers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia import knowledge, registry
from pymicroglia.video import tracked_cell_video
from pymicroglia.visualisation.overlays import tracked_cell_image
from pymicroglia.visualisation.tracked_outlines import outer_boundaries, overlay


def stacks(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "original_photons.tif"
    labels = tmp_path / "motion_labels.tif"
    tifffile.imwrite(
        raw,
        np.arange(6 * 7 * 7, dtype=np.uint16).reshape(6, 1, 7, 7),
        imagej=True, metadata={"axes": "TCYX", "finterval": 1800.0,
                               "tunit": "s"})
    values = np.zeros((3, 7, 7), np.uint16)
    values[0, 2:4, 2:4] = 1
    values[1, 3:5, 3:5] = 1
    values[2, 1:3, 4:6] = 2
    tifffile.imwrite(labels, values, photometric="minisblack")
    return raw, labels


def test_boundaries_are_outside_identity_coloured_and_frame_specific():
    values = np.zeros((2, 7, 7), np.uint16)
    values[0, 3, 3] = 1
    values[1, 1, 1] = 2
    boundaries = outer_boundaries(values)
    assert boundaries[0, 3, 3] == 0
    assert np.count_nonzero(boundaries[0] == 1) == 4
    assert np.count_nonzero(boundaries[1] == 2) == 4
    assert np.count_nonzero(boundaries[0] == 2) == 0
    with pytest.raises(ValueError, match="whole number"):
        outer_boundaries(values, width_px=1.5)


def test_video_forwards_all_display_choices_over_original_photons(
        tmp_path: Path, monkeypatch):
    raw, labels = stacks(tmp_path)
    seen = {}

    def render(source, **options):
        seen["source"] = source
        seen.update(options)
        return {"output": str(tmp_path / "movie.mp4")}

    monkeypatch.setattr("pymicroglia.video.stack_to_video", render)
    result = tracked_cell_video(
        raw, labels=labels, output_dir=tmp_path / "video",
        source_frame_offset=2, frame_interval_h=0.5, hours_per_second=8,
        outline_width_px=2, outline_opacity=0.6,
        outline_colours=[(1, 2, 3), (4, 5, 6)], smooth_frames=5,
        smooth_sigma_px=1.2, black_percentile=10, white_percentile=99,
        timestamp=False, crf=18)

    assert Path(seen["source"]) == raw
    assert seen["first_frame"] == 2 and seen["frames"] == 3
    assert seen["hours_per_second"] == 8
    assert seen["smooth_frames"] == 5 and seen["smooth_sigma_px"] == 1.2
    assert seen["auto_black_percentile"] == 10
    assert seen["auto_white_percentile"] == 99
    assert seen["timestamp"] is False and seen["crf"] == 18
    painter = seen["outline"]
    assert painter.colour_by_label is True
    assert painter.width_px == 2 and painter.opacity == 0.6
    assert painter.palette == ((1, 2, 3), (4, 5, 6))
    assert result["display_only"] is True


def test_still_uses_the_selected_tracked_frame_and_source_offset(
        tmp_path: Path, monkeypatch):
    raw, labels = stacks(tmp_path)
    seen = {}

    def render(source, **options):
        seen["source"] = source
        seen.update(options)
        return {"output": str(tmp_path / "still.png")}

    monkeypatch.setattr("auto_organotypic.image.stack_to_image", render)
    result = tracked_cell_image(raw, labels=labels, frame_index=1,
                                source_frame_offset=2,
                                output_dir=tmp_path / "image")
    assert Path(seen["source"]) == raw
    assert seen["when"] == 4
    assert seen["outline"].labels.ndim == 2
    assert result["tracked_frame_index"] == 1
    assert result["display_only"] is True


def test_still_renders_through_the_shared_renderer(tmp_path: Path):
    raw, labels = stacks(tmp_path)
    result = tracked_cell_image(
        raw, labels=labels, frame_index=1, source_frame_offset=2,
        frame_interval_h=0.5, timestamp=False,
        output_dir=tmp_path / "rendered", output_name="tracked",
        smooth_sigma_px=0)
    assert Path(result["output"]).is_file()
    assert Path(result["output"]).suffix == ".png"
    assert result["outline"]["colour"] == "per_identity"


def test_display_actions_are_live_and_documented():
    assert registry.REGISTRY.resolve("tracked_cell_video") is tracked_cell_video
    assert registry.REGISTRY.resolve("tracked_cell_image") is tracked_cell_image
    for name in ("tracked_cell_video", "tracked_cell_image"):
        details = knowledge.describe(name)
        assert details["pending"] is False
        names = {row["name"] for row in details["params"]}
        assert {"outline_width_px", "outline_opacity", "outline_colours",
                "black_percentile", "white_percentile"} <= names
