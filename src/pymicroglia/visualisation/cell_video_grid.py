"""A full-duration moving-cell video using Auto-Organotypic's video grid."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from auto_organotypic import video_grid
from auto_organotypic.render.outlines import (DEFAULT_COLOUR,
                                              DEFAULT_OPACITY,
                                              DEFAULT_WIDTH_PX)

from .cell_tiles import cell_tiles


def cell_video_grid(raw, labels, *, output_dir=None, output_name=None,
                    overwrite: bool = False,
                    source_frame_offset: int = 0,
                    crop_basis: str = "largest_cell", crop: str = "tight",
                    crop_rectangle_px: tuple[int, int] | None = None,
                    trace_channel: int = 1,
                    max_trace_gap_h: float = 4.0,
                    missing_centre: str = "interpolate",
                    show_outline: bool = True,
                    outline_colour: Any = DEFAULT_COLOUR,
                    outline_width_px: int = DEFAULT_WIDTH_PX,
                    outline_opacity: float = DEFAULT_OPACITY,
                    display_options: Mapping[str, Any] | None = None,
                    **video_options) -> dict[str, Any]:
    """Play every requested source frame with each cell centred in its tile.

    The labels are the selected videos eligibility view. By default every
    cell follows the same original source-frame clock; explicit alignment,
    playback, labels, contrast and encoding options go to the shared renderer.
    The default is a purple photon ramp, top-left names, and cyan outlines
    from the observed mask. Pass ``show_outline=False`` to hide them.
    """
    video_options = {**dict(display_options or {}), **video_options}
    crop_size_px = video_options.pop("crop_size_px", None)
    if crop_rectangle_px is not None:
        if crop_size_px is not None:
            raise ValueError("give crop_size_px or crop_rectangle_px, not both")
        crop_size_px = crop_rectangle_px
    video_options.setdefault("overwrite", overwrite)
    missing_centre = video_options.pop("missing_centre", missing_centre)
    show_outline = video_options.pop("show_outline", show_outline)
    outline = video_options.pop("outline", show_outline)
    outline_colour = video_options.pop("outline_colour", outline_colour)
    outline_width_px = video_options.pop("outline_width_px", outline_width_px)
    outline_opacity = video_options.pop("outline_opacity", outline_opacity)
    video_options.setdefault("lut", "dluc_purple")
    tiles = cell_tiles(
        raw, labels, source_frame_offset=source_frame_offset,
        frame_interval_h=video_options.get("frame_interval_h"),
        crop_basis=crop_basis, crop=crop, crop_size_px=crop_size_px,
        trace_channel=trace_channel, missing_centre=missing_centre,
        outline=outline, outline_colour=outline_colour,
        outline_width_px=outline_width_px,
        outline_opacity=outline_opacity, unavailable_label="NO MASK")
    align = video_options.get("align", "start")
    if isinstance(align, str) and align.strip().lower() in (
            "best", "onset", "peak", "trough"):
        prepared = []
        for tile in tiles:
            times, values = (np.asarray(one, float) for one in tile.trace)
            known = np.flatnonzero(np.isfinite(values))
            if known.size < 4:
                raise ValueError(f"{tile.name} has no trace for phase alignment")
            interval = float(np.median(np.diff(times)))
            missing = np.flatnonzero(~np.isfinite(values))
            # The phase detector is the upstream one. Only short internal
            # tracking gaps may be bridged in its display-only trace.
            if any(index < known[0] or index > known[-1] for index in missing):
                raise ValueError(f"{tile.name} has an unobserved recording edge; choose align='start' or a shorter window")
            if missing.size:
                groups = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
                if any(len(group) * interval > float(max_trace_gap_h)
                       for group in groups):
                    raise ValueError(f"{tile.name} has a tracking gap too long for phase alignment")
            filled = np.interp(np.arange(len(values)), known, values[known])
            prepared.append(replace(tile, trace=(times, filled),
                                    provenance={**tile.provenance,
                                                "interpolated_trace_frames":
                                                [int(one) for one in missing]}))
        tiles = prepared
    report = video_grid.stack_to_video_grid(
        tiles, output_dir=output_dir, output_name=output_name, **video_options)
    report["cell_grid"] = {
        "source": str(Path(raw)),
        "labels": (str(Path(labels)) if isinstance(labels, (str, Path))
                   else "in_memory"),
        "cell_identities": [tile.key for tile in tiles],
        "source_frame_offset": int(source_frame_offset),
        "crop_basis": crop_basis if crop_size_px is None else "pixels",
        "crop": crop if crop_size_px is None else None,
        "crop_size_px": list(crop_size_px) if crop_size_px is not None else None,
        "clock": "original source frame order" if align == "start" else str(align),
        "missing_centre": missing_centre,
        "outline": bool(outline),
        "outline_colour": (list(outline_colour) if not isinstance(outline_colour, str)
                           else outline_colour),
        "outline_width_px": int(outline_width_px),
        "outline_opacity": float(outline_opacity),
    }
    return report
