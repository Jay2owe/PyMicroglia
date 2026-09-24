"""A still grid of tracked cells, drawn by Auto-Organotypic's shared renderer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from auto_organotypic import grid, timebase
from auto_organotypic.render.outlines import (DEFAULT_COLOUR,
                                              DEFAULT_OPACITY,
                                              DEFAULT_WIDTH_PX)

from .cell_tiles import cell_tiles
from .cell_selection import select_cell_tiles
from .cell_display import prepare_cell_display


def _runs(indices: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in indices:
        index = int(value)
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def _cycle_trace(tile, *, first_frame: int, frames: int, period_h: float,
                 max_gap_h: float) -> tuple[Any, dict[str, Any], Any]:
    times, values = (np.asarray(one, float) for one in tile.trace)
    interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    if interval <= 0:
        raise ValueError("cell cycle grid needs a positive photon-frame interval")
    window = np.arange(int(first_frame), len(times) if frames < 0 else
                       min(len(times), int(first_frame) + int(frames)))
    if not window.size:
        raise ValueError("the selected frame window is empty")
    missing = ~np.isfinite(values)
    short, invalid = [], []
    known = np.flatnonzero(~missing)
    for run in _runs(np.flatnonzero(missing)):
        internal = bool(known.size and run[0] > known[0] and run[-1] < known[-1])
        if internal and len(run) * interval <= float(max_gap_h):
            short.extend(run)
        else:
            invalid.extend(run)
    filled = (np.interp(np.arange(len(values)), known, values[known])
              if known.size else np.zeros(len(values), float))
    detail: dict[str, Any] = {
        "interpolated_trace_frames": short,
        "ineligible_trace_frames": invalid,
        "trace_use": "display-only cycle selection; image frames never interpolated",
    }
    selected = None
    if known.size >= 4:
        try:
            chosen = timebase.choose_day(
                filled[window], times[window], period_h=period_h,
                skip_settling=bool(int(window[0]) == 0))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            chosen = None
        if chosen is not None:
            invalid_in_window = {index - int(window[0]) for index in invalid
                                 if index in window}
            eligible = [one for one in chosen["cycles"]
                        if float(one["score"]) > 0 and not any(
                            frame in invalid_in_window for frame in range(
                                int(one["frames"][0]), int(one["frames"][1]) + 1))]
            if int(window[0]) == 0:
                nonsettling = [one for one in eligible if not one["settling"]]
                eligible = nonsettling or eligible
            selected = max(eligible, key=lambda one: one["score"], default=None)
            if selected is not None:
                detail.update({"chosen_cycle": int(selected["cycle"]),
                               "score": float(selected["score"]),
                               "selected_trace_frames": [
                                   int(window[0] + selected["frames"][0]),
                                   int(window[0] + selected["frames"][1])],
                               "origin_h": float(times[window[0]] + selected["start_hours"]),
                               "rule": timebase.DEFAULT_DAY_RULE})
    if selected is None:
        detail["status"] = "cycle unavailable"
        alignment = float(times[window[0]])
        unavailable = {index: "cycle unavailable" for index in range(len(values))}
    else:
        detail["status"] = "selected"
        # Let the upstream renderer make and record the same choice whenever
        # no long tracking gap disqualifies its winner.
        alignment = ("best" if not invalid else detail["origin_h"])
        unavailable = dict(tile.unavailable_frames)
    prepared = replace(tile, trace=(times, filled),
                       unavailable_frames=unavailable,
                       provenance={**tile.provenance, "cycle_selection": detail})
    return alignment, detail, prepared


def cell_image_grid(raw, labels, *, output_dir=None, output_name=None,
                    overwrite: bool = False,
                    source_frame_offset: int = 0,
                    crop_basis: str = "largest_cell", crop: str = "tight",
                    crop_rectangle_px: tuple[int, int] | None = None,
                    fill_tile: bool = True,
                    shared_time: str | None = None,
                    event_hour: float | None = None,
                    max_trace_gap_h: float = 4.0,
                    trace_channel: int = 1,
                    centre_method: str = "intensity_weighted",
                    significant_period_only: bool = False,
                    period_recipe: Mapping[str, Any] | str | Path | None = None,
                    period_decisions: Mapping[int | str, Mapping[str, Any]] | None = None,
                    max_gap_frames: int | None = None,
                    max_missing_frames: int | None = None,
                    max_missing_fraction: float | None = None,
                    show_outline: bool = True,
                    outline_colour: Any = DEFAULT_COLOUR,
                    outline_width_px: int = DEFAULT_WIDTH_PX,
                    outline_opacity: float = DEFAULT_OPACITY,
                    mask_style: str = "outline",
                    mask_opacity: float = 0.35,
                    display_filter: str | Mapping[str, Any] | None = None,
                    scale_bar: bool = True,
                    um_per_px: float | None = None,
                    scale_bar_um: float | None = None,
                    display_options: Mapping[str, Any] | None = None,
                    **grid_options) -> dict[str, Any]:
    """Draw all identities in the supplied images label view as grid rows.

    Default columns show each cell's own best scored cycle on circadian time.
    ``shared_time='recording'`` compares source hours; ``'event'`` compares
    hours relative to ``event_hour``. All other visual options pass directly to
    :func:`auto_organotypic.grid.stack_to_grid`. Optional quality limits and
    ``significant_period_only`` select rows before the largest-cell crop is
    sized; the default keeps every identity. The observed mask is outlined by
    default. ``mask_style='fill'`` tints its interior with ``mask_opacity``.
    ``display_filter={'method': 'a104'}`` filters only rendered full frames;
    cell traces and cycle selection remain on the original photons.
    Still crops use each observed mask's intensity-weighted centroid by default.
    The original source frame remains the picture at each selected time.
    """
    if shared_time not in (None, "recording", "event"):
        raise ValueError("shared_time must be None, 'recording' or 'event'")
    if shared_time == "event" and event_hour is None:
        raise ValueError("shared_time='event' needs event_hour in source hours")
    grid_options = {**dict(display_options or {}), **grid_options}
    fill_tile = grid_options.pop("fill_tile", fill_tile)
    grid_options.setdefault("scale_bar", scale_bar)
    if um_per_px is not None:
        grid_options.setdefault("um_per_px", um_per_px)
    if scale_bar_um is not None:
        grid_options.setdefault("scale_bar_um", scale_bar_um)
    source_um_per_px = grid_options.pop("um_per_px", None) if fill_tile else None
    show_outline = grid_options.pop("show_outline", show_outline)
    show_outline = grid_options.pop("outline", show_outline)
    outline_colour = grid_options.pop("outline_colour", outline_colour)
    outline_width_px = grid_options.pop("outline_width_px", outline_width_px)
    outline_opacity = grid_options.pop("outline_opacity", outline_opacity)
    mask_style = grid_options.pop("mask_style", mask_style)
    mask_opacity = grid_options.pop("mask_opacity", mask_opacity)
    display_filter = grid_options.pop("display_filter", display_filter)
    centre_method = grid_options.pop("centre_method", centre_method)
    crop_size_px = grid_options.pop("crop_size_px", None)
    if crop_rectangle_px is not None:
        if crop_size_px is not None:
            raise ValueError("give crop_size_px or crop_rectangle_px, not both")
        crop_size_px = crop_rectangle_px
    grid_options.setdefault("overwrite", overwrite)
    if not np.isfinite(float(max_trace_gap_h)) or float(max_trace_gap_h) < 0:
        raise ValueError("max_trace_gap_h must be nonnegative")
    selected, selection = None, None
    if (significant_period_only or period_recipe is not None or
            period_decisions is not None or
            max_gap_frames is not None or max_missing_frames is not None or
            max_missing_fraction is not None):
        candidates = cell_tiles(
            raw, labels, source_frame_offset=source_frame_offset,
            frame_interval_h=grid_options.get("frame_interval_h"),
            crop_basis="own_cell", trace_channel=trace_channel)
        selected, selection = select_cell_tiles(
            candidates, raw=raw, significant_period_only=significant_period_only,
            period_recipe=period_recipe, period_decisions=period_decisions,
            max_gap_frames=max_gap_frames,
            max_missing_frames=max_missing_frames,
            max_missing_fraction=max_missing_fraction)
    display_raw, a104_range, display_report = prepare_cell_display(
        raw, display_filter)
    if a104_range is not None:
        grid_options.setdefault("display_range", a104_range)
        grid_options.setdefault("lut", "dluc_purple")
    tiles = cell_tiles(
        raw, labels, source_frame_offset=source_frame_offset,
        frame_interval_h=grid_options.get("frame_interval_h"),
        display_raw=display_raw,
        include_identities=selected,
        crop_basis=crop_basis, crop=crop, crop_size_px=crop_size_px,
        fill_tile=fill_tile, um_per_px=source_um_per_px,
        trace_channel=trace_channel, centre_method=centre_method,
        outline=show_outline,
        outline_colour=outline_colour, outline_width_px=outline_width_px,
        outline_opacity=outline_opacity, mask_style=mask_style,
        mask_opacity=mask_opacity)
    if display_raw is not None and tiles:
        grid_options.setdefault("frame_interval_h",
                                tiles[0].provenance["frame_interval_h"])
    defaults: dict[str, Any] = {"moments": 6}
    if "when" in grid_options and "moments" not in grid_options:
        defaults.pop("moments")
    selections = []
    if shared_time is None:
        defaults.update(time_scale="ct", between="cycle")
        settings = {**defaults, **grid_options}
        period = settings.get("period_h") or timebase.DEFAULT_PERIOD_H
        if isinstance(period, str):
            if period != "fit":
                raise ValueError("period_h must be numeric or 'fit'")
            fitted = []
            for tile in tiles:
                times, values = tile.trace
                finite = np.isfinite(values)
                if finite.sum() >= 4:
                    filled = np.interp(np.arange(len(values)), np.flatnonzero(finite),
                                       np.asarray(values)[finite])
                    fitted.append(timebase.fitted_period_hours(filled, times))
            period = float(np.median(fitted)) if fitted else timebase.DEFAULT_PERIOD_H
            settings["period_h"] = period
        alignments = {}
        prepared = []
        for tile in tiles:
            alignment, detail, ready = _cycle_trace(
                tile, first_frame=int(settings.get("first_frame", 0)),
                frames=int(settings.get("frames", -1)), period_h=float(period),
                max_gap_h=float(max_trace_gap_h))
            alignments[str(tile)] = alignment
            selections.append({"identity": tile.key, **detail})
            prepared.append(ready)
        settings.setdefault("align", alignments)
        if all(one["status"] == "cycle unavailable" for one in selections):
            settings.update(time_scale="elapsed", between="window", align="start")
    else:
        settings = {**defaults, "between": "window", **grid_options}
        settings["time_scale"] = "elapsed"
        settings["align"] = (0.0 if shared_time == "recording"
                             else float(event_hour))
        if shared_time == "event":
            settings.setdefault("timestamp_format", "Event {total_hours:+.1f} h")
        prepared = tiles
    report = grid.stack_to_grid(
        prepared, output_dir=output_dir, output_name=output_name, **settings)
    report["cell_grid"] = {
        "source": str(Path(raw)),
        "labels": (str(Path(labels)) if isinstance(labels, (str, Path))
                   else "in_memory"),
        "shared_time": shared_time or "own_best_cycle",
        "event_hour": float(event_hour) if event_hour is not None else None,
        "cycle_selection": selections,
        "cell_identities": [tile.key for tile in tiles],
        "source_frame_offset": int(source_frame_offset),
        "max_trace_gap_h": float(max_trace_gap_h),
        "centre_method": centre_method,
        "fill_tile": bool(fill_tile),
        "mask_style": mask_style if show_outline else "none",
        "outline_colour": (list(outline_colour) if not isinstance(outline_colour, str)
                           else outline_colour),
        "outline_width_px": int(outline_width_px),
        "outline_opacity": float(outline_opacity),
        "mask_opacity": float(mask_opacity),
        "display_filter": display_report,
        "selection": selection,
    }
    return report
