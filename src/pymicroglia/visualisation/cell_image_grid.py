"""A still grid of tracked cells, drawn by Auto-Organotypic's shared renderer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from auto_organotypic import grid, timebase

from .cell_tiles import cell_tiles


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
                    source_frame_offset: int = 0,
                    crop_basis: str = "largest_cell", crop: str = "tight",
                    crop_size_px: tuple[int, int] | None = None,
                    crop_rectangle_px: tuple[int, int] | None = None,
                    shared_time: str | None = None,
                    event_hour: float | None = None,
                    max_trace_gap_h: float = 4.0,
                    trace_channel: int = 1,
                    display_options: Mapping[str, Any] | None = None,
                    **grid_options) -> dict[str, Any]:
    """Draw all identities in the supplied images label view as grid rows.

    Default columns show each cell's own best scored cycle on circadian time.
    ``shared_time='recording'`` compares source hours; ``'event'`` compares
    hours relative to ``event_hour``. All other visual options pass directly to
    :func:`auto_organotypic.grid.stack_to_grid`.
    """
    if shared_time not in (None, "recording", "event"):
        raise ValueError("shared_time must be None, 'recording' or 'event'")
    if shared_time == "event" and event_hour is None:
        raise ValueError("shared_time='event' needs event_hour in source hours")
    if crop_rectangle_px is not None:
        if crop_size_px is not None:
            raise ValueError("give crop_size_px or crop_rectangle_px, not both")
        crop_size_px = crop_rectangle_px
    grid_options = {**dict(display_options or {}), **grid_options}
    if not np.isfinite(float(max_trace_gap_h)) or float(max_trace_gap_h) < 0:
        raise ValueError("max_trace_gap_h must be nonnegative")
    tiles = cell_tiles(
        raw, labels, source_frame_offset=source_frame_offset,
        frame_interval_h=grid_options.get("frame_interval_h"),
        crop_basis=crop_basis, crop=crop, crop_size_px=crop_size_px,
        trace_channel=trace_channel)
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
    }
    return report
