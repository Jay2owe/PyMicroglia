"""The quality-control figures the science stages leave notes asking for.

Four figures, each drawn from artefacts a stage already stored, and each
answering a question somebody actually asks before trusting a run:

``registration_figure``
    Did the drift get taken out, and where did it fail? Drawn from the
    per-frame shift and residual table registration writes, never from pixels.
``cosmic_ray_preview``
    Which pixels were replaced? Stage 06 recorded ``"preview_drawn_by": "stage
    09, from the stored mask and event table"`` rather than drawing it there,
    because the mask is the artefact and the picture is a reading of it.
``channel_figure``
    Which channel is which, and do they stay apart? An overlap here is the
    difference between a microglial rhythm and a bleed-through.
``frames_figure``
    The first, middle and last frame of every channel, decoded. Cheap, and it
    catches a wrongly ordered hyperstack before six hours of processing do.

Nothing here computes. Every module under ``visualisation/`` is barred from
importing scipy, scikit-image or any module of this package that *produces* an
artefact; these read what a stage stored and draw it. The two transforms that
do appear — a percentile display range and ``log1p`` — change what a reader
sees and never a number that leaves the module. Both are the engines' own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from . import panels as _panels

__all__ = [
    "DLUC_CMAP",
    "STRUCTURAL_CMAP",
    "READS",
    "registration_figure",
    "cosmic_ray_preview",
    "channel_figure",
    "frames_figure",
]

#: The stage names of the artefacts these figures read, spelled out rather than
#: imported. Importing ``filtering`` to learn what it calls its own stage would
#: make a figure depend on the module that *produces* the artefact, which is the
#: one thing this package's layout exists to prevent. The cost is that the two
#: could drift apart, so a test asserts every name here still matches the
#: constant the producing module exports.
READS: dict[str, str] = {
    "registration": "registration",
    "cosmic_mask": "cosmic_rays",
    "cosmic_events": "cosmic_rays_events",
}

#: Colormap names, not colour tables. Matplotlib owns these; the house palette
#: has no opinion about a continuous map and inventing one here would be a
#: local palette by another name.
DLUC_CMAP = "magma"
STRUCTURAL_CMAP = "bone"

_PREVIEW_PERCENTILES = (0.5, 99.5)


def _series(source):
    """The pixels, opened read-only.

    Reading a file is retrieval, not computation: a figure is allowed to open
    what it draws. Producing an artefact is what it may not do.
    """
    from .. import series

    return series.open_series(source)


def _stored(stage: str, source, *, required: bool = True):
    from .. import store

    return store.resolve(stage, source, required=required)


def _channels(opened, channels) -> dict[str, int | None]:
    """Which channel is which, from an override, a stored decision or the stats.

    ``metadata`` describes a file; it produces no artefact of its own, so a
    figure may ask it. Falling back to positional names keeps a figure drawable
    on a stack nobody has assigned yet, which is exactly when somebody wants to
    look at one.
    """
    from .. import metadata

    if isinstance(channels, Mapping):
        assignment = dict(channels)
    elif channels:
        assignment = dict(metadata.parse_channel_spec(channels, opened.shape[1]))
    else:
        assigned = metadata.assign_channels(opened)
        assignment = {"dluc": assigned.dluc, "struct": assigned.struct,
                      "bf": assigned.bf}
    if not any(index is not None for index in assignment.values()):
        return {f"channel_{i}": i for i in range(opened.shape[1])}
    return assignment


def _ordered(assignment: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Named channels in file order, skipping the roles this stack has not got."""
    return [(name, int(index))
            for name, index in sorted(
                ((n, i) for n, i in assignment.items() if i is not None),
                key=lambda pair: pair[1])]


def _numbers(column: Sequence[Any]) -> list[float]:
    """A stored table column as floats. Tables come back as text-typed rows."""
    out = []
    for value in column:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


# ------------------------------------------------------------- registration
def registration_figure(source, *, output_dir=None, output_name=None,
                        overwrite: bool = False, shifts=None,
                        theme: str = _panels.HOUSE_THEME,
                        fig_width_in: float = 11.0,
                        panel_height_in: float = 2.2, dpi: int = 150,
                        claim: str = "") -> dict[str, Any]:
    """Drift per frame, and what was left after correcting it.

    Three rows: the x and y shift applied to each frame, the residual
    misalignment after correction, and the frame-to-frame jump. The residual is
    the one that matters — a large shift is a stage that moved, a large
    residual is a registration that did not work.
    """
    stored = _stored(READS["registration"], source) if shifts is None else None
    table = stored.load() if stored is not None else dict(shifts)

    frames = _numbers(table.get("frame", range(len(next(iter(table.values()))))))
    rows = [
        ("shift, px", [("shift_x_px", "dluc"), ("shift_y_px", "rfp")],
         "applied translation"),
        ("residual, px", [("residual_x_px", "gfp"), ("residual_y_px", "cyan")],
         "what correction left behind"),
        ("|residual|, px", [("residual_magnitude_px", "plum")],
         "the number the pass/fail threshold is read from"),
        ("phase peak", [("phase_peak_quality", "olive")],
         "how confident each frame's alignment was"),
    ]
    present = [row for row in rows
               if any(name in table for name, _ in row[1])]
    stack = _panels.stack(len(present), height_per_panel=panel_height_in,
                          width=fig_width_in, head_in=0.8, theme=theme)

    drawn: dict[str, list[float]] = {"frame": frames}
    for axis, (label, series, note) in zip(stack, present):
        for column, colour_name in series:
            if column not in table:
                continue
            values = _numbers(table[column])
            axis.plot(frames, values, lw=1.6,
                      color=_panels.colour(colour_name), label=column)
            drawn[column] = values
        _panels.zero_line(axis)
        _panels.label(axis, y=label, title=note)
        if len(series) > 1:
            _panels.legend(axis, columns=len(series))

    _panels.label(stack.last, x="frame")
    stack.tighten()
    stack.title("Registration", subtitle="translation applied, and what it left")
    return _panels.save_for(
        stack.figure, source, drawn, stage="registration_figure",
        output_dir=output_dir, output_name=output_name or "registration_qc",
        overwrite=overwrite, dpi=dpi, artefacts=[stored],
        claim=claim or "registration removed the drift it measured",
        settings={"theme": theme})


# --------------------------------------------------------------- cosmic ray
def cosmic_ray_preview(source, *, output_dir=None, output_name=None,
                       overwrite: bool = False, frame: int | None = None,
                       channel: int = 0, cleaned=None, mask=None,
                       theme: str = _panels.HOUSE_THEME, dpi: int = 150,
                       fig_width_in: float = 15.0, fig_height_in: float = 5.4,
                       claim: str = "") -> dict[str, Any]:
    """Before, after, and exactly which pixels changed.

    The frame shown is the one the stored event table says lost the most
    pixels, because a preview of a quiet frame proves nothing. The right-hand
    panel is the replacement mask itself, so a reader can see whether the
    filter took cosmic rays or took cells.
    """
    import numpy as np

    events = _stored(READS["cosmic_events"], source, required=False)
    stored_mask = _stored(READS["cosmic_mask"], source, required=False)
    counts = events.load() if events is not None else {}

    per_frame = _events_per_frame(counts)
    if frame is None:
        frame = _busiest_frame(per_frame)
    frame = int(frame)

    with _series(source) as opened:
        before = np.asarray(opened.frame(frame, channel), float)
    after = _frame_of(cleaned, frame, channel)
    replaced = _frame_of(mask, frame, 0)
    if replaced is None and stored_mask is not None:
        replaced = _frame_of(stored_mask.load(), frame, 0)

    images = [(before, "before", DLUC_CMAP)]
    if after is not None:
        images.append((after, "after", DLUC_CMAP))
    if replaced is not None:
        images.append((np.asarray(replaced, float), "replaced pixels", "gray"))

    grid = _panels.grid(1, len(images), width=fig_width_in,
                        height=fig_height_in, theme=theme)
    low, high = np.percentile(before, _PREVIEW_PERCENTILES)
    for axis, (array, title, cmap) in zip(grid, images):
        if title == "replaced pixels":
            _panels.image(axis, array, cmap=cmap, vmin=0.0, vmax=1.0,
                          title=title)
        else:
            _panels.image(axis, array, cmap=cmap, vmin=low, vmax=high,
                          title=title)
    grid.figure.tight_layout()
    changed = 0 if replaced is None else int(np.count_nonzero(replaced))
    grid.title(f"Cosmic-ray removal, frame {frame}",
               subtitle=f"{changed} pixels replaced in this frame")

    table = {
        "frame": [float(f) for f in per_frame["frame"]],
        "events": [float(v) for v in per_frame["events"]],
        "pixels_replaced": [float(v) for v in per_frame["pixels"]],
        "shown": [1.0 if int(f) == frame else 0.0
                  for f in per_frame["frame"]],
    }
    if not per_frame["frame"]:
        table = {"frame": [float(frame)], "events": [0.0],
                 "pixels_replaced": [float(changed)], "shown": [1.0]}
    return _panels.save_for(
        grid.figure, source, table, stage="cosmic_ray_preview",
        output_dir=output_dir,
        output_name=output_name or "cosmic_ray_preview", overwrite=overwrite,
        dpi=dpi, artefacts=[events, stored_mask],
        claim=claim or "the cosmic-ray filter replaced isolated spikes and "
                       "not cells",
        settings={"frame": frame, "channel": channel, "theme": theme})


def _events_per_frame(counts: Mapping[str, Sequence[Any]]) -> dict[str, list]:
    """The event table folded to one row per frame.

    The stored table is one row per replaced spike — frame, peak position, area,
    the value before and after. What a preview needs is how many landed on each
    frame, which is a regrouping of stored rows and not a new measurement.
    """
    frames = counts.get("frame_zero_based") or counts.get("frame") or []
    areas = counts.get("grown_area_px") or [1] * len(list(frames))
    tally: dict[int, list[float]] = {}
    for frame, area in zip(_numbers(frames), _numbers(areas)):
        row = tally.setdefault(int(frame), [0.0, 0.0])
        row[0] += 1.0
        row[1] += area
    order = sorted(tally)
    return {"frame": order,
            "events": [tally[f][0] for f in order],
            "pixels": [tally[f][1] for f in order]}


def _busiest_frame(per_frame: Mapping[str, Sequence[float]]) -> int:
    """The frame that lost the most pixels. A preview of a quiet frame proves nothing."""
    pixels = list(per_frame.get("pixels", ()))
    if not pixels:
        return 0
    best = max(range(len(pixels)), key=lambda i: pixels[i])
    return int(per_frame["frame"][best])


def _frame_of(value, frame: int, channel: int):
    """One plane out of an array, a path, or ``None``."""
    import numpy as np

    if value is None:
        return None
    if isinstance(value, (str, Path)):
        with _series(value) as opened:
            return np.asarray(opened.frame(frame, channel), float)
    array = np.asarray(value)
    if array.ndim == 4:
        return np.asarray(array[frame, channel], float)
    if array.ndim == 3:
        return np.asarray(array[frame], float)
    return np.asarray(array, float)


# ------------------------------------------------------------------ channels
def channel_figure(source, *, output_dir=None, output_name=None,
                   overwrite: bool = False, channels=None, frame: int = 0,
                   theme: str = _panels.HOUSE_THEME, dpi: int = 150,
                   fig_width_in: float = 15.0, fig_height_in: float = 5.4,
                   claim: str = "") -> dict[str, Any]:
    """Which channel is which, and how far apart their intensities sit.

    The lab has already had a run in which the structural and the
    bioluminescence channel were the other way round, and the rhythm that came
    out of it was the microscope's. This is the cheapest check that catches it:
    the frames side by side, and the intensity histogram of each.
    """
    import numpy as np

    with _series(source) as opened:
        assignment = _channels(opened, channels)
        planes = {name: np.asarray(opened.frame(int(frame), int(index)), float)
                  for name, index in _ordered(assignment)}

    names = list(planes)
    grid = _panels.grid(1, len(names) + 1, width=fig_width_in,
                        height=fig_height_in, theme=theme)
    cycle = _panels.colours("semantic", max(len(names), 1))
    for axis, name in zip(grid, names):
        _panels.image(axis, planes[name], cmap=DLUC_CMAP,
                      title=f"{name} (channel {assignment[name]})")

    table: dict[str, list[float]] = {}
    histogram = grid[len(names)]
    for position, name in enumerate(names):
        values = planes[name].ravel()
        counts, edges = np.histogram(values, bins=80)
        centres = 0.5 * (edges[:-1] + edges[1:])
        histogram.plot(centres, counts, lw=1.8, color=cycle[position],
                       label=name)
        table[f"{name}_intensity"] = [float(v) for v in centres]
        table[f"{name}_pixels"] = [float(v) for v in counts]
    histogram.set_yscale("log")
    _panels.label(histogram, x="intensity", y="pixels",
                  title="separation between channels")
    _panels.legend(histogram, columns=1)
    grid.figure.tight_layout()
    grid.title("Channel identity and separation",
               subtitle=f"frame {frame}")

    return _panels.save_for(
        grid.figure, source, table, stage="channel_figure",
        output_dir=output_dir, output_name=output_name or "channel_identity",
        overwrite=overwrite, dpi=dpi,
        claim=claim or "the channels are the ones named and stay apart",
        settings={"channels": dict(assignment), "frame": int(frame),
                  "theme": theme})


# -------------------------------------------------------------------- frames
def frames_figure(source, *, output_dir=None, output_name=None,
                  overwrite: bool = False, channels=None,
                  theme: str = _panels.HOUSE_THEME, dpi: int = 150,
                  fig_width_in: float = 16.0, fig_height_in: float = 9.0,
                  claim: str = "") -> dict[str, Any]:
    """First, middle and last frame of every channel, decoded.

    One row per time point, one column per channel, with the hour on each
    panel. It costs three plane reads and catches a hyperstack whose axes were
    read in the wrong order before anything expensive runs on it.
    """
    import numpy as np

    with _series(source) as opened:
        count = opened.shape[0]
        chosen = sorted({0, count // 2, max(count - 1, 0)})
        assignment = _channels(opened, channels)
        order = _ordered(assignment)
        times = opened.meta.times_h
        planes = {(t, index): np.asarray(opened.frame(t, index), float)
                  for t in chosen for _, index in order}
        channel_count = opened.shape[1]

    grid = _panels.grid(len(chosen), len(order), width=fig_width_in,
                        height=fig_height_in, theme=theme, constrained=True)
    table: dict[str, list[float]] = {"frame": [float(t) for t in chosen]}
    for row, t in enumerate(chosen):
        hour = float(times[t]) if times is not None and t < len(times) else float("nan")
        for column, (name, index) in enumerate(order):
            axis = grid[row * len(order) + column]
            plane = planes[(t, index)]
            _panels.image(axis, plane, cmap=DLUC_CMAP,
                          title=f"{name}, frame {t}\n{hour:.2f} h")
            table.setdefault(f"{name}_mean", []).append(float(plane.mean()))
            table.setdefault(f"{name}_max", []).append(float(plane.max()))
    grid.title("First, middle and last frame",
               subtitle=f"{len(order)} channels, {count} frames")

    return _panels.save_for(
        grid.figure, source, table, stage="frames_figure",
        output_dir=output_dir,
        output_name=output_name or "frames_first_mid_last",
        overwrite=overwrite, dpi=dpi,
        claim=claim or "the stack decodes to the channels and times its "
                       "metadata claims",
        settings={"frames": [int(t) for t in chosen],
                  "channels": dict(assignment), "theme": theme})


# --------------------------------------------------------------------- saving
# Every figure here leaves by ``panels.save_for``: same default folder, same
# provenance bundle, same closing of the figure. There is no second save path
# in this module and there must not be one.
