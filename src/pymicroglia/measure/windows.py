"""Before and after, for every metric, without remeasuring anything.

Every summary this package writes covers the whole recording. If something
changes partway through -- a drug goes in, a stimulus starts -- the before and
the after are averaged into one number and the change partly cancels itself
out. Nothing needs remeasuring to fix that: the per-cell-per-frame table
already holds every frame separately, so a window is a filter on it and a
windowed summary is the existing roll-up run over that filter.

Three tables come out, long rather than wide:

* ``cell_summary_windowed`` - the per-cell roll-up, once per window;
* ``frame_summary_windowed`` - the same for the per-frame roll-up;
* ``window_change`` - one row per cell per window per metric, carrying that
  cell's value against its own value in the named baseline window.

Ported from Motion's ``analysis/windows.py`` on 2026-09-21. The span a window
names is resolved by :func:`auto_organotypic.windows.resolve`, the one
grammar every stage of that package uses, so that a window said in hours is
cut at the same frame here as in a trim or a split there: half-open in time,
inclusive in frames. Motion's own rule -- ``from <= x < to`` on the ``hours``
or ``frame_index`` column -- is what that resolves to, frame for frame.

No pandas at import time: this module is imported by the action registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from auto_organotypic import windows as _spans

from .declare import Output
from .spec import WindowSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

    from .context import MeasurementContext

__all__ = ["WINDOWED", "frame_span", "frame_mask", "window_extent",
           "windowed_summaries", "window"]

#: The three tables this file builds, declared the way ``summarise.ROLLUPS``
#: declares the three it builds.
WINDOWED: tuple[Output, ...] = (
    Output("cell_summary_windowed", grain=("identity", "window")),
    Output("frame_summary_windowed", grain=("frame_index", "window")),
    Output("window_change", grain=("identity", "window", "metric")),
)


def _as_span(window: WindowSpec) -> _spans.Window:
    """A window in Auto-Organotypic's grammar: half-open hours, inclusive frames.

    Motion's frame window is half-open too (``from_frame <= i < to_frame``,
    zero-based), so it is said as the one-based inclusive span
    ``from_frame + 1 .. to_frame``.
    """
    if window.in_frames:
        return _spans.Window(str(int(window.from_frame) + 1), str(int(window.to_frame)))
    return _spans.Window(f"{float(window.from_hours):g}h", f"{float(window.to_hours):g}h")


def frame_span(window: WindowSpec, *, frames: int, minutes_per_frame: float,
               source_frame_offset: int = 0) -> tuple[int, int] | None:
    """Zero-based label frames ``[first, last]`` a window names, or ``None``.

    ``None`` is a window that reaches no frame of this recording: it is
    skipped, and the run record says how many frames it held. Hours count
    from the start of the *source* recording, as Motion's ``hours`` column
    does, so the offset between label frame 0 and source frame 0 is applied.
    """
    interval_s = float(minutes_per_frame) * 60.0
    total = int(frames) + int(source_frame_offset)
    try:
        found = _spans.resolve(_as_span(window), frames=total, interval_s=interval_s)
    except _spans.BadWindow:
        return None
    first = found.first - 1 - int(source_frame_offset)
    last = found.last - 1 - int(source_frame_offset)
    first, last = max(first, 0), min(last, int(frames) - 1)
    if first > last:
        return None
    return first, last


def frame_mask(table: "pd.DataFrame", window: WindowSpec, *,
               minutes_per_frame: float | None = None,
               source_frame_offset: int = 0) -> "np.ndarray":
    """Which rows of a table fall inside one window.

    Half-open, ``from <= x < to``. A window declared in frames is masked on
    ``frame_index`` and one declared in hours on ``hours``, exactly as Motion
    did -- the resolved span is the same set of frames, and masking on the
    column keeps the selection free of any rounding between what the reader
    wrote and what was selected.
    """
    import numpy as np

    if window.in_frames:
        values = table["frame_index"].to_numpy()
        return (values >= window.from_frame) & (values < window.to_frame)
    values = table["hours"].to_numpy()
    return (values >= window.from_hours) & (values < window.to_hours)


def window_extent(window: WindowSpec, context: "MeasurementContext") -> tuple[int, float]:
    """How many of the recording's frames a window holds, and how long it is."""
    span = frame_span(window, frames=context.n_frames,
                      minutes_per_frame=context.scale.minutes_per_frame,
                      source_frame_offset=context.source_frame_offset)
    held = 0 if span is None else span[1] - span[0] + 1
    if window.in_frames:
        length = (window.to_frame - window.from_frame) * context.scale.minutes_per_frame / 60.0
    else:
        length = float(window.to_hours - window.from_hours)
    return int(held), float(length)


def _metric_statistics(summary: "pd.DataFrame") -> list[tuple[str, str, str]]:
    """The ``{metric}_{statistic}`` columns of a roll-up, and their two parts."""
    from .summarise import SUMMARY_METRICS, _STATISTICS

    found = []
    for metric in SUMMARY_METRICS:
        for statistic in _STATISTICS:
            column = f"{metric}_{statistic}"
            if column in summary.columns:
                found.append((column, metric, statistic))
    return found


_CHANGE_COLUMNS = ["identity", "window", "baseline_window", "metric", "statistic",
                   "value", "baseline_value", "change", "ratio"]


def _changes(cell_summary: "pd.DataFrame", windows: list[WindowSpec]) -> "pd.DataFrame":
    """Each window's values against its baseline window's, per cell per metric.

    A cell absent from one of the two windows gets no row at all rather than a
    row of blanks. A ratio against a baseline of zero is undefined, not large.
    """
    import numpy as np
    import pandas as pd

    paired = [w for w in windows if w.baseline is not None]
    if not paired or cell_summary.empty:
        return pd.DataFrame(columns=_CHANGE_COLUMNS)
    columns = _metric_statistics(cell_summary)
    by_window = {name: block.set_index("identity")
                 for name, block in cell_summary.groupby("window", sort=False)}

    rows = []
    for window in paired:
        here = by_window.get(window.name)
        there = by_window.get(window.baseline)
        if here is None or there is None:
            continue
        shared = here.index.intersection(there.index)
        if not len(shared):
            continue
        for column, metric, statistic in columns:
            value = here.loc[shared, column].to_numpy(dtype=float)
            base = there.loc[shared, column].to_numpy(dtype=float)
            ratio = np.divide(value, base, out=np.full_like(value, np.nan),
                              where=base != 0)
            rows.append(pd.DataFrame({
                "identity": shared,
                "window": window.name,
                "baseline_window": window.baseline,
                "metric": metric,
                "statistic": statistic,
                "value": value,
                "baseline_value": base,
                "change": value - base,
                "ratio": ratio,
            }))
    if not rows:
        return pd.DataFrame(columns=_CHANGE_COLUMNS)
    changes = pd.concat(rows, ignore_index=True)
    return changes.sort_values(["identity", "window", "metric", "statistic"]).reset_index(
        drop=True)


def windowed_summaries(
    cell_frame: "pd.DataFrame",
    context: "MeasurementContext",
    windows: list[WindowSpec],
) -> dict[str, "pd.DataFrame"]:
    """The existing roll-ups, once per window, stacked with a ``window`` column.

    The roll-ups are called with an empty table dictionary on purpose: a
    number folded in from a whole-recording table would repeat one answer as
    though it were a windowed one.
    """
    import pandas as pd

    from .summarise import build_cell_summary, build_frame_summary

    if not windows:
        return {}

    cell_rows, frame_rows = [], []
    for window in windows:
        held, length = window_extent(window, context)
        subset = cell_frame[frame_mask(cell_frame, window)]
        if subset.empty:
            # A window that caught nothing contributes nothing; the run
            # record carries every declared window with the frames it held.
            continue

        cells = build_cell_summary(subset, {}, context)
        cells.insert(0, "window", window.name)
        cells["window_frames"] = held
        cells["window_hours"] = length
        cells["window_coverage"] = cells["observed_frames"] / held
        cell_rows.append(cells)

        frames = build_frame_summary(subset, context, {})
        frames = frames[frame_mask(frames, window)].copy()
        frames.insert(0, "window", window.name)
        frames["window_frames"] = held
        frames["window_hours"] = length
        frame_rows.append(frames)

    cell_summary = (pd.concat(cell_rows, ignore_index=True, sort=False)
                    if cell_rows else pd.DataFrame(columns=["window", "identity"]))
    frame_summary = (pd.concat(frame_rows, ignore_index=True, sort=False)
                     if frame_rows else pd.DataFrame(columns=["window", "frame_index"]))
    return {
        "cell_summary_windowed": cell_summary,
        "frame_summary_windowed": frame_summary,
        "window_change": _changes(cell_summary, windows),
    }


# ---------------------------------------------------------------- the action
def window(run_dir, *, windows: Sequence[Mapping[str, Any]] = (),
           verify_hashes: bool = True) -> dict[str, Any]:
    """Compute windowed summaries for a run that already exists.

    Separate from ``measure`` so that windows can be declared after the
    measuring is done, which is the usual order. It re-reads each movie's
    inputs to rebuild the measurement context -- the per-frame roll-up counts
    pixels in the label stack -- but re-measures nothing: the numbers come from
    the run's own ``cell_frame.csv``.

    ``windows`` is a list of window blocks in the configuration's own shape;
    empty means the windows the run's manifest recorded. Refuses a movie whose
    ``windows/<stem>/`` already holds a table, in line with immutable runs.
    """
    import pandas as pd

    from .. import store
    from .inputs import load_movie
    from .run import (MEASURE_FOLDER, WINDOWS_FOLDER, read_manifest,
                      write_manifest, write_windows)
    from .spec import MeasureConfig, MovieSpec, parse_windows

    run = _run_path(run_dir)
    manifest = read_manifest(run)
    settings = dict(manifest.get("settings", {}))
    declared = parse_windows(windows, "windows") if windows else \
        parse_windows(settings.get("windows"), "windows")
    config = MeasureConfig.from_parts(
        [MovieSpec.from_dict(entry["spec"]) for entry in manifest.get("movies", [])],
        frame_interval_min=settings["frame_interval_min"],
        microns_per_pixel=settings.get("microns_per_pixel"),
        calibration_tiffs=settings.get("calibration_tiffs", ()),
        module_params=settings.get("modules", {}),
        verify_hashes=verify_hashes,
        conditions=manifest.get("design"),
        windows=declared,
    )
    if not windows:
        # Only the shared block is replaced by the argument; a movie's own
        # windows, recorded in its spec, still apply when nothing was passed.
        pass
    else:
        for movie in config.movies:
            movie.windows = []

    written: dict[str, dict] = {}
    for entry in manifest.get("movies", []):
        movie = config.movie(entry["stem"])
        applying = config.windows_for(movie)
        if not applying:
            continue
        target = run / WINDOWS_FOLDER / movie.stem
        existing = [name for name in (o.name for o in WINDOWED)
                    if (target / f"{name}.csv").exists()]
        if existing:
            raise FileExistsError(
                f"{target} already holds {existing}; analysis runs are "
                "immutable, delete them deliberately or window a new run")
        context, _ = load_movie(config, movie)
        cell_frame = pd.read_csv(run / MEASURE_FOLDER / movie.stem / "cell_frame.csv")
        condition = str(entry.get("condition", {}).get("condition", ""))
        source = store.fingerprint(movie.labels)
        tables = write_windows(run, movie, context, cell_frame, applying, condition,
                               source=source,
                               params={"run": manifest["run"]["run_label"],
                                       "stem": movie.stem})
        entry.setdefault("tables", {}).update(tables)
        entry["windows"] = [
            {"name": w.name, "baseline": w.baseline, "description": w.description,
             "frames": window_extent(w, context)[0],
             "hours": window_extent(w, context)[1]}
            for w in applying]
        written[movie.stem] = tables
    manifest["settings"]["windows"] = [w.as_dict() for w in declared]
    write_manifest(run, manifest)
    return {"run": str(run), "windows": [w.as_dict() for w in declared],
            "written": written}


def _run_path(run_dir):
    from pathlib import Path

    run = Path(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(f"{run} is not a run folder")
    return run
