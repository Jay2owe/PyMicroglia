"""The exact table a trace panel draws, and the panel action that assembles it.

This is the computing half of the trace panel; ``visualisation.traces`` is the
drawing half. They are apart because no module under ``visualisation/`` may
compute — a figure that cannot compute cannot grow a private helper stack, and
``PyFLASH/plotting.py`` reached 31,497 lines by growing one per plot.

So the arrow points this way and only this way: a science module may draw, a
figure module may not measure. Everything in here — reading the CSVs, choosing
which trace goes on which panel, the rolling or polynomial baseline, the
normalisation, the display smoothing, the standard deviation over the
trustworthy interior — happens before a figure exists, and the result is handed
over as a :class:`~pymicroglia.visualisation.traces.PanelTable`.

The maths is not reimplemented here. ``rolling_baseline``, ``polynomial_baseline``
and ``window_length`` are ``pymicroglia.tracing``'s, which are the ones the
pipeline already measures with; a second copy would eventually disagree with the
first.

Ported from ``trace_panel_figure.py``. Its settings keep their internal names —
``window_h``, ``cut_stage``, ``show_raw`` — because ``example_panels.json`` and
every saved spec beside it write settings under exactly those keys, and a rename
would turn a saved run into an error message.

Three defaults decide whether the figure is honest, and all three are the
engine's:

``normalise="window_mean"``
    Divides by each trace's mean over the plotted window. ``"baseline"`` is the
    textbook dF/F and is singular wherever the rolling baseline nears zero; on
    the validation recording one cell's baseline fell to 5% of its mean and
    produced a +634% spike that flattened its whole panel.
``cut_stage="after"``
    Detrend on everything, then crop. ``"before"`` re-detrends inside the crop,
    so the ends you kept become the untrustworthy ones.
``shade_edges=True``
    The grey blocks mark where the baseline came from a truncated window.
    Hiding them does not make those ends comparable to the middle.
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from auto_organotypic import baselines as _baselines
from auto_organotypic import conventions as _conventions

from . import tracing as _tracing

__all__ = [
    "METHOD_VERSION",
    "PANEL_STAGE",
    "read_trace_csv",
    "parse_panel",
    "auto_panels",
    "pretty_label",
    "detrend",
    "normalise",
    "ticks_between",
    "describe",
    "prepare",
    "trace_panel",
]

METHOD_VERSION = "2026-08-19-window-mean-panel-table"
PANEL_STAGE = "trace_panel"

#: The engine's own defaults, under the engine's own keys. A JSON spec's
#: ``settings`` block writes into this dictionary, which is why the names here
#: must not drift from the ones in ``example_panels.json``.
DEFAULTS: dict[str, Any] = {
    "time_column": "hours",
    "method": "rolling",
    "window_h": 24.0,
    "degree": 6,
    "normalise": "window_mean",
    "as_percent": True,
    "scale": 1.0,
    "poly_edge_h": 12.0,
    "time_start": None,
    "time_end": None,
    "cut_stage": "after",
    "smooth": 3,
    "show_raw": True,
    "raw_colour": "raw",
    "raw_linewidth": 0.6,
    "trace_linewidth": 2.1,
    "default_colour": "dluc",
    "cycle": [],
    "overflow_cmap": "turbo",
    "shade_edges": True,
    "shade_colour": "shade",
    "show_zero_line": True,
    "show_sd": True,
    "share_y": False,
    "xtick_interval_h": 24.0,
    "xtick_origin_h": 0.0,
    "xtick_hours": [],
    "vline_interval_h": 24.0,
    "vline_origin_h": 0.0,
    "vline_hours": [],
    "vline_colour": "tab:green",
    "vline_alpha": 0.45,
    "vline_linewidth": 0.7,
    "fig_width_in": 11.5,
    "panel_height_in": 2.05,
    "title_height_in": 1.0,
    "title": "",
    "subtitle": "",
    "xlabel": "",
    "ylabel_suffix": "",
    "legend_when_merged": True,
    "dpi": 150,
}


# ------------------------------------------------------------------ loading
def read_trace_csv(path, time_column: str = "hours", offset_h: float = 0.0
                   ) -> tuple[Any, dict[str, Any]]:
    """A wide numeric CSV as ``(times, {column: values})``.

    ``utf-8-sig`` because a CSV that has been through Excel carries a byte
    order mark, and a leading ``#`` is stripped because several of the
    protocols write their header commented.
    """
    import numpy as np

    path = Path(path)
    with open(path, "r", encoding="utf-8-sig") as handle:
        header = handle.readline().rstrip("\n\r")
    names = [c.strip().lstrip("#").strip() for c in header.split(",")]
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if time_column not in names:
        shown = ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
        raise ValueError(f"{path.name} has no {time_column!r} column. "
                         f"Columns: {shown}")
    index = names.index(time_column)
    times = data[:, index] + float(offset_h)
    columns = {name: data[:, i] for i, name in enumerate(names) if i != index}
    return times, columns


def load_sources(entries: Sequence[Any], time_column: str, offset_h: float
                 ) -> dict[str, dict[str, Any]]:
    """``NAME=path`` handles resolved to loaded sources, in the order given.

    The handle is split off only when it is a bare word, so a drive-letter path
    is never split on its own separator: a handle holds no colon or slash, and
    ``C:`` fails on the colon. A handle may start with a digit — recording
    names such as ``20260810_1432`` are real.
    """
    sources: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        text = str(entry)
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)=(.+)$", text)
        name, raw = ((match.group(1), match.group(2))
                     if match and not Path(text).exists()
                     else (f"s{position + 1}", text))
        path = Path(raw.strip()).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        times, columns = read_trace_csv(path, time_column, offset_h)
        sources[name] = {"path": path, "t": times, "columns": columns,
                         "time_column": time_column, "offset": float(offset_h)}
    return sources


# --------------------------------------------------------------- panel spec
TRACE_RE = re.compile(
    r"^\s*(?:(?P<source>[A-Za-z0-9_.-]+)\s*:\s*)?"           # optional source:
    r"(?P<column>[^#\[\]]+?)"                                # column or glob
    r"(?:\s*#(?P<colour>[0-9A-Fa-f]{3,8}|[A-Za-z:. ]+?))?"   # optional #colour
    r"(?:\s*\[(?P<legend>[^\]]*)\])?\s*$")                   # optional [legend]


def pretty_label(column: str) -> str:
    """``cell_1_processed`` -> ``cell 1``; ``whole_field_dFF`` -> ``whole field``."""
    stem = re.sub(r"_(processed|raw|baseline|dFF|dff)$", "", column)
    return stem.replace("_", " ")


def parse_panel(text: str, sources: Mapping[str, Any], index: int
                ) -> dict[str, Any]:
    """One panel string parsed.

    Grammar: ``[label =] trace [+ trace ...]`` where a trace is
    ``[source:]column[#colour][[legend]]``. A glob inside one panel *merges*
    every match onto that panel's axis; ``auto_columns`` instead gives every
    match its own panel.
    """
    label, body = "", text
    if "=" in text:
        head, rest = text.split("=", 1)
        if ":" not in head and "+" not in head and "#" not in head:
            label, body = head.strip(), rest

    traces: list[dict[str, Any]] = []
    for piece in body.split("+"):
        if not piece.strip():
            continue
        found = TRACE_RE.match(piece)
        if not found:
            raise ValueError(
                f"cannot parse trace {piece.strip()!r} in panel {text!r}. "
                f"Expected [source:]column[#colour][[legend]]")
        parts = found.groupdict()
        name = parts["source"] or (list(sources)[0] if len(sources) == 1
                                   else None)
        if name is None:
            raise ValueError(
                f"trace {piece.strip()!r} names no source and there is more "
                f"than one CSV. Prefix it, e.g. "
                f"{list(sources)[0]}:{parts['column'].strip()}")
        if name not in sources:
            raise ValueError(f"unknown source {name!r}. "
                             f"Known: {', '.join(sources)}")
        pattern = parts["column"].strip()
        available = sources[name]["columns"]
        matched = [c for c in available if fnmatch.fnmatchcase(c, pattern)] or (
            [pattern] if pattern in available else [])
        if not matched:
            shown = ", ".join(list(available)[:10])
            raise ValueError(f"no column matches {pattern!r} in source "
                             f"{name!r}. Available: {shown}"
                             + (" ..." if len(available) > 10 else ""))
        colour = parts["colour"]
        if colour and re.fullmatch(r"[0-9A-Fa-f]{3,8}", colour):
            colour = "#" + colour
        for column in matched:
            traces.append({
                "source": name, "column": column, "colour": colour,
                "legend": (parts["legend"] if parts["legend"] is not None
                           else (column if len(matched) > 1 else None))})

    if not traces:
        raise ValueError(f"panel {text!r} selects no trace.")
    if not label:
        label = (pretty_label(traces[0]["column"]) if len(traces) == 1
                 else f"panel {index + 1}")
    return {"label": label, "traces": traces}


def auto_panels(sources: Mapping[str, Any], pattern: str) -> list[dict[str, Any]]:
    """One panel per matching column, in file order."""
    panels = []
    for name, source in sources.items():
        for column in source["columns"]:
            if fnmatch.fnmatchcase(column, pattern):
                panels.append({"label": pretty_label(column),
                               "traces": [{"source": name, "column": column,
                                           "colour": None, "legend": None}]})
    if not panels:
        raise ValueError(f"no column anywhere matches {pattern!r}.")
    return panels


def apply_spec_json(spec_path, sources: dict[str, Any],
                    settings: dict[str, Any]) -> list[dict[str, Any]]:
    """A JSON spec: the only route to per-panel y limits, notes and detrends."""
    spec_path = Path(spec_path).expanduser().resolve()
    document = json.loads(spec_path.read_text(encoding="utf-8"))

    for key, value in (document.get("settings") or {}).items():
        if key not in settings:
            raise ValueError(f"unknown setting {key!r} in {spec_path.name}. "
                             f"Known: {', '.join(sorted(settings))}")
        settings[key] = value

    base = spec_path.parent
    for name, entry in (document.get("sources") or {}).items():
        path = Path(entry["path"])
        path = path if path.is_absolute() else (base / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        column = entry.get("time_column", settings["time_column"])
        offset = float(entry.get("time_offset_h", 0.0))
        times, columns = read_trace_csv(path, column, offset)
        sources[name] = {"path": path, "t": times, "columns": columns,
                         "time_column": column, "offset": offset}

    panels: list[dict[str, Any]] = []
    for index, entry in enumerate(document.get("panels") or []):
        if isinstance(entry, str):
            panels.append(parse_panel(entry, sources, index))
            continue
        traces: list[dict[str, Any]] = []
        for trace in entry["traces"]:
            if isinstance(trace, str):
                traces += parse_panel(trace, sources, index)["traces"]
                continue
            if trace["column"] not in sources[trace["source"]]["columns"]:
                raise ValueError(f"column {trace['column']!r} is not in "
                                 f"source {trace['source']!r}.")
            traces.append({"source": trace["source"], "column": trace["column"],
                           "colour": trace.get("colour"),
                           "legend": trace.get("legend")})
        panel = {"label": entry.get("label") or pretty_label(traces[0]["column"]),
                 "traces": traces}
        for key in ("ylabel", "note", "ylim", "bold", "detrend",
                    "detrend_window_h", "normalise"):
            if key in entry:
                panel[key] = entry[key]
        panels.append(panel)
    return panels


# ------------------------------------------------------------- the numbers
def detrend(times_h, values, method: str, window_h: float, degree: int,
            poly_edge_h: float) -> tuple[Any, float]:
    """``(baseline, edge_hours)``: the baselines are Auto-Organotypic's.

    ``rolling_baseline``, ``polynomial_baseline`` and ``window_length`` are
    :mod:`auto_organotypic.baselines`, called by that name since 2026-09-16
    rather than through ``tracing``'s re-export, so this module holds no
    baseline of its own. What is the panel's is ``edge_hours``: the stretch
    at each end where the baseline came from a truncated window, and where
    the detrended trace is not comparable to the middle. The panel shades
    it by the window actually used (half its length in samples for the
    rolling mean, a quarter of the span capped by ``poly_edge_h`` for the
    polynomial), which is a narrower margin than
    :func:`auto_organotypic.baselines.detrend_edge_hours` reports, and the
    frozen panel fixtures hold it.
    """
    import numpy as np

    times = np.asarray(times_h, float)
    values = np.asarray(values, float)
    if method == "none":
        return np.zeros_like(values), 0.0
    if method == "rolling":
        step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        length = _baselines.window_length(window_h, times) if len(times) > 1 else 1
        return _baselines.rolling_baseline(values, length), (length // 2) * step
    if method == "polynomial":
        span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        return (_baselines.polynomial_baseline(times, values, int(degree)),
                min(float(poly_edge_h), max(0.0, span / 4.0)))
    raise ValueError(f"unknown detrend method {method!r}. "
                     f"Use rolling, polynomial or none.")


def normalise(values, baseline, mode: str) -> tuple[Any, float]:
    """The detrended trace divided. Returns ``(values, denominator)``.

    Stays here on purpose. ``window_mean`` divides the residual by the
    window's mean and ``baseline`` by the baseline itself, point by point;
:func:`auto_organotypic.trace_plot.normalise_values` rescales an
    already-detrended trace (min-max, z-score, or not at all) and has no
    mode that divides by a baseline, so there is nothing there for these
    two to become calls to.
    """
    import numpy as np

    residual = np.asarray(values, float) - np.asarray(baseline, float)
    if mode == "none":
        return residual, 1.0
    if mode == "window_mean":
        denominator = float(np.nanmean(values))
        if not np.isfinite(denominator) or denominator <= 0:
            return np.full_like(residual, np.nan), float("nan")
        return residual / denominator, denominator
    if mode == "baseline":
        divisor = np.where(np.asarray(baseline, float) > 0, baseline, np.nan)
        return residual / divisor, float(np.nanmean(baseline))
    raise ValueError(f"unknown normalisation {mode!r}. "
                     f"Use window_mean, baseline or none.")


def ticks_between(low: float, high: float, interval: float, origin: float):
    import numpy as np

    if not interval or interval <= 0:
        return None
    first = origin + np.ceil((low - origin) / interval) * interval
    if first > high:
        return np.array([])
    return np.arange(first, high + interval * 1e-9, interval)


def describe(settings: Mapping[str, Any], step_h: float | None = None
             ) -> tuple[str, str, str]:
    """Title, subtitle and x label generated from the settings in force.

    Generated rather than fixed so a figure cannot claim a detrend it did not
    do. The subtitle names the denominator, which is the one thing about a
    dF/F panel a reader has to know and the one thing a legend never says.
    """
    if settings["method"] == "rolling":
        what = f"{settings['window_h']:g} h rolling baseline"
        if step_h:
            length = max(int(round(float(settings["window_h"]) / step_h)) | 1, 1)
            what += f" ({length} frames = {length * step_h:.1f} h)"
    elif settings["method"] == "polynomial":
        what = f"degree {settings['degree']} polynomial detrend"
    else:
        what = "no detrend"

    lead = "dF/F" if settings["normalise"] != "none" else "signal"
    title = f"{lead}, {what}"
    if settings["normalise"] == "window_mean":
        subtitle = ("normalised by each trace's window mean, not its "
                    "instantaneous baseline")
    elif settings["normalise"] == "baseline":
        subtitle = ("normalised by each trace's instantaneous baseline "
                    "(textbook dF/F; unstable where the baseline nears zero)")
    else:
        subtitle = "raw units, not normalised"
    if settings["shade_edges"] and settings["method"] != "none":
        subtitle += "; shaded = truncated baseline window"

    xlabel = "hours"
    if settings["vline_hours"]:
        xlabel = "hours (vertical lines as listed)"
    elif settings["vline_interval_h"]:
        xlabel = f"hours (lines every {settings['vline_interval_h']:g} h)"
    return title, subtitle, xlabel


# ---------------------------------------------------------------- assembly
def prepare(panels: Sequence[Mapping[str, Any]], sources: Mapping[str, Any],
            settings: Mapping[str, Any]):
    """Every value the figure will draw, with nothing left to compute."""
    import numpy as np

    from .visualisation import panels as _grammar
    from .visualisation import traces as _traces

    low, high = settings["time_start"], settings["time_end"]
    all_times = [sources[trace["source"]]["t"]
                 for panel in panels for trace in panel["traces"]]
    low = float(min(t.min() for t in all_times)) if low is None else float(low)
    high = float(max(t.max() for t in all_times)) if high is None else float(high)
    if high <= low:
        raise ValueError(f"empty time window: {low} to {high} h.")

    built: list[Any] = []
    for panel in panels:
        palette = _traces.panel_colours(
            len(panel["traces"]),
            default=settings["default_colour"],
            overflow=settings["overflow_cmap"]) if not settings["cycle"] else (
            _explicit_palette(len(panel["traces"]), settings))
        drawn = []
        for position, trace in enumerate(panel["traces"]):
            source = sources[trace["source"]]
            times = source["t"]
            values = source["columns"][trace["column"]]
            keep = np.isfinite(values)
            if settings["cut_stage"] == "before":
                keep &= (times >= low) & (times <= high)
            used_t, used_y = times[keep], values[keep]
            if used_t.size < 2:
                raise ValueError(
                    f"trace {trace['source']}:{trace['column']} has fewer than "
                    f"two finite samples in the requested window.")

            method = panel.get("detrend", settings["method"])
            window = panel.get("detrend_window_h", settings["window_h"])
            baseline, edge_h = detrend(used_t, used_y, method, window,
                                       settings["degree"],
                                       settings["poly_edge_h"])
            mode = panel.get("normalise", settings["normalise"])
            value, denominator = normalise(used_y, baseline, mode)
            if settings["as_percent"] and mode != "none":
                value = 100.0 * value
            if settings["scale"] != 1.0:
                value = settings["scale"] * value
            smoothed = (_tracing.rolling_baseline(value, settings["smooth"])
                        if settings["smooth"] and settings["smooth"] > 1
                        else np.array(value, copy=True))

            inner = ((used_t >= max(low, used_t[0] + edge_h))
                     & (used_t <= min(high, used_t[-1] - edge_h)))
            if not inner.any():
                inner = np.ones_like(used_t, bool)

            drawn.append(_traces.DrawnTrace(
                source=trace["source"], column=trace["column"], times_h=used_t,
                values=value, smoothed=smoothed,
                colour=_grammar.resolve_colour(trace["colour"]
                                               or palette[position]),
                legend=trace["legend"], edge_h=float(edge_h),
                sd=float(np.nanstd(value[inner])), denominator=denominator,
                detrend=method, detrend_window_h=float(window)))

        built.append(_traces.DrawnPanel(
            label=panel["label"], traces=drawn, note=str(panel.get("note", "")),
            ylabel=str(panel.get("ylabel", "")), ylim=panel.get("ylim"),
            bold=bool(panel.get("bold", False))))

    first = list(sources.values())[0]["t"]
    step = float(np.median(np.diff(first))) if first.size > 1 else None
    title, subtitle, xlabel = describe(settings, step)
    unit = "%" if ((settings["as_percent"] and settings["normalise"] != "none")
                   or settings["scale"] == 100.0) else ""
    suffix = settings["ylabel_suffix"] or (
        "dF/F %" if (settings["normalise"] != "none" and settings["as_percent"])
        else "dF/F" if settings["normalise"] != "none"
        else "dF/F %" if settings["scale"] == 100.0 else "signal")

    return _traces.PanelTable(
        panels=built, window_h=(low, high),
        title=settings["title"] or title,
        subtitle=settings["subtitle"] if settings["subtitle"] else subtitle,
        xlabel=settings["xlabel"] or xlabel,
        ylabel_suffix=suffix, unit=unit, settings=dict(settings),
        xtick_hours=(np.asarray(settings["xtick_hours"], float)
                     if settings["xtick_hours"] else
                     ticks_between(low, high, settings["xtick_interval_h"],
                                   settings["xtick_origin_h"])),
        vline_hours=(np.asarray(settings["vline_hours"], float)
                     if settings["vline_hours"] else
                     ticks_between(low, high, settings["vline_interval_h"],
                                   settings["vline_origin_h"])))


def _default_colour(colour):
    """The first trace's colour: the caller's, else the run's registry, else
    the house dLuc.

    Inside a run of the chain :mod:`auto_organotypic.conventions` is
    current, and its trace colour for the run's first channel is what the
    chain's own trace figures and movies are drawn in, so a cell's trace
    on this panel matches them. Outside one the answer is the house colour
    it always was.
    """
    if colour is not None:
        return colour
    registry = _conventions.current()
    if registry is not None and registry.channels:
        return registry.trace_colour(registry.channels[0])
    return "dluc"


def _explicit_palette(n: int, settings: Mapping[str, Any]) -> list[str]:
    """A caller's own cycle, by house name or by value, in the order given."""
    from .visualisation import panels as _grammar

    cycle = [_grammar.resolve_colour(name) for name in settings["cycle"]]
    first = _grammar.resolve_colour(settings["default_colour"])
    if n <= 1 or not cycle:
        return [first] * max(n, 1)
    if n <= len(cycle) or not settings["overflow_cmap"]:
        return [first] + [cycle[k % len(cycle)] for k in range(1, n)]
    return [first] + _grammar.overflow_colours(
        n - 1, cmap=settings["overflow_cmap"])


# ------------------------------------------------------------- the action
def trace_panel(source=None, *, output_dir=None, output_name=None,
                overwrite: bool = False, input_csvs: Sequence[Any] = (),
                time_column: str = "hours", time_offset_h: float = 0.0,
                output_path: str = "", output_formats: Sequence[str] = ("png",),
                write_plotted_csv: bool = True, write_provenance: bool = True,
                write_bundle: bool = True, claim: str = "",
                panel_specs: Sequence[str] = (), auto_columns: str = "*_processed",
                spec_json: str = "", detrend_method: str = "rolling",
                detrend_window_h: float = 24.0, poly_degree: int = 6,
                normalise: str = "window_mean", as_percent: bool = True,
                value_scale: float = 1.0, time_start_h: float | None = None,
                time_end_h: float | None = None, time_cut_stage: str = "after",
                smooth_frames: int = 3, show_raw_trace: bool = True,
                raw_colour: str = "raw", raw_linewidth: float = 0.6,
                trace_linewidth: float = 2.1, colour: str | None = None,
                colour_cycle: Sequence[str] = (),
                colour_overflow_cmap: str = "turbo", shade_edges: bool = True,
                shade_colour: str = "shade", poly_edge_h: float = 12.0,
                show_zero_line: bool = True, show_sd_label: bool = True,
                share_y: bool = False, xtick_interval_h: float = 24.0,
                xtick_origin_h: float = 0.0, xtick_hours: Sequence[float] = (),
                vline_interval_h: float = 24.0, vline_origin_h: float = 0.0,
                vline_hours: Sequence[float] = (), vline_colour: str = "tab:green",
                vline_alpha: float = 0.45, vline_linewidth: float = 0.7,
                fig_width_in: float = 11.5, panel_height_in: float = 2.05,
                title_height_in: float = 1.0, title: str = "", subtitle: str = "",
                xlabel: str = "", ylabel_suffix: str = "",
                legend_when_merged: bool = True, dpi: int = 150,
                figure_profile: str = "master",
                figure_safe_columns: Sequence[str] = (),
                public_sources: Mapping[str, str] | None = None,
                dpi_preset: str | None = None,
                render_preset: str | None = None,
                render_width_in: float | None = None,
                render_height_in: float | None = None,
                format_options: Mapping[str, Any] | None = None,
                allow_reencode: bool = False,
                proof: bool = False,
                required_grades: Sequence[str] = (),
                signing_key_path: str | None = None,
                signing_password_env: str | None = None,
                trust_policy_path: str | None = None,
                encrypted_sections: Sequence[str] = (),
                encryption_password_env: str | None = None,
                recipient_file: str | None = None,
                broker_policy_path: str | None = None,
                theme: str = "engine") -> dict[str, Any]:
    """Draw a stacked trace panel from one or more trace CSVs.

    ``source`` is the common argument every action takes and is treated as one
    more input CSV, so ``trace_panel(path)`` works; ``input_csvs`` takes the
    rest, each optionally as ``NAME=path`` to give it a handle the panel
    grammar can refer to.

    ``theme`` defaults to ``"engine"``, which leaves Matplotlib's own defaults
    alone so the output matches the figure ``trace_panel_figure.py`` already
    produces. Pass a kit theme name — ``"pyflash"`` — to draw it in the house
    look instead. That moves the type sizes and the spines, which is a visible
    change and so is never the default on a figure somebody already has.
    """
    from .visualisation import panels as _grammar
    from .visualisation import traces as _traces

    settings = dict(DEFAULTS)
    settings.update({
        "time_column": time_column, "method": detrend_method,
        "window_h": float(detrend_window_h), "degree": int(poly_degree),
        "normalise": normalise, "as_percent": bool(as_percent),
        "scale": float(value_scale), "poly_edge_h": float(poly_edge_h),
        "time_start": time_start_h, "time_end": time_end_h,
        "cut_stage": time_cut_stage, "smooth": int(smooth_frames),
        "show_raw": bool(show_raw_trace), "raw_colour": raw_colour,
        "raw_linewidth": float(raw_linewidth),
        "trace_linewidth": float(trace_linewidth),
        "default_colour": _default_colour(colour),
        "cycle": list(colour_cycle), "overflow_cmap": colour_overflow_cmap,
        "shade_edges": bool(shade_edges), "shade_colour": shade_colour,
        "show_zero_line": bool(show_zero_line), "show_sd": bool(show_sd_label),
        "share_y": bool(share_y), "xtick_interval_h": float(xtick_interval_h),
        "xtick_origin_h": float(xtick_origin_h), "xtick_hours": list(xtick_hours),
        "vline_interval_h": float(vline_interval_h),
        "vline_origin_h": float(vline_origin_h), "vline_hours": list(vline_hours),
        "vline_colour": vline_colour, "vline_alpha": float(vline_alpha),
        "vline_linewidth": float(vline_linewidth),
        "fig_width_in": float(fig_width_in),
        "panel_height_in": float(panel_height_in),
        "title_height_in": float(title_height_in), "title": title,
        "subtitle": subtitle, "xlabel": xlabel, "ylabel_suffix": ylabel_suffix,
        "legend_when_merged": bool(legend_when_merged), "dpi": int(dpi),
    })

    entries = ([source] if source is not None else []) + list(input_csvs)
    sources = load_sources(entries, time_column, time_offset_h) if entries else {}

    if spec_json:
        built = apply_spec_json(spec_json, sources, settings)
    elif panel_specs:
        built = [parse_panel(text, sources, i)
                 for i, text in enumerate(panel_specs)]
    else:
        built = auto_panels(sources, auto_columns)
    if not sources:
        raise ValueError("no input. Pass a CSV as source= or input_csvs=, or "
                         "a JSON spec as spec_json=.")

    table = prepare(built, sources, settings)
    figure = _traces.draw(
        table, fig_width_in=settings["fig_width_in"],
        panel_height_in=settings["panel_height_in"],
        title_height_in=settings["title_height_in"],
        share_y=settings["share_y"], show_raw=settings["show_raw"],
        raw_linewidth=settings["raw_linewidth"],
        trace_linewidth=settings["trace_linewidth"],
        shade_edges=settings["shade_edges"],
        show_zero_line=settings["show_zero_line"], show_sd=settings["show_sd"],
        legend_when_merged=settings["legend_when_merged"],
        vline_colour=settings["vline_colour"], vline_alpha=settings["vline_alpha"],
        vline_linewidth=settings["vline_linewidth"], theme=theme)

    target = _output_path(output_path, output_dir, output_name, sources)
    written = _grammar.save(
        figure, target, table=table.as_table(),
        sources=[{"name": name, "path": entry["path"],
                  "rows": int(entry["t"].size)}
                 for name, entry in sources.items()],
        claim=claim or f"{len(table.panels)} trace panels, "
                       f"{table.window_h[0]:.2f}-{table.window_h[1]:.2f} h",
        settings=settings, formats=output_formats, dpi=settings["dpi"],
        figure_profile=figure_profile,
        figure_safe_columns=(list(figure_safe_columns) or None),
        public_sources=public_sources,
        dpi_preset=dpi_preset, render_preset=render_preset,
        width=render_width_in, height=render_height_in,
        format_options=format_options, allow_reencode=allow_reencode,
        proof=proof, required_grades=required_grades,
        signing_key_path=signing_key_path,
        signing_password_env=signing_password_env,
        trust_policy_path=trust_policy_path,
        encrypted_sections=encrypted_sections,
        encryption_password_env=encryption_password_env,
        recipient_file=recipient_file,
        broker_policy_path=broker_policy_path,
        # The engine replaced by default and refused only with
        # ``--no-overwrite``. This package refuses by default, as every other
        # action here does: a re-run silently replacing the figure you were
        # comparing against is the mistake that rule exists for.
        overwrite=bool(overwrite), bundle=bool(write_bundle),
        drawn=table.as_records())

    _close(figure)
    if not write_plotted_csv:
        written["table"].unlink(missing_ok=True)
    if not write_provenance:
        written["provenance"].unlink(missing_ok=True)

    return {
        "figures": [str(p) for p in written["figures"]],
        "table": str(written["table"]) if write_plotted_csv else None,
        "provenance": str(written["provenance"]) if write_provenance else None,
        "bundle": str(written["bundle"]) if written["bundle"] else None,
        "proof": written.get("proof"),
        "panels": len(table.panels),
        "traces": sum(len(panel.traces) for panel in table.panels),
        "window_h": list(table.window_h),
        "method_version": METHOD_VERSION,
        "sd_percent": [[trace.sd for trace in panel.traces]
                       for panel in table.panels],
    }


def _output_path(output_path: str, output_dir, output_name,
                 sources: Mapping[str, Any]) -> Path:
    if output_path:
        return Path(output_path).expanduser().resolve()
    first = Path(list(sources.values())[0]["path"]) if sources else Path("panels")
    stem = str(output_name) if output_name else f"{first.stem}_panels"
    folder = Path(output_dir).expanduser() if output_dir else first.parent
    return (folder / stem).with_suffix(".png").resolve()


def _close(figure) -> None:
    """Release the figure. A batch that draws a hundred otherwise keeps all of them."""
    from matplotlib import pyplot

    pyplot.close(figure)
