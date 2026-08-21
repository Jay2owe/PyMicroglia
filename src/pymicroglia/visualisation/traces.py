"""The stacked trace panel: one row per panel, one or more traces on each.

Ported from ``trace_panel_figure.py``, which generalised ``traces_figure()`` in
the dLuc pipeline. It draws the figure and nothing else — every number it puts
on the page arrives already computed, in a :class:`PanelTable` built by
``pymicroglia.trace_tables``. That split is the point of this stage: a figure
that cannot compute cannot grow a private helper stack, because there is
nothing for the helpers to do.

What that leaves here is genuinely only drawing:

* the grey blocks over the stretch where the rolling baseline was built from a
  truncated window, and so where the detrended trace is not comparable to the
  middle;
* the periodic vertical lines and the x ticks;
* the thin unsmoothed line under each trace, and the thick smoothed one;
* the zero line, the y label, the corner note and the standard-deviation
  annotation;
* a legend, on panels holding more than one trace.

Colours come from ``analysis_kit.style`` by name. Three constants changed
meaning in the port and nothing else did:

    DEFAULT_COLOUR = "#a340d1"   ->  colour("dluc")
    COLOUR_CYCLE   = [six hex]   ->  cycle("semantic")
    UNDERLAY_GREY  = "0.72"      ->  GREY_LEVEL["raw"]

All three resolve to the values the engine spelled out, so the port is a change
of where the colour is written down and not of what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import panels as _panels

__all__ = [
    "DrawnTrace",
    "DrawnPanel",
    "PanelTable",
    "panel_colours",
    "draw",
]

#: Names, not values. ``colour("dluc")`` is ``#a340d1``; writing the hex here
#: is how a lab ends up with four different reds.
DEFAULT_COLOUR_NAME = "dluc"
CYCLE_GROUP = "semantic"
RAW_GREY = "raw"
SHADE_GREY = "shade"


@dataclass
class DrawnTrace:
    """One line on one panel, with every value it is drawn from.

    ``values`` is what the y axis shows; ``smoothed`` is the thick line over
    it. Both are already detrended, normalised and scaled — this module never
    touches either.
    """

    source: str
    column: str
    times_h: Any
    values: Any
    smoothed: Any
    colour: str
    legend: str | None = None
    edge_h: float = 0.0
    sd: float = float("nan")
    denominator: float = float("nan")
    detrend: str = ""
    detrend_window_h: float = 0.0

    @property
    def name(self) -> str:
        return self.legend or self.column

    def as_record(self) -> dict[str, Any]:
        """What the engine wrote about this trace in its provenance file.

        The detrend, the shaded width, the denominator and the spread — the
        four numbers that say what the percentage on the axis is a percentage
        *of*, and which part of it a reader may believe.
        """
        return {"source": self.source, "column": self.column,
                "colour": self.colour, "detrend": self.detrend,
                "detrend_window_h": self.detrend_window_h,
                "edge_shaded_h": self.edge_h,
                "denominator": (None if self.denominator != self.denominator
                                else self.denominator),
                "sd": self.sd}


@dataclass
class DrawnPanel:
    """One row of the stack."""

    label: str
    traces: list[DrawnTrace] = field(default_factory=list)
    note: str = ""
    ylabel: str = ""
    ylim: tuple[float, float] | None = None
    bold: bool = False


@dataclass
class PanelTable:
    """Everything the figure draws, and the settings it was drawn under.

    Handed over whole rather than assembled here. A caller holding one of these
    already holds the figure's exact plotted data, which is what makes the
    provenance bundle free instead of extra work.
    """

    panels: list[DrawnPanel]
    window_h: tuple[float, float]
    title: str = ""
    subtitle: str = ""
    xlabel: str = "hours"
    ylabel_suffix: str = ""
    unit: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    xtick_hours: Any = None
    vline_hours: Any = None

    def as_records(self) -> list[dict[str, Any]]:
        """One entry per panel, each listing what was drawn on it."""
        return [{"label": panel.label,
                 "traces": [trace.as_record() for trace in panel.traces]}
                for panel in self.panels]

    def as_table(self) -> dict[str, list[float]]:
        """The exact plotted values, as the CSV that goes beside the figure.

        Column names carry the panel and the trace, because two panels may draw
        the same column under different detrends and a reader has to be able to
        tell which row of the figure a column belongs to.
        """
        import re

        table: dict[str, list[float]] = {}
        for panel in self.panels:
            tag = re.sub(r"[^a-z0-9]+", "_", panel.label.lower()).strip("_")
            for trace in panel.traces:
                stem = f"{tag}__{trace.column}"
                table[f"{stem}_hours"] = [float(v) for v in trace.times_h]
                table[f"{stem}_value"] = [float(v) for v in trace.values]
                table[f"{stem}_smoothed"] = [float(v) for v in trace.smoothed]
        return table


def panel_colours(n: int, *, default: str = DEFAULT_COLOUR_NAME,
                  group: str = CYCLE_GROUP, overflow: str | None = None
                  ) -> list[str]:
    """Colours for one panel's traces, all distinct.

    The first trace is the house dLuc purple whatever else is on the panel, so
    a single-trace panel never changes colour when a second trace joins it.
    Beyond the cycle's length the remainder is sampled from a colormap rather
    than repeating: repeating gives the seventh trace the first one's colour,
    and on an eight-recording overlay two lines then cannot be told apart.
    """
    first = _panels.colour(default)
    if n <= 1:
        return [first]
    cycle = _panels.colours(group)
    if n <= len(cycle) or overflow == "":
        return [first] + [cycle[k % len(cycle)] for k in range(1, n)]
    return [first] + _panels.overflow_colours(n - 1, cmap=overflow or None)


def draw(table: PanelTable, *, fig_width_in: float = 11.5,
         panel_height_in: float = 2.05, title_height_in: float = 1.0,
         share_y: bool = False, show_raw: bool = True,
         raw_linewidth: float = 0.6, trace_linewidth: float = 2.1,
         shade_edges: bool = True, show_zero_line: bool = True,
         show_sd: bool = True, legend_when_merged: bool = True,
         vline_colour: str = "tab:green", vline_alpha: float = 0.45,
         vline_linewidth: float = 0.7, theme: str = _panels.ENGINE_THEME,
         label_size: float = 10.0, note_size: float = 9.0,
         tick_size: float = 9.0, xlabel_size: float = 11.0,
         title_size: float = 13.0) -> Any:
    """Draw the stack and hand back the figure. Saving is ``panels.save``'s job.

    Every size here is the engine's, so ``theme="engine"`` reproduces its
    output. Passing a kit theme name instead applies the house look, which
    moves the type and the spines and is therefore a visible change — worth
    making deliberately, never by default on a figure somebody already has.
    """
    low, high = float(table.window_h[0]), float(table.window_h[1])
    stack = _panels.stack(len(table.panels), height_per_panel=panel_height_in,
                          width=fig_width_in, head_in=title_height_in,
                          share_x=True, share_y=share_y, theme=theme)
    shade = _panels.grey(SHADE_GREY)
    raw_grey = _panels.grey(RAW_GREY)

    for axis, panel in zip(stack, table.panels):
        if shade_edges:
            _shade(axis, panel, low, high, shade)
        _panels.vlines(axis, table.vline_hours, colour_name=vline_colour,
                       alpha=vline_alpha, width=vline_linewidth)
        for trace in panel.traces:
            if show_raw:
                axis.plot(trace.times_h, trace.values, lw=raw_linewidth,
                          color=raw_grey, zorder=1)
            axis.plot(trace.times_h, trace.smoothed, lw=trace_linewidth,
                      color=trace.colour, zorder=2, label=trace.name)
        if show_zero_line:
            _panels.zero_line(axis)
        axis.set_xlim(low, high)
        _panels.ticks(axis, table.xtick_hours)
        if panel.ylim:
            axis.set_ylim(*panel.ylim)
        _panels.label(
            axis,
            y=panel.ylabel or f"{panel.label}\n{table.ylabel_suffix}",
            title=_corner(panel, show_sd, table.unit) or None,
            bold=panel.bold or panel.label.startswith("WHOLE"),
            size=label_size, title_size=note_size)
        if legend_when_merged and len(panel.traces) > 1:
            _panels.legend(axis, columns=len(panel.traces))
        axis.tick_params(labelsize=tick_size)

    _panels.label(stack.last, x=table.xlabel, size=xlabel_size - 1)
    stack.tighten()
    stack.title(table.title, subtitle=table.subtitle, size=title_size)
    return stack.figure


def _shade(axis, panel: DrawnPanel, low: float, high: float,
           colour: str) -> None:
    """Grey over the ends where the baseline window was truncated.

    Per trace, not per panel: two recordings merged onto one axis may start at
    different hours, and each one's unreliable stretch is its own.
    """
    for trace in panel.traces:
        if trace.edge_h <= 0 or len(trace.times_h) == 0:
            continue
        start = float(trace.times_h[0])
        end = float(trace.times_h[-1])
        axis.axvspan(max(start, low), min(start + trace.edge_h, high),
                     color=colour, lw=0, zorder=0)
        axis.axvspan(max(end - trace.edge_h, low), min(end, high),
                     color=colour, lw=0, zorder=0)


def _corner(panel: DrawnPanel, show_sd: bool, unit: str) -> str:
    """The panel's note and its standard deviations, in the engine's order.

    The sd is a spread over the trustworthy interior, unsmoothed. It is not an
    amplitude and makes no noise correction, which is why it is a label in the
    corner rather than a result.
    """
    parts = []
    if panel.note:
        parts.append(panel.note)
    if show_sd and panel.traces:
        parts.append("   ".join(f"sd {trace.sd:.1f}{unit}"
                                for trace in panel.traces))
    return "   |   ".join(parts)
