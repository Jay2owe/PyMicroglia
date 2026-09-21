"""How each cell's track begins and ends, and where one becomes two.

Every other module measures a cell while it is on screen. This one measures the
two frames the rest of the package steps over: the frame a name first appears
and the frame it was last seen. A register of arrivals and departures kept
beside the attendance sheet - who joined, who left, and whether anyone saw them
go.

Those two frames are where the biology a time-lapse is run for actually shows:
a cell dividing, a cell dying, a cell arriving. They are also where a tracker
fails, and the two look identical in the label stack. A name that stops is a
name that stops; nothing in the pixels says whether the cell died, walked out of
the field, hid inside a neighbour, or simply dimmed below threshold with the
tracker losing it. So this module writes down what it can see and says how sure
that is, in a column, on every row.

``event_status`` is the column that carries the honesty:

``observed``
    the frame itself settles it - the outline is against the edge of the field,
    so the cell entered or left.
``censored``
    the recording started or ended here, so how the track began or ended was
    never imaged.
``candidate``
    a rule matched: areas conserved across a separation, or the ground went dark
    with nothing left on it. The rule and its thresholds are configuration and
    are recorded in the manifest. A candidate is a case to look at, not a
    finding.
``unproven``
    something happened and none of the rules fits it. Left as it is rather than
    forced into the nearest name.

The rules, in full, so that changing one is a decision rather than a discovery:

* a track that starts in contact with a name that was already there, whose two
  areas together account for that name's previous area, is a **division
  candidate** - the parent gets ``divided``, the new name gets ``born``;
* a track that ends mid-field with no foreground left where it was, named or
  unclaimed, is a **death candidate** - nothing is there any more;
* a track that ends mid-field with foreground still on that ground is ``lost``,
  which is a tracking failure and not a biological ending, and saying so is the
  point of separating the two;
* an outline touching the border of the field at its first or last frame is an
  arrival or a departure and is ``observed``, whatever else is going on.

Without an ``unclaimed`` stack the death rule cannot run at all - a cell that
vanished and a cell whose foreground nobody claimed are the same picture - so
every mid-field ending is written ``vanished`` and ``unproven`` instead. A
module that quietly reported fewer deaths because an input was missing would be
worse than one that says it cannot tell.

Ported from Motion's ``analysis/modules/lifecycle.py`` on 2026-09-21 (stage 04 of the Motion port); the arithmetic is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pymicroglia.measure.context import MeasurementContext
from pymicroglia.measure.declare import Column, Output, measurement

#: Bumped when this module's arithmetic changes; recorded in the run record.
METHOD_VERSION = "2026-09-21-lifecycle-v1"


#: Every threshold the rules use. All of them are ``lifecycle`` module
#: parameters, so a run that disagrees changes them in its configuration and the
#: manifest records what it used.
DEFAULTS = {
    # How close two outlines have to be to count as touching. Two pixels, not
    # one: adjacent cells are routinely separated by a single background pixel
    # where the segmentation cut between them.
    "contact_dilation_px": 2,
    # An outline this close to the edge of the field is against it. Cells enter
    # and leave through the border, and a mask one pixel short of it has still
    # arrived from outside.
    "border_margin_px": 2,
    # A separation counts as area-conserving when the two parts together are
    # within this fraction of what the parent held the frame before. Wide,
    # because a dividing cell rounds up and the outline changes with it.
    "area_conservation_tolerance": 0.4,
    # Both parts must still be named this many frames later. One frame of a
    # second name is a segmentation flicker, not a division.
    "min_frames_after_split": 3,
    # Foreground still on the ground a cell held, as a fraction of its last
    # area, above which the ending is `lost` rather than a death candidate.
    "residual_foreground_fraction": 0.25,
    # How many of a cell's last frames the fading summary is taken over. It is
    # reported, never tested against: whether a cell faded is for the reader.
    "fade_frames": 3,
}


#: The words this module is allowed to use for what happened, grouped by where
#: in a track they can occur. Declared rather than written inline so that the
#: figures, the tests and the documentation all read one list.
STARTS = ("present_at_start", "entered_field", "born", "appeared")
ENDS = ("present_at_end", "left_field", "died", "absorbed", "lost", "vanished")
DURING = ("divided",)
EVENTS = STARTS + ENDS + DURING

#: How much the pixels settle. See the module docstring.
STATUSES = ("observed", "censored", "candidate", "unproven")


PRODUCES = (
    # lifecycle_events - one row per cell per event
    Column("event", "What happened to this name", "arrival, departure or division", "reference"),
    Column("event_status", "How far the frames settle it", "observed, censored, candidate or unproven", "reference"),
    Column("partner_identity", "The other cell in this event", "identity", "reference"),
    Column("partner_area_px", "The other cell's outline size in this frame", "px", "morphology"),
    Column("parent_area_before_px", "What the parent held the frame before", "px", "morphology"),
    Column("area_conservation", "Both parts together over the parent's previous area", "fraction", "morphology"),
    Column("residual_foreground_px", "Foreground still on this ground afterwards", "px", "unclaimed"),
    Column("residual_foreground_fraction", "That foreground over the last outline", "fraction", "unclaimed"),
    Column("area_change_over_last_frames", "Size over the closing frames, last against first", "fraction", "morphology"),
    Column("frames_together", "Frames both parts stayed named after separating", "frames", "reference"),
    # lifecycle_cells - one row per cell
    Column("start_event", "How this track began", "arrival", "reference"),
    Column("start_status", "How far the frames settle the beginning", "", "reference"),
    Column("start_frame", "The frame this track began", "frame", "reference"),
    Column("start_hours", "When this track began", "h", "reference"),
    Column("end_event", "How this track ended", "departure", "reference"),
    Column("end_status", "How far the frames settle the ending", "", "reference"),
    Column("end_frame", "The frame this track ended", "frame", "reference"),
    Column("end_hours", "When this track ended", "h", "reference"),
    Column("parent_identity", "The cell this one separated from", "identity", "reference"),
    Column("absorbed_into_identity", "The cell this one was absorbed into", "identity", "reference"),
    Column("children", "Names that separated from this cell", "count", "reference"),
    Column("first_division_frame", "The frame this cell first divided", "frame", "reference"),
    Column("first_division_hours", "When this cell first divided", "h", "reference"),
    Column("frames_named", "Frames this name was on screen", "frames", "named"),
    Column("observed_hours", "Hours between first and last frame", "h", "reference"),
)


#: Events are their own grain: one row per cell per event, several to a cell and
#: none for a cell that starts at the first frame and runs to the last without
#: dividing. The per-cell table is at cell grain but is deliberately not folded
#: into ``cell_summary`` - it is about the two frames outside every summary, and
#: a reader looking for why a cell ended should find one file, not a pair of
#: columns among four hundred.
WRITES = (
    Output("lifecycle_events", grain=("identity", "frame_index", "event")),
    Output("lifecycle_cells",  grain=("identity",)),
)


def _params(context: MeasurementContext) -> dict:
    params = {**DEFAULTS, **context.module_params("lifecycle")}
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(
            "lifecycle does not accept " + ", ".join(unknown)
            + "; it accepts " + ", ".join(sorted(DEFAULTS))
        )
    for name in ("contact_dilation_px", "border_margin_px", "min_frames_after_split",
                 "fade_frames"):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"lifecycle.{name} must be a whole number of at least zero")
        params[name] = int(value)
    for name in ("area_conservation_tolerance", "residual_foreground_fraction"):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or not 0 <= value <= 1:
            raise ValueError(f"lifecycle.{name} must be a fraction between zero and one")
        params[name] = float(value)
    return params


def _areas(labels: np.ndarray) -> tuple[list[dict[int, int]], list[set[int]]]:
    """Pixels per name per frame, and who is on screen, in one pass."""
    per_frame: list[dict[int, int]] = []
    on_screen: list[set[int]] = []
    for frame in labels:
        counts = np.bincount(frame.ravel())
        present = {int(value): int(counts[value]) for value in np.flatnonzero(counts) if value}
        per_frame.append(present)
        on_screen.append(set(present))
    return per_frame, on_screen


def _touches_border(frame: np.ndarray, identity: int, margin: int) -> bool:
    """Whether this name's outline reaches the edge of the imaged field."""
    mask = frame == identity
    if not mask.any():
        return False
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    height, width = mask.shape
    return bool(rows[0] <= margin or columns[0] <= margin
                or rows[-1] >= height - 1 - margin or columns[-1] >= width - 1 - margin)


def _neighbours(frame: np.ndarray, identity: int, dilation: int) -> dict[int, int]:
    """Other names within ``dilation`` pixels of this one, and how many pixels each holds there.

    Cropped to the cell's own bounding box first: the field is mostly empty and
    dilating a whole frame per event costs more than every other rule together.
    """
    from scipy import ndimage as ndi

    mask = frame == identity
    if not mask.any():
        return {}
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    top = max(int(rows[0]) - dilation - 1, 0)
    bottom = min(int(rows[-1]) + dilation + 2, mask.shape[0])
    left = max(int(columns[0]) - dilation - 1, 0)
    right = min(int(columns[-1]) + dilation + 2, mask.shape[1])
    window = frame[top:bottom, left:right]
    near = window == identity
    if dilation:
        near = ndi.binary_dilation(near, iterations=dilation)
    found = window[near & (window != identity) & (window > 0)]
    if found.size == 0:
        return {}
    values, counts = np.unique(found, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _foreground(context: MeasurementContext, frame_index: int) -> np.ndarray:
    """Every foreground pixel of one frame, named or not.

    ``None`` when the movie declared no unclaimed stack, which is what stops the
    death rule from running rather than letting it run on half the evidence.
    """
    if context.unclaimed is None:
        return None
    return (context.labels[frame_index] > 0) | (context.unclaimed[frame_index] > 0)


@measurement(
    name="lifecycle",
    description="How each name's track begins and ends, and which separations conserve a parent's area",
    requires=("labels",),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = _params(context)
    labels = context.labels
    n_frames = context.n_frames
    areas, on_screen = _areas(labels)
    identities = sorted({identity for frame in on_screen for identity in frame})
    hours = context.frame_table()["hours"].to_numpy(float)

    seen = {identity: [index for index in range(n_frames) if identity in on_screen[index]]
            for identity in identities}
    first = {identity: frames[0] for identity, frames in seen.items() if frames}
    last = {identity: frames[-1] for identity, frames in seen.items() if frames}

    rows: list[dict] = []
    parents: dict[int, int] = {}
    children: dict[int, int] = {identity: 0 for identity in identities}

    for identity in identities:
        rows.append(_start(identity, context, params, areas, on_screen,
                           first, last, seen, hours, parents, children))
        rows.append(_end(identity, context, params, areas, on_screen,
                         first, last, seen, hours))

    # A division is one occurrence written twice: once against the name that
    # arrived and once against the name it came from, so a figure of one cell's
    # own history shows the cell dividing rather than only its offspring being
    # born.
    for child, parent in parents.items():
        birth = next(row for row in rows
                     if row["identity"] == child and row["event"] == "born")
        rows.append({**birth, "identity": parent, "event": "divided",
                     "partner_identity": child,
                     "area_px": birth["partner_area_px"],
                     "partner_area_px": birth["area_px"]})

    events = pd.DataFrame(rows, columns=[
        "identity", "frame_index", "hours", "event", "event_status", "partner_identity",
        "touches_border", "area_px", "partner_area_px", "parent_area_before_px",
        "area_conservation", "residual_foreground_px", "residual_foreground_fraction",
        "area_change_over_last_frames", "frames_together",
    ]).sort_values(["identity", "frame_index", "event"]).reset_index(drop=True)

    return {"lifecycle_events": events,
            "lifecycle_cells": _cells(events, identities, seen, first, last,
                                      hours, parents, children)}


def _start(identity, context, params, areas, on_screen, first, last, seen,
           hours, parents, children) -> dict:
    """The frame this name first appears, and the best account of why."""
    frame_index = first[identity]
    row = {"identity": identity, "frame_index": frame_index,
           "hours": float(hours[frame_index]), "partner_identity": pd.NA,
           "area_px": areas[frame_index][identity],
           "partner_area_px": pd.NA, "parent_area_before_px": pd.NA,
           "area_conservation": np.nan, "residual_foreground_px": pd.NA,
           "residual_foreground_fraction": np.nan,
           "area_change_over_last_frames": np.nan, "frames_together": pd.NA,
           "touches_border": False}

    if frame_index == 0:
        return {**row, "event": "present_at_start", "event_status": "censored"}

    row["touches_border"] = _touches_border(
        context.labels[frame_index], identity, params["border_margin_px"])
    if row["touches_border"]:
        return {**row, "event": "entered_field", "event_status": "observed"}

    division = _division(identity, frame_index, context, params, areas,
                         on_screen, last, seen)
    if division is not None:
        parents[identity] = division["partner_identity"]
        children[division["partner_identity"]] = children.get(division["partner_identity"], 0) + 1
        return {**row, **division, "event": "born", "event_status": "candidate"}

    return {**row, "event": "appeared", "event_status": "unproven"}


def _division(identity, frame_index, context, params, areas, on_screen, last, seen) -> dict | None:
    """The area-conserving separation rule, or ``None`` when nothing fits it.

    Each candidate parent is judged on its own: the pair that arrived here
    together must account for what that parent held one frame earlier, and both
    must still be named far enough on to rule out a flicker of the segmentation.
    Where several fit, the closest conservation wins, and the runner-up is not
    recorded - a cell in a crowd is exactly the case a reader should look at,
    which is what ``candidate`` is for.
    """
    previous = frame_index - 1
    touching = _neighbours(context.labels[frame_index], identity,
                           params["contact_dilation_px"])
    area = areas[frame_index][identity]
    best = None
    for partner, _ in sorted(touching.items(), key=lambda item: -item[1]):
        if partner not in on_screen[previous] or partner not in on_screen[frame_index]:
            continue
        before = areas[previous][partner]
        if before <= 0:
            continue
        conservation = (area + areas[frame_index][partner]) / before
        if abs(conservation - 1.0) > params["area_conservation_tolerance"]:
            continue
        together = min(last[identity], last[partner]) - frame_index + 1
        if together < params["min_frames_after_split"]:
            continue
        if best is None or abs(conservation - 1.0) < abs(best["area_conservation"] - 1.0):
            best = {"partner_identity": partner,
                    "partner_area_px": areas[frame_index][partner],
                    "parent_area_before_px": before,
                    "area_conservation": float(conservation),
                    "frames_together": int(together)}
    return best


def _end(identity, context, params, areas, on_screen, first, last, seen, hours) -> dict:
    """The frame this name was last seen, and the best account of why it stopped."""
    frame_index = last[identity]
    area = areas[frame_index][identity]
    frames = seen[identity]
    closing = frames[-params["fade_frames"]:]
    change = (areas[closing[-1]][identity] / areas[closing[0]][identity]
              if areas[closing[0]][identity] else np.nan)
    row = {"identity": identity, "frame_index": frame_index,
           "hours": float(hours[frame_index]), "partner_identity": pd.NA,
           "area_px": area, "partner_area_px": pd.NA,
           "parent_area_before_px": pd.NA, "area_conservation": np.nan,
           "residual_foreground_px": pd.NA, "residual_foreground_fraction": np.nan,
           "area_change_over_last_frames": float(change), "frames_together": pd.NA,
           "touches_border": False}

    if frame_index == context.n_frames - 1:
        return {**row, "event": "present_at_end", "event_status": "censored"}

    row["touches_border"] = _touches_border(
        context.labels[frame_index], identity, params["border_margin_px"])
    if row["touches_border"]:
        return {**row, "event": "left_field", "event_status": "observed"}

    ground = context.labels[frame_index] == identity
    after = context.labels[frame_index + 1]
    taken = after[ground]
    taken = taken[taken > 0]
    if taken.size:
        values, counts = np.unique(taken, return_counts=True)
        winner = int(values[int(np.argmax(counts))])
        if counts.max() >= area * (1.0 - params["residual_foreground_fraction"]):
            return {**row, "event": "absorbed", "event_status": "candidate",
                    "partner_identity": winner,
                    "partner_area_px": areas[frame_index + 1].get(winner, 0)}

    foreground = _foreground(context, frame_index + 1)
    if foreground is None:
        # No unclaimed stack: a cell that went and a cell nobody claimed are the
        # same picture, and this module will not pick between them.
        return {**row, "event": "vanished", "event_status": "unproven"}

    residual = int(np.count_nonzero(foreground & ground))
    row["residual_foreground_px"] = residual
    row["residual_foreground_fraction"] = float(residual / area) if area else np.nan
    if residual <= area * params["residual_foreground_fraction"]:
        return {**row, "event": "died", "event_status": "candidate"}
    return {**row, "event": "lost", "event_status": "unproven"}


def _cells(events, identities, seen, first, last, hours, parents, children) -> pd.DataFrame:
    """One row per name: how its track began, how it ended, and what it held.

    Every moment is written twice, as a frame and as an hour. The hour is what a
    reader wants; the frame is what a join wants, because every other table in
    the run is keyed on ``frame_index``. Tagging a cell with its ending and then
    asking what its intensity did for six hours beforehand should be a merge,
    not an arithmetic exercise in two different clocks.

    Both relatives are carried here as well: ``parent_identity`` for a cell that
    separated from another, ``absorbed_into_identity`` for one that ended inside
    another. Those are the two events where the cell's own trace stops meaning
    anything on its own, so the other cell has to be one column away.
    """
    starts = events[events["event"].isin(STARTS)].set_index("identity")
    ends = events[events["event"].isin(ENDS)].set_index("identity")
    divisions = (events[events["event"] == "divided"]
                 .sort_values("frame_index").groupby("identity").first())
    rows = []
    for identity in identities:
        frames = seen[identity]
        span = last[identity] - first[identity] + 1
        divided = divisions.loc[identity] if identity in divisions.index else None
        absorbed = ends.loc[identity, "partner_identity"]             if ends.loc[identity, "event"] == "absorbed" else pd.NA
        rows.append({
            "identity": identity,
            "start_event": starts.loc[identity, "event"],
            "start_status": starts.loc[identity, "event_status"],
            "start_frame": int(first[identity]),
            "start_hours": float(hours[first[identity]]),
            "end_event": ends.loc[identity, "event"],
            "end_status": ends.loc[identity, "event_status"],
            "end_frame": int(last[identity]),
            "end_hours": float(hours[last[identity]]),
            "parent_identity": parents.get(identity, pd.NA),
            "absorbed_into_identity": absorbed,
            "children": int(children.get(identity, 0)),
            "first_division_frame": (int(divided["frame_index"])
                                     if divided is not None else pd.NA),
            "first_division_hours": (float(divided["hours"])
                                     if divided is not None else np.nan),
            "frames_named": len(frames),
            "gap_frames": int(span - len(frames)),
            "observed_hours": float(hours[last[identity]] - hours[first[identity]]),
        })
    return pd.DataFrame(rows, columns=[
        "identity", "start_event", "start_status", "start_frame", "start_hours",
        "end_event", "end_status", "end_frame", "end_hours",
        "parent_identity", "absorbed_into_identity", "children",
        "first_division_frame", "first_division_hours",
        "frames_named", "gap_frames", "observed_hours",
    ])

