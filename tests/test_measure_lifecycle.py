"""What the lifecycle module is allowed to call a division, a death and a loss.

Every case here is drawn by hand: rectangles on a small field, moved and removed
on known frames. That is the only way to test a rule whose whole purpose is to
separate cases that look identical in a label stack - on real pixels there is no
answer to check against, which is exactly why the module writes a status column
instead of a verdict.

Run with ``python -m pytest analysis/test_lifecycle.py``.

Ported from Motion's ``analysis/test_lifecycle.py`` on 2026-09-21 (stage 04 of the Motion port).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymicroglia.measure.modules import lifecycle
from pymicroglia.measure import MeasurementContext, Scale


FRAMES, HEIGHT, WIDTH = 20, 60, 60


def _context(labels: np.ndarray, *, unclaimed: np.ndarray | None = None,
             params: dict | None = None) -> MeasurementContext:
    """A movie of nothing but the outlines the case needs.

    ``raw`` is the labels themselves: no rule in this module reads the signal,
    and a synthetic intensity would only invite one to start.
    """
    return MeasurementContext(
        stem="synthetic",
        labels=labels,
        raw=(labels > 0).astype(np.uint16) * 100,
        scale=Scale(minutes_per_frame=30, microns_per_pixel=None),
        identities=sorted({int(v) for v in np.unique(labels) if v}),
        unclaimed=unclaimed,
        params={"lifecycle": dict(params or {})},
    )


def _blank(unclaimed: bool = False) -> np.ndarray:
    return np.zeros((FRAMES, HEIGHT, WIDTH), dtype=np.uint16)


def _events(context: MeasurementContext) -> pd.DataFrame:
    return lifecycle.measure(context)["lifecycle_events"]


def _row(events: pd.DataFrame, identity: int, event: str) -> pd.Series:
    found = events[(events["identity"] == identity) & (events["event"] == event)]
    assert len(found) == 1, f"expected one {event!r} for cell {identity}, got {len(found)}"
    return found.iloc[0]


# ----------------------------------------------------------------- divisions

def _dividing_pair(*, split_frame: int = 8, persist: int = 20,
                   daughter_area: int = 48) -> np.ndarray:
    """One 96-pixel cell that becomes two touching cells of the same total area."""
    labels = _blank()
    for frame in range(split_frame):
        labels[frame, 20:28, 20:32] = 1                       # 8 x 12 = 96 px
    for frame in range(split_frame, min(persist, FRAMES)):
        labels[frame, 20:28, 20:26] = 1                       # 8 x 6  = 48 px
        width = max(daughter_area // 8, 1)
        labels[frame, 20:28, 27:27 + width] = 2
    return labels


def test_an_area_conserving_separation_is_a_division_candidate() -> None:
    events = _events(_context(_dividing_pair()))
    born = _row(events, 2, "born")
    divided = _row(events, 1, "divided")
    assert born["event_status"] == "candidate"
    assert born["partner_identity"] == 1
    assert divided["partner_identity"] == 2
    assert born["frame_index"] == divided["frame_index"] == 8
    assert born["area_conservation"] == pytest.approx(1.0, abs=0.01)


def test_a_separation_that_loses_most_of_the_parent_is_not_a_division() -> None:
    """Two parts far smaller than the parent are a segmentation change, not a split."""
    events = _events(_context(_dividing_pair(daughter_area=8)))
    assert _row(events, 2, "appeared")["event_status"] == "unproven"
    assert events[events["event"] == "divided"].empty


def test_a_second_name_that_flickers_for_one_frame_is_not_a_division() -> None:
    events = _events(_context(_dividing_pair(persist=9)))
    assert _row(events, 2, "appeared")["event_status"] == "unproven"


def test_a_cell_appearing_alone_mid_field_is_left_unproven() -> None:
    labels = _blank()
    labels[:, 10:18, 10:18] = 1
    labels[8:, 40:48, 40:48] = 2
    events = _events(_context(labels))
    assert _row(events, 2, "appeared")["event_status"] == "unproven"


# ------------------------------------------------------------------- endings

def _vanishing_cell(*, until: int = 12, leaves_foreground: bool = False) -> tuple:
    labels = _blank()
    labels[:, 10:18, 10:18] = 1                                # runs throughout
    for frame in range(until):
        labels[frame, 30:38, 30:38] = 2
    unclaimed = _blank()
    if leaves_foreground:
        unclaimed[until:, 30:38, 30:38] = 1
    return labels, unclaimed


def test_ground_that_goes_dark_is_a_death_candidate() -> None:
    labels, unclaimed = _vanishing_cell()
    row = _row(_events(_context(labels, unclaimed=unclaimed)), 2, "died")
    assert row["event_status"] == "candidate"
    assert row["residual_foreground_px"] == 0


def test_ground_that_keeps_its_foreground_is_a_loss_not_a_death() -> None:
    labels, unclaimed = _vanishing_cell(leaves_foreground=True)
    events = _events(_context(labels, unclaimed=unclaimed))
    assert _row(events, 2, "lost")["event_status"] == "unproven"
    assert events[events["event"] == "died"].empty


def test_without_an_unclaimed_stack_no_ending_is_called_a_death() -> None:
    """The two cases are one picture, so the module refuses to choose between them."""
    labels, _ = _vanishing_cell()
    events = _events(_context(labels))
    assert _row(events, 2, "vanished")["event_status"] == "unproven"
    assert events[events["event"].isin(["died", "lost"])].empty


def test_a_cell_taken_over_by_its_neighbour_is_absorbed() -> None:
    labels = _blank()
    labels[:, 20:28, 20:28] = 1
    for frame in range(10):
        labels[frame, 20:28, 28:36] = 2
    labels[10:, 20:28, 28:36] = 1                              # cell 1 takes the ground
    row = _row(_events(_context(labels, unclaimed=_blank())), 2, "absorbed")
    assert row["event_status"] == "candidate" and row["partner_identity"] == 1


# ------------------------------------------------- the edges of the recording

def test_the_field_edge_settles_an_arrival_and_a_departure() -> None:
    labels = _blank()
    for frame in range(6, 14):
        labels[frame, 0:8, 20:28] = 1                          # against the top edge
    events = _events(_context(labels, unclaimed=_blank()))
    assert _row(events, 1, "entered_field")["event_status"] == "observed"
    assert _row(events, 1, "left_field")["event_status"] == "observed"


def test_the_recording_edges_are_censored_not_events() -> None:
    labels = _blank()
    labels[:, 20:28, 20:28] = 1
    events = _events(_context(labels, unclaimed=_blank()))
    assert _row(events, 1, "present_at_start")["event_status"] == "censored"
    assert _row(events, 1, "present_at_end")["event_status"] == "censored"


# ------------------------------------------------------------ the cell ledger

def test_every_cell_gets_one_beginning_and_one_ending() -> None:
    labels = _dividing_pair()
    tables = lifecycle.measure(_context(labels, unclaimed=_blank()))
    cells = tables["lifecycle_cells"]
    assert list(cells["identity"]) == [1, 2]
    assert cells.set_index("identity").loc[2, "parent_identity"] == 1
    assert cells.set_index("identity").loc[1, "children"] == 1
    assert set(cells["start_event"]) <= set(lifecycle.STARTS)
    assert set(cells["end_event"]) <= set(lifecycle.ENDS)


def test_a_cell_row_carries_every_moment_as_a_frame_as_well_as_an_hour() -> None:
    """The hour is for the reader; the frame is what every other table joins on."""
    labels = _dividing_pair()
    cells = lifecycle.measure(_context(labels, unclaimed=_blank()))["lifecycle_cells"]
    parent = cells.set_index("identity").loc[1]
    assert parent["start_frame"] == 0 and parent["end_frame"] == labels.shape[0] - 1
    assert parent["first_division_frame"] == 8                 # where the pair splits
    assert parent["first_division_hours"] == 8 * 0.5
    daughter = cells.set_index("identity").loc[2]
    assert daughter["start_frame"] == 8
    assert pd.isna(daughter["first_division_frame"])           # it never divided


def test_a_cell_that_ends_inside_another_names_the_one_that_took_it() -> None:
    labels = _blank()
    labels[:, 20:28, 20:28] = 1
    for frame in range(10):
        labels[frame, 20:28, 28:36] = 2
    labels[10:, 20:28, 28:36] = 1
    cells = lifecycle.measure(_context(labels, unclaimed=_blank()))["lifecycle_cells"]
    taken = cells.set_index("identity").loc[2]
    assert taken["end_event"] == "absorbed" and taken["absorbed_into_identity"] == 1
    assert pd.isna(cells.set_index("identity").loc[1, "absorbed_into_identity"])


def test_a_gap_is_counted_without_becoming_an_event() -> None:
    labels = _blank()
    labels[:, 20:28, 20:28] = 1
    labels[6:9, 20:28, 20:28] = 0                              # three missing frames
    tables = lifecycle.measure(_context(labels, unclaimed=_blank()))
    assert tables["lifecycle_cells"].loc[0, "gap_frames"] == 3
    assert set(tables["lifecycle_events"]["event"]) == {"present_at_start", "present_at_end"}


# ------------------------------------------------------------- configuration

def test_an_unknown_setting_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="lifecycle does not accept"):
        _events(_context(_blank(), params={"divide_threshold": 2}))


def test_a_tolerance_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match="area_conservation_tolerance"):
        _events(_context(_blank(), params={"area_conservation_tolerance": 4}))


def test_the_conservation_tolerance_is_what_decides_a_division() -> None:
    labels = _dividing_pair(daughter_area=24)                  # 75% of the parent
    strict = _events(_context(labels, params={"area_conservation_tolerance": 0.1}))
    loose = _events(_context(labels, params={"area_conservation_tolerance": 0.5}))
    assert strict[strict["event"] == "born"].empty
    assert not loose[loose["event"] == "born"].empty
