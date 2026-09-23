"""Original tissue occupancy and event reducers, without drawing."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

MAP_ASSIGNMENT_LABELS = {
    "occupancy_weighted_mean": "average weighted by occupancy time",
    "equal_mean": "equal average across occupants",
    "first": "first occupant",
    "last": "last occupant",
    "most_frequent": "most frequent occupant",
}

def metric_assignment(method: str) -> str:
    """Validate and canonicalise the way overlapping cell values are combined."""
    chosen = str(method).strip().lower().replace("-", "_")
    chosen = {
        "dominant": "most_frequent", "mode": "most_frequent",
        "mean": "equal_mean", "average": "equal_mean",
        "weighted_mean": "occupancy_weighted_mean",
        "weighted_average": "occupancy_weighted_mean",
    }.get(chosen, chosen)
    if chosen not in MAP_ASSIGNMENT_LABELS:
        raise ValueError("--map-assignment must be " + ", ".join(MAP_ASSIGNMENT_LABELS))
    return chosen

def owner_assignment(
    labels: Any,
    method: str = "first",
    *,
    first_owner: Any = None,
) -> np.ndarray:
    """Choose which cell carries a value at every ever-occupied field pixel.

    ``first`` reproduces the tissue-tectonics ownership map. ``last`` uses the
    final named occupant, while ``most_frequent`` uses the identity present for
    the most frames. Ties in ``most_frequent`` go to the smaller identity so a
    repeated build cannot change because dictionary order changed.
    """
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError("owner assignment needs a frame-by-row-by-column label stack")
    chosen = str(method).strip().lower().replace("-", "_")
    aliases = {"dominant": "most_frequent", "mode": "most_frequent"}
    chosen = aliases.get(chosen, chosen)
    if chosen not in {"first", "last", "most_frequent"}:
        raise ValueError("owner assignment must be first, last, or most_frequent")

    if chosen == "first" and first_owner is not None:
        owners = np.asarray(first_owner)
        if owners.shape != stack.shape[1:]:
            raise ValueError("first_owner must match the label field shape")
        return owners.astype(np.int64, copy=True)

    owners = np.zeros(stack.shape[1:], dtype=np.int64)
    if chosen == "first":
        for frame in stack:
            take = (owners == 0) & (frame > 0)
            owners[take] = frame[take]
        return owners
    if chosen == "last":
        for frame in stack:
            take = frame > 0
            owners[take] = frame[take]
        return owners

    largest = np.zeros(stack.shape[1:], dtype=np.int32)
    identities = sorted(int(value) for value in np.unique(stack) if value)
    for identity in identities:
        count = np.count_nonzero(stack == identity, axis=0).astype(np.int32)
        take = count > largest
        owners[take] = identity
        largest[take] = count[take]
    return owners

def _value_limits(values: Any, range_mode: str) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("a spatial value map needs at least one finite value")
    mode = str(range_mode).strip().lower()
    if mode == "robust":
        low, high = map(float, np.nanpercentile(finite, [2.0, 98.0]))
    elif mode == "full":
        low, high = float(finite.min()), float(finite.max())
    else:
        raise ValueError("map range must be robust or full")
    if high <= low:
        padding = max(abs(low) * 0.05, 0.5)
        low, high = low - padding, high + padding
    return low, high

def first_coverage_time(labels: Any, hours: Any) -> np.ndarray:
    """Elapsed hours when each eventually occupied pixel was first covered."""
    stack = np.asarray(labels)
    times = np.asarray(hours, dtype=float)
    if stack.ndim != 3 or times.shape != (len(stack),):
        raise ValueError("first coverage needs a frame-by-row-by-column label stack and one time per frame")
    if not len(stack) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("first-coverage times must be finite and strictly increasing")
    if not np.isfinite(stack).all() or np.any(stack < 0):
        raise ValueError("labels must be finite and nonnegative; missing observations are not background")

    occupied = stack > 0
    visited = occupied.any(axis=0)
    first_indices = np.argmax(occupied, axis=0)
    elapsed = np.full(stack.shape[1:], np.nan, dtype=float)
    elapsed[visited] = times[first_indices[visited]] - times[0]
    return elapsed

def cumulative_occupancy(labels: Any, hours: Any) -> np.ndarray:
    """Cumulative occupied hours per pixel, pooling all nonzero cell identities.

    Integrate observed occupancy between successive timestamps using the mean
    of the two endpoint states. This is equivalent to assigning half each
    observed interval to each endpoint; never extrapolate beyond the recording.
    The first frame has zero accumulated elapsed time. Missing observations are
    not accepted as background, and identities cannot double-count a pixel.
    """
    stack = np.asarray(labels)
    times = np.asarray(hours, float)
    if stack.ndim != 3 or len(stack) < 2 or times.shape != (len(stack),):
        raise ValueError("cumulative occupancy needs at least two labelled frames and one time per frame")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("occupancy times must be finite and strictly increasing")
    if not np.isfinite(stack).all() or np.any(stack < 0):
        raise ValueError("labels must be finite and nonnegative; missing observations are not background")
    occupied = (stack > 0).astype(float)
    increments = (occupied[:-1] + occupied[1:]) * .5 * np.diff(times)[:, None, None]
    return np.concatenate([np.zeros_like(increments[:1]), np.cumsum(increments, axis=0)])

def _event_identities(event: Any) -> list[int]:
    """Positive identities named on one merge/split event row."""
    identities = [int(event.identity)]
    raw = getattr(event, "candidate_identities", "")
    if not pd.isna(raw):
        for token in str(raw).split("|"):
            token = token.strip()
            if token and token.lower() != "nan":
                identities.append(int(float(token)))
    return sorted({identity for identity in identities if identity > 0})

def split_event_areas(
    labels: Any,
    events: pd.DataFrame,
    *,
    event_type: str = "contact_separate",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Count recorded contact-separation transition footprints at each pixel.

    One event area is the union of its named cells' last connected footprint and
    first separated footprint. This localises the transition without treating a
    tracker split as biological cell division.
    """
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError("split-event areas need a frame-by-row-by-column label stack")
    required = {
        "event_id", "event_type", "identity", "candidate_identities",
        "gap_end_frame_index", "post_frame_index",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError("split-event table is missing " + ", ".join(missing))

    selected = events.loc[events["event_type"].astype(str).eq(event_type)].copy()
    counts = np.zeros(stack.shape[1:], dtype=np.uint32)
    records: list[dict[str, Any]] = []
    for event in selected.itertuples(index=False):
        connected_frame = int(event.gap_end_frame_index)
        separated_frame = int(event.post_frame_index)
        if not (0 <= connected_frame < len(stack) and 0 <= separated_frame < len(stack)):
            raise ValueError(
                f"split event {event.event_id} names frames outside the label movie"
            )
        identities = _event_identities(event)
        area = (
            np.isin(stack[connected_frame], identities)
            | np.isin(stack[separated_frame], identities)
        )
        counts += area.astype(np.uint32)
        records.append({
            "event_id": str(event.event_id),
            "event_type": str(event.event_type),
            "identity": int(event.identity),
            "candidate_identities": "|".join(
                str(identity) for identity in identities
                if identity != int(event.identity)
            ),
            "connected_frame_index": connected_frame,
            "separated_frame_index": separated_frame,
            "event_area_px": int(np.count_nonzero(area)),
        })
    columns = [
        "event_id", "event_type", "identity", "candidate_identities",
        "connected_frame_index", "separated_frame_index", "event_area_px",
    ]
    return counts, pd.DataFrame(records, columns=columns)