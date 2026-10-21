"""Deterministic changes in saved cell footprints, with optional translation removal."""
import numpy as np
import pandas as pd
from .mask_io import labels as read_labels, clock

METHOD_VERSION = "2026-09-28-mask-changes-v1"


def anchor(mask):
    yy, xx = np.nonzero(mask)
    return (int(np.floor(xx.mean() + .5)), int(np.floor(yy.mean() + .5))) if len(xx) else None


def compare(previous, current, *, coordinates="cell"):
    """Return gained/lost pixel coordinates and retained counts for one identity.

    Cell coordinates remove only whole-pixel translation. Output coordinates
    are painted around the current mask. Missing either mask returns None.
    """
    if coordinates not in {"cell", "recording"}:
        raise ValueError("coordinates must be 'cell' or 'recording'")
    before, after = np.asarray(previous, bool), np.asarray(current, bool)
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("compared masks must have the same row and column axes")
    old_anchor, new_anchor = anchor(before), anchor(after)
    if old_anchor is None or new_anchor is None:
        return None
    shift = np.asarray(new_anchor) - old_anchor if coordinates == "cell" else np.zeros(2, int)
    old = {(int(x + shift[0]), int(y + shift[1])) for y, x in np.argwhere(before)}
    now = {(int(x), int(y)) for y, x in np.argwhere(after)}
    return {"gained": sorted(now - old, key=lambda p: (p[1], p[0])),
            "lost": sorted(old - now, key=lambda p: (p[1], p[0])),
            "retained_px": len(now & old), "shift_xy": shift.tolist()}


def analyse(labels, *, identities=None, coordinates="cell", source_frame_offset=0,
            frame_interval_h=.5):
    """Return separate transition summaries and gained/lost pixel tables.

    Only consecutive native observations are compared; gaps are explicit rows
    with missing counts. Identities are never assigned, removed or renumbered.
    """
    if coordinates not in {"cell", "recording"}:
        raise ValueError("coordinates must be 'cell' or 'recording'")
    offset, cadence = clock(source_frame_offset, frame_interval_h)
    labels = read_labels(labels)
    available = [int(value) for value in np.unique(labels) if value]
    selected = available if identities is None else list(dict.fromkeys(map(int, identities)))
    if set(selected) - set(available):
        raise ValueError("requested identities are absent from saved labels")
    rows, pixels = [], []
    for identity in selected:
        for frame in range(len(labels)):
            current = labels[frame] == identity
            previous = labels[frame - 1] == identity if frame else np.zeros_like(current)
            delta = compare(previous, current, coordinates=coordinates)
            native = frame + offset
            rows.append({"identity": identity, "source_frame_index": native,
                "frame_index": frame, "hours": native * cadence,
                "comparison_available": delta is not None, "coordinates": coordinates,
                "gained_px": len(delta["gained"]) if delta else np.nan,
                "lost_px": len(delta["lost"]) if delta else np.nan,
                "held_px": delta["retained_px"] if delta else np.nan})
            if delta is not None:
                for kind in ("gained", "lost"):
                    pixels.extend({"identity": identity, "source_frame_index": native,
                        "frame_index": frame, "hours": native * cadence,
                        "kind": kind, "x_px": x, "y_px": y} for x, y in delta[kind])
    return {"summary": pd.DataFrame(rows, columns=["identity", "source_frame_index", "frame_index",
                "hours", "comparison_available", "coordinates", "gained_px", "lost_px", "held_px"]),
            "pixels": pd.DataFrame(pixels, columns=["identity", "source_frame_index", "frame_index",
                "hours", "kind", "x_px", "y_px"])}
