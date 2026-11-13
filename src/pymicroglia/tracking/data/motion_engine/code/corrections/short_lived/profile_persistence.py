"""Measure whole-movie identity persistence before any end-stage changes."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
import sys
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack  # noqa: E402


def consecutive_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in map(int, indices[1:]):
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    return float(np.mean(yy)), float(np.mean(xx))


# Speed (folded from persistence_profile_reuse): every identity's per-frame
# area is counted in one np.unique pass per frame, and an anchor's centroid is
# read inside the identity's find_objects box. Every pixel of the name lies in
# the box, so np.nonzero of the box plus its offset gives the same coordinates
# in the same row-major order and the means are identical. Frames that are not
# 2-D non-negative integer labels, and names without a box, use centroid().


def _identity_areas(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Each nonzero identity's pixel count in every frame."""
    result: dict[int, np.ndarray] = {}
    for t, frame in enumerate(labels):
        values, counts = np.unique(frame, return_counts=True)
        for value, count in zip(values, counts):
            identity = int(value)
            if not identity:
                continue
            if identity not in result:
                result[identity] = np.zeros(len(labels), dtype=int)
            result[identity][t] = count
    return result


def _frame_boxes(store: OrderedDict, key, frame):
    """find_objects of one integer label frame (False when it is not one), kept
    for the 64 latest frames."""
    boxes = store.get(key)
    if boxes is not None:
        store.move_to_end(key)
        return boxes
    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.dtype.kind not in "iu" or (
            frame.dtype.kind == "i" and frame.size and frame.min() < 0):
        boxes = False
    else:
        boxes = find_objects(frame)
    store[key] = boxes
    if len(store) > 64:
        store.popitem(last=False)
    return boxes


def _box_centroid(frame, identity: int, boxes) -> tuple[float, float] | None:
    """The centroid of `frame == identity` read inside its box, or None when it
    has no box."""
    if boxes is False or not 0 < identity <= len(boxes) or boxes[identity - 1] is None:
        return None
    y, x = boxes[identity - 1]
    yy, xx = np.nonzero(np.asarray(frame)[y, x] == identity)
    yy += y.start
    xx += x.start
    return float(np.mean(yy)), float(np.mean(xx))


class _Centroids(OrderedDict):
    """One profile call's centroids by (frame, identity), with its frame boxes."""

    def __init__(self):
        super().__init__()
        self.boxes = OrderedDict()


def _cached_centroid(labels: np.ndarray, frame: int, identity: int,
                     cache: _Centroids) -> tuple[float, float]:
    key = (frame, identity)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = _box_centroid(labels[frame], identity,
                          _frame_boxes(cache.boxes, frame, labels[frame]))
    if value is None:
        value = centroid(labels[frame] == identity)
    cache[key] = value
    if len(cache) > 32768:
        cache.popitem(last=False)
    return value


def profile(labels: np.ndarray, authority: pd.DataFrame, params: dict
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.dtype.kind not in "iu":
        return _profile_whole(labels, authority, params)
    offset = int(params["source_frame_offset"])
    cfg = params["profile"]
    window = int(cfg["local_area_window_frames"])
    fraction = float(cfg["deficit_area_fraction"])
    maximum_gap = int(cfg["maximum_bracket_gap_frames"])
    authority_by_id = authority.set_index("identity").to_dict("index")
    profiles: list[dict] = []
    deficits: list[dict] = []
    area_index = _identity_areas(labels)
    centroids = _Centroids()
    identities = sorted(area_index)
    for identity in identities:
        areas = area_index[identity]
        present = np.flatnonzero(areas > 0)
        runs = consecutive_runs(present)
        auth = authority_by_id.get(identity, {})
        profiles.append({
            "identity": identity,
            "frames_present": int(len(present)),
            "first_output_frame": int(present[0] + 1),
            "last_output_frame": int(present[-1] + 1),
            "first_source_imagej_frame": int(present[0] + 1 + offset),
            "last_source_imagej_frame": int(present[-1] + 1 + offset),
            "span_frames": int(present[-1] - present[0] + 1),
            "run_count": int(len(runs)),
            "maximum_consecutive_frames": int(max(e - s + 1 for s, e in runs)),
            "edge_censored": bool(present[0] == 0 or present[-1] == len(labels) - 1),
            "median_area_px": float(np.median(areas[present])),
            "minimum_area_px": int(np.min(areas[present])),
            "maximum_area_px": int(np.max(areas[present])),
            "total_identity_pixels": int(np.sum(areas)),
            "confidence_layer": auth.get("confidence_layer", "unknown"),
            "priority_rank": auth.get("priority_rank", ""),
            "permanently_birth_small": auth.get("permanently_birth_small", ""),
        })
        if len(present) < int(cfg["persistent_minimum_frames"]):
            continue
        for frame in range(1, len(labels) - 1):
            before = present[present < frame]
            after = present[present > frame]
            if not len(before) or not len(after):
                continue
            previous, following = int(before[-1]), int(after[0])
            gap = following - previous - 1
            if gap > maximum_gap:
                continue
            local = areas[max(0, frame - window):min(len(labels), frame + window + 1)]
            local = local[local > 0]
            if not len(local):
                continue
            reference = float(np.median(local))
            current = int(areas[frame])
            if current > 0 and current >= fraction * reference:
                continue
            prev_y, prev_x = _cached_centroid(labels, previous, identity, centroids)
            next_y, next_x = _cached_centroid(labels, following, identity, centroids)
            alpha = (frame - previous) / max(following - previous, 1)
            deficits.append({
                "identity": identity,
                "output_frame": frame + 1,
                "source_imagej_frame": frame + 1 + offset,
                "deficit_type": "missing" if current == 0 else "shrunken",
                "current_area_px": current,
                "reference_area_px": reference,
                "area_fraction": current / max(reference, 1.0),
                "previous_anchor_output_frame": previous + 1,
                "next_anchor_output_frame": following + 1,
                "bracket_gap_frames": gap,
                "predicted_y": prev_y + alpha * (next_y - prev_y),
                "predicted_x": prev_x + alpha * (next_x - prev_x),
            })
    return pd.DataFrame(profiles), pd.DataFrame(deficits)


def _profile_whole(labels: np.ndarray, authority: pd.DataFrame,
                   params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """profile for anything but a 3-D integer label stack, counting each
    identity in every frame."""
    offset = int(params["source_frame_offset"])
    cfg = params["profile"]
    window = int(cfg["local_area_window_frames"])
    fraction = float(cfg["deficit_area_fraction"])
    maximum_gap = int(cfg["maximum_bracket_gap_frames"])
    authority_by_id = authority.set_index("identity").to_dict("index")
    profiles: list[dict] = []
    deficits: list[dict] = []
    identities = sorted(set(map(int, np.unique(labels))) - {0})
    for identity in identities:
        areas = np.asarray(
            [np.count_nonzero(frame == identity) for frame in labels], int)
        present = np.flatnonzero(areas > 0)
        runs = consecutive_runs(present)
        auth = authority_by_id.get(identity, {})
        profiles.append({
            "identity": identity,
            "frames_present": int(len(present)),
            "first_output_frame": int(present[0] + 1),
            "last_output_frame": int(present[-1] + 1),
            "first_source_imagej_frame": int(present[0] + 1 + offset),
            "last_source_imagej_frame": int(present[-1] + 1 + offset),
            "span_frames": int(present[-1] - present[0] + 1),
            "run_count": int(len(runs)),
            "maximum_consecutive_frames": int(max(e - s + 1 for s, e in runs)),
            "edge_censored": bool(present[0] == 0 or present[-1] == len(labels) - 1),
            "median_area_px": float(np.median(areas[present])),
            "minimum_area_px": int(np.min(areas[present])),
            "maximum_area_px": int(np.max(areas[present])),
            "total_identity_pixels": int(np.sum(areas)),
            "confidence_layer": auth.get("confidence_layer", "unknown"),
            "priority_rank": auth.get("priority_rank", ""),
            "permanently_birth_small": auth.get("permanently_birth_small", ""),
        })
        if len(present) < int(cfg["persistent_minimum_frames"]):
            continue
        for frame in range(1, len(labels) - 1):
            before = present[present < frame]
            after = present[present > frame]
            if not len(before) or not len(after):
                continue
            previous, following = int(before[-1]), int(after[0])
            gap = following - previous - 1
            if gap > maximum_gap:
                continue
            local = areas[max(0, frame - window):min(len(labels), frame + window + 1)]
            local = local[local > 0]
            if not len(local):
                continue
            reference = float(np.median(local))
            current = int(areas[frame])
            if current > 0 and current >= fraction * reference:
                continue
            prev_y, prev_x = centroid(labels[previous] == identity)
            next_y, next_x = centroid(labels[following] == identity)
            alpha = (frame - previous) / max(following - previous, 1)
            deficits.append({
                "identity": identity,
                "output_frame": frame + 1,
                "source_imagej_frame": frame + 1 + offset,
                "deficit_type": "missing" if current == 0 else "shrunken",
                "current_area_px": current,
                "reference_area_px": reference,
                "area_fraction": current / max(reference, 1.0),
                "previous_anchor_output_frame": previous + 1,
                "next_anchor_output_frame": following + 1,
                "bracket_gap_frames": gap,
                "predicted_y": prev_y + alpha * (next_y - prev_y),
                "predicted_x": prev_x + alpha * (next_x - prev_x),
            })
    return pd.DataFrame(profiles), pd.DataFrame(deficits)

