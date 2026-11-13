"""Restore a large one-frame foreign name with two-sided soma and mask evidence.

This additive producer reads only approved masks, raw frames, soma points and
protected pixels. It has no feedback, well, identity, track or region selector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage.measure import label as equal_value_components

import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_component_area: int = 150
    minimum_two_sided_overlap: float = .65
    maximum_step_radii: float = 1.0
    minimum_raw_peak_retention: float = .5
    minimum_pair_valley_ratio: float = .85
    raw_peak_radius_px: int = 2


def _pixel_owner(frame: np.ndarray, point) -> int:
    y, x = round(float(point.y)), round(float(point.x))
    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
        return int(frame[y, x])
    return 0


def _raw_peak(frame: np.ndarray, point, radius: int) -> float:
    y, x = round(float(point.y)), round(float(point.x))
    return float(np.max(frame[max(0, y-radius):min(frame.shape[0], y+radius+1),
                              max(0, x-radius):min(frame.shape[1], x+radius+1)]))


def _resident_continuity(track: dict[int, object], t: int, owner: int,
                         labels: np.ndarray, raw: np.ndarray,
                         params: Params) -> bool:
    if any(u not in track for u in (t-1, t, t+1)):
        return False
    points = [track[u] for u in (t-1, t, t+1)]
    if any(str(point.state) == "dormant" for point in points):
        return False
    if _pixel_owner(labels[t-1], points[0]) != owner or _pixel_owner(labels[t+1], points[2]) != owner:
        return False
    for a, b in zip(points[:-1], points[1:]):
        radius = max((float(a.radius_px)+float(b.radius_px))/2, 1.)
        if np.hypot(float(a.x)-float(b.x), float(a.y)-float(b.y)) > params.maximum_step_radii*radius:
            return False
    peaks = [_raw_peak(raw[t+delta], point, params.raw_peak_radius_px)
             for delta, point in zip((-1, 0, 1), points)]
    return (np.isfinite(peaks).all() and peaks[1] > 0
            and peaks[1] >= params.minimum_raw_peak_retention
            * float(np.median((peaks[0], peaks[2]))))


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or labels.shape != protected.shape
            or raw.shape[:3] != labels.shape):
        raise ValueError("labels, raw and protected pixels must align")
    strong = points[(points.state.astype(str) == "observed")
                    & points.strong.astype(bool)]
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    by_track = {int(track_id): {int(p.frame): p for p in group.itertuples(index=False)}
                for track_id, group in points.groupby("track_id")}
    candidate = labels.copy()
    remaining = np.bincount(labels.ravel().astype(np.int64))
    audit, applications = [], []
    for t in range(1, len(labels)-1):
        frame = labels[t]
        components = equal_value_components(frame, connectivity=2, background=0)
        areas = np.bincount(components.ravel())
        at_frame = by_frame.get(t, strong.iloc[0:0])
        resident_ids: dict[int, list] = {}
        for point in at_frame.itertuples(index=False):
            y, x = round(float(point.y)), round(float(point.x))
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                cid = int(components[y, x])
                if cid:
                    resident_ids.setdefault(cid, []).append(point)
        for cid in np.flatnonzero(areas >= params.minimum_component_area):
            cid = int(cid)
            if cid == 0 or cid not in resident_ids:
                continue
            mask = components == cid
            old_name = int(frame[mask][0])
            before_counts = np.bincount(labels[t-1][mask].astype(np.int64))
            after_counts = np.bincount(labels[t+1][mask].astype(np.int64))
            threshold = params.minimum_two_sided_overlap * int(areas[cid])
            owners = set(np.flatnonzero(before_counts >= threshold)) & set(
                np.flatnonzero(after_counts >= threshold))
            owners.discard(0)
            owners.discard(old_name)
            if len(owners) != 1:
                continue
            new_name = int(next(iter(owners)))
            base = {"frame": t+1, "component": cid,
                    "old_name": old_name, "new_name": new_name,
                    "component_area": int(areas[cid]),
                    "before_overlap": float(before_counts[new_name]/areas[cid]),
                    "after_overlap": float(after_counts[new_name]/areas[cid]),
                    "strong_residents": len(resident_ids[cid])}
            reason = ""
            if np.any(protected[t] & mask):
                reason = "accepted_pixel_in_component"
            elif remaining[old_name] <= areas[cid]:
                reason = "would_remove_established_name"
            elif any(not _resident_continuity(by_track[int(point.track_id)], t,
                                               new_name, labels, raw, params)
                     for point in resident_ids[cid]):
                reason = "strong_soma_not_bookended"
            elif len(resident_ids[cid]) > 1:
                for i, point in enumerate(resident_ids[cid]):
                    for other in resident_ids[cid][i+1:]:
                        valley = physical._valley_ratio(
                            raw[t], (float(point.x), float(point.y)),
                            (float(other.x), float(other.y)))
                        if not np.isfinite(valley) or valley < params.minimum_pair_valley_ratio:
                            reason = "independent_raw_somata_share_component"
                            break
                    if reason:
                        break
            if not reason and not np.all(candidate[t][mask] == old_name):
                reason = "overlapping_prior_proposal"
            if reason:
                audit.append({**base, "outcome": "rejected", "reason": reason})
                continue
            candidate[t][mask] = new_name
            remaining[old_name] -= int(areas[cid])
            audit.append({**base, "outcome": "applied", "reason": ""})
            applications.append(base)
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("large name flicker crossed an accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
