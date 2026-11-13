"""Restore two consecutive foreign-owner soma seats between early-owner bookends.

This layer runs on the accepted twelfth-round labels. Discovery and selection
use physical tracks and raw signal, never feedback marks or selected IDs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import bridge_strong_owner_flickers as one_frame
import complete_owner_handoff_episodes_r10_a002 as partition
import owner_consensus
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    maximum_step_radii: float = 0.85
    maximum_interpolation_error_radii: float = 0.5
    minimum_relative_point_evidence: float = 0.6
    minimum_relative_raw_peak: float = 0.6
    maximum_adjacent_area_ratio: float = 2.0
    raw_peak_radius_px: int = 2


def _continuous_seats(seats: list, raw: np.ndarray, params: Params) -> bool:
    if len(seats) != 4:
        return False
    for a, b in zip(seats[:-1], seats[1:]):
        radius = max((float(a.radius_px) + float(b.radius_px)) / 2, 1.0)
        step = float(np.hypot(float(a.x) - float(b.x),
                              float(a.y) - float(b.y))) / radius
        if step > params.maximum_step_radii:
            return False
    for i, row in enumerate(seats[1:3], 1):
        fraction = i / 3
        x = (1 - fraction) * float(seats[0].x) + fraction * float(seats[3].x)
        y = (1 - fraction) * float(seats[0].y) + fraction * float(seats[3].y)
        residual = float(np.hypot(float(row.x) - x, float(row.y) - y))
        if residual / max(float(row.radius_px), 1.0) > params.maximum_interpolation_error_radii:
            return False
    evidence = [float(row.evidence) for row in seats]
    if not all(np.isfinite(evidence)):
        return False
    reference = float(np.median((evidence[0], evidence[3])))
    if any(value < params.minimum_relative_point_evidence * reference
           for value in evidence[1:3]):
        return False
    peaks = [one_frame._peak(raw[int(row.frame)], row,
                             params.raw_peak_radius_px) for row in seats]
    reference_peak = float(np.median((peaks[0], peaks[3])))
    return (reference_peak > 0 and all(np.isfinite(peaks))
            and all(value >= params.minimum_relative_raw_peak * reference_peak
                    for value in peaks[1:3]))


def _frame_proposal(candidate: np.ndarray, raw: np.ndarray,
                    protected: np.ndarray, row, owner: int, host: int,
                    track_id: int, qualified: set[int], by_frame: dict,
                    params: partition.Params,
                    reference_area: float) -> tuple[np.ndarray, str] | None:
    t = int(row.frame)
    source = candidate[t]
    mask = partition._at_component(source, host, row)
    if mask is None:
        return None
    others = list(by_frame[t].itertuples(index=False))
    if any(int(other.track_id) != track_id
           and int(other.track_id) in qualified
           and int(other.accepted_owner) == owner
           and bool(other.strong)
           and partition._distinct(row, other, params)
           for other in others):
        return None
    companions = [other for other in others
                  if int(other.track_id) != track_id
                  and int(other.track_id) in qualified
                  and bool(other.strong)
                  and partition._inside(mask, other)
                  and partition._distinct(row, other, params)]
    if any(int(other.accepted_owner) != host for other in companions):
        return None
    if len(companions) > 1:
        selected = partition._partition_many(mask, raw[t], row, companions, params)
        method = "two_frame_multi_soma_raw_watershed"
    elif companions:
        selected = partition._partition(mask, raw[t], row, companions[0], params)
        method = "two_frame_two_soma_raw_watershed"
    else:
        if partition._unexpected_second_core(mask):
            return None
        selected = mask
        method = "two_frame_single_soma_component"
    if selected is None:
        return None
    size = int(selected.sum())
    if not (params.minimum_area_ratio * reference_area <= size
            <= params.maximum_area_ratio * reference_area):
        return None
    if np.any(protected[t] & selected):
        return None
    trial = source.copy()
    trial[selected] = owner
    if owner_consensus.count_new_duplicate_components(source[None], trial[None]):
        return None
    return selected, method


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray | None = None,
            params: Params = Params(),
            partition_params: partition.Params = partition.Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape:
        raise ValueError("raw and labels must align")
    if protected is None:
        protected = np.zeros_like(labels, bool)
    if protected.shape != labels.shape:
        raise ValueError("protected pixels must align")

    candidate = labels.copy()
    scored = physical.attach_owners(points, candidate)
    scored["strong"] = scored.strong.astype(bool)
    scored["physically_visible"] = scored.physically_visible.astype(bool)
    scored["in_encounter"] = scored.in_encounter.astype(bool)
    visible = scored[scored.physically_visible].copy()
    by_frame = {int(t): group for t, group in visible.groupby("frame")}
    qualified = partition._qualified_references(
        scored, partition_params.minimum_reference_strong_frames)
    original_names = np.unique(labels)
    audits, applications, frames = [], [], []
    occupied = np.zeros_like(labels, bool)

    for track_id, group in visible.groupby("track_id", sort=True):
        track_id = int(track_id)
        if track_id not in qualified:
            continue
        group = group.sort_values("frame")
        anchor = one_frame._early_anchor(group)
        if anchor is None:
            continue
        owner = int(anchor["owner"])
        rows = {int(row.frame): row for row in group.itertuples(index=False)}
        for first, left in rows.items():
            seats = [rows.get(first + offset) for offset in range(4)]
            if any(row is None for row in seats):
                continue
            left, mid1, mid2, right = seats
            host = int(mid1.accepted_owner)
            if (int(left.accepted_owner) != owner
                    or int(right.accepted_owner) != owner
                    or host <= 0 or host == owner
                    or int(mid2.accepted_owner) != host
                    or not all(str(row.state) == "observed" and bool(row.strong)
                               for row in seats)):
                continue
            if not _continuous_seats(seats, raw, params):
                continue
            areas = []
            for bookend in (left, right):
                mask = partition._at_component(candidate[int(bookend.frame)],
                                               owner, bookend)
                if mask is None:
                    break
                areas.append(int(mask.sum()))
            if len(areas) != 2:
                continue
            reference_area = float(np.median(areas))
            proposed = []
            for row in (mid1, mid2):
                item = _frame_proposal(candidate, raw, protected, row, owner,
                                       host, track_id, qualified, by_frame,
                                       partition_params, reference_area)
                if item is None or np.any(occupied[int(row.frame)] & item[0]):
                    proposed = []
                    break
                proposed.append(item)
            if len(proposed) != 2:
                continue
            sizes = [int(mask.sum()) for mask, _ in proposed]
            if max(sizes) / min(sizes) > params.maximum_adjacent_area_ratio:
                continue
            base = {"track_id": track_id, "early_owner": owner,
                    "early_first_frame": int(anchor["first"]) + 1,
                    "early_last_frame": int(anchor["last"]) + 1,
                    "episode_first_frame": int(mid1.frame) + 1,
                    "episode_last_frame": int(mid2.frame) + 1,
                    "wrong_frames": 2}
            for row, (mask, method), size in zip((mid1, mid2), proposed, sizes):
                t = int(row.frame)
                candidate[t][mask] = owner
                occupied[t][mask] = True
                applications.append({**base, "frame": t + 1,
                                     "former_owner": host, "method": method,
                                     "changed_pixels": size})
                frames.append({**base, "frame": t + 1,
                               "former_owner": host, "status": "applied",
                               "changed_pixels": size})
            audits.append({**base, "outcome": "applied",
                           "reason": "two_frame_early_owner_gap",
                           "changed_pixels": sum(sizes),
                           "supported_frames": 2, "coverage": 1.0,
                           "unresolved_frames": 0})

    if (not np.array_equal(labels > 0, candidate > 0)
            or np.any((candidate != labels) & protected)
            or not np.array_equal(original_names, np.unique(candidate))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("two-frame bridge crossed an accepted boundary")
    return (candidate, pd.DataFrame(audits),
            pd.DataFrame(applications), pd.DataFrame(frames))
