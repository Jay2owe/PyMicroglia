"""Bridge raw-visible soma frames with one temporarily dormant track point.

The approved twelfth-round result is the parent. A two-frame owner-gap layer
runs first; this separate layer adds only raw-supported single-frame dropouts.
Neither layer selects feedback identities, recordings, frames or regions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import bridge_strong_owner_flickers as early
import bridge_two_frame_owner_gaps as pair
import complete_owner_handoff_episodes_r10_a002 as partition
import owner_consensus
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    maximum_step_radii: float = 0.85
    maximum_midpoint_error_radii: float = 0.5
    minimum_relative_raw_peak: float = 0.5
    raw_peak_radius_px: int = 2


def _raw_visible_bridge(prev, row, following, raw: np.ndarray,
                        params: Params) -> bool:
    seats = (prev, row, following)
    for a, b in zip(seats[:-1], seats[1:]):
        radius = max((float(a.radius_px) + float(b.radius_px)) / 2, 1.0)
        step = float(np.hypot(float(a.x) - float(b.x),
                              float(a.y) - float(b.y))) / radius
        if step > params.maximum_step_radii:
            return False
    midpoint_error = float(np.hypot(
        float(row.x) - (float(prev.x) + float(following.x)) / 2,
        float(row.y) - (float(prev.y) + float(following.y)) / 2))
    if midpoint_error / max(float(row.radius_px), 1.0) > params.maximum_midpoint_error_radii:
        return False
    peaks = [early._peak(raw[int(seat.frame)], seat,
                         params.raw_peak_radius_px) for seat in seats]
    reference = float(np.median((peaks[0], peaks[2])))
    return (reference > 0 and all(np.isfinite(peaks))
            and peaks[1] >= params.minimum_relative_raw_peak * reference)


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

    candidate, audits, applications, frames = pair.correct(
        labels, raw, points, protected, partition_params=partition_params)
    scored = physical.attach_owners(points, candidate)
    scored["strong"] = scored.strong.astype(bool)
    scored["physically_visible"] = scored.physically_visible.astype(bool)
    scored["in_encounter"] = scored.in_encounter.astype(bool)
    visible = scored[scored.physically_visible].copy()
    by_frame = {int(t): group for t, group in visible.groupby("frame")}
    qualified = partition._qualified_references(
        scored, partition_params.minimum_reference_strong_frames)
    extra_audits, extra_applications, extra_frames = [], [], []

    for track_id, group in scored.groupby("track_id", sort=True):
        track_id = int(track_id)
        if track_id not in qualified:
            continue
        group = group.sort_values("frame")
        anchor = early._early_anchor(group)
        if anchor is None:
            continue
        owner = int(anchor["owner"])
        rows = {int(row.frame): row for row in group.itertuples(index=False)}
        for t, row in rows.items():
            host = int(row.accepted_owner)
            if (str(row.state) != "dormant" or bool(row.physically_visible)
                    or host <= 0 or host == owner):
                continue
            prev, following = rows.get(t - 1), rows.get(t + 1)
            if (prev is None or following is None
                    or int(prev.accepted_owner) != owner
                    or int(following.accepted_owner) != owner
                    or not bool(prev.physically_visible)
                    or not bool(following.physically_visible)
                    or str(prev.state) != "observed"
                    or str(following.state) != "observed"):
                continue
            if not _raw_visible_bridge(prev, row, following, raw, params):
                continue
            source = candidate[t]
            mask = partition._at_component(source, host, row)
            if mask is None:
                continue
            others = list(by_frame.get(t, pd.DataFrame()).itertuples(index=False))
            if any(int(other.track_id) != track_id
                   and int(other.track_id) in qualified
                   and int(other.accepted_owner) == owner
                   and bool(other.strong)
                   and partition._distinct(row, other, partition_params)
                   for other in others):
                continue
            companions = [other for other in others
                          if int(other.track_id) != track_id
                          and int(other.track_id) in qualified
                          and bool(other.strong)
                          and partition._inside(mask, other)
                          and partition._distinct(row, other, partition_params)]
            if any(int(other.accepted_owner) != host for other in companions):
                continue
            if len(companions) > 1:
                selected = partition._partition_many(
                    mask, raw[t], row, companions, partition_params)
                method = "dormant_raw_multi_soma_watershed"
            elif companions:
                selected = partition._partition(
                    mask, raw[t], row, companions[0], partition_params)
                method = "dormant_raw_two_soma_watershed"
            else:
                if partition._unexpected_second_core(mask):
                    continue
                selected = mask
                method = "dormant_raw_single_soma_component"
            if selected is None:
                continue
            reference_areas = []
            for neighbor in (prev, following):
                reference = partition._at_component(
                    candidate[int(neighbor.frame)], owner, neighbor)
                if reference is not None:
                    reference_areas.append(int(reference.sum()))
            if len(reference_areas) != 2:
                continue
            reference_area = float(np.median(reference_areas))
            size = int(selected.sum())
            if not (partition_params.minimum_area_ratio * reference_area <= size
                    <= partition_params.maximum_area_ratio * reference_area):
                continue
            if np.any(protected[t] & selected) or np.any((candidate[t] != labels[t]) & selected):
                continue
            trial = source.copy()
            trial[selected] = owner
            if owner_consensus.count_new_duplicate_components(source[None], trial[None]):
                continue
            candidate[t] = trial
            base = {"track_id": track_id, "early_owner": owner,
                    "early_first_frame": int(anchor["first"]) + 1,
                    "early_last_frame": int(anchor["last"]) + 1,
                    "episode_first_frame": t + 1,
                    "episode_last_frame": t + 1,
                    "wrong_frames": 1}
            extra_audits.append({**base, "outcome": "applied",
                                 "reason": "raw_visible_dormant_soma_bridge",
                                 "changed_pixels": size,
                                 "supported_frames": 1, "coverage": 1.0,
                                 "unresolved_frames": 0})
            extra_applications.append({**base, "frame": t + 1,
                                       "former_owner": host, "method": method,
                                       "changed_pixels": size})
            extra_frames.append({**base, "frame": t + 1,
                                 "former_owner": host, "status": "applied",
                                 "changed_pixels": size})

    if (not np.array_equal(labels > 0, candidate > 0)
            or np.any((candidate != labels) & protected)
            or not np.array_equal(np.unique(labels), np.unique(candidate))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("dormant bridge crossed an accepted boundary")
    return (candidate,
            pd.concat([audits, pd.DataFrame(extra_audits)], ignore_index=True),
            pd.concat([applications, pd.DataFrame(extra_applications)],
                      ignore_index=True),
            pd.concat([frames, pd.DataFrame(extra_frames)], ignore_index=True))
