"""Fill isolated weak observations between two approved strong soma seats.

This is a layer after the accepted multi-soma owner rule. It never selects a
track, identity, recording, frame or coordinate from feedback annotations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import complete_owner_handoff_episodes_r10_a002 as parent_rule
import owner_consensus
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_strong_owner_frames: int = 10
    maximum_step_radii: float = 0.85
    maximum_midpoint_error_radii: float = 0.5
    minimum_relative_point_evidence: float = 0.6
    minimum_relative_raw_peak: float = 0.6
    raw_peak_radius_px: int = 2


def _peak(frame: np.ndarray, row, radius: int) -> float:
    x, y = int(round(float(row.x))), int(round(float(row.y)))
    h, w = frame.shape
    patch = frame[max(0, y - radius):min(h, y + radius + 1),
                  max(0, x - radius):min(w, x + radius + 1)]
    return float(np.max(patch)) if patch.size else 0.0


def _continuity(prev, row, following, raw: np.ndarray,
                params: Params) -> bool:
    points = (prev, row, following)
    for a, b in zip(points[:-1], points[1:]):
        radius = max((float(a.radius_px) + float(b.radius_px)) / 2, 1.0)
        step = float(np.hypot(float(a.x) - float(b.x),
                              float(a.y) - float(b.y))) / radius
        if step > params.maximum_step_radii:
            return False
    radius = max(float(row.radius_px), 1.0)
    midpoint_error = float(np.hypot(
        float(row.x) - (float(prev.x) + float(following.x)) / 2,
        float(row.y) - (float(prev.y) + float(following.y)) / 2)) / radius
    if midpoint_error > params.maximum_midpoint_error_radii:
        return False
    evidence = [float(p.evidence) for p in points]
    if (not all(np.isfinite(evidence)) or
            evidence[1] < params.minimum_relative_point_evidence
            * np.median([evidence[0], evidence[2]])):
        return False
    peak = [_peak(raw[int(p.frame)], p, params.raw_peak_radius_px)
            for p in points]
    return (np.isfinite(peak[1]) and peak[1] > 0
            and peak[1] >= params.minimum_relative_raw_peak
            * np.median([peak[0], peak[2]]))


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray | None = None,
            params: Params = Params(),
            partition_params: parent_rule.Params = parent_rule.Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape:
        raise ValueError("raw and labels must align")
    if protected is None:
        protected = np.zeros_like(labels, bool)
    if protected.shape != labels.shape:
        raise ValueError("protected pixels must align")

    candidate, audits, applications, frame_audit = parent_rule.correct(
        labels, raw, points, protected, partition_params)
    scored = physical.attach_owners(points, candidate)
    scored["strong"] = scored.strong.astype(bool)
    scored["physically_visible"] = scored.physically_visible.astype(bool)
    visible = scored[scored.physically_visible].copy()
    by_frame = {int(t): group for t, group in visible.groupby("frame")}
    qualified = parent_rule._qualified_references(
        scored, partition_params.minimum_reference_strong_frames)
    prior_unique = np.unique(candidate)
    extra_audits, extra_applications, extra_frames = [], [], []

    for track_id, group in visible.groupby("track_id", sort=True):
        track_id = int(track_id)
        if track_id not in qualified:
            continue
        group = group.sort_values("frame")
        rows = {int(row.frame): row for row in group.itertuples(index=False)}
        owner_counts = (group[(group.state.astype(str) == "observed")
                              & group.strong]
                        .accepted_owner.astype(int).value_counts())
        for t, row in rows.items():
            host = int(row.accepted_owner)
            if host <= 0 or (str(row.state) == "observed" and bool(row.strong)):
                continue
            prev, following = rows.get(t - 1), rows.get(t + 1)
            if prev is None or following is None:
                continue
            owner = int(prev.accepted_owner)
            if (owner <= 0 or owner == host or int(following.accepted_owner) != owner
                    or str(prev.state) != "observed" or not bool(prev.strong)
                    or str(following.state) != "observed" or not bool(following.strong)
                    or int(owner_counts.get(owner, 0))
                    < params.minimum_strong_owner_frames):
                continue
            if not _continuity(prev, row, following, raw, params):
                continue
            source = candidate[t]
            mask = parent_rule._at_component(source, host, row)
            if mask is None:
                continue
            if any(int(other.track_id) != track_id
                   and int(other.track_id) in qualified
                   and int(other.accepted_owner) == owner
                   and bool(other.strong)
                   and parent_rule._distinct(row, other, partition_params)
                   for other in by_frame[t].itertuples(index=False)):
                continue
            companions = [other for other in by_frame[t].itertuples(index=False)
                          if int(other.track_id) != track_id
                          and int(other.track_id) in qualified
                          and bool(other.strong)
                          and parent_rule._inside(mask, other)
                          and parent_rule._distinct(row, other, partition_params)]
            if any(int(other.accepted_owner) != host for other in companions):
                continue
            if len(companions) > 1:
                selected = parent_rule._partition_many(
                    mask, raw[t], row, companions, partition_params)
                method = "bounded_weak_multi_soma_raw_watershed"
            elif companions:
                selected = parent_rule._partition(
                    mask, raw[t], row, companions[0], partition_params)
                method = "bounded_weak_two_soma_raw_watershed"
            else:
                if parent_rule._unexpected_second_core(mask):
                    continue
                selected = mask
                method = "bounded_weak_single_soma_component"
            if selected is None:
                continue
            reference_areas = []
            for neighbour in (prev, following):
                ref = parent_rule._at_component(
                    candidate[int(neighbour.frame)], owner, neighbour)
                if ref is not None:
                    reference_areas.append(int(ref.sum()))
            if len(reference_areas) != 2:
                continue
            area = float(np.median(reference_areas))
            size = int(selected.sum())
            if not (partition_params.minimum_area_ratio * area <= size
                    <= partition_params.maximum_area_ratio * area):
                continue
            if np.any(protected[t] & selected):
                continue
            if np.any((candidate[t] != labels[t]) & selected):
                continue
            trial_frame = source.copy()
            trial_frame[selected] = owner
            if owner_consensus.count_new_duplicate_components(
                    source[None], trial_frame[None]):
                continue
            candidate[t] = trial_frame
            base = {"track_id": track_id, "early_owner": owner,
                    "early_first_frame": t,
                    "early_last_frame": t,
                    "episode_first_frame": t + 1,
                    "episode_last_frame": t + 1,
                    "wrong_frames": 1}
            extra_audits.append({**base, "outcome": "applied",
                                 "reason": "bounded_weak_soma_bridge",
                                 "changed_pixels": size,
                                 "supported_frames": 1,
                                 "coverage": 1.0,
                                 "unresolved_frames": 0})
            extra_applications.append({**base, "frame": t + 1,
                                       "former_owner": host,
                                       "method": method,
                                       "changed_pixels": size})
            extra_frames.append({**base, "frame": t + 1,
                                 "former_owner": host,
                                 "status": "applied", "changed_pixels": size})

    if (not np.array_equal(labels > 0, candidate > 0)
            or np.any((candidate != labels) & protected)
            or not np.array_equal(prior_unique, np.unique(candidate))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("weak bridge crossed an accepted boundary")
    if extra_frames and not frame_audit.empty:
        changed_keys = {(int(row["track_id"]), int(row["frame"]))
                        for row in extra_frames}
        keep = [((int(row.track_id), int(row.frame)) not in changed_keys)
                for row in frame_audit.itertuples(index=False)]
        frame_audit = frame_audit.loc[keep].copy()
    return (candidate,
            pd.concat([audits, pd.DataFrame(extra_audits)], ignore_index=True),
            pd.concat([applications, pd.DataFrame(extra_applications)], ignore_index=True),
            pd.concat([frame_audit, pd.DataFrame(extra_frames)], ignore_index=True))
