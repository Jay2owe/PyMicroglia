"""Extend a proven shared-soma split through a short adjacent contact.

The prior applications are produced by earlier general layers. They serve as
temporal evidence for the same physical track, not as feedback or cell targets.
The established dual-soma geometry, raw partition and duplicate guards still
decide whether each adjacent frame is admissible.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import owner_consensus
import recover_stable_dual_soma_contacts as dual


@dataclass(frozen=True)
class Params:
    minimum_prior_run_frames: int = 3
    maximum_edge_distance_frames: int = 2
    minimum_reference_area_percentile: float = 0.6


def _run_edges(applications: pd.DataFrame, minimum_length: int
               ) -> dict[tuple[int, int], list[tuple[int, int]]]:
    required = {"track_id", "reference_owner", "frame"}
    if not required.issubset(applications.columns):
        raise ValueError(f"prior applications require {sorted(required)}")
    runs = {}
    for (track, owner), group in applications.groupby(
            ["track_id", "reference_owner"], sort=True):
        frames = sorted(set(group.frame.astype(int)))
        current, found = [], []
        for frame in [*frames, -2]:
            if current and frame != current[-1] + 1:
                if len(current) >= minimum_length:
                    found.append((current[0], current[-1]))
                current = []
            if frame > 0:
                current.append(frame)
        if found:
            runs[(int(track), int(owner))] = found
    return runs


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            prior_applications: pd.DataFrame,
            protected: np.ndarray | None = None,
            params: Params = Params(),
            dual_params: dual.Params = dual.Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if protected is None:
        protected = np.zeros_like(labels, bool)
    if protected.shape != labels.shape:
        raise ValueError("protected pixels must align")
    runs = _run_edges(prior_applications, params.minimum_prior_run_frames)
    candidate = labels.copy()
    if not runs:
        return candidate, pd.DataFrame(), pd.DataFrame()

    proposed, _, applications = dual.correct(
        labels, raw, points, protected,
        params=replace(dual_params, minimum_contact_run_frames=1,
                       minimum_reference_area_percentile=
                       params.minimum_reference_area_percentile))
    if applications.empty:
        return candidate, pd.DataFrame(), pd.DataFrame()

    per_frame = applications.frame.astype(int).value_counts().to_dict()
    audit, accepted = [], []
    for app in applications.itertuples(index=False):
        frame = int(app.frame)
        key = (int(app.track_id), int(app.reference_owner))
        prior = runs.get(key, [])
        distance = min((min(abs(frame - first), abs(frame - last))
                        for first, last in prior), default=10 ** 9)
        reason = None
        if not prior:
            reason = "no_sustained_prior_split"
        elif not 1 <= distance <= params.maximum_edge_distance_frames:
            reason = "outside_short_contact_edge"
        elif per_frame[frame] != 1:
            reason = "multiple_simultaneous_proposals"
        t = frame - 1
        changed = proposed[t] != labels[t]
        if reason is None and int(changed.sum()) != int(app.changed_pixels):
            reason = "proposal_pixels_ambiguous"
        if reason is None and np.any(protected[t] & changed):
            reason = "protected_pixel"
        if reason is None:
            trial = candidate[t].copy()
            trial[changed] = proposed[t][changed]
            if owner_consensus.count_new_duplicate_components(candidate[t:t + 1],
                                                              trial[None]):
                reason = "duplicate_soma"
            else:
                candidate[t] = trial
                accepted.append({
                    "track_id": key[0], "reference_owner": key[1],
                    "former_owner": int(app.former_owner), "frame": frame,
                    "changed_pixels": int(changed.sum()),
                    "distance_from_prior_run": distance,
                })
        audit.append({"track_id": key[0], "reference_owner": key[1],
                      "frame": frame, "distance_from_prior_run": distance,
                      "status": "applied" if reason is None else reason})

    if (not np.array_equal(labels > 0, candidate > 0)
            or np.any((candidate != labels) & protected)
            or not np.array_equal(np.unique(labels), np.unique(candidate))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("contact-edge layer crossed an accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(accepted)
