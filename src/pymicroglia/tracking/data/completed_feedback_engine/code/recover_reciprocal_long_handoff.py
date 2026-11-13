"""Reciprocal handoff on the accepted parent, with no new identity names."""
from __future__ import annotations

import numpy as np
import pandas as pd

import recover_dim_displaced_name_paths as precursor


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate, audits, applications = precursor.correct(labels, raw, points)
    if applications.empty:
        return candidate, audits, applications
    candidate = candidate.copy()
    audits = audits.copy()
    applications = applications.copy()
    rejected_tracks = set()
    for audit in audits[audits.outcome == "applied"].itertuples(index=False):
        track = int(audit.track_id)
        earlier = int(audit.earlier_name)
        later = int(audit.later_name)
        transient = int(audit.residual_name)
        path = applications[applications.track_id.astype(int) == track]
        exclusive = all(
            int(np.count_nonzero(labels[int(row.frame)-1] == later)) == int(row.target_pixels)
            and not np.any(candidate[int(row.frame)-1] == later)
            for row in path.itertuples(index=False)
        )
        if not exclusive:
            for row in path.itertuples(index=False):
                t = int(row.frame) - 1
                target = (labels[t] == later) & (candidate[t] == earlier)
                residual = candidate[t] == transient
                candidate[t][target | residual] = labels[t][target | residual]
            rejected_tracks.add(track)
            continue
        for row in path.itertuples(index=False):
            t = int(row.frame) - 1
            candidate[t][candidate[t] == transient] = later
        audits.loc[audits.track_id == track, "residual_name"] = later
        applications.loc[applications.track_id == track, "residual_new_name"] = later
    if rejected_tracks:
        audits.loc[audits.track_id.isin(rejected_tracks), "outcome"] = "rejected"
        audits.loc[audits.track_id.isin(rejected_tracks), "reason"] = "displaced_name_not_exclusive"
        applications = applications[~applications.track_id.isin(rejected_tracks)].copy()
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("reciprocal handoff changed foreground")
    if set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels))):
        raise AssertionError("reciprocal handoff introduced a name")
    return candidate, audits, applications
