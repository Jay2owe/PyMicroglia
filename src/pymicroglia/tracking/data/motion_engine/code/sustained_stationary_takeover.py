"""Reapply stationary soma ownership only to sustained, unmodified episodes.

An unchanged, repeated soma site can expose a later name takeover after other
tracking repairs. Discovery is field-wide. Numeric names, wells, frames,
coordinates, feedback and review cases are not accepted as parameters.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import stationary_reconciliation as stationary


ALLOWED_PARAMETERS = {
    "targeting_mode", "minimum_component_soma_area_fraction",
    "minimum_reference_movie_fraction", "minimum_episode_movie_fraction",
}
AUDIT_COLUMNS = [
    "event_id", "resident_identity", "observed_identity", "takeover_frame",
    "last_matched_frame", "matched_components", "reference_observed_frames",
    "changed_pixels", "protected_pixels", "decision", "reason",
]


def _parameters(overrides: dict | None) -> dict:
    values = dict(overrides or {})
    unknown = set(values) - ALLOWED_PARAMETERS
    if unknown:
        raise ValueError("stationary takeover received target or unknown keys: "
                         + ", ".join(sorted(unknown)))
    if values.get("targeting_mode", "field_wide_discovery") != "field_wide_discovery":
        raise ValueError("stationary takeover must discover the full field")
    result = {
        "minimum_component_soma_area_fraction": .05,
        "minimum_reference_movie_fraction": .02,
        "minimum_episode_movie_fraction": .08,
    }
    result.update(values)
    for key in ALLOWED_PARAMETERS - {"targeting_mode"}:
        if not 0 <= float(result[key]) <= 1:
            raise ValueError(f"{key} must be a fraction")
    return result


def correct(labels: np.ndarray, raw: np.ndarray, protected: np.ndarray,
            points: pd.DataFrame, overrides: dict | None = None
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return candidate labels, every discovered decision, and applied pixels."""
    params = _parameters(overrides)
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected arrays must align")
    radii = points.loc[points.strong.astype(bool) &
                       points.state.eq("observed"), "radius_px"].to_numpy(float)
    if not len(radii):
        raise ValueError("no strong physical soma radii for recording calibration")
    median_soma_area = math.pi * float(np.median(radii)) ** 2
    minimum_component_px = max(1, int(math.ceil(
        median_soma_area * float(params["minimum_component_soma_area_fraction"]))))
    reference_frames = max(2, int(math.ceil(
        len(labels) * float(params["minimum_reference_movie_fraction"]))))
    episode_frames = max(2, int(math.ceil(
        len(labels) * float(params["minimum_episode_movie_fraction"]))))
    engine_params = stationary.reconciliation_params({
        "minimum_component_px": minimum_component_px,
        "minimum_reference_duration_fraction": float(
            params["minimum_reference_movie_fraction"]),
    })
    _, _, _, events, evidence = stationary.discover_stationary_takeovers(
        labels, raw, engine_params)
    candidate = labels.copy()
    decisions: list[dict] = []
    selected = []
    for event in events.itertuples(index=False):
        subset = evidence[evidence.event_id == event.event_id]
        frame = subset.review_frame.to_numpy(int) - 1
        y = subset.y.to_numpy(int)
        x = subset.x.to_numpy(int)
        protected_pixels = int(np.count_nonzero(protected[frame, y, x]))
        reasons = []
        if event.decision != "accepted_field_wide_takeover":
            reasons.append("engine_" + str(event.decision))
        if int(event.reference_observed_frames) < reference_frames:
            reasons.append("short_reference")
        if int(event.matched_components) < episode_frames:
            reasons.append("short_takeover")
        if protected_pixels:
            reasons.append("previously_accepted_pixels")
        if not len(subset):
            reasons.append("no_changed_pixels")
        decision = "eligible" if not reasons else "rejected"
        decisions.append({
            "event_id": event.event_id,
            "resident_identity": int(event.resident_identity),
            "observed_identity": int(event.observed_identity),
            "takeover_frame": int(event.takeover_frame),
            "last_matched_frame": int(event.last_matched_frame),
            "matched_components": int(event.matched_components),
            "reference_observed_frames": int(event.reference_observed_frames),
            "changed_pixels": len(subset),
            "protected_pixels": protected_pixels,
            "decision": decision,
            "reason": "eligible" if not reasons else "|".join(reasons),
        })
        if decision == "eligible":
            selected.append(subset)
    applied = pd.concat(selected, ignore_index=True) if selected else \
        evidence.iloc[:0].copy()
    if len(applied):
        frame = applied.review_frame.to_numpy(int) - 1
        y = applied.y.to_numpy(int)
        x = applied.x.to_numpy(int)
        if len(set(zip(frame, y, x))) != len(applied):
            raise AssertionError("stationary proposals overlap")
        candidate[frame, y, x] = applied.resident_identity.to_numpy(labels.dtype)
    if np.any((candidate != labels) & protected):
        raise AssertionError("stationary takeover reversed an accepted pixel")
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("stationary takeover changed detected foreground")
    return candidate, pd.DataFrame(decisions, columns=AUDIT_COLUMNS), applied
