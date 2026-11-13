"""Restore an ownerless opening soma from its long subsequent named history.

The producer uses physical tracks, accepted outlines and raw pixels only.
It never receives review feedback or identifiers naming a target case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import recover_short_ownerless_soma_runs as core_rule
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class Params:
    minimum_reference_frames: int = 20
    minimum_reference_movie_fraction: float = .5
    minimum_reference_independent_fraction: float = .85
    minimum_peak_retention: float = .7
    minimum_historical_outline_fraction: float = .65
    maximum_historical_outline_fraction: float = 1.2
    minimum_raw_threshold_fraction: float = .08
    historical_outline_intensity_quantile: float = .1


def _following_run(group: pd.DataFrame) -> pd.DataFrame:
    """Return the contiguous named run starting on the second movie frame."""
    later = group[(group.frame.astype(int).ge(1))
                  & group.strong.astype(bool)
                  & group.physically_visible.astype(bool)
                  & group.state.astype(str).eq("observed")
                  & group.accepted_owner.astype(int).gt(0)]
    for run in core_rule._consecutive_runs(later):
        if int(run.frame.min()) == 1 and run.accepted_owner.astype(int).nunique() == 1:
            return run
    return group.iloc[0:0]


def _point_component(frame: np.ndarray, owner: int, point):
    y, x = int(round(float(point.y))), int(round(float(point.x)))
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return None
    if frame[y, x] != owner:
        return None
    components, _ = ndi.label(frame == owner, STRUCTURE)
    index = int(components[y, x])
    return components == index if index else None

# Speed (folded from boundary_frame_checks): correct copies the movie and its
# ledger and writes cores only into the frames it records as applied, so every
# other frame of the copies equals the input. For integer movies the changed-
# label and ledger tests are the same on the written frames alone, and an
# unchanged movie has the same names; the name test runs on the whole movies as
# before when anything changed. Other inputs take the whole-movie tests.
def _boundary_crossed(candidate, labels, candidate_ledger, ledger, frames):
    arrays = (candidate, labels, candidate_ledger, ledger)
    if not (all(isinstance(a, np.ndarray) for a in arrays)
            and candidate.shape == labels.shape == candidate_ledger.shape == ledger.shape
            and labels.ndim >= 1 and all(a.dtype.kind in 'biu' for a in arrays)
            and all(isinstance(t, int) and 0 <= t < len(labels) for t in frames)):
        changed = candidate != labels
        return bool(np.any(changed & (labels != 0))
                    or np.any((candidate_ledger != ledger) & ~changed)
                    or not np.array_equal(np.unique(candidate), np.unique(labels)))
    any_changed = False
    for t in sorted(frames):
        changed = candidate[t] != labels[t]
        if np.any(changed & (labels[t] != 0)):
            return True
        if np.any((candidate_ledger[t] != ledger[t]) & ~changed):
            return True
        any_changed = any_changed or bool(np.any(changed))
    # Unwritten frames are equal; an unchanged movie has the same names.
    return any_changed and not np.array_equal(np.unique(candidate), np.unique(labels))


# Speed (folded from boundary_frame_checks): the tracks with exactly one frame-0
# point, from the frame-0 points correct already selects, so the others are
# skipped before their rows are indexed.
def _boundary_single_first(first_points):
    counts = first_points["track_id"].value_counts(dropna=True)
    return set(counts.index[counts == 1].tolist())


def correct(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: Params = Params()
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != unclaimed.shape:
        raise ValueError("raw, labels and unclaimed ledger must align")
    result, result_ledger = labels.copy(), unclaimed.copy()
    scored = physical.attach_owners(points, labels)
    for field in ("strong", "physically_visible", "in_encounter"):
        scored[field] = scored[field].astype(bool)
    first_points = scored[scored.frame.astype(int).eq(0)]
    audits, applications = [], []
    min_frames = max(params.minimum_reference_frames,
                     int(np.ceil(params.minimum_reference_movie_fraction * len(labels))))
    _single_first = _boundary_single_first(first_points)
    for track, group in scored.groupby("track_id", sort=True):
        if track not in _single_first:
            continue
        first = group[group.frame.astype(int).eq(0)]
        if len(first) != 1:
            continue
        point = first.iloc[0]
        if (not bool(point.physically_visible) or not bool(point.strong)
                or bool(point.in_encounter) or str(point.state) != "observed"
                or int(point.accepted_owner) != 0):
            continue
        reference = _following_run(group)
        if (len(reference) < min_frames
                or float((~reference.in_encounter).mean())
                   < params.minimum_reference_independent_fraction):
            continue
        owner = int(reference.accepted_owner.iloc[0])
        reason = ""
        historical_areas, historical_low_ratios, historical_peaks = [], [], []
        for q in reference[~reference.in_encounter].itertuples(index=False):
            t = int(q.frame)
            component = _point_component(labels[t], owner, q)
            if component is None:
                continue
            peak = core_rule._core(raw[t], q, .35)[2]
            if peak <= 0:
                continue
            historical_areas.append(int(component.sum()))
            historical_low_ratios.append(float(np.quantile(
                raw[t][component] / peak, params.historical_outline_intensity_quantile)))
            historical_peaks.append(peak)
        if len(historical_areas) < min_frames * .85:
            reason = "insufficient_accepted_outline_history"
        else:
            threshold = max(params.minimum_raw_threshold_fraction,
                            float(np.median(historical_low_ratios)))
            core, region, peak, area_ratio = core_rule._core(raw[0], point, threshold)
            outline_area = int(np.count_nonzero(core))
            median_outline_area = float(np.median(historical_areas))
            peak_retention = peak / max(float(np.median(historical_peaks)), 1)
            if (not outline_area
                    or peak_retention < params.minimum_peak_retention
                    or not params.minimum_historical_outline_fraction * median_outline_area
                       <= outline_area <= params.maximum_historical_outline_fraction * median_outline_area):
                reason = "raw_soma_outside_historical_range"
            elif np.any(result[0][region][core] > 0):
                reason = "accepted_outline_collision"
            elif np.any(result[0] == owner):
                reason = "identity_on_other_body"
            else:
                for other in first_points.itertuples(index=False):
                    if int(other.track_id) == int(track) or not bool(other.physically_visible):
                        continue
                    y, x = int(round(float(other.y))), int(round(float(other.x)))
                    if (region[0].start <= y < region[0].stop
                            and region[1].start <= x < region[1].stop
                            and core[y-region[0].start, x-region[1].start]):
                        reason = "other_physical_point_inside"
                        break
        if not reason:
            result[0][region][core] = owner
            result_ledger[0][region][core] = 0
            applications.append({"track_id": int(track), "reference_owner": owner,
                                 "frame": 1, "added_pixels": outline_area,
                                 "reference_frames": len(reference),
                                 "reference_independent_fraction":
                                 float((~reference.in_encounter).mean()),
                                 "reference_outline_area": median_outline_area,
                                 "raw_threshold_fraction": threshold,
                                 "peak_retention": peak_retention,
                                 "core_area_ratio": area_ratio})
        audits.append({"track_id": int(track), "reference_owner": owner,
                       "reference_frames": len(reference),
                       "applied": not bool(reason), "refusal": reason})
    if _boundary_crossed(result, labels, result_ledger, unclaimed, {0} if applications else set()):
        raise AssertionError("leading-soma layer crossed an accepted-label boundary")
    return result, result_ledger, pd.DataFrame(audits), pd.DataFrame(applications)
