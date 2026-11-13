"""Recover a sustained missing soma after an independently named track run.

The field-wide producer receives no feedback, coordinates, names, or selected
recordings. It only adds a raw-supported body to previously empty pixels;
every accepted named pixel remains byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class Params:
    minimum_reference_run_frames: int = 10
    minimum_ownerless_run_frames: int = 10
    maximum_reference_gap_frames: int = 1
    minimum_strong_fraction: float = .9
    minimum_median_peak_retention: float = .7
    minimum_median_core_area_ratio: float = .45
    minimum_median_core_retention: float = .7
    minimum_frame_peak_retention: float = .4
    maximum_core_area_ratio: float = 2.0
    minimum_application_fraction: float = .8
    core_peak_fraction: float = .35


def _consecutive_runs_whole(rows: pd.DataFrame) -> list[pd.DataFrame]:
    rows = rows.sort_values("frame")
    if rows.empty:
        return []
    boundaries = np.flatnonzero(np.diff(rows.frame.astype(int).to_numpy()) != 1) + 1
    return [rows.loc[indices] for indices in np.split(rows.index.to_numpy(), boundaries)
            if len(indices)]

# Speed (folded from ownerless_run_slices): with a unique plain integer row
# index (not a RangeIndex, which a slice would keep where rows.loc gives a plain
# Index) each label sits at its own position in the sorted rows, so
# rows.iloc[start:end] holds the same rows in the same order, with the same
# index, columns and dtypes; every piece between boundaries is non-empty, as
# np.split's were. The sort, the empty test and the integer frame cast run in
# the original order, so the same errors are raised. Runs are only read. Other
# indexes keep _consecutive_runs_whole.
def _consecutive_runs(rows: pd.DataFrame) -> list[pd.DataFrame]:
    if not (type(rows.index) is pd.Index and rows.index.is_unique and rows.index.dtype.kind in 'iu'):
        return _consecutive_runs_whole(rows)
    rows = rows.sort_values("frame")
    if rows.empty:
        return []
    boundaries = (np.flatnonzero(np.diff(rows.frame.astype(int).to_numpy()) != 1) + 1).tolist()
    if type(rows.index) is not pd.Index:
        # A slice keeps a RangeIndex, where rows.loc gives a plain Index.
        return [rows.loc[indices] for indices in np.split(rows.index.to_numpy(), boundaries) if len(indices)]
    return [rows.iloc[start:end] for start, end in zip([0] + boundaries, boundaries + [len(rows)])]



def _reference(group: pd.DataFrame, onset: int) -> tuple[int, pd.DataFrame]:
    early = group[(group.frame.astype(int) < onset)
                  & group.physically_visible.astype(bool)
                  & group.strong.astype(bool)
                  & ~group.in_encounter.astype(bool)
                  & group.state.astype(str).eq("observed")
                  & group.accepted_owner.astype(int).gt(0)]
    best_owner, best = 0, early.iloc[0:0]
    for owner, owned in early.groupby("accepted_owner", sort=True):
        for run in _consecutive_runs(owned):
            if len(run) > len(best) or (len(run) == len(best) and len(run)
                                       and int(run.frame.max()) > int(best.frame.max())):
                best_owner, best = int(owner), run
    return best_owner, best


def _core(raw_frame: np.ndarray, point, peak_fraction: float
          ) -> tuple[np.ndarray, tuple[slice, slice], float, float]:
    radius = max(2.0, float(point.radius_px))
    extent = int(np.ceil(2 * radius))
    y, x = float(point.y), float(point.x)
    y0, y1 = max(0, int(y) - extent), min(raw_frame.shape[0], int(y) + extent + 1)
    x0, x1 = max(0, int(x) - extent), min(raw_frame.shape[1], int(x) + extent + 1)
    region = slice(y0, y1), slice(x0, x1)
    local = raw_frame[region]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.hypot(yy - y, xx - x)
    near = distance <= radius
    peak = float(np.max(local[near]))
    if peak <= 0:
        return np.zeros_like(local, bool), region, 0.0, 0.0
    seed_y, seed_x = np.argwhere((local == peak) & near)[0]
    binary = (distance <= 2 * radius) & (local >= peak_fraction * peak) & (local > 0)
    components, _ = ndi.label(binary, STRUCTURE)
    component = int(components[seed_y, seed_x])
    core = components == component if component else np.zeros_like(binary)
    area_ratio = float(np.count_nonzero(core) / (np.pi * max(float(point.radius_px), 1) ** 2))
    return core, region, peak, area_ratio

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


# Speed (folded from boundary_frame_checks): np.clip of a Python integer gives
# the same integer as min and max.
def _boundary_clip(value, high):
    if type(value) is int and high >= 0 and -(1 << 62) < value < (1 << 62):
        return min(max(value, 0), high)
    return int(np.clip(value, 0, high))


def correct(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: Params = Params()
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != unclaimed.shape:
        raise ValueError("raw, labels and unclaimed ledger must align")
    scored = physical.attach_owners(points, labels)
    for field in ("strong", "physically_visible", "in_encounter"):
        scored[field] = scored[field].astype(bool)
    by_frame = {int(t): group for t, group in scored.groupby("frame")}
    proposals = []
    for track, group in scored.groupby("track_id", sort=True):
        visible = group[group.physically_visible]
        zero = visible[visible.accepted_owner.astype(int).eq(0)]
        for episode in _consecutive_runs(zero):
            if len(episode) < params.minimum_ownerless_run_frames:
                continue
            onset = int(episode.frame.min())
            owner, reference = _reference(group, onset)
            if (owner <= 0 or len(reference) < params.minimum_reference_run_frames
                    or onset - int(reference.frame.max()) - 1 > params.maximum_reference_gap_frames
                    or float(episode.strong.mean()) < params.minimum_strong_fraction):
                continue
            reference_raw = [_core(raw[int(row.frame)], row, params.core_peak_fraction)
                             for row in reference.itertuples(index=False)]
            median_peak = float(np.median([item[2] for item in reference_raw]))
            median_area = float(np.median([item[3] for item in reference_raw]))
            episode_raw = [_core(raw[int(row.frame)], row, params.core_peak_fraction)
                           for row in episode.itertuples(index=False)]
            median_peak_retention = float(np.median(
                [item[2] / max(median_peak, 1) for item in episode_raw]))
            median_core_area_ratio = float(np.median([item[3] for item in episode_raw]))
            median_core_retention = median_core_area_ratio / max(median_area, .01)
            if (median_peak_retention < params.minimum_median_peak_retention
                    or median_core_area_ratio < params.minimum_median_core_area_ratio
                    or median_core_retention < params.minimum_median_core_retention):
                continue
            proposals.append((int(track), owner, episode, median_peak,
                              median_peak_retention, median_core_area_ratio,
                              median_core_retention, episode_raw))

    candidate, candidate_unclaimed = labels.copy(), unclaimed.copy()
    audits, applications = [], []
    for (track, owner, episode, reference_peak, peak_retention, area_ratio,
         area_retention, episode_raw) in proposals:
        selected, refusals = [], []
        for row, (core, region, peak, core_ratio) in zip(
                episode.itertuples(index=False), episode_raw):
            t = int(row.frame)
            reason = None
            if not np.any(core) or core_ratio > params.maximum_core_area_ratio:
                reason = "core_area_invalid"
            elif peak / max(reference_peak, 1) < params.minimum_frame_peak_retention:
                reason = "raw_peak_too_dim"
            elif np.any(candidate[t][region][core] > 0):
                reason = "core_touches_existing_outline"
            elif np.any(candidate[t] == owner):
                reason = "identity_on_other_body"
            else:
                for other in by_frame.get(t, scored.iloc[0:0]).itertuples(index=False):
                    if int(other.track_id) == track or not bool(other.physically_visible):
                        continue
                    yy = _boundary_clip(round(float(other.y)), labels.shape[1] - 1)
                    xx = _boundary_clip(round(float(other.x)), labels.shape[2] - 1)
                    if region[0].start <= yy < region[0].stop and region[1].start <= xx < region[1].stop:
                        if core[yy - region[0].start, xx - region[1].start]:
                            reason = "other_physical_point_in_core"
                            break
            if reason is None:
                selected.append((t, core, region, peak, core_ratio))
            else:
                refusals.append(f"{t + 1}:{reason}")
        if len(selected) / len(episode) < params.minimum_application_fraction:
            audits.append({
                "track_id": track, "reference_owner": owner,
                "first_frame": int(episode.frame.min()) + 1,
                "last_frame": int(episode.frame.max()) + 1,
                "visible_episode_frames": len(episode),
                "selected_frames": len(selected),
                "median_peak_retention": peak_retention,
                "median_core_area_radius_ratio": area_ratio,
                "median_core_area_retention": area_retention,
                "status": "insufficient_safe_frame_coverage",
                "refusals": "|".join(refusals),
            })
            continue
        for t, core, region, peak, core_ratio in selected:
            selected_pixels = int(np.count_nonzero(core))
            candidate[t][region][core] = owner
            candidate_unclaimed[t][region][core] = 0
            applications.append({
                "track_id": track, "reference_owner": owner,
                "frame": t + 1, "added_pixels": selected_pixels,
                "raw_peak": peak, "core_area_radius_ratio": core_ratio,
            })
        audits.append({
            "track_id": track, "reference_owner": owner,
            "first_frame": int(episode.frame.min()) + 1,
            "last_frame": int(episode.frame.max()) + 1,
            "visible_episode_frames": len(episode),
            "selected_frames": len(selected),
            "median_peak_retention": peak_retention,
            "median_core_area_radius_ratio": area_ratio,
            "median_core_area_retention": area_retention,
            "status": "applied",
            "refusals": "|".join(refusals),
        })
    if _boundary_crossed(candidate, labels, candidate_unclaimed, unclaimed, {int(a["frame"]) - 1 for a in applications}):
        raise AssertionError("ownerless-soma layer crossed an accepted label boundary")
    return candidate, candidate_unclaimed, pd.DataFrame(audits), pd.DataFrame(applications)
