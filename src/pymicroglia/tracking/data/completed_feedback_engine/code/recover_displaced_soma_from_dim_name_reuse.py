"""Keep a strong displaced soma's name when a dim outline reuses that name.

All candidates are discovered from masks, raw intensities, and soma tracks.
Recording names, literal identities, coordinates, and saved feedback are absent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


CONNECTIVITY = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    stable_observations: int = 10
    minimum_stable_owner_observations: int = 9
    minimum_branch_observations: int = 10
    minimum_branch_owner_observations: int = 8
    minimum_source_area: int = 60
    minimum_area_ratio: float = .5
    maximum_area_ratio: float = 2.
    maximum_displacement_radii: float = 5.5
    maximum_initial_residual_fraction: float = .4
    minimum_residual_distance_radii: float = 3.
    maximum_residual_brightness_ratio: float = .25
    minimum_corrected_pixels: int = 500


def _component(frame: np.ndarray, name: int, point) -> np.ndarray | None:
    cc, _ = ndi.label(frame == name, CONNECTIVITY)
    y, x = round(float(point.y)), round(float(point.x))
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return None
    index = int(cc[y, x])
    return cc == index if index else None


def _components(frame: np.ndarray, name: int) -> list[np.ndarray]:
    cc, count = ndi.label(frame == name, CONNECTIVITY)
    return [cc == index for index in range(1, count + 1)]


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or raw.shape != labels.shape or protected.shape != labels.shape:
        raise ValueError("aligned labels, raw, and protected masks required")
    strong = points[(points.state.eq("observed")) & points.strong.astype(bool)].copy()
    yx = np.rint(strong[["y", "x"]].to_numpy(float)).astype(int)
    if (np.any(yx[:, 0] < 0) or np.any(yx[:, 0] >= labels.shape[1])
            or np.any(yx[:, 1] < 0) or np.any(yx[:, 1] >= labels.shape[2])):
        raise ValueError("soma point outside labels")
    strong["owner"] = labels[strong.frame.to_numpy(int), yx[:, 0], yx[:, 1]]
    by_track = {int(track): group.sort_values("frame")
                for track, group in strong.groupby("track_id")}
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    heads: dict[int, list[tuple[int, pd.DataFrame]]] = {}
    for track, group in by_track.items():
        if len(group) >= params.minimum_branch_observations:
            heads.setdefault(int(group.frame.iloc[0]), []).append((track, group))

    proposals = []
    audits = []
    for old_track, history in by_track.items():
        tail = history.iloc[-params.stable_observations:]
        if len(tail) != params.stable_observations:
            continue
        old_name = int(tail.owner.iloc[-1])
        if (old_name <= 0
                or int(tail.owner.eq(old_name).sum()) < params.minimum_stable_owner_observations
                or not np.all(np.diff(tail.frame.to_numpy(int)) == 1)):
            continue
        previous = history.iloc[-1]
        t = int(previous.frame) + 1
        if t >= len(labels) or t not in heads:
            continue
        holders = by_frame.get(t)
        if holders is not None and bool(holders.owner.eq(old_name).any()):
            continue
        old_mask = _component(labels[t-1], old_name, previous)
        if old_mask is None or int(old_mask.sum()) < params.minimum_source_area:
            continue
        old_area = int(old_mask.sum())
        initial_residuals = _components(labels[t], old_name)
        if len(initial_residuals) != 1:
            continue
        initial = initial_residuals[0]
        if int(initial.sum()) >= params.maximum_initial_residual_fraction * old_area:
            continue
        old_y, old_x = ndi.center_of_mass(initial)
        old_radius = max(float(previous.radius_px), 1.)
        if np.hypot(old_y-float(previous.y), old_x-float(previous.x))/old_radius < params.minimum_residual_distance_radii:
            continue
        for new_track, branch in heads[t]:
            if new_track == old_track:
                continue
            head = branch.iloc[0]
            new_name = int(head.owner)
            if new_name <= 0 or new_name == old_name:
                continue
            earlier = by_frame.get(t-1)
            if earlier is not None and bool(earlier.owner.eq(new_name).any()):
                continue
            if int(branch.owner.iloc[:params.minimum_branch_observations].eq(new_name).sum()) < params.minimum_branch_owner_observations:
                continue
            if not bool(branch.owner.eq(new_name).all()):
                continue
            radius = max((float(previous.radius_px)+float(head.radius_px))/2, 1.)
            step = np.hypot(float(head.y-previous.y), float(head.x-previous.x))/radius
            if step > params.maximum_displacement_radii:
                continue
            first_target = _component(labels[t], new_name, head)
            if first_target is None:
                continue
            area_ratio = float(first_target.sum())/old_area
            if not params.minimum_area_ratio <= area_ratio <= params.maximum_area_ratio:
                continue
            # Every strong branch observation must keep the new name. No other
            # strong soma may hold either name during this proposed path.
            target_frames = {}
            brightness = []
            reason = "supported"
            for row in branch.itertuples(index=False):
                f = int(row.frame)
                group = by_frame.get(f)
                if group is not None and bool(group[
                        (group.track_id != new_track) &
                        (group.owner.isin((old_name, new_name)))].shape[0]):
                    reason = "competing_strong_holder"; break
                mask = _component(labels[f], new_name, row)
                if mask is None or np.any(mask & protected[f]):
                    reason = "branch_component_or_protection"; break
                target_frames[f] = mask
                brightness.append(float(np.quantile(raw[f][mask], .9)))
            if reason != "supported" or not target_frames:
                continue
            reference_brightness = float(np.median(brightness))
            residual_frames = {}
            for f in range(t, len(labels)):
                remaining = _components(labels[f], old_name)
                if len(remaining) != 1:
                    reason = "residual_missing_or_split"; break
                mask = remaining[0]
                group = by_frame.get(f)
                if group is not None:
                    rows = group[group.owner == old_name]
                    if len(rows):
                        reason = "strong_old_name_holder"; break
                if (np.any(mask & protected[f])
                        or float(np.quantile(raw[f][mask], .9)) >
                        params.maximum_residual_brightness_ratio * reference_brightness):
                    reason = "residual_bright_or_protected"; break
                residual_frames[f] = mask
            if reason != "supported":
                continue
            changed_pixels = sum(int(mask.sum()) for mask in target_frames.values()) + sum(
                int(mask.sum()) for mask in residual_frames.values())
            if changed_pixels < params.minimum_corrected_pixels:
                continue
            seed = {"old_track": old_track, "new_track": new_track,
                    "old_name": old_name, "new_name": new_name,
                    "first_frame": t+1, "last_frame": len(labels),
                    "changed_pixels": changed_pixels}
            proposals.append((seed, target_frames, residual_frames))

    # Conflicting proposals are ambiguous regardless of tracker number or
    # iteration order. Reject all participants rather than tie-breaking on IDs.
    conflicted = set()
    for i, (left, targets_a, residuals_a) in enumerate(proposals):
        for j in range(i+1, len(proposals)):
            right, targets_b, residuals_b = proposals[j]
            if left["old_name"] == right["old_name"] or left["new_name"] == right["new_name"]:
                conflicted.update((i, j))
                continue
            if any(np.any(a & b) for family in (targets_a, residuals_a)
                   for f, a in family.items()
                   for b in (targets_b.get(f), residuals_b.get(f)) if b is not None):
                conflicted.update((i, j))
    candidate = labels.copy()
    applications = []
    for i, (seed, target_frames, residual_frames) in enumerate(proposals):
        if i in conflicted:
            audits.append({**seed, "outcome": "rejected", "reason": "conflicting_proposal"})
            continue
        for f, mask in residual_frames.items():
            candidate[f][mask] = 0
            applications.append({**seed, "frame": f+1, "action": "retire_dim_residual",
                                 "pixels": int(mask.sum())})
        for f, mask in target_frames.items():
            candidate[f][mask] = seed["old_name"]
            applications.append({**seed, "frame": f+1, "action": "restore_strong_soma",
                                 "pixels": int(mask.sum())})
        audits.append({**seed, "outcome": "applied", "reason": "displaced_soma_and_dim_name_reuse"})
    if np.any((candidate != labels) & protected):
        raise AssertionError("displaced soma correction crossed an accepted pixel")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
