"""Restore a stable soma through a nearby tracker branch and coupled name reuse.

Discovery uses masks, raw images and soma tracks. No recording, identity,
track, coordinate, frame or feedback selector is part of the rule. It stops at
the first unresolved contact instead of transferring a neighbour's outline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_stable_observations: int = 20
    minimum_branch_observations: int = 8
    maximum_branch_start_lag: int = 2
    maximum_branch_start_step_radii: float = 2.5
    minimum_corrected_frames: int = 8
    minimum_corrected_pixels: int = 500
    minimum_target_basin_area: int = 25
    minimum_other_basin_area: int = 20
    maximum_other_valley_ratio: float = .6
    maximum_target_step_radii: float = 3.
    maximum_alt_name_lookback: int = 10
    maximum_alt_step_radii_per_frame: float = 1.5
    watershed_sigma_px: float = 1.


def _component(labels: np.ndarray, t: int, name: int, point,
               cache: dict) -> np.ndarray | None:
    key = (t, name)
    if key not in cache:
        cache[key] = ndi.label(labels[t] == name, STRUCTURE)[0]
    cc = cache[key]
    y, x = round(float(point.y)), round(float(point.x))
    if not (0 <= y < cc.shape[0] and 0 <= x < cc.shape[1]):
        return None
    index = int(cc[y, x])
    return cc == index if index else None


def _inside(mask: np.ndarray, point) -> bool:
    y, x = round(float(point.y)), round(float(point.x))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _valley(frame: np.ndarray, a, b) -> float:
    distance = np.hypot(float(a.x)-float(b.x), float(a.y)-float(b.y))
    values = []
    for weight in np.linspace(0, 1, max(3, int(np.ceil(distance))+1)):
        y = np.clip(round((1-weight)*float(a.y)+weight*float(b.y)), 0, frame.shape[0]-1)
        x = np.clip(round((1-weight)*float(a.x)+weight*float(b.x)), 0, frame.shape[1]-1)
        values.append(float(frame[int(y), int(x)]))
    return min(values[1:-1])/max(min(values[0], values[-1]), 1.)


def _basin(mask: np.ndarray, frame: np.ndarray, members: list, others: list,
           params: Params) -> np.ndarray | None:
    if not others:
        return mask
    if any(min(_valley(frame, target, other) for target in members)
           > params.maximum_other_valley_ratio for other in others):
        return None
    yy, xx = np.where(mask)
    box = np.s_[max(0, int(yy.min())-2):min(mask.shape[0], int(yy.max())+3),
                max(0, int(xx.min())-2):min(mask.shape[1], int(xx.max())+3)]
    local = mask[box]
    markers = np.zeros(local.shape, np.uint16)
    for mark, point in [(1, point) for point in members] + [
            (index, point) for index, point in enumerate(others, 2)]:
        y = round(float(point.y))-box[0].start
        x = round(float(point.x))-box[1].start
        if not (0 <= y < local.shape[0] and 0 <= x < local.shape[1]
                and local[y, x] and markers[y, x] == 0):
            return None
        markers[y, x] = mark
    assigned = watershed(-ndi.gaussian_filter(frame[box].astype(np.float32),
                                               params.watershed_sigma_px),
                         markers, mask=local, connectivity=STRUCTURE)
    if (int((assigned == 1).sum()) < params.minimum_target_basin_area or
            any(int((assigned == index).sum()) < params.minimum_other_basin_area
                for index in range(2, len(others)+2))):
        return None
    selected = np.zeros(mask.shape, bool)
    selected[box] = assigned == 1
    return selected


def _alternate_name(holder, t: int, original: int, by_track: dict,
                    strong_by_frame: dict, labels: np.ndarray,
                    params: Params) -> int | None:
    history = by_track[int(holder.track_id)]
    before = history[(history.frame < t) &
                     (history.frame >= t-params.maximum_alt_name_lookback) &
                     history.owner.gt(0) & history.owner.ne(original)]
    if before.empty:
        return None
    last = before.iloc[-1]
    alternate = int(last.owner)
    if any(int(p.owner) == alternate and int(p.track_id) != int(holder.track_id)
           for p in strong_by_frame.get(t, ())):
        return None
    distance = np.hypot(float(last.x)-float(holder.x),
                        float(last.y)-float(holder.y))
    radius = max((float(last.radius_px)+float(holder.radius_px))/2, 1.)
    if distance/radius/max(t-int(last.frame), 1) > params.maximum_alt_step_radii_per_frame:
        return None
    return alternate


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected pixels must align")
    scored = points.copy()
    coords = np.rint(scored[["y", "x"]].to_numpy(float)).astype(int)
    if (np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= labels.shape[1])
            or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= labels.shape[2])):
        raise ValueError("soma point outside image")
    scored["owner"] = labels[scored.frame.to_numpy(int), coords[:, 0], coords[:, 1]]
    strong = scored[(scored.state.astype(str) == "observed") &
                    scored.strong.astype(bool)].copy()
    by_track = {int(track): group.sort_values("frame")
                for track, group in strong.groupby("track_id")}
    strong_by_frame = {int(t): list(group.itertuples(index=False))
                       for t, group in strong.groupby("frame")}
    cache = {}
    proposals = []
    audit = []
    for seed_track, group in by_track.items():
        named = group[group.owner.gt(0)]
        if len(named) < params.minimum_stable_observations+1:
            continue
        initial_name = int(named.owner.iloc[0])
        different = np.flatnonzero(named.owner.to_numpy(int) != initial_name)
        if len(different) == 0 or int(different[0]) < params.minimum_stable_observations:
            continue
        first = named.iloc[int(different[0])]
        t0 = int(first.frame)
        prior = named.iloc[int(different[0])-1]
        if t0-int(prior.frame) != 1:
            continue
        first_owner = int(first.owner)
        branches = []
        for branch_track, branch in by_track.items():
            if branch_track == seed_track or len(branch) < params.minimum_branch_observations:
                continue
            head = branch.iloc[0]
            if not (t0 <= int(head.frame) <= t0+params.maximum_branch_start_lag
                    and int(head.owner) == first_owner):
                continue
            distance = np.hypot(float(head.x)-float(prior.x),
                                float(head.y)-float(prior.y))
            radius = max((float(head.radius_px)+float(prior.radius_px))/2, 1.)
            if distance/radius <= params.maximum_branch_start_step_radii:
                branches.append((distance/radius, branch_track))
        if len(branches) != 1:
            continue
        _, branch_track = branches[0]
        branch = by_track[branch_track].set_index("frame")
        seed = group.set_index("frame")
        base = {"seed_track": seed_track, "branch_track": branch_track,
                "first_frame": t0+1, "initial_name": initial_name}
        changes = []
        stop = "end_of_branch"
        previous = prior
        for t in range(t0, min(len(labels), int(branch.index.max())+1)):
            if t == t0:
                target = first
            elif t in branch.index:
                target = branch.loc[t]
            else:
                stop = "branch_not_strong"; break
            if isinstance(target, pd.DataFrame) or int(target.owner) <= 0:
                stop = "branch_owner_missing"; break
            step = np.hypot(float(target.x)-float(previous.x),
                            float(target.y)-float(previous.y))
            radius = max((float(target.radius_px)+float(previous.radius_px))/2, 1.)
            if step/radius > params.maximum_target_step_radii:
                stop = "branch_motion_jump"; break
            owner = int(target.owner)
            component = _component(labels, t, owner, target, cache)
            if component is None or np.any(component & protected[t]):
                stop = "unavailable_or_protected_source"; break
            members = [p for p in strong_by_frame.get(t, ())
                       if int(p.track_id) in (seed_track, branch_track)
                       and _inside(component, p)]
            if not any(int(p.track_id) == int(target.track_id) for p in members):
                stop = "target_not_in_source"; break
            others = [p for p in strong_by_frame.get(t, ())
                      if int(p.track_id) not in (seed_track, branch_track)
                      and _inside(component, p)]
            selected = _basin(component, raw[t], members, others, params)
            if selected is None:
                stop = "unresolved_shared_outline"; break
            displaced = []
            if owner != initial_name:
                for holder in strong_by_frame.get(t, ()):
                    if int(holder.owner) != initial_name or _inside(selected, holder):
                        continue
                    alternate = _alternate_name(holder, t, initial_name,
                                                by_track, strong_by_frame,
                                                labels, params)
                    holder_mask = (_component(labels, t, initial_name,
                                              holder, cache) if alternate else None)
                    if (alternate is None or holder_mask is None
                            or np.any(holder_mask & protected[t])
                            or np.any(holder_mask & selected)
                            or any(_inside(holder_mask, p) and
                                   int(p.track_id) != int(holder.track_id)
                                   for p in strong_by_frame.get(t, ()))):
                        stop = "unresolved_other_holder"; break
                    displaced.append((holder_mask, alternate))
            if stop == "unresolved_other_holder":
                break
            changes.append((t, owner, selected, displaced))
            previous = target
        total = sum(int(mask.sum())+sum(int(other.sum()) for other, _ in displaced)
                    for _, _, mask, displaced in changes)
        if len(changes) < params.minimum_corrected_frames or total < params.minimum_corrected_pixels:
            audit.append({**base, "outcome": "rejected", "reason": stop,
                          "eligible_frames": len(changes), "changed_pixels": 0})
            continue
        proposals.append((total, base, changes, stop))
    candidate = labels.copy()
    applications = []
    for total, base, changes, stop in sorted(proposals, key=lambda row: -row[0]):
        if any(np.any(candidate[t][mask] != owner) or
               any(np.any(candidate[t][other] != base["initial_name"])
                   for other, _ in displaced)
               for t, owner, mask, displaced in changes):
            audit.append({**base, "outcome": "rejected",
                          "reason": "overlapping_proposal", "eligible_frames": 0,
                          "changed_pixels": 0})
            continue
        for t, owner, mask, displaced in changes:
            candidate[t][mask] = base["initial_name"]
            count = int(mask.sum())
            for other, alternate in displaced:
                candidate[t][other] = alternate
                count += int(other.sum())
            applications.append({**base, "frame": t+1, "source_owner": owner,
                                 "target_pixels": int(mask.sum()),
                                 "displaced_pixels": count-int(mask.sum()),
                                 "changed_pixels": count})
        audit.append({**base, "outcome": "applied", "reason": stop,
                      "eligible_frames": len(changes), "changed_pixels": total})
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("stable soma branch correction crossed accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
