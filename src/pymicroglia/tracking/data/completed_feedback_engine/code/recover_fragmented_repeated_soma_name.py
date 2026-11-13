"""Restore a repeatedly renamed soma when its former name has fragmented holders.

This extends the approved terminal-name restoration to multiple later names.
The physical target must be stable and each changed component must contain
only its strong soma. All other components, including name holders, stay put.
No recording, feedback, identity, track, frame or region selects a correction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_visible_fraction: float = .90
    minimum_strong_fraction: float = .80
    maximum_first_visible_frame: int = 3
    maximum_last_visible_gap: int = 1
    minimum_anchor_frames: int = 20
    minimum_later_labelled_frames: int = 30
    minimum_later_names: int = 3
    maximum_step_p95_radii_per_frame: float = 1.0
    maximum_step_radii_per_frame: float = 2.5
    minimum_holder_frames: int = 20
    maximum_one_holder_fraction: float = .5


def _runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        frame, owner = int(row.frame), int(row.accepted_owner)
        if runs and runs[-1]["name"] == owner and runs[-1]["last"] + 1 == frame:
            runs[-1]["last"] = frame
            runs[-1]["rows"].append(row)
        else:
            runs.append({"name": owner, "first": frame, "last": frame, "rows": [row]})
    return runs


def _component(frame: np.ndarray, name: int, row) -> np.ndarray | None:
    parts, _ = ndi.label(frame == name, STRUCTURE)
    y, x = round(float(row.y)), round(float(row.x))
    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and parts[y, x]:
        return parts == int(parts[y, x])
    yy, xx = np.where(parts > 0)
    if len(yy):
        distance = np.hypot(yy-float(row.y), xx-float(row.x))
        best = int(np.argmin(distance))
        if float(distance[best]) <= max(1., float(row.radius_px)):
            return parts == int(parts[yy[best], xx[best]])
    return None


def correct(labels: np.ndarray, points: pd.DataFrame, protected: np.ndarray,
            params: Params = Params()) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != protected.shape:
        raise ValueError("aligned complete label and protected movies required")
    scored = physical.attach_owners(points, labels)
    scored["physically_visible"] = scored.physically_visible.astype(bool)
    scored["strong"] = scored.strong.astype(bool)
    visible = scored[scored.physically_visible].copy()
    strong = scored[scored.state.astype(str).eq("observed") & scored.strong].copy()
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    candidate = labels.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    audit, applications = [], []

    for track, group in visible.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        if (len(group) < params.minimum_visible_fraction * len(labels)
                or int(group.frame.min()) > params.maximum_first_visible_frame
                or int(group.frame.max()) < len(labels)-1-params.maximum_last_visible_gap
                or int(group.strong.sum()) < params.minimum_strong_fraction * len(labels)):
            continue
        anchors = [run for run in _runs(group)
                   if run["name"] > 0 and len(run["rows"]) >= params.minimum_anchor_frames
                   and sum(bool(row.strong) for row in run["rows"]) >= .8*len(run["rows"])]
        if not anchors:
            continue
        anchor = anchors[0]
        first_name, switch = int(anchor["name"]), int(anchor["last"])+1
        later = group[group.frame.astype(int).ge(switch)]
        later_named = later[later.accepted_owner.astype(int).gt(0)]
        other_names = sorted(set(later_named.accepted_owner.astype(int))-{first_name})
        if (len(later_named) < params.minimum_later_labelled_frames
                or len(other_names) < params.minimum_later_names
                or later_named.accepted_owner.astype(int).eq(first_name).any()):
            continue
        path = group[group.frame.astype(int).ge(int(anchor["first"]))]
        xy = path[["y", "x"]].to_numpy(dtype=float)
        radii = np.maximum(path.radius_px.to_numpy(dtype=float), 1.)
        gaps = np.maximum(np.diff(path.frame.to_numpy(dtype=float)), 1.)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(
            (radii[:-1]+radii[1:])/2, 1.) / gaps
        if (len(steps) and (np.percentile(steps, 95) > params.maximum_step_p95_radii_per_frame
                            or np.max(steps) > params.maximum_step_radii_per_frame)):
            continue

        records = []
        reason = "supported"
        for row in later_named.itertuples(index=False):
            t, old_name = int(row.frame), int(row.accepted_owner)
            if str(row.state) != "observed" or not bool(row.strong):
                reason = "target_not_strong_observed"
                break
            mask = _component(labels[t], old_name, row)
            if mask is None:
                reason = "target_has_no_outline"
                break
            for other in by_frame.get(t, pd.DataFrame()).itertuples(index=False):
                if int(other.track_id) == int(track):
                    continue
                y, x = round(float(other.y)), round(float(other.x))
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    reason = "target_outline_contains_other_strong_soma"
                    break
            if reason != "supported":
                break
            if np.any(protected[t] & mask) or np.any(occupied[t] & mask):
                reason = "target_touches_prior_correction"
                break
            records.append((t, old_name, mask, int(mask.sum())))

        holder_frames = 0
        holder_tracks: Counter[int] = Counter()
        if reason == "supported":
            for t in range(switch, len(labels)):
                # A holder is spatially separate from the physical target.
                target_at_t = group[group.frame.astype(int).eq(t)]
                if target_at_t.empty:
                    continue
                target_row = target_at_t.iloc[0]
                target_xy = np.array((float(target_row.y), float(target_row.x)))
                parts, count = ndi.label(labels[t] == first_name, STRUCTURE)
                present = False
                for component in range(1, count+1):
                    mask = parts == component
                    yy, xx = np.where(mask)
                    if not len(yy) or np.linalg.norm(np.array((yy.mean(), xx.mean()))-target_xy) < 4*max(float(target_row.radius_px), 1.):
                        continue
                    present = True
                    for other in by_frame.get(t, pd.DataFrame()).itertuples(index=False):
                        if int(other.track_id) == int(track):
                            continue
                        y, x = round(float(other.y)), round(float(other.x))
                        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                            holder_tracks[int(other.track_id)] += 1
                holder_frames += int(present)
            if holder_frames < params.minimum_holder_frames:
                reason = "first_name_not_recurrently_held_elsewhere"
            elif max(holder_tracks.values(), default=0) >= params.maximum_one_holder_fraction*holder_frames:
                reason = "coherent_first_name_holder"

        audit.append(dict(track_id=int(track), first_name=first_name,
                          anchor_first_frame=int(anchor["first"])+1,
                          anchor_last_frame=int(anchor["last"])+1,
                          later_names="|".join(map(str, other_names)),
                          later_labelled_frames=len(later_named),
                          target_pixels=sum(r[3] for r in records),
                          holder_frames=holder_frames,
                          largest_holder_track_frames=max(holder_tracks.values(), default=0),
                          outcome="applied" if reason == "supported" else "rejected",
                          reason=reason))
        if reason != "supported":
            continue
        for t, old_name, mask, area in records:
            candidate[t][mask] = first_name
            occupied[t][mask] = True
            applications.append(dict(track_id=int(track), frame=t+1,
                                     old_name=old_name, restored_name=first_name,
                                     changed_pixels=area))
    if np.any((candidate != labels) & protected) or not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("fragmented repeated-name correction crossed protected boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
