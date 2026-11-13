"""Resume an approved stable soma branch after one raw-supported absent frame.

This opt-in layer reads the preceding general layer's application ledger. It
uses no feedback, recording, name, track, location or frame selector. A weak
predicted soma must occupy almost none of a neighbouring outline before a
strong contiguous return may receive its established name.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import recover_stable_soma_branch_handoffs as prior


@dataclass(frozen=True)
class Params:
    maximum_absent_basin_fraction: float = .05
    minimum_absent_other_basin_area: int = 40
    minimum_return_frames: int = 5
    minimum_return_pixels: int = 500
    maximum_step_radii: float = 3.
    partition: prior.Params = prior.Params()


def _absent_fraction(labels: np.ndarray, raw: np.ndarray, t: int, point,
                     strong_points: list, cache: dict) -> tuple[float, int] | None:
    y, x = round(float(point.y)), round(float(point.x))
    name = int(labels[t, y, x])
    if name <= 0:
        return None
    mask = prior._component(labels, t, name, point, cache)
    if mask is None:
        return None
    others = [other for other in strong_points
              if int(other.track_id) != int(point.track_id)
              and prior._inside(mask, other)]
    if len(others) != 1:
        return None
    other = others[0]
    yy, xx = np.where(mask)
    box = np.s_[max(0, int(yy.min())-2):min(mask.shape[0], int(yy.max())+3),
                max(0, int(xx.min())-2):min(mask.shape[1], int(xx.max())+3)]
    markers = np.zeros(mask[box].shape, np.uint8)
    oy, ox = round(float(other.y)), round(float(other.x))
    markers[y-box[0].start, x-box[1].start] = 1
    markers[oy-box[0].start, ox-box[1].start] = 2
    assigned = watershed(-ndi.gaussian_filter(raw[t][box].astype(np.float32), 1.),
                         markers, mask=mask[box], connectivity=prior.STRUCTURE)
    other_area = int((assigned == 2).sum())
    return int((assigned == 1).sum())/max(int(mask.sum()), 1), other_area


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, prior_audit: pd.DataFrame,
            params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected pixels must align")
    if prior_audit.empty:
        return labels.copy(), pd.DataFrame(), pd.DataFrame()
    scored = points.copy()
    yx = np.rint(scored[["y", "x"]].to_numpy(float)).astype(int)
    if (np.any(yx[:, 0] < 0) or np.any(yx[:, 0] >= labels.shape[1])
            or np.any(yx[:, 1] < 0) or np.any(yx[:, 1] >= labels.shape[2])):
        raise ValueError("soma point outside image")
    scored["owner"] = labels[scored.frame.to_numpy(int), yx[:, 0], yx[:, 1]]
    all_tracks = {int(track): group.set_index("frame")
                  for track, group in scored.groupby("track_id")}
    strong = scored[(scored.state.astype(str) == "observed") &
                    scored.strong.astype(bool)]
    strong_tracks = {int(track): group.sort_values("frame")
                     for track, group in strong.groupby("track_id")}
    strong_by_frame = {int(t): list(group.itertuples(index=False))
                       for t, group in strong.groupby("frame")}
    cache = {}
    proposals, audits = [], []
    for old in prior_audit.itertuples(index=False):
        if str(old.outcome) != "applied" or str(old.reason) != "branch_not_strong":
            continue
        branch_id = int(old.branch_track)
        anchor_name = int(old.initial_name)
        gap = int(old.first_frame) - 1 + int(old.eligible_frames)
        base = {"seed_track": int(old.seed_track), "branch_track": branch_id,
                "gap_frame": gap + 1, "initial_name": anchor_name}
        branch = all_tracks.get(branch_id)
        if branch is None or gap not in branch.index or gap + 1 not in branch.index:
            continue
        weak = branch.loc[gap]
        next_point = branch.loc[gap + 1]
        if (isinstance(weak, pd.DataFrame) or isinstance(next_point, pd.DataFrame)
                or (str(weak.state) == "observed" and bool(weak.strong))
                or str(next_point.state) != "observed" or not bool(next_point.strong)):
            continue
        absence = _absent_fraction(labels, raw, gap, weak,
                                   strong_by_frame.get(gap, []), cache)
        if (absence is None or absence[0] > params.maximum_absent_basin_fraction
                or absence[1] < params.minimum_absent_other_basin_area):
            audits.append({**base, "outcome": "rejected",
                           "reason": "gap_has_no_exclusive_absence_evidence",
                           "changed_pixels": 0})
            continue
        frames = []
        reason = "end_of_track"
        previous = weak
        for t in range(gap + 1, min(int(branch.index.max()) + 1, len(labels))):
            if t not in branch.index:
                reason = "missing_branch_point"; break
            target = branch.loc[t]
            if (isinstance(target, pd.DataFrame) or str(target.state) != "observed"
                    or not bool(target.strong) or int(target.owner) <= 0):
                reason = "branch_not_strong"; break
            distance = np.hypot(float(target.x)-float(previous.x),
                                float(target.y)-float(previous.y))
            radius = max((float(target.radius_px)+float(previous.radius_px))/2, 1.)
            if distance/radius > params.maximum_step_radii:
                reason = "soma_step_too_large"; break
            owner = int(target.owner)
            if owner == anchor_name:
                reason = "already_restored"; break
            mask = prior._component(labels, t, owner, target, cache)
            if mask is None or np.any(mask & protected[t]):
                reason = "unavailable_or_protected_source"; break
            members = [p for p in strong_by_frame.get(t, ())
                       if int(p.track_id) == branch_id and prior._inside(mask, p)]
            others = [p for p in strong_by_frame.get(t, ())
                      if int(p.track_id) != branch_id and prior._inside(mask, p)]
            if len(members) != 1:
                reason = "branch_not_unique_in_source"; break
            selected = prior._basin(mask, raw[t], members, others,
                                    params.partition)
            if selected is None:
                reason = "raw_partition_unresolved"; break
            displaced = []
            for holder in strong_by_frame.get(t, ()):
                if int(holder.owner) != anchor_name or prior._inside(selected, holder):
                    continue
                alternate = prior._alternate_name(
                    holder, t, anchor_name, strong_tracks, strong_by_frame,
                    labels, params.partition)
                other_mask = (prior._component(labels, t, anchor_name,
                                               holder, cache) if alternate else None)
                if (alternate is None or other_mask is None
                        or np.any(other_mask & protected[t])
                        or np.any(other_mask & selected)
                        or any(prior._inside(other_mask, p) and
                               int(p.track_id) != int(holder.track_id)
                               for p in strong_by_frame.get(t, ()))):
                    reason = "unresolved_other_holder"; break
                displaced.append((other_mask, alternate))
            if reason == "unresolved_other_holder":
                break
            frames.append((t, owner, selected, displaced))
            previous = target
        total = sum(int(mask.sum())+sum(int(other.sum()) for other, _ in displaced)
                    for _, _, mask, displaced in frames)
        if len(frames) < params.minimum_return_frames or total < params.minimum_return_pixels:
            audits.append({**base, "outcome": "rejected", "reason": reason,
                           "eligible_frames": len(frames), "changed_pixels": 0})
            continue
        proposals.append((total, base, frames, reason, absence))
    candidate = labels.copy()
    applications = []
    for total, base, frames, reason, absence in sorted(proposals, key=lambda item: -item[0]):
        if any(np.any(candidate[t][mask] != owner) or
               any(np.any(candidate[t][other] != base["initial_name"])
                   for other, _ in displaced)
               for t, owner, mask, displaced in frames):
            audits.append({**base, "outcome": "rejected", "reason": "overlapping_proposal",
                           "eligible_frames": 0, "changed_pixels": 0})
            continue
        for t, owner, mask, displaced in frames:
            candidate[t][mask] = base["initial_name"]
            count = int(mask.sum())
            for other, alternate in displaced:
                candidate[t][other] = alternate
                count += int(other.sum())
            applications.append({**base, "frame": t + 1,
                                 "source_owner": owner,
                                 "target_pixels": int(mask.sum()),
                                 "displaced_pixels": count-int(mask.sum()),
                                 "changed_pixels": count,
                                 "gap_basin_fraction": absence[0]})
        audits.append({**base, "outcome": "applied", "reason": reason,
                       "eligible_frames": len(frames), "changed_pixels": total,
                       "gap_basin_fraction": absence[0]})
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("post-contact branch recovery crossed accepted boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
