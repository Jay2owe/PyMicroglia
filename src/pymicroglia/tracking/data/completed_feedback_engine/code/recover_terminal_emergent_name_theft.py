"""Restore terminal soma ownership while naming a separable emerging soma.

Discovery uses strong physical tracks, accepted masks and raw soma basins.
No movie, track, identity, coordinate, region, frame or feedback selector is
an input to the rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_early_frames: int = 8
    minimum_terminal_frames: int = 4
    maximum_terminal_end_gap: int = 1
    maximum_target_step_p95_radii: float = .9
    minimum_holder_stolen_frames: int = 2
    maximum_holder_early_lead_frames: int = 6
    minimum_shared_host_frames: int = 2
    minimum_separation_sum_radii: float = 3.0
    maximum_raw_valley_ratio: float = .85
    minimum_basin_area_ratio: float = .2
    maximum_basin_area_ratio: float = 4.0
    maximum_residual_distance_radii: float = 3.0
    watershed_sigma_px: float = 1.0


def _strong_points(points: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    strong = points[points.state.astype(str).eq("observed")
                    & points.strong.astype(bool)].copy()
    if strong.empty:
        return strong.assign(owner=pd.Series(dtype=int))
    t = strong.frame.to_numpy(int)
    y = np.rint(strong.y.to_numpy(float)).astype(int)
    x = np.rint(strong.x.to_numpy(float)).astype(int)
    valid = (0 <= t) & (t < len(labels)) & (0 <= y) & (y < labels.shape[1]) & (0 <= x) & (x < labels.shape[2])
    strong = strong.iloc[np.flatnonzero(valid)].copy()
    strong["owner"] = labels[t[valid], y[valid], x[valid]]
    return strong


def _runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for point in group.sort_values("frame").itertuples(index=False):
        t, name = int(point.frame), int(point.owner)
        if runs and runs[-1]["name"] == name and runs[-1]["last"] + 1 == t:
            runs[-1]["last"] = t
            runs[-1]["frames"] += 1
        else:
            runs.append({"name": name, "first": t, "last": t, "frames": 1})
    return runs


def _component(frame: np.ndarray, name: int, point) -> np.ndarray | None:
    y, x = round(float(point.y)), round(float(point.x))
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]
            and int(frame[y, x]) == int(name)):
        return None
    parts, _ = ndi.label(frame == int(name), STRUCTURE)
    cid = int(parts[y, x])
    return parts == cid if cid else None


def _inside(mask: np.ndarray, point) -> bool:
    y, x = round(float(point.y)), round(float(point.x))
    return bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                and mask[y, x])


def discover(labels: np.ndarray, points: pd.DataFrame,
             params: Params = Params()) -> tuple[list[dict], pd.DataFrame]:
    strong = _strong_points(points, labels)
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    by_track = {int(track): group.sort_values("frame")
                for track, group in strong.groupby("track_id")}
    proposals: list[dict] = []
    audit: list[dict] = []
    for track, group in by_track.items():
        runs = _runs(group)
        if len(runs) < 2:
            continue
        early, terminal = runs[-2:]
        old, wrong = int(early["name"]), int(terminal["name"])
        if (old <= 0 or wrong <= 0 or old == wrong
                or early["frames"] < params.minimum_early_frames
                or terminal["frames"] < params.minimum_terminal_frames
                or terminal["last"] < len(labels)-1-params.maximum_terminal_end_gap
                or early["last"]+1 != terminal["first"]):
            continue
        steps = np.linalg.norm(np.diff(group[["x", "y"]].to_numpy(float), axis=0), axis=1)
        radii = np.maximum(group.radius_px.to_numpy(float)[:-1], 1.)
        p95 = float(np.percentile(steps/radii, 95)) if len(steps) else 0.
        if p95 > params.maximum_target_step_p95_radii:
            continue
        late = group[group.frame.astype(int).between(terminal["first"], terminal["last"])]
        candidates = {}
        for point in late.itertuples(index=False):
            frame = int(point.frame)
            for other in by_frame.get(frame, group.iloc[0:0]).itertuples(index=False):
                if int(other.track_id) == track or int(other.owner) != old:
                    continue
                distance = float(np.hypot(float(point.x)-float(other.x),
                                          float(point.y)-float(other.y)))
                scale = float(point.radius_px)+float(other.radius_px)
                if distance >= params.minimum_separation_sum_radii*scale:
                    candidates.setdefault(int(other.track_id), []).append(frame)
        for holder, stolen_frames in sorted(candidates.items()):
            if len(stolen_frames) < params.minimum_holder_stolen_frames:
                continue
            holder_group = by_track[holder]
            frames = holder_group.frame.to_numpy(int)
            if (len(frames) < 4 or np.any(np.diff(frames) != 1)
                    or int(frames.min()) < terminal["first"]-params.maximum_holder_early_lead_frames):
                continue
            before = holder_group[holder_group.frame.astype(int).lt(terminal["first"])]
            before_names = before.owner.astype(int)
            before_names = before_names[(before_names > 0) & (before_names != old)]
            if len(before_names) < params.minimum_shared_host_frames or before_names.nunique() != 1:
                continue
            host = int(before_names.iloc[0])
            if (holder_group.owner.astype(int).isin([old, host]).sum()
                    != len(holder_group)):
                continue
            shared = 0
            for point in before.itertuples(index=False):
                frame = int(point.frame)
                if int(point.owner) != host:
                    continue
                mask = _component(labels[frame], host, point)
                if mask is not None and any(
                    int(other.track_id) != holder and _inside(mask, other)
                    for other in by_frame.get(frame, group.iloc[0:0]).itertuples(index=False)
                ):
                    shared += 1
            if shared < params.minimum_shared_host_frames:
                continue
            proposal = {
                "target_track": track, "holder_track": holder,
                "old_name": old, "wrong_name": wrong, "host_name": host,
                "first_terminal_frame": terminal["first"]+1,
                "terminal_frames": terminal["frames"],
                "holder_first_frame": int(frames.min())+1,
                "holder_last_frame": int(frames.max())+1,
                "holder_stolen_frames": len(stolen_frames),
                "holder_shared_frames": shared,
                "target_step_p95_radii": p95,
                "target_points": late, "holder_points": holder_group,
                "strong_by_frame": by_frame,
            }
            proposals.append(proposal)
            audit.append({key: value for key, value in proposal.items()
                          if key not in ("target_points", "holder_points", "strong_by_frame")})
    return proposals, pd.DataFrame(audit)


def _holder_basin(mask: np.ndarray, raw_frame: np.ndarray, holder,
                  others: list, reference_area: float,
                  params: Params) -> np.ndarray | None:
    if not others:
        basin = mask
    else:
        target_seed = physical._marker_pixel(
            mask, float(holder.x), float(holder.y))
        if target_seed is None:
            return None
        markers = np.zeros(mask.shape, np.uint16)
        markers[target_seed] = 1
        nearest = min(others, key=lambda p: np.hypot(
            float(p.x)-float(holder.x), float(p.y)-float(holder.y)))
        valley = physical._valley_ratio(
            raw_frame, (float(holder.x), float(holder.y)),
            (float(nearest.x), float(nearest.y)))
        if not np.isfinite(valley) or valley > params.maximum_raw_valley_ratio:
            return None
        for other in others:
            seed = physical._marker_pixel(
                mask, float(other.x), float(other.y))
            if seed is None or markers[seed]:
                continue
            markers[seed] = int(markers.max())+1
        if int(markers.max()) < 2:
            return None
        elevation = -ndi.gaussian_filter(
            raw_frame.astype(np.float32), params.watershed_sigma_px)
        assigned = watershed(elevation, markers=markers, mask=mask,
                             connectivity=STRUCTURE)
        basin = assigned == 1
    area = int(basin.sum())
    if (area < 8 or ndi.label(basin, STRUCTURE)[1] != 1
            or not params.minimum_basin_area_ratio
            <= area / max(reference_area, 1.)
            <= params.maximum_basin_area_ratio):
        return None
    return basin


def _build_proposal(labels: np.ndarray, raw: np.ndarray,
                    protected: np.ndarray, proposal: dict,
                    params: Params) -> tuple[list[dict], str]:
    target, holder = [], []
    old, wrong, host = (proposal["old_name"], proposal["wrong_name"],
                        proposal["host_name"])
    by_frame = proposal["strong_by_frame"]
    for point in proposal["target_points"].itertuples(index=False):
        t = int(point.frame)
        mask = _component(labels[t], wrong, point)
        if (mask is None or np.any(mask & protected[t])
                or any(int(other.track_id) != proposal["target_track"]
                       and _inside(mask, other)
                       for other in by_frame[t].itertuples(index=False))):
            return [], "target_not_complete_single_soma"
        target.append({"frame": t, "mask": mask, "old_name": wrong,
                       "kind": "restore_stable_target"})
    isolated = []
    for point in proposal["holder_points"].itertuples(index=False):
        t = int(point.frame)
        if int(point.owner) == old:
            mask = _component(labels[t], old, point)
            if mask is None:
                return [], "holder_component_missing"
            isolated.append(int(mask.sum()))
    if not isolated:
        return [], "holder_reference_area_missing"
    reference_area = float(np.median(isolated))
    for point in proposal["holder_points"].itertuples(index=False):
        t, name = int(point.frame), int(point.owner)
        mask = _component(labels[t], name, point)
        if mask is None:
            return [], "holder_component_missing"
        others = [other for other in by_frame[t].itertuples(index=False)
                  if int(other.track_id) != proposal["holder_track"]
                  and _inside(mask, other)]
        basin = _holder_basin(mask, raw[t], point, others,
                              reference_area, params)
        if basin is None or np.any(basin & protected[t]):
            return [], "holder_raw_basin_unsupported_or_protected"
        if name == old:
            # Small detached pieces of the stolen name near the emerging
            # soma follow its basin when they contain no other strong soma.
            parts, count = ndi.label(labels[t] == old, STRUCTURE)
            extra = np.zeros(mask.shape, bool)
            for cid in range(1, count+1):
                part = parts == cid
                if np.any(part & mask) or int(part.sum()) > .25*reference_area:
                    continue
                if np.any(part & protected[t]) or any(
                    _inside(part, other) for other in by_frame[t].itertuples(index=False)
                    if int(other.track_id) != proposal["holder_track"]):
                    continue
                yy, xx = np.where(part)
                if len(yy) and np.hypot(xx.mean()-float(point.x),
                                       yy.mean()-float(point.y)) <= (
                                           params.maximum_residual_distance_radii
                                           * float(point.radius_px)):
                    extra |= part
            basin = basin | extra
        holder.append({"frame": t, "mask": basin, "old_name": name,
                       "kind": "name_emerging_soma"})
    if (len(target) != proposal["terminal_frames"]
            or len(holder) != proposal["holder_last_frame"]
                              - proposal["holder_first_frame"]+1):
        return [], "incomplete_target_or_holder_path"
    return [*holder, *target], "supported"


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or labels.shape != protected.shape
            or raw.shape[0] < len(labels) or raw.shape[1:] != labels.shape[1:]):
        raise ValueError("labels, raw and protected masks must align")
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    proposals, discovered = discover(labels, points, params)
    audits, applications = [], []
    next_name = int(labels.max())+1
    for proposal in proposals:
        records, reason = _build_proposal(labels, raw, protected,
                                          proposal, params)
        if reason == "supported" and any(
                np.any(occupied[row["frame"]] & row["mask"])
                for row in records):
            reason = "overlaps_other_proposal"
        if reason == "supported" and next_name > np.iinfo(labels.dtype).max:
            reason = "no_identity_value_remains"
        audit = {
            key: value for key, value in proposal.items()
            if key not in ("target_points", "holder_points", "strong_by_frame")
        }
        audit.update({
            "outcome": "applied" if reason == "supported" else "rejected",
            "reason": reason,
            "target_pixels": sum(int(row["mask"].sum()) for row in records
                                 if row["kind"] == "restore_stable_target"),
            "holder_pixels": sum(int(row["mask"].sum()) for row in records
                                 if row["kind"] == "name_emerging_soma"),
            "new_holder_name": next_name if reason == "supported" else 0,
        })
        audits.append(audit)
        if reason != "supported":
            continue
        for row in records:
            t, mask = row["frame"], row["mask"]
            name = (next_name if row["kind"] == "name_emerging_soma"
                    else int(proposal["old_name"]))
            candidate[t][mask] = name
            occupied[t][mask] = True
            applications.append({
                "kind": row["kind"], "target_track": proposal["target_track"],
                "holder_track": proposal["holder_track"], "frame": t+1,
                "old_name": row["old_name"], "new_name": name,
                "changed_pixels": int(mask.sum()),
            })
        next_name += 1
    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or int(changed.sum()) != sum(row["changed_pixels"] for row in applications)):
        raise AssertionError("terminal emergent correction crossed accepted boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
