"""Carry a stable name through an exchange when the old physical holder ends.

The producer receives accepted masks, raw images, physical points and a
protected-pixel mask. It never receives feedback or chosen cell identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import recover_short_ownerless_soma_runs as core_rule
import recover_stable_dual_soma_contacts as dual
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class Params:
    minimum_jump_radii: float = 3.0
    minimum_destination_reference_frames: int = 20
    maximum_destination_reference_gap: int = 2
    minimum_old_holder_run_frames: int = 3
    minimum_later_run_frames: int = 20
    maximum_later_step_p95_radii: float = 1.2
    minimum_later_raw_peak_retention: float = .1
    minimum_later_raw_core_area_ratio: float = .4
    maximum_old_holder_weak_tail_frames: int = 2


def _reference(group: pd.DataFrame, owner: int, onset: int, params: Params):
    qualified = group[(group.frame.astype(int).lt(onset))
                      & group.accepted_owner.astype(int).eq(owner)
                      & group.strong.astype(bool)
                      & group.physically_visible.astype(bool)
                      & ~group.in_encounter.astype(bool)
                      & group.state.astype(str).eq("observed")]
    for run in reversed(core_rule._consecutive_runs(qualified)):
        if (len(run) >= params.minimum_destination_reference_frames
                and 0 <= onset-int(run.frame.max())-1
                   <= params.maximum_destination_reference_gap):
            return run
    return group.iloc[0:0]


def _run_ending_before(group: pd.DataFrame, owner: int, onset: int) -> pd.DataFrame:
    prior = group[group.frame.astype(int).lt(onset)].sort_values("frame")
    keep = []
    expected = onset-1
    for row in reversed(list(prior.itertuples(index=False))):
        if (int(row.frame) != expected or int(row.accepted_owner) != owner
                or not bool(row.physically_visible)):
            break
        keep.append(row)
        expected -= 1
    return pd.DataFrame.from_records(list(reversed(keep)), columns=group.columns)


def _run_starting_at(group: pd.DataFrame, owner: int, onset: int) -> pd.DataFrame:
    later = group[group.frame.astype(int).ge(onset)].sort_values("frame")
    keep = []
    expected = onset
    for row in later.itertuples(index=False):
        if (int(row.frame) != expected or int(row.accepted_owner) != owner
                or not bool(row.physically_visible)):
            break
        keep.append(row)
        expected += 1
    return pd.DataFrame.from_records(keep, columns=group.columns)


def _component_near(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    binary = frame == owner
    parts, count = ndi.label(binary, STRUCTURE)
    if count != 1:
        return None
    ys, xs = np.where(binary)
    if not len(ys):
        return None
    distance = np.min(np.hypot(ys-float(point.y), xs-float(point.x)))
    if distance > max(1.0, float(point.radius_px)):
        return None
    return binary


def _other_point_inside(component: np.ndarray, target_track: int,
                        frame_points: pd.DataFrame) -> bool:
    for point in frame_points.itertuples(index=False):
        if int(point.track_id) == target_track or not bool(point.physically_visible):
            continue
        y, x = int(round(float(point.y))), int(round(float(point.x)))
        if 0 <= y < component.shape[0] and 0 <= x < component.shape[1] and component[y,x]:
            return True
    return False


def correct(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, protected: np.ndarray,
            params: Params = Params()
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != unclaimed.shape or labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, unclaimed, raw and protected arrays must align")
    result = labels.copy()
    scored = physical.attach_owners(points, labels)
    for field in ("strong", "physically_visible", "in_encounter"):
        scored[field] = scored[field].astype(bool)
    independent_references = dual._references(scored, dual.Params())
    named = scored[scored.accepted_owner.astype(int).gt(0)
                   & scored.physically_visible.astype(bool)]
    unique = named.groupby(["accepted_owner", "frame"]).filter(
        lambda group: len(group) == 1 and group.track_id.nunique() == 1)
    by_track = {int(track): group.sort_values("frame")
                for track, group in scored.groupby("track_id")}
    by_track_frame = {(int(row.track_id), int(row.frame)): row
                      for row in scored.itertuples(index=False)}
    by_frame = {int(frame): group for frame, group in scored.groupby("frame")}
    audits, applications = [], []
    for jumping_name, group in unique.groupby("accepted_owner", sort=True):
        ordered = list(group.sort_values("frame").itertuples(index=False))
        for old, new in zip(ordered, ordered[1:]):
            t = int(new.frame)
            if t != int(old.frame)+1 or int(new.track_id) == int(old.track_id):
                continue
            old_track, new_track = int(old.track_id), int(new.track_id)
            radii = float(np.hypot(float(new.x)-float(old.x),
                                   float(new.y)-float(old.y))
                          / max((float(old.radius_px)+float(new.radius_px))/2, 1))
            if radii < params.minimum_jump_radii:
                continue
            old_now = by_track_frame.get((old_track,t))
            if old_now is None or not bool(old_now.physically_visible):
                continue
            earlier_name = int(old_now.accepted_owner)
            if earlier_name <= 0 or earlier_name == int(jumping_name):
                continue
            reference = _reference(by_track[new_track], earlier_name, t, params)
            old_run = _run_ending_before(by_track[old_track], int(jumping_name), t)
            later = _run_starting_at(by_track[new_track], int(jumping_name), t)
            if (len(reference) < params.minimum_destination_reference_frames
                    or len(old_run) < params.minimum_old_holder_run_frames
                    or len(later) < params.minimum_later_run_frames):
                continue
            old_future = by_track[old_track][
                by_track[old_track].frame.astype(int).gt(t)]
            old_visible = old_future[old_future.physically_visible.astype(bool)]
            old_strong = old_visible[
                old_visible.strong.astype(bool)
                & old_visible.state.astype(str).eq("observed")]
            weak_tail_frames = old_visible.frame.astype(int).tolist()
            later_rows = list(later.itertuples(index=False))
            competing_holders = any(
                int(other.track_id) != new_track
                and int(other.accepted_owner) == int(jumping_name)
                and bool(other.physically_visible) and bool(other.strong)
                and str(other.state) == "observed"
                for point in later_rows
                for other in by_frame[int(point.frame)].itertuples(index=False))
            independently_named_old_future = any(
                int(anchor["owner"]) == int(jumping_name)
                and any(bool(point.strong) and bool(point.physically_visible)
                        and str(point.state) == "observed" and int(point.frame) > t
                        for point in by_track[int(track)].itertuples(index=False))
                for track, anchor in independent_references.items())
            if (len(old_strong)
                    or len(weak_tail_frames) > params.maximum_old_holder_weak_tail_frames
                    or weak_tail_frames != list(range(t+1,t+1+len(weak_tail_frames)))
                    or competing_holders or independently_named_old_future):
                audits.append({"jumping_name":int(jumping_name),
                               "earlier_name":earlier_name,"old_track":old_track,
                               "new_track":new_track,"exchange_frame":t+1,
                               "reference_frames":len(reference),
                               "later_frames":len(later),"status":"refused",
                               "refusals":"old_name_not_physically_retired"})
                continue
            track_steps = []
            for a,b in zip(later_rows,later_rows[1:]):
                distance = float(np.hypot(float(a.x)-float(b.x),float(a.y)-float(b.y)))
                track_steps.append(distance/max((float(a.radius_px)+float(b.radius_px))/2,1))
            if track_steps and np.quantile(track_steps,.95)>params.maximum_later_step_p95_radii:
                continue
            ref_peaks = [core_rule._core(raw[int(q.frame)],q,.35)[2]
                         for q in reference.itertuples(index=False)]
            median_peak = float(np.median(ref_peaks))
            safe = []
            refusals = []
            old_component = _component_near(result[t],earlier_name,old_now)
            if old_component is None or _other_point_inside(
                    old_component,old_track,by_frame[t]) or np.any(protected[t][old_component]):
                refusals.append(f"{t+1}:old_holder_component_unsafe")
            for point in later_rows:
                frame = int(point.frame)
                component = _component_near(result[frame],int(jumping_name),point)
                if component is None:
                    refusals.append(f"{frame+1}:destination_component_ambiguous")
                    continue
                if (_other_point_inside(component,new_track,by_frame[frame])
                        or np.any(protected[frame][component])):
                    refusals.append(f"{frame+1}:destination_component_protected_or_shared")
                    continue
                if frame > t and np.any(result[frame] == earlier_name):
                    refusals.append(f"{frame+1}:earlier_name_elsewhere")
                    continue
                core, region, peak, area = core_rule._core(raw[frame],point,.35)
                if (peak/max(median_peak,1) < params.minimum_later_raw_peak_retention
                        or area < params.minimum_later_raw_core_area_ratio
                        or not np.any(core)):
                    refusals.append(f"{frame+1}:raw_soma_unsupported")
                    continue
                safe.append((frame,component,peak/max(median_peak,1),area))
            if refusals or len(safe)!=len(later):
                audits.append({"jumping_name":int(jumping_name),"earlier_name":earlier_name,
                               "old_track":old_track,"new_track":new_track,
                               "exchange_frame":t+1,"reference_frames":len(reference),
                               "later_frames":len(later),"status":"refused",
                               "refusals":"|".join(refusals)})
                continue
            # All masks come from the frozen parent. The swap is atomic at t.
            for frame, component, peak_retention, area in safe:
                pixels = int(component.sum())
                result[frame][component] = earlier_name
                applications.append({"kind":"destination_path","frame":frame+1,
                                     "jumping_name":int(jumping_name),
                                     "earlier_name":earlier_name,"old_track":old_track,
                                     "new_track":new_track,"changed_pixels":pixels,
                                     "peak_retention":peak_retention,
                                     "core_area_ratio":area})
            result[t][old_component] = int(jumping_name)
            applications.append({"kind":"reciprocal_source","frame":t+1,
                                 "jumping_name":int(jumping_name),
                                 "earlier_name":earlier_name,"old_track":old_track,
                                 "new_track":new_track,
                                 "changed_pixels":int(old_component.sum()),
                                 "peak_retention":np.nan,"core_area_ratio":np.nan})
            audits.append({"jumping_name":int(jumping_name),"earlier_name":earlier_name,
                           "old_track":old_track,"new_track":new_track,
                           "exchange_frame":t+1,"reference_frames":len(reference),
                           "later_frames":len(later),"status":"applied","refusals":""})
    changed = result != labels
    if (np.any(changed & protected)
            or not np.array_equal(result>0,labels>0)
            or not np.array_equal(np.unique(result),np.unique(labels))):
        raise AssertionError("reciprocal exchange crossed a protected boundary")
    return result, unclaimed.copy(), pd.DataFrame(audits), pd.DataFrame(applications)
