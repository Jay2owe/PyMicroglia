"""Partition brief wrong-name contacts only for established larger cells.

Only physical points, accepted masks, raw images and a prior-correction mask
are producer inputs. Saved feedback never selects a cell or frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import complete_owner_handoff_episodes_r10_a002 as body
import owner_consensus
import recover_stable_dual_soma_contacts as dual
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    flank_window_frames: int = 5
    minimum_named_flank_frames: int = 3
    minimum_other_somata: int = 2
    maximum_other_somata: int = 3
    minimum_reference_area_ratio: float = .5
    maximum_reference_area_ratio: float = 2.0
    minimum_host_residual_pixels: int = 20
    minimum_reference_area_percentile: float = .6


def _flank_count(group: pd.DataFrame, owner: int, t: int, before: bool,
                 params: Params) -> int:
    lo, hi = ((t-params.flank_window_frames,t-1) if before
              else (t+1,t+params.flank_window_frames))
    rows = group[group.frame.astype(int).between(lo,hi)
                 & group.accepted_owner.astype(int).eq(owner)
                 & group.physically_visible.astype(bool)
                 & group.strong.astype(bool)
                 & group.state.astype(str).eq("observed")]
    return len(rows)


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params(),
            anchor_params: dual.Params = dual.Params(),
            partition_params: body.Params = body.Params()
            ) -> tuple[np.ndarray,pd.DataFrame,pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected mask must align")
    scored = physical.attach_owners(points,labels)
    for field in ("strong","physically_visible","in_encounter"):
        scored[field] = scored[field].astype(bool)
    anchors = dual._references(scored,anchor_params)
    by_track = {int(track):group.sort_values("frame")
                for track,group in scored.groupby("track_id")}
    reference_areas = {
        track:dual._reference_area(labels,by_track[track],int(anchor["owner"]),anchor)
        for track,anchor in anchors.items()}
    area_population = np.array([area for area in reference_areas.values() if area>0],float)
    by_frame = {int(t):group for t,group in scored[
        scored.physically_visible & scored.strong
        & scored.state.astype(str).eq("observed")].groupby("frame")}
    proposals, audits = [], []
    for track,group in by_track.items():
        if track not in anchors:
            continue
        owner = int(anchors[track]["owner"])
        reference_area = reference_areas[track]
        if reference_area <= 0:
            continue
        # Match the accepted multi-resident layer's relative large-cell
        # boundary; a movie's own cell-size distribution sets the scale.
        if (not len(area_population) or float(np.mean(area_population<=reference_area))
                < params.minimum_reference_area_percentile):
            continue
        for point in group.itertuples(index=False):
            t, host = int(point.frame),int(point.accepted_owner)
            if (host <= 0 or host == owner or not bool(point.strong)
                    or not bool(point.physically_visible)
                    or str(point.state) != "observed"):
                continue
            before = _flank_count(group,owner,t,True,params)
            after = _flank_count(group,owner,t,False,params)
            if (before < params.minimum_named_flank_frames
                    or after < params.minimum_named_flank_frames):
                continue
            reason = ""
            mask = body._at_component(labels[t],host,point)
            if mask is None:
                reason = "host_component_missing"
            else:
                seats = [other for other in by_frame.get(t,scored.iloc[0:0]).itertuples(index=False)
                         if int(other.track_id)!=track and body._inside(mask,other)
                         and body._distinct(point,other,partition_params)]
                if (not params.minimum_other_somata<=len(seats)<=params.maximum_other_somata
                        or any(int(other.accepted_owner)!=host for other in seats)
                        or any(not body._distinct(a,b,partition_params)
                               for i,a in enumerate(seats) for b in seats[i+1:])):
                    reason = "other_somata_not_distinct_or_shared"
                elif any(int(other.accepted_owner)==owner
                         and body._distinct(point,other,partition_params)
                         for other in by_frame.get(t,scored.iloc[0:0]).itertuples(index=False)):
                    reason = "reference_owner_on_distinct_soma"
                else:
                    basin = body._partition_many(mask,raw[t],point,seats,partition_params)
                    if basin is None:
                        reason = "raw_partition_failed"
                    else:
                        size = int(basin.sum())
                        remaining = mask & ~basin
                        host_has_soma = any(body._inside(remaining,other) for other in seats)
                        if (not params.minimum_reference_area_ratio*reference_area <= size
                                <= params.maximum_reference_area_ratio*reference_area):
                            reason = "basin_size_outside_reference"
                        elif (int(remaining.sum())<params.minimum_host_residual_pixels
                              or not host_has_soma):
                            reason = "host_residual_without_soma"
                        elif np.any(protected[t]&basin):
                            reason = "previous_correction_overlap"
                        else:
                            trial = labels[t].copy()
                            trial[basin] = owner
                            if owner_consensus.count_new_duplicate_components(
                                    labels[t][None],trial[None]):
                                reason = "duplicate_name"
                            else:
                                proposals.append((t,track,owner,host,basin,size,
                                                  reference_area,before,after))
            audits.append({"track_id":track,"reference_owner":owner,
                           "former_owner":host,"frame":t+1,
                           "before_named_frames":before,"after_named_frames":after,
                           "status":"proposed" if not reason else "refused",
                           "reason":reason})
    # Resolve simultaneous proposals against the same frozen pixels together.
    candidate = labels.copy()
    applications = []
    for index,(t,track,owner,host,basin,size,area,before,after) in enumerate(proposals):
        if any(t==other[0] and np.any(basin&other[4])
               for j,other in enumerate(proposals) if j!=index):
            audits.append({"track_id":track,"reference_owner":owner,
                           "former_owner":host,"frame":t+1,
                           "before_named_frames":before,"after_named_frames":after,
                           "status":"refused","reason":"overlapping_simultaneous_proposals"})
            continue
        candidate[t][basin] = owner
        applications.append({"track_id":track,"reference_owner":owner,
                             "former_owner":host,"frame":t+1,
                             "changed_pixels":size,"reference_area_px":area,
                             "before_named_frames":before,"after_named_frames":after})
    changed = candidate != labels
    if (not np.array_equal(candidate>0,labels>0)
            or np.any(changed&protected)
            or not np.array_equal(np.unique(candidate),np.unique(labels))
            or owner_consensus.count_new_duplicate_components(labels,candidate)):
        raise AssertionError("flanked contact layer crossed a protected boundary")
    return candidate,pd.DataFrame(audits),pd.DataFrame(applications)
