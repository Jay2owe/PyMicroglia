"""Separate a persistent small soma from an independently stable larger cell.

Discovery uses only full physical tracks, accepted masks, raw images, and the
original unclaimed-mask ledger. Feedback, movie names, cell IDs and coordinates
never select a correction. Unsupported whole paths are left unchanged.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import complete_owner_handoff_episodes_r10_a002 as body
import recover_stable_dual_soma_contacts as dual
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_anchor_strong_frames: int = 10
    minimum_large_radius_px: float = 4.0
    minimum_radius_ratio: float = 2.0
    minimum_satellite_visible_frames: int = 50
    minimum_shared_raw_frames: int = 50
    maximum_median_shared_component_px: float = 600.0
    minimum_area_ratio: float = .25
    maximum_area_ratio: float = 4.0
    minimum_raw_gap_contrast_ratio: float = 3.0
    maximum_soma_to_basin_distance_px: float = 1.5


def _strong_observed(point) -> bool:
    return (bool(point.physically_visible) and bool(point.strong)
            and str(point.state) == "observed")


def _raw_separate(raw: np.ndarray, a, b) -> bool:
    distance = float(np.hypot(float(a.x)-float(b.x),float(a.y)-float(b.y)))
    if distance < 4.0 or not body._distinct(a,b,body.Params()):
        return False
    valley = physical._valley_ratio(raw,(float(a.x),float(a.y)),
                                    (float(b.x),float(b.y)))
    return bool(np.isfinite(valley) and valley<=body.Params().maximum_raw_valley_ratio)


def _at_unclaimed_component(frame: np.ndarray, point) -> np.ndarray | None:
    y,x=round(float(point.y)),round(float(point.x))
    if not (0<=y<frame.shape[0] and 0<=x<frame.shape[1]):
        return None
    original_id=int(frame[y,x])
    if original_id<=0:
        return None
    components,_=ndi.label(frame==original_id,body.STRUCTURE)
    component=int(components[y,x])
    return components==component if component>0 else None


def _gap_contrast(raw: np.ndarray, point) -> float:
    y,x=round(float(point.y)),round(float(point.x))
    local=raw[max(0,y-12):min(raw.shape[0],y+13),
              max(0,x-12):min(raw.shape[1],x+13)]
    return float(raw[y,x])/max(float(np.quantile(local,.75)),1.0)


def _discover(labels: np.ndarray,raw: np.ndarray,scored: pd.DataFrame,
              params: Params) -> list[dict]:
    anchors=dual._references(scored,dual.Params())
    visible=scored[scored.physically_visible.astype(bool)]
    groups={int(track):g.sort_values("frame") for track,g in visible.groupby("track_id")}
    by_name: dict[int,list[int]]={}
    for track,anchor in anchors.items():
        if int(anchor["support_frames"])>=params.minimum_anchor_strong_frames:
            by_name.setdefault(int(anchor["owner"]),[]).append(int(track))
    candidates=[]
    for name,tracks in by_name.items():
        for a_id,b_id in itertools.combinations(sorted(tracks),2):
            a,b=groups[a_id],groups[b_id]
            a_radius=float(a.radius_px.median())
            b_radius=float(b.radius_px.median())
            if a_radius>=b_radius:
                large_id,small_id,large_radius,small_radius=a_id,b_id,a_radius,b_radius
            else:
                large_id,small_id,large_radius,small_radius=b_id,a_id,b_radius,a_radius
            small=groups[small_id]
            if (large_radius<params.minimum_large_radius_px
                    or large_radius/max(small_radius,1.0)<params.minimum_radius_ratio
                    or len(small)<params.minimum_satellite_visible_frames):
                continue
            large_strong={int(p.frame):p for p in groups[large_id].itertuples(index=False)
                          if _strong_observed(p)}
            small_strong={int(p.frame):p for p in small.itertuples(index=False)
                          if _strong_observed(p)}
            qualifying=[]
            areas=[]
            for frame in sorted(large_strong.keys() & small_strong.keys()):
                large_point,small_point=large_strong[frame],small_strong[frame]
                if (int(large_point.accepted_owner)!=name
                        or int(small_point.accepted_owner)!=name):
                    continue
                component=body._at_component(labels[frame],name,large_point)
                if (component is None or not body._inside(component,small_point)
                        or not _raw_separate(raw[frame],large_point,small_point)):
                    continue
                qualifying.append(frame)
                areas.append(int(component.sum()))
            if (len(qualifying)<params.minimum_shared_raw_frames
                    or float(np.median(areas))>params.maximum_median_shared_component_px):
                continue
            candidates.append({"large_track":large_id,"satellite_track":small_id,
                               "shared_name":name,"shared_raw_frames":len(qualifying),
                               "median_shared_component_px":float(np.median(areas)),
                               "large_radius_px":large_radius,
                               "satellite_radius_px":small_radius})
    return sorted(candidates,key=lambda c:(-c["shared_raw_frames"],
                                            c["satellite_track"]))


def _propose_path(labels: np.ndarray,unclaimed: np.ndarray,raw: np.ndarray,
                  by_frame: dict[int,list],by_track: dict[int,list],
                  protected: np.ndarray,candidate: dict,params: Params
                  ) -> tuple[list[dict],str]:
    track=candidate["satellite_track"]
    path=by_track[track]
    proposals=[]
    gap_original_ids=set()
    for target in path:
        frame=int(target.frame)
        owner=int(target.accepted_owner)
        if owner<=0:
            part=_at_unclaimed_component(unclaimed[frame],target)
            if part is None:
                return [],"ownerless_gap_without_original_mask"
            gap_original_ids.add(int(unclaimed[frame,round(float(target.y)),
                                               round(float(target.x))]))
            if (np.any(labels[frame][part]!=0)
                    or np.any(protected[frame]&part)
                    or _gap_contrast(raw[frame],target)<params.minimum_raw_gap_contrast_ratio):
                return [],"ownerless_gap_not_independently_supported"
            if any(int(p.track_id)!=track and _strong_observed(p)
                   and body._inside(part,p) for p in by_frame[frame]):
                return [],"ownerless_gap_contains_another_soma"
            method="original_unclaimed_mask"
        else:
            component=body._at_component(labels[frame],owner,target)
            if component is None:
                return [],"visible_soma_without_component"
            companions=[p for p in by_frame[frame]
                        if int(p.track_id)!=track and _strong_observed(p)
                        and body._inside(component,p)]
            if any(not _raw_separate(raw[frame],target,p) for p in companions):
                return [],"component_has_inseparable_strong_soma"
            if not companions:
                part=component
                method="sole_soma"
            elif len(companions)==1:
                part=body._partition(component,raw[frame],target,
                                     companions[0],body.Params())
                method="two_soma_partition"
            else:
                part=body._partition_many(component,raw[frame],target,
                                          companions,body.Params())
                method="multi_soma_partition"
            if part is None:
                return [],"raw_partition_failed"
            if np.any(part & protected[frame]):
                return [],"protected_accepted_pixel"
        if not body._inside(part,target):
            nearest=physical._marker_pixel(part,float(target.x),float(target.y))
            if (nearest is None or float(np.hypot(
                    nearest[0]-float(target.y),nearest[1]-float(target.x)))
                    >params.maximum_soma_to_basin_distance_px):
                return [],"proposed_basin_misses_soma"
        proposals.append({"frame":frame,"part":part,"owner":owner,
                          "method":method,"area_px":int(part.sum()),
                          "point":target})
    if len(gap_original_ids)>1:
        return [],"ownerless_gaps_do_not_share_original_object"
    reference=float(np.median([p["area_px"] for p in proposals]))
    if reference<=0 or any(not params.minimum_area_ratio<=p["area_px"]/reference
                           <=params.maximum_area_ratio for p in proposals):
        return [],"path_basin_area_inconsistent"
    return proposals,"supported"


def correct(labels: np.ndarray,unclaimed: np.ndarray,raw: np.ndarray,
            points: pd.DataFrame,protected: np.ndarray,params: Params=Params()
            ) -> tuple[np.ndarray,np.ndarray,pd.DataFrame,pd.DataFrame]:
    if not (labels.shape==unclaimed.shape==raw.shape==protected.shape):
        raise ValueError("labels, unclaimed, raw and protected must align")
    scored=physical.attach_owners(points,labels)
    for field in ("physically_visible","strong"):
        scored[field]=scored[field].astype(bool)
    by_frame={int(frame):list(g.itertuples(index=False))
              for frame,g in scored.groupby("frame")}
    by_track={int(track):list(g[g.physically_visible].sort_values("frame")
                              .itertuples(index=False))
              for track,g in scored.groupby("track_id")}
    candidate_labels=labels.copy()
    candidate_unclaimed=unclaimed.copy()
    occupied=np.zeros_like(labels,bool)
    audits,applications=[],[]
    claimed_tracks=set()
    for candidate in _discover(labels,raw,scored,params):
        track=candidate["satellite_track"]
        if track in claimed_tracks:
            audits.append({**candidate,"outcome":"rejected",
                           "reason":"track_already_claimed"})
            continue
        proposals,reason=_propose_path(labels,unclaimed,raw,by_frame,
                                       by_track,protected,candidate,params)
        if reason=="supported" and any(occupied[p["frame"]][p["part"]].any()
                                       for p in proposals):
            reason="overlaps_another_new_identity"
        if reason!="supported":
            audits.append({**candidate,"outcome":"rejected","reason":reason})
            continue
        new_name=int(candidate_labels.max())+1
        if new_name>np.iinfo(candidate_labels.dtype).max:
            raise OverflowError("no label value remains for a new cell")
        for p in proposals:
            frame=p["frame"]
            candidate_labels[frame][p["part"]]=new_name
            occupied[frame][p["part"]]=True
            if p["owner"]==0:
                candidate_unclaimed[frame][p["part"]]=0
            applications.append({"large_track":candidate["large_track"],
                                 "satellite_track":track,"frame":frame+1,
                                 "old_name":p["owner"],"new_name":new_name,
                                 "area_px":p["area_px"],"method":p["method"]})
        claimed_tracks.add(track)
        audits.append({**candidate,"outcome":"applied","reason":"supported",
                       "new_name":new_name,"path_frames":len(proposals),
                       "gap_frames":sum(p["owner"]==0 for p in proposals),
                       "changed_pixels":sum(p["area_px"] for p in proposals)})
    return (candidate_labels,candidate_unclaimed,
            pd.DataFrame(audits),pd.DataFrame(applications))
