"""Restore a whole short A-B...B-A owner episode on one physical soma.

Discovery is fieldwide and feedback-free. Accepted masks, raw frames, strong
soma tracks and protected accepted pixels are the only decision inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as equal_value_components
from skimage.segmentation import watershed

import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_episode_frames: int = 2
    maximum_episode_frames: int = 5
    minimum_source_area: int = 100
    maximum_step_radii: float = 1.
    minimum_raw_peak_retention: float = .75
    minimum_mask_overlap: float = .70
    maximum_cross_soma_valley_ratio: float = .5
    minimum_target_basin_area: int = 25
    minimum_other_basin_area: int = 8
    watershed_sigma_px: float = 1.


def _runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for point in group.itertuples(index=False):
        t, owner = int(point.frame), int(point.owner)
        if runs and runs[-1]["name"] == owner and runs[-1]["last"]+1 == t:
            runs[-1]["last"] = t
            runs[-1]["points"].append(point)
        else:
            runs.append({"name": owner, "first": t, "last": t,
                         "points": [point]})
    return runs


def _peak(raw_frame: np.ndarray, point) -> float:
    y, x = round(float(point.y)), round(float(point.x))
    return float(np.max(raw_frame[max(0,y-2):min(raw_frame.shape[0],y+3),
                                  max(0,x-2):min(raw_frame.shape[1],x+3)]))


def _component(comps: np.ndarray, point) -> np.ndarray | None:
    y, x = round(float(point.y)), round(float(point.x))
    if not (0 <= y < comps.shape[0] and 0 <= x < comps.shape[1]):
        return None
    cid = int(comps[y, x])
    return comps == cid if cid > 0 else None


def _watershed_target(mask: np.ndarray, raw_frame: np.ndarray,
                      target, residents: list, params: Params
                      ) -> tuple[np.ndarray | None, int, str]:
    others = [point for point in residents
              if int(point.track_id) != int(target.track_id)]
    if not others:
        return mask, 0, "sole_soma"
    for point in others:
        valley = physical._valley_ratio(
            raw_frame, (float(target.x), float(target.y)),
            (float(point.x), float(point.y)))
        if (not np.isfinite(valley)
                or valley > params.maximum_cross_soma_valley_ratio):
            return None, 0, "not_raw_separable"
    yy, xx = np.where(mask)
    box = np.s_[max(0,int(yy.min())-2):min(mask.shape[0],int(yy.max())+3),
                max(0,int(xx.min())-2):min(mask.shape[1],int(xx.max())+3)]
    local = mask[box]
    markers = np.zeros(local.shape, np.uint16)
    target_marker = 0
    for index, point in enumerate(residents, 1):
        seed = physical._marker_pixel(local, float(point.x)-box[1].start,
                                      float(point.y)-box[0].start)
        if seed is None or markers[seed]:
            return None, 0, "nonunique_raw_seed"
        markers[seed] = index
        if int(point.track_id) == int(target.track_id):
            target_marker = index
    if not target_marker:
        raise AssertionError("target soma missing from its component")
    assigned = watershed(
        -ndi.gaussian_filter(raw_frame[box].astype(np.float32),
                             params.watershed_sigma_px),
        markers, mask=local, connectivity=np.ones((3,3), bool))
    basin = assigned == target_marker
    area = int(basin.sum())
    if area < params.minimum_target_basin_area:
        return None, 0, "target_basin_too_small"
    other_areas = [int((assigned == index).sum())
                   for index, point in enumerate(residents, 1)
                   if int(point.track_id) != int(target.track_id)]
    if min(other_areas) < params.minimum_other_basin_area:
        return None, 0, "other_soma_basin_too_small"
    whole = np.zeros(mask.shape, bool)
    whole[box] = basin
    return whole, min(other_areas), "raw_watershed"


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or labels.shape != protected.shape
            or raw.shape[:3] != labels.shape):
        raise ValueError("labels, raw and protected pixels must align")
    strong = points[(points.state.astype(str) == "observed")
                    & points.strong.astype(bool)].copy()
    coords = np.rint(strong[["y", "x"]].to_numpy(float)).astype(int)
    if np.any(coords[:,0]<0) or np.any(coords[:,0]>=labels.shape[1]) or np.any(coords[:,1]<0) or np.any(coords[:,1]>=labels.shape[2]):
        raise ValueError("strong soma point lies outside label field")
    strong["owner"] = labels[strong.frame.to_numpy(int),coords[:,0],coords[:,1]]
    by_frame = {int(t): group for t,group in strong.groupby("frame")}
    components: dict[int,np.ndarray] = {}
    proposals, audits = [], []
    for track_id, group in strong.groupby("track_id"):
        runs = _runs(group.sort_values("frame"))
        for previous, episode, following in zip(runs[:-2],runs[1:-1],runs[2:]):
            if (not params.minimum_episode_frames <= len(episode["points"]) <= params.maximum_episode_frames
                    or previous["name"] <= 0 or episode["name"] <= 0
                    or previous["name"] != following["name"]
                    or previous["name"] == episode["name"]
                    or previous["last"]+1 != episode["first"]
                    or episode["last"]+1 != following["first"]):
                continue
            target_name = int(previous["name"])
            source_name = int(episode["name"])
            all_points = [previous["points"][-1],*episode["points"],following["points"][0]]
            base = {"track_id":int(track_id),"first_frame":episode["first"]+1,
                    "last_frame":episode["last"]+1,
                    "episode_frames":len(episode["points"]),
                    "old_name":source_name,"new_name":target_name}
            positions = np.asarray([(float(p.y),float(p.x)) for p in all_points])
            radii = np.asarray([max(float(p.radius_px),1.) for p in all_points])
            steps = np.linalg.norm(np.diff(positions,axis=0),axis=1)
            step_radii = steps/np.maximum((radii[:-1]+radii[1:])/2,1.)
            if np.max(step_radii) > params.maximum_step_radii:
                audits.append({**base,"outcome":"rejected","reason":"soma_step_too_large",
                               "changed_pixels":0})
                continue
            peaks = [_peak(raw[int(p.frame)],p) for p in all_points]
            retention = min(peaks[1:-1])/max(float(np.median((peaks[0],peaks[-1]))),1.)
            if not np.isfinite(retention) or retention < params.minimum_raw_peak_retention:
                audits.append({**base,"outcome":"rejected","reason":"raw_peak_not_retained",
                               "changed_pixels":0})
                continue
            masks = []
            reason = ""
            for point in all_points:
                t = int(point.frame)
                if t not in components:
                    components[t] = equal_value_components(
                        labels[t],connectivity=2,background=0)
                mask = _component(components[t],point)
                if mask is None:
                    reason = "soma_without_outline"
                    break
                masks.append(mask)
            if not reason:
                overlap = [float(np.sum(a&b))/max(1,min(int(a.sum()),int(b.sum())))
                           for a,b in zip(masks[:-1],masks[1:])]
                if min(overlap) < params.minimum_mask_overlap:
                    reason = "physical_outline_not_continuous"
            if reason:
                audits.append({**base,"outcome":"rejected","reason":reason,
                               "changed_pixels":0})
                continue
            frame_parts=[]
            for point,mask in zip(episode["points"],masks[1:-1]):
                t=int(point.frame)
                if int(mask.sum()) < params.minimum_source_area:
                    reason="source_outline_too_small"
                    break
                if np.any(mask&protected[t]):
                    reason="accepted_pixel_in_source_outline"
                    break
                residents=[p for p in by_frame[t].itertuples(index=False)
                           if mask[round(float(p.y)),round(float(p.x))]]
                if any(int(p.owner)==target_name and not mask[round(float(p.y)),round(float(p.x))]
                       for p in by_frame[t].itertuples(index=False)):
                    reason="other_strong_holder_of_return_name"
                    break
                selected,other_area,mode=_watershed_target(mask,raw[t],point,residents,params)
                if selected is None:
                    reason=mode
                    break
                yy,xx=np.where(selected)
                if not len(yy):
                    reason="empty_target_basin"
                    break
                if not np.all(labels[t,yy,xx]==source_name):
                    reason="source_name_not_uniform"
                    break
                frame_parts.append({"frame":t,"yy":yy,"xx":xx,
                                    "source_area":int(mask.sum()),
                                    "changed_pixels":len(yy),
                                    "other_basin_min":other_area,"mode":mode})
            if reason or len(frame_parts)!=len(episode["points"]):
                audits.append({**base,"outcome":"rejected",
                               "reason":reason or "incomplete_episode","changed_pixels":0})
                continue
            total=sum(part["changed_pixels"] for part in frame_parts)
            first=frame_parts[0]
            proposals.append({"base":base,"parts":frame_parts,"total":total,
                              "order":(-total,episode["first"],int(first["yy"].min()),
                                       int(first["xx"].min()))})
    candidate=labels.copy()
    remaining=np.bincount(labels.ravel().astype(np.int64))
    applications=[]
    for proposal in sorted(proposals,key=lambda item:item["order"]):
        base,parts,total=proposal["base"],proposal["parts"],proposal["total"]
        old=int(base["old_name"])
        if remaining[old]<=total:
            reason="would_remove_established_name"
        elif any(np.any(candidate[part["frame"],part["yy"],part["xx"]]!=old)
                 for part in parts):
            reason="overlaps_prior_episode"
        else:
            reason=""
        if reason:
            audits.append({**base,"outcome":"rejected","reason":reason,
                           "changed_pixels":0})
            continue
        for part in parts:
            t=part["frame"]
            candidate[t,part["yy"],part["xx"]]=int(base["new_name"])
            applications.append({**base,"frame":t+1,"changed_pixels":part["changed_pixels"],
                                 "source_area":part["source_area"],
                                 "other_basin_min":part["other_basin_min"],
                                 "mode":part["mode"]})
        remaining[old]-=total
        audits.append({**base,"outcome":"applied","reason":"complete_supported_episode",
                       "changed_pixels":total})
    if (np.any((candidate!=labels)&protected)
            or not np.array_equal(candidate>0,labels>0)
            or not np.array_equal(np.unique(candidate),np.unique(labels))):
        raise AssertionError("short owner excursion crossed accepted boundary")
    return candidate,pd.DataFrame(audits),pd.DataFrame(applications)
