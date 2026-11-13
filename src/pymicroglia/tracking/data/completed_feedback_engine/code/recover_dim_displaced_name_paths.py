"""Restore a long physical name path when its old name survives on dim remnants.

Discovery uses tracks, current masks and raw signal only. It has no feedback,
recording, cell identity or region selector. This is an experimental layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import separable_merge_recovery as physical

STRUCTURE=np.ones((3,3),bool)


@dataclass(frozen=True)
class Params:
    minimum_early_strong_frames:int=10
    minimum_later_strong_frames:int=30
    minimum_residual_frames:int=20
    maximum_residual_area_fraction:float=.5
    maximum_residual_brightness_fraction:float=.5


def _runs(group:pd.DataFrame)->list[tuple[int,int,int]]:
    runs=[]
    for p in group.sort_values("frame").itertuples(index=False):
        t,name=int(p.frame),int(p.accepted_owner)
        if name<=0:
            return []
        if runs and runs[-1][0]==name and runs[-1][2]+1==t:
            runs[-1]=(name,runs[-1][1],t)
        else:
            runs.append((name,t,t))
    return runs


def _component(frame:np.ndarray,name:int,point)->np.ndarray|None:
    y,x=round(float(point.y)),round(float(point.x))
    if not(0<=y<frame.shape[0] and 0<=x<frame.shape[1]) or int(frame[y,x])!=name:
        return None
    parts,_=ndi.label(frame==name,STRUCTURE)
    return parts==parts[y,x]


def _strong_seats(mask:np.ndarray,frame_points:pd.DataFrame)->int:
    count=0
    for p in frame_points.itertuples(index=False):
        y,x=round(float(p.y)),round(float(p.x))
        if 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]:
            count+=1
    return count


def correct(labels:np.ndarray,raw:np.ndarray,points:pd.DataFrame,
            params:Params=Params())->tuple[np.ndarray,pd.DataFrame,pd.DataFrame]:
    if labels.shape!=raw.shape or labels.ndim!=3:
        raise ValueError("label and raw movies must align")
    scored=physical.attach_owners(points,labels)
    strong=scored[(scored.state.astype(str)=="observed")&scored.strong.astype(bool)]
    visible=scored[scored.state.astype(str).isin(["observed","latent_visible"])]
    by_frame={int(t):g for t,g in strong.groupby("frame")}
    empty=strong.iloc[0:0]
    candidate=labels.copy()
    audits=[]
    applications=[]
    claimed_earlier_names=set()
    for track,group in strong.groupby("track_id",sort=True):
        runs=_runs(group)
        if len(runs)!=2:
            continue
        early,later=runs
        early_name,early_first,early_last=early
        later_name,later_first,later_last=later
        if (early_name==later_name or early_last+1!=later_first or
            early_last-early_first+1<params.minimum_early_strong_frames or
            later_last-later_first+1<params.minimum_later_strong_frames):
            continue
        audit={"track_id":int(track),"earlier_name":early_name,"later_name":later_name,
               "earlier_first_frame":early_first+1,"earlier_last_frame":early_last+1,
               "later_first_frame":later_first+1,"later_last_frame":later_last+1,
               "later_frames":later_last-later_first+1}
        if early_name in claimed_earlier_names:
            audits.append({**audit,"outcome":"rejected","reason":"earlier_name_already_claimed"})
            continue
        path=visible[(visible.track_id.astype(int)==int(track))&
                     visible.frame.astype(int).between(later_first,later_last)]
        if len(path)!=audit["later_frames"] or not path.strong.astype(bool).all():
            audits.append({**audit,"outcome":"rejected","reason":"later_visible_path_has_weak_or_missing_frame"})
            continue
        proposals=[]
        residual_frames=0
        reason="supported"
        for point in path.sort_values("frame").itertuples(index=False):
            t=int(point.frame)
            target=_component(labels[t],later_name,point)
            if target is None:
                reason="target_soma_outside_named_component"
                break
            if _strong_seats(target,by_frame.get(t,empty))!=1:
                reason="target_component_has_other_strong_soma"
                break
            target_area=int(target.sum())
            target_q95=float(np.percentile(raw[t][target],95))
            residual=labels[t]==early_name
            if residual.any():
                residual_frames+=1
                parts,n=ndi.label(residual,STRUCTURE)
                for i in range(1,n+1):
                    holder=parts==i
                    if (int(holder.sum())>params.maximum_residual_area_fraction*target_area or
                        float(np.percentile(raw[t][holder],95))>
                        params.maximum_residual_brightness_fraction*target_q95):
                        reason="old_name_has_independent_large_or_bright_holder"
                        break
            if reason!="supported":
                break
            proposals.append((t,target,residual,target_area,int(residual.sum())))
        if reason=="supported" and residual_frames<params.minimum_residual_frames:
            reason="old_name_residual_not_recurrent"
        if reason!="supported":
            audits.append({**audit,"outcome":"rejected","reason":reason,
                           "proposed_frames":len(proposals),"residual_frames":residual_frames})
            continue
        residual_name=int(candidate.max())+1
        if residual_name>np.iinfo(candidate.dtype).max:
            raise OverflowError("no label value remains for residual identity")
        for t,target,residual,target_area,residual_area in proposals:
            candidate[t][residual]=residual_name
            candidate[t][target]=early_name
            applications.append({"track_id":int(track),"frame":t+1,
                                 "target_old_name":later_name,"target_new_name":early_name,
                                 "residual_old_name":early_name,
                                 "residual_new_name":residual_name,
                                 "target_pixels":target_area,
                                 "residual_pixels":residual_area,
                                 "changed_pixels":target_area+residual_area})
        claimed_earlier_names.add(early_name)
        audits.append({**audit,"outcome":"applied","reason":"supported",
                       "residual_name":residual_name,"residual_frames":residual_frames,
                       "changed_pixels":sum(p[3]+p[4] for p in proposals)})
    if not np.array_equal(candidate>0,labels>0):
        raise AssertionError("dim-residual handoff changed foreground")
    return candidate,pd.DataFrame(audits),pd.DataFrame(applications)
