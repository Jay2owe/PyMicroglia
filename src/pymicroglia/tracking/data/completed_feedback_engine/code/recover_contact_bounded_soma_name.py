"""Restore a short first-name anchor through a long clean run and contacts.

Only geometry, raw signal and physical tracker continuity decide eligibility.
Feedback, movie names, identity numbers and regions are not inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

STRUCTURE=np.ones((3,3),bool)


def _runs(frames:np.ndarray,names:np.ndarray)->list[dict]:
    runs=[]
    for t,name in zip(frames,names):
        t,name=int(t),int(name)
        if runs and runs[-1]["name"]==name and runs[-1]["last"]+1==t:
            runs[-1]["last"]=t
            runs[-1]["length"]+=1
        else:
            runs.append({"name":name,"first":t,"last":t,"length":1})
    return runs


def _source(frame:np.ndarray,name:int,y:int,x:int)->np.ndarray|None:
    parts,_=ndi.label(frame==name,STRUCTURE)
    return parts==parts[y,x] if parts[y,x] else None


def _raw_target_basin(source:np.ndarray,image:np.ndarray,target,
                      others:list)->tuple[np.ndarray|None,str,float]:
    y,x=round(float(target.y)),round(float(target.x))
    signal=ndi.gaussian_filter(image.astype(np.float32),.8)
    ys,xs=np.where(source)
    y0,y1=int(ys.min()),int(ys.max())+1
    x0,x1=int(xs.min()),int(xs.max())+1
    markers=np.zeros((y1-y0,x1-x0),np.int32)
    markers[y-y0,x-x0]=1
    valley_ratios=[]
    for index,other in enumerate(others,2):
        oy,ox=round(float(other.y)),round(float(other.x))
        if markers[oy-y0,ox-x0]:
            return None,"overlapping_soma_seeds",np.nan
        markers[oy-y0,ox-x0]=index
        distance=float(np.hypot(y-oy,x-ox))
        if distance/max(float(target.radius_px),float(other.radius_px),1.)<1.5:
            return None,"somata_not_separable_by_radius",np.nan
        count=max(2,int(np.ceil(distance*4)))
        line_y=np.rint(np.linspace(y,oy,count)).astype(int)
        line_x=np.rint(np.linspace(x,ox,count)).astype(int)
        valley=float(signal[line_y,line_x].min())
        weaker=min(float(signal[y,x]),float(signal[oy,ox]))
        valley_ratios.append(valley/max(weaker,1.))
    worst=max(valley_ratios)
    if worst>=.6:
        return None,"weak_raw_valley",worst
    local_source=source[y0:y1,x0:x1]
    basins=watershed(-signal[y0:y1,x0:x1],markers,mask=local_source,
                     connectivity=STRUCTURE)
    if int(np.count_nonzero(basins))!=int(source.sum()):
        return None,"watershed_does_not_cover_source",worst
    for index in range(1,len(others)+2):
        basin=basins==index
        if int(basin.sum())<8 or ndi.label(basin,STRUCTURE)[1]!=1:
            return None,"disconnected_or_tiny_raw_basin",worst
    target_mask=np.zeros_like(source)
    target_mask[y0:y1,x0:x1]=basins==1
    return target_mask,"eligible",worst


def correct(labels:np.ndarray,raw:np.ndarray,points:pd.DataFrame,
            protected:np.ndarray)->tuple[np.ndarray,pd.DataFrame,pd.DataFrame]:
    if labels.shape!=raw.shape or labels.shape!=protected.shape or labels.ndim!=3:
        raise ValueError("aligned complete labels, raw images and protection required")
    candidate=labels.copy()
    strong=points[(points.state=="observed") & points.strong.astype(bool)].copy()
    coords=np.rint(strong[["y","x"]].to_numpy()).astype(int)
    strong["owner"]=labels[strong.frame.to_numpy(dtype=int),coords[:,0],coords[:,1]]
    by_frame={int(t):group for t,group in strong.groupby("frame")}
    proposals=[]
    audit=[]
    for track,group in strong.groupby("track_id",sort=True):
        group=group.sort_values("frame")
        frames=group.frame.to_numpy(dtype=int)
        names=group.owner.to_numpy(dtype=int)
        if (len(group)<30 or np.any(np.diff(frames)!=1) or np.any(names<=0)):
            continue
        runs=_runs(frames,names)
        first=runs[0]
        if first["length"]<5 or len(runs)<3:
            continue
        mains=[i for i,run in enumerate(runs[1:],1)
               if run["length"]>=20 and run["name"]!=first["name"]]
        if len(mains)!=1:
            continue
        main_idx=mains[0]
        main=runs[main_idx]
        later=runs[1:]
        contacts=[run for i,run in enumerate(runs[1:],1) if i!=main_idx]
        if (not contacts or any(run["length"]>5 for run in contacts)
                or sum(run["length"] for run in contacts)>10
                or any(run["name"]==first["name"] for run in later)
                or sum(run["length"] for run in later)<25):
            continue
        records=[]
        reason="eligible"
        main_areas=[]
        contact_count=0
        for pt in group.itertuples(index=False):
            t=int(pt.frame)
            if t<=first["last"]:
                continue
            y,x=round(float(pt.y)),round(float(pt.x))
            name=int(pt.owner)
            source=_source(labels[t],name,y,x)
            if source is None:
                reason="source_component_missing"
                break
            others=[]
            for other in by_frame[t].itertuples(index=False):
                if int(other.track_id)==int(track):
                    continue
                oy,ox=round(float(other.y)),round(float(other.x))
                if source[oy,ox]:
                    others.append(other)
            if main["first"]<=t<=main["last"]:
                if name!=main["name"] or others:
                    reason="main_run_not_single_soma"
                    break
                target=source
                method="whole_main_outline"
                valley=np.nan
                main_areas.append(int(target.sum()))
            else:
                if not others:
                    reason="contact_without_other_soma"
                    break
                target,reason,valley=_raw_target_basin(source,raw[t],pt,others)
                if target is None:
                    break
                method="multi_soma_raw_basin"
                contact_count+=1
            if np.any(target&protected[t]):
                reason="previously_accepted_pixel"
                break
            records.append({"frame":t,"source_name":name,"restored_name":first["name"],
                            "target":target,"source_pixels":int(source.sum()),
                            "changed_pixels":int(target.sum()),"method":method,
                            "other_strong_somata":len(others),"valley_ratio":valley})
        if reason=="eligible" and len(records)!=len(group)-first["length"]:
            reason="incomplete_later_path"
        if reason=="eligible" and contact_count!=sum(run["length"] for run in contacts):
            reason="incomplete_contact_partition"
        if reason=="eligible":
            middle=float(np.median(main_areas))
            if any(not .25*middle<=r["changed_pixels"]<=2*middle
                   for r in records if r["method"]=="multi_soma_raw_basin"):
                reason="contact_basin_size_disagrees_with_main_soma"
        audit.append({"track_id":int(track),"first_name":first["name"],
                      "main_name":main["name"],"first_run_frames":first["length"],
                      "main_run_frames":main["length"],"contact_frames":contact_count,
                      "target_pixels":sum(r["changed_pixels"] for r in records),
                      "outcome":"proposed" if reason=="eligible" else "rejected",
                      "reason":reason})
        if reason=="eligible":
            proposals.append({"track_id":int(track),"records":records})
    conflicting=set()
    for i,left in enumerate(proposals):
        for j in range(i+1,len(proposals)):
            right=proposals[j]
            left_by_frame={r["frame"]:r["target"] for r in left["records"]}
            if any(t in left_by_frame and np.any(left_by_frame[t]&r["target"])
                   for r in right["records"] for t in [r["frame"]]):
                conflicting.update((i,j))
    applications=[]
    for index,proposal in enumerate(proposals):
        if index in conflicting:
            for row in audit:
                if row["track_id"]==proposal["track_id"]:
                    row.update(outcome="rejected",reason="overlapping_eligible_paths")
            continue
        for record in proposal["records"]:
            t=record["frame"]
            target=record["target"]
            if np.any(candidate[t][target]!=record["source_name"]):
                raise AssertionError("source labels changed during proposal application")
            candidate[t][target]=record["restored_name"]
            applications.append({"track_id":proposal["track_id"],"frame":t+1,
                                 "source_name":record["source_name"],
                                 "restored_name":record["restored_name"],
                                 "source_pixels":record["source_pixels"],
                                 "changed_pixels":record["changed_pixels"],
                                 "method":record["method"],
                                 "other_strong_somata":record["other_strong_somata"],
                                 "valley_ratio":record["valley_ratio"]})
    for row in audit:
        if row["outcome"]=="proposed":
            row["outcome"]="applied"
    return candidate,pd.DataFrame(audit),pd.DataFrame(applications)
