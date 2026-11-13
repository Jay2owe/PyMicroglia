"""Fill one-frame identity gaps using adjacent corrected soma components."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

STRUCTURE = np.ones((3,3),bool)


def _component(frame: np.ndarray, name: int, y: int, x: int) -> np.ndarray | None:
    parts,_ = ndi.label(frame==name,STRUCTURE)
    return parts==parts[y,x] if parts[y,x] else None


def correct(labels: np.ndarray, points: pd.DataFrame,
            prior_applications: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray,pd.DataFrame,pd.DataFrame]:
    candidate = labels.copy()
    audit, applications = [], []
    if prior_applications.empty:
        return candidate,pd.DataFrame(audit),pd.DataFrame(applications)
    strong = points[(points.state=="observed") & points.strong.astype(bool)].copy()
    indexed = strong.set_index(["track_id","frame"])
    by_frame = {int(t): group for t,group in strong.groupby("frame")}
    for track, group in prior_applications.groupby("track_id"):
        group = group.sort_values("frame")
        frames = group.frame.to_numpy(dtype=int)
        if group.old_name.nunique()!=1 or group.restored_name.nunique()!=1:
            continue
        old_name = int(group.old_name.iloc[0])
        restored = int(group.restored_name.iloc[0])
        for left,right in zip(frames[:-1],frames[1:]):
            if right-left!=2:
                continue
            t = int(left)  # 1-based left + 1, converted to zero-based.
            keys = [(int(track),t-1),(int(track),t),(int(track),t+1)]
            if any(key not in indexed.index for key in keys):
                continue
            left_pt,middle_pt,right_pt = (indexed.loc[key] for key in keys)
            if any(isinstance(pt,pd.DataFrame) for pt in (left_pt,middle_pt,right_pt)):
                continue
            ly,lx=round(float(left_pt.y)),round(float(left_pt.x))
            my,mx=round(float(middle_pt.y)),round(float(middle_pt.x))
            ry,rx=round(float(right_pt.y)),round(float(right_pt.x))
            reason="eligible"
            if int(labels[t,my,mx])!=0:
                reason="gap_soma_not_background"
            before = _component(labels[t-1],restored,ly,lx)
            after = _component(labels[t+1],restored,ry,rx)
            if before is None or after is None:
                reason="flanking_restored_body_missing"
            parts,n=ndi.label(labels[t]==old_name,STRUCTURE)
            candidates=[]
            radius=max(1.,.5*float(middle_pt.radius_px))
            for idx in range(1,n+1):
                mask=parts==idx
                ys,xs=np.where(mask)
                distance=float(np.hypot(ys-my,xs-mx).min())
                if distance<=radius:
                    candidates.append((mask,distance))
            if len(candidates)!=1:
                reason="no_unique_nearby_old_name_body"
            overlap_before=overlap_after=np.nan
            area=0
            if reason=="eligible":
                mask,distance=candidates[0]
                area=int(mask.sum())
                overlap_before=float(np.count_nonzero(mask&before))/max(min(area,int(before.sum())),1)
                overlap_after=float(np.count_nonzero(mask&after))/max(min(area,int(after.sum())),1)
                median_flank=.5*(int(before.sum())+int(after.sum()))
                if (overlap_before<.6 or overlap_after<.6
                        or not .5*median_flank<=area<=2.*median_flank):
                    reason="weak_temporal_body_overlap_or_area"
                elif np.any(mask&protected[t]):
                    reason="previously_accepted_pixel"
                elif any(mask[round(float(pt.y)),round(float(pt.x))]
                         for pt in by_frame[t].itertuples(index=False)
                         if int(pt.track_id)!=int(track)):
                    reason="other_strong_soma_in_body"
            audit.append({"track_id":int(track),"frame":t+1,
                          "old_name":old_name,"restored_name":restored,
                          "nearby_components":len(candidates),"candidate_pixels":area,
                          "overlap_before":overlap_before,"overlap_after":overlap_after,
                          "outcome":"applied" if reason=="eligible" else "rejected",
                          "reason":reason})
            if reason!="eligible":
                continue
            if np.any(candidate[t][mask]!=old_name):
                raise AssertionError("overlapping temporal bridge")
            candidate[t][mask]=restored
            applications.append({"track_id":int(track),"frame":t+1,
                                 "old_name":old_name,"restored_name":restored,
                                 "changed_pixels":area})
    return candidate,pd.DataFrame(audit),pd.DataFrame(applications)
