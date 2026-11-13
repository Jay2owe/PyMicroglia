"""Carry established ownership through an overlapping detector restart.

The old detector has substantial stable evidence before the new detector exists.
Both observe one component at the handoff, then the old detector ends and the
new one receives another name. Independent earlier somata remain competitors.
No feedback, recording, identity, frame or region targets enter discovery.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import resident_takeover as owner
import anchored_soma_owner_recovery_a002_snapshot as established
from shared_parent_soma_ownership_a002 import _core_intersects

STRUCTURE=np.ones((3,3),bool)



def _shared_body_basin(raw,mask,members,others,sigma):
    if not others:return mask
    smooth=ndi.gaussian_filter(raw.astype(np.float32),sigma)
    markers=np.zeros(mask.shape,np.int16)
    for point in members:
        marker=established.soma._peak_marker(smooth,mask,point,others)
        if marker is not None:markers[marker]=1
    if not np.any(markers==1):return None
    for i,other in enumerate(others,2):
        marker=established.soma._peak_marker(smooth,mask,other)
        if marker is None or markers[marker]:return None
        markers[marker]=i
    return watershed(-smooth,markers=markers,mask=mask)==1


def correct(labels,raw,points,protected):
    if labels.shape!=raw.shape or labels.shape!=protected.shape:
        raise ValueError('Aligned complete movies required')
    scored=owner.attach_owners(points,labels)
    strong=scored[scored.strong.astype(bool)&scored.state.eq('observed')]
    visible=scored[scored.physically_visible.astype(bool)]
    groups={int(tr):g.sort_values('frame').set_index('frame',drop=False) for tr,g in scored.groupby('track_id')}
    byframe={int(t):list(g.itertuples(index=False)) for t,g in visible.groupby('frame')}
    minimum=established._minimum(len(labels),established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    window=established._minimum(len(labels),established.DEFAULTS['minimum_source_anchor_movie_fraction'])
    sigma=max(.5,float(strong.radius_px.median())/4)
    proposals=[]
    for tr,g in groups.items():
        end=g.iloc[-1];t=int(end.frame);name=int(end.candidate_owner)
        hist=g[g.strong.astype(bool)&g.state.eq('observed')&g.candidate_owner.gt(0)]
        if (not end.strong or end.state!='observed' or name<=0 or t>=len(labels)-1
                or len(hist)<minimum or not hist.candidate_owner.eq(name).all()):continue
        parent=established.soma._component(labels[t],name,end)
        if parent is None:continue
        successors=[]
        for p in byframe.get(t,[]):
            other_track=int(p.track_id)
            if other_track==tr or not p.strong or p.state!='observed' or int(p.candidate_owner)!=name:continue
            y,x=round(p.y),round(p.x)
            if not (0<=y<parent.shape[0] and 0<=x<parent.shape[1] and parent[y,x]):continue
            other=groups[other_track]
            # New evidence must arrive after the resident was established;
            # another peak present from the outset is not a detector restart.
            birth=int(other.frame.min())
            if len(hist[hist.frame<birth])<minimum or t+1 not in other.index:continue
            nxt=other.loc[t+1]
            if not nxt.strong or nxt.state!='observed' or int(nxt.candidate_owner) in (0,name):continue
            # This boundary covers reuse of an existing identity. A genuinely
            # new name has a different birth decision in the accepted pathway.
            incoming=int(nxt.candidate_owner)
            if not np.any(strong.frame.lt(t+1)&strong.candidate_owner.eq(incoming)):continue
            earlier=other[(other.frame<=t)&other.strong.astype(bool)&other.state.eq('observed')]
            foreign=earlier[earlier.candidate_owner.gt(0)&earlier.candidate_owner.ne(name)]
            if len(foreign)>=window:continue
            if len(other[(other.frame>t)&other.strong.astype(bool)&other.state.eq('observed')])<window:continue
            successors.append(p)
        if len(successors)==1:proposals.append((tr,name,t,parent,successors[0],len(hist)))
    claims={};audits=[];parts_cache={}
    def parts(t,name):
        if (t,name) not in parts_cache:parts_cache[t,name]=ndi.label(labels[t]==name,STRUCTURE)[0]
        return parts_cache[t,name]
    for tr,name,start,parent,successor,support in proposals:
        previous=parent;joined={tr,int(successor.track_id)};foreign_tracks=set()
        row=dict(anchor=name,predecessor=tr,successor=int(successor.track_id),first_frame=start+2,
                 earlier_strong_frames=support,last_supported_frame=start+1,stop_reason='movie_end',split_frames=0)
        for t in range(start+1,len(labels)):
            yy,xx=np.where(previous);choices=[]
            for current in np.unique(labels[t,yy,xx]):
                if current<=0:continue
                cc=parts(t,int(current))
                for cid in np.unique(cc[yy,xx]):
                    if cid<=0:continue
                    mask=cc==cid
                    members=[p for p in byframe.get(t,[]) if _core_intersects(mask,p)]
                    own=[];others=[]
                    for p in members:
                        track=int(p.track_id)
                        if track not in joined and track not in foreign_tracks:
                            h=groups[track]
                            h=h[(h.frame<t)&h.strong.astype(bool)&h.state.eq('observed')&h.candidate_owner.gt(0)&h.candidate_owner.ne(name)]
                            if any(len(g)>=window for _,g in h.groupby('candidate_owner')):foreign_tracks.add(track)
                        (others if track in foreign_tracks else own).append(p)
                    if not own:continue
                    # Multiple measured peaks can belong to one soma; all its
                    # known peaks share a marker identity against other cells.
                    basin=_shared_body_basin(raw[t],mask,own,others,sigma)
                    if basin is None or not basin.any():continue
                    overlap=basin&previous
                    if not overlap.any():continue
                    recurrent=[]
                    for p in own:
                        track=int(p.track_id)
                        if (track not in joined or not p.strong or p.state!='observed'
                                or int(p.candidate_owner)!=int(current) or t-1 not in groups[track].index):continue
                        old=groups[track].loc[t-1]
                        if old.physically_visible and _core_intersects(previous,old):
                            recurrent.append(float(p.evidence) if hasattr(p,'evidence') else float(raw[t,round(p.y),round(p.x)]))
                    continuity=(bool(recurrent),max(recurrent) if recurrent else 0.)
                    choices.append((continuity[0],continuity[1],float(raw[t][overlap].sum(dtype=np.float64)),int(overlap.sum()),int(current),basin,own,len(others)))
            if not choices:
                row['stop_reason']='no_supported_component_overlap';break
            best=max(c[:4] for c in choices)
            winners=[c for c in choices if c[:4]==best]
            if len(winners)!=1:
                row['stop_reason']='ambiguous_continuation';break
            recurrent,_,_,overlap,current,mask,members,competitors=winners[0]
            stronger=[]
            for prior_name in np.unique(labels[t-1][mask]):
                if prior_name<=0:continue
                cc=parts(t-1,int(prior_name))
                for cid,count in zip(*np.unique(cc[mask],return_counts=True)):
                    if cid>0 and count>overlap and not np.any((cc==cid)&previous):stronger.append(cc==cid)
            independent_predecessor=False
            for prior_mask in stronger:
                for p in byframe.get(t-1,[]):
                    track=int(p.track_id)
                    if track in joined or not p.strong or not _core_intersects(prior_mask,p):continue
                    h=groups[track]
                    h=h[(h.frame<t)&h.strong.astype(bool)&h.state.eq('observed')&h.candidate_owner.gt(0)&h.candidate_owner.ne(name)]
                    if any(len(g)>=window for _,g in h.groupby('candidate_owner')):independent_predecessor=True
            if stronger and (not recurrent or independent_predecessor):
                row['stop_reason']='stronger_independent_predecessor';break
            previous=mask;joined.update(int(p.track_id) for p in members if _core_intersects(mask,p))
            row['last_supported_frame']=t+1;row['split_frames']+=bool(competitors)
            if current==name or np.any(mask&protected[t]):continue
            yy,xx=np.where(mask);key=(t,current,int(yy[0]),int(xx[0]))
            claims.setdefault(key,[]).append((name,int(successor.track_id),yy,xx))
        audits.append(row)
    out=labels.copy();applications=[]
    for key,items in claims.items():
        if len({item[0] for item in items})!=1:continue
        t,source,_,_=key;name,tr,yy,xx=items[0]
        out[t,yy,xx]=name
        applications.append(dict(frame=t+1,source_identity=source,restored_identity=name,
            changed_pixels=len(yy),raw_track=tr,role='overlapping_detector_soma_memory'))
    assert not np.any((out!=labels)&protected)
    assert np.array_equal(out>0,labels>0)
    return out,pd.DataFrame(audits,columns=['anchor','predecessor','successor','first_frame','earlier_strong_frames','last_supported_frame','stop_reason','split_frames']),pd.DataFrame(applications,columns=['frame','source_identity','restored_identity','changed_pixels','raw_track','role'])
