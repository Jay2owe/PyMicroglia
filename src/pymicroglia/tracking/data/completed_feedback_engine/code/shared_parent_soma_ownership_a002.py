"""Conserve an established soma's ownership when its outline fragments.

Discovery uses measured shared ancestry, paired child mass and parent size.
Continuation requires component overlap and physical observations, independent
of detector numbers and incoming labels. No feedback or literal targets enter.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import resident_takeover as owner
import anchored_soma_owner_recovery_a002_snapshot as established

STRUCTURE=np.ones((3,3),bool)


def _inside(mask,p):
    y,x=round(float(p.y)),round(float(p.x))
    return 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and bool(mask[y,x])


def _core_intersects(mask,p):
    # A weak outline can miss the fitted centre while crossing the measured soma.
    radius=float(p.radius_px)
    x0=max(0,math.floor(p.x-radius));x1=min(mask.shape[1],math.ceil(p.x+radius)+1)
    y0=max(0,math.floor(p.y-radius));y1=min(mask.shape[0],math.ceil(p.y+radius)+1)
    yy,xx=np.ogrid[y0:y1,x0:x1]
    return bool(np.any(mask[y0:y1,x0:x1]&((xx-p.x)**2+(yy-p.y)**2<=radius**2)))


def correct(labels,raw,points,protected):
    if labels.shape!=raw.shape or labels.shape!=protected.shape:
        raise ValueError('Aligned complete movies required')
    scored=owner.attach_owners(points,labels)
    visible=scored[scored.physically_visible.astype(bool)]
    strong=visible[visible.strong.astype(bool)&visible.state.eq('observed')]
    by_frame={int(t):list(g.itertuples(index=False)) for t,g in visible.groupby('frame')}
    by_track={int(tr):g.sort_values('frame').set_index('frame',drop=False) for tr,g in visible.groupby('track_id')}
    minimum=established._minimum(len(labels),established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    window=established._minimum(len(labels),established.DEFAULTS['minimum_source_anchor_movie_fraction'])
    gap=math.ceil(len(labels)*established.DEFAULTS['maximum_blank_movie_fraction'])
    proposals=owner.discover_proposals(scored,dict(minimum_resident_run_frames=minimum,
        minimum_takeover_run_frames=window,maximum_adjoining_blank_frames=gap,
        maximum_endpoint_step_radii=established.DEFAULTS['maximum_endpoint_step_radii']))
    audits=[];claims={};cache={}
    def component(t,name,p):
        return established.soma._component(labels[t],name,p)
    def parts(t,name):
        key=(t,name)
        if key not in cache:cache[key]=ndi.label(labels[t]==name,STRUCTURE)[0]
        return cache[key]
    for q in proposals[proposals.precohort_status.eq('eligible')].itertuples(index=False):
        first=int(q.first_frame);track=int(q.track_id);name=int(q.resident_owner);source=int(q.source_owner)
        group=by_track[track];p=group.loc[first]
        past=group[(group.frame<first)&group.candidate_owner.eq(name)]
        past=past[past.frame>=first-int(q.resident_run_frames)-gap]
        if len(past)<minimum or int(past.frame.max())!=first-1:continue
        prior_sources=strong[(strong.frame<first)&strong.candidate_owner.eq(source)]
        if any(len(g)>=window for _,g in prior_sources.groupby('track_id')):continue
        parent=component(first-1,name,past.iloc[-1])
        if parent is None:continue
        holders=[h for h in by_frame.get(first,[]) if int(h.track_id)!=track
                 and int(h.candidate_owner)==name and bool(h.strong)
                 and math.hypot(h.x-p.x,h.y-p.y)>max(h.radius_px,p.radius_px)]
        if len(holders)!=1 or not _inside(parent,holders[0]):continue
        holder=holders[0];other_track=int(holder.track_id);other=by_track[other_track]
        shared=[];reference_areas=[]
        for old in past.itertuples(index=False):
            t=int(old.frame)
            if t not in other.index or not old.strong:continue
            h=other.loc[t]
            if not h.strong or int(h.candidate_owner)!=name:continue
            mask=component(t,name,old)
            if mask is not None and _inside(mask,h):
                shared.append(t);reference_areas.append(int(mask.sum()))
        if len(shared)<minimum:continue
        # Separate pre-existing identities cannot become eligible through a merge.
        earlier_other=strong[(strong.track_id==other_track)&(strong.frame<first)&strong.candidate_owner.gt(0)&strong.candidate_owner.ne(name)]
        if any(len(g)>=minimum for _,g in earlier_other.groupby('candidate_owner')):continue
        combined=[];dominance=[];seed=None
        for t in range(first,min(first+window,len(labels))):
            if t not in group.index or t not in other.index:break
            a=group.loc[t];b=other.loc[t]
            if int(a.candidate_owner)!=source or int(b.candidate_owner)!=name:break
            ma=component(t,source,a);mb=component(t,name,b)
            if ma is None or mb is None or np.any(ma&mb):break
            area_a=int(ma.sum());area_b=int(mb.sum())
            signal_a=float(raw[t][ma].sum(dtype=np.float64));signal_b=float(raw[t][mb].sum(dtype=np.float64))
            dominance.append(area_a>area_b and signal_a>signal_b)
            combined.append(area_a+area_b)
            if seed is None:seed=ma
        if len(combined)!=window or not all(dominance):continue
        combined_median=float(np.median(combined))
        if not min(reference_areas)<=combined_median<=max(reference_areas):continue
        row=dict(anchor=name,source=source,raw_track=track,retained_fragment_track=other_track,
                 first_frame=first+1,shared_reference_frames=len(shared),comparison_frames=window,
                 reference_area_min=min(reference_areas),reference_area_max=max(reference_areas),
                 combined_child_area_median=combined_median,last_supported_frame=first,
                 decision='qualified_shared_parent_dominance',stop_reason='movie_end')
        previous=seed;memory_names={name,source}
        for t in range(first,len(labels)):
            if t==first:choices=[(source,seed)]
            else:
                yy,xx=np.where(previous);choices=[]
                for current in np.unique(labels[t,yy,xx]):
                    if current<=0:continue
                    cc=parts(t,int(current))
                    for cid in np.unique(cc[yy,xx]):
                        if cid>0:choices.append((int(current),cc==cid))
            viable=[]
            for current,mask in choices:
                members=[p for p in by_frame.get(t,[]) if _core_intersects(mask,p)]
                if not members:continue
                independent=False
                for member in members:
                    history=strong[(strong.track_id==int(member.track_id))&(strong.frame<t)
                        &strong.candidate_owner.gt(0)&~strong.candidate_owner.isin(memory_names)]
                    if any(len(g)>=minimum for _,g in history.groupby('candidate_owner')):
                        independent=True;break
                if independent:continue
                overlap=mask&previous
                signal=float(raw[t][overlap].sum(dtype=np.float64))
                viable.append((signal,int(overlap.sum()),current,mask))
            if not viable:
                row['stop_reason']='no_independently_supported_overlap';break
            best=max((v[0],v[1]) for v in viable)
            winners=[v for v in viable if (v[0],v[1])==best]
            if len(winners)!=1:
                row['stop_reason']='ambiguous_overlap';break
            _,_,current,mask=winners[0]
            # A reciprocal overlap check prevents a small piece from annexing a
            # different large object which already had a stronger predecessor.
            if t>first:
                overlap=int(np.count_nonzero(mask&previous))
                for prior_name in np.unique(labels[t-1][mask]):
                    if prior_name<=0:continue
                    cc=parts(t-1,int(prior_name))
                    ids,counts=np.unique(cc[mask],return_counts=True)
                    if any(cid>0 and count>overlap for cid,count in zip(ids,counts)):
                        row['stop_reason']='stronger_other_predecessor';break
                if row['stop_reason']=='stronger_other_predecessor':break
            previous=mask;memory_names.add(current);row['last_supported_frame']=t+1
            if current==name or np.any(mask&protected[t]):continue
            yy,xx=np.where(mask);key=(t,current,int(yy[0]),int(xx[0]))
            claims.setdefault(key,[]).append((name,track,yy,xx))
        audits.append(row)
    output=labels.copy();applications=[]
    for key,items in claims.items():
        if len({item[0] for item in items})!=1:continue
        t,source,_,_=key;name,track,yy,xx=items[0]
        output[t,yy,xx]=name
        applications.append(dict(frame=t+1,source_identity=source,restored_identity=name,
            changed_pixels=len(yy),raw_track=track,role='shared_parent_dominant_soma'))
    assert not np.any((output!=labels)&protected)
    assert np.array_equal(output>0,labels>0)
    return output,pd.DataFrame(audits,columns=['anchor','source','raw_track','retained_fragment_track','first_frame','shared_reference_frames','comparison_frames','reference_area_min','reference_area_max','combined_child_area_median','last_supported_frame','decision','stop_reason']),pd.DataFrame(applications,columns=['frame','source_identity','restored_identity','changed_pixels','raw_track','role'])
