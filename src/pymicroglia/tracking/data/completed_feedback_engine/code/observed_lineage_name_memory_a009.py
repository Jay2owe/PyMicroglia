"""Restore ownership only where independent observed cores support membership.

Detector restart ancestry can establish a lost name. An inherited outline name
cannot propagate that correction after the observed physical path ends: the
outline tracker may have handed the name to another, already-correct cell.
Unobserved continuations remain unresolved; no feedback is a production input.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import resident_takeover as owner
import predecessor_name_memory_a002 as predecessor
import anchored_soma_owner_recovery_a002_snapshot as established

STRUCTURE=np.ones((3,3),bool)


def _parts(plane, name):
    parts,count=ndi.label(plane==name,STRUCTURE)
    result=[]
    for index,box in enumerate(ndi.find_objects(parts),1):
        if box is None:continue
        yy,xx=np.where(parts[box]==index)
        yy=yy+box[0].start;xx=xx+box[1].start
        result.append(dict(yy=yy,xx=xx,x=float(xx.mean()),y=float(yy.mean()),
                           radius=math.sqrt(len(yy)/math.pi)))
    return result


def correct(labels,raw,points,protected):
    if labels.shape!=raw.shape or labels.shape!=protected.shape:
        raise ValueError('Aligned movie, raw frames and protected pixels required')
    merged,links,original=predecessor.preserve_predecessors(labels,points)
    scored=owner.attach_owners(merged,labels)
    minimum=established._minimum(len(labels),established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    gap=math.ceil(len(labels)*established.DEFAULTS['maximum_blank_movie_fraction'])
    limit=established._recording_step_limit(scored)
    proposals=owner.discover_proposals(scored,dict(minimum_resident_run_frames=minimum,
        minimum_takeover_run_frames=established._minimum(len(labels),established.DEFAULTS['minimum_shared_contact_movie_fraction']),
        maximum_adjoining_blank_frames=gap,maximum_endpoint_step_radii=established.DEFAULTS['maximum_endpoint_step_radii']))
    seeds=[];refusals=[]
    for q in proposals[proposals.precohort_status.eq('eligible')].itertuples(index=False):
        first=int(q.first_frame);track=int(q.track_id);name=int(q.resident_owner)
        p=scored[(scored.track_id==track)&(scored.frame==first)]
        if len(p)!=1:continue
        p=p.iloc[0];old=original[(original.frame==first)&(original.x==p.x)&(original.y==p.y)]
        if len(old)!=1:continue
        old_track=int(old.iloc[0].track_id)
        before=established._exclusive_earlier(original,labels,old_track,name,first,gap)
        after=established._exclusive_earlier(scored,labels,track,name,first,gap)
        if before<minimum<=after and old_track!=track:
            holders=scored[(scored.frame==first)&scored.track_id.ne(track)&scored.candidate_owner.eq(name)&scored.strong.astype(bool)&scored.physically_visible]
            survives=any(math.hypot(float(h.x-p.x),float(h.y-p.y))>max(float(h.radius_px),float(p.radius_px)) for h in holders.itertuples(index=False))
            collision=False
            source=int(p.candidate_owner)
            mask=established.soma._component(labels[first],source,p) if source>0 else None
            if first>0 and mask is not None:
                previous=scored[(scored.frame==first-1)&scored.physically_visible]
                def inside(point):
                    iy,ix=round(point.y),round(point.x)
                    return 0<=iy<mask.shape[0] and 0<=ix<mask.shape[1] and mask[iy,ix]
                prior_target=previous[previous.track_id.eq(track)&previous.candidate_owner.eq(name)]
                prior_source=previous[previous.track_id.ne(track)&previous.candidate_owner.eq(source)&previous.strong.astype(bool)]
                collision=any(inside(q) for q in prior_target.itertuples(index=False)) and any(inside(q) for q in prior_source.itertuples(index=False))
            if survives or collision:
                refusals.append(dict(anchor=name,raw_track=track,first_frame=first+1,
                    minimum_observed_frames=minimum,maximum_blank_frames=gap,
                    measured_speed_limit=limit,observations=[],extensions=[],decision='original_named_soma_survives' if survives else 'two_prior_named_cores_enter_one_outline'))
            else:seeds.append((track,name,first))
    output=labels.copy();audits=refusals;applications=[];claims={};cache={}
    def components(t,name):
        key=(t,name)
        if key not in cache:cache[key]=_parts(labels[t],name)
        return cache[key]
    def claim(t,c,source,target,track,role):
        if np.any(protected[t,c['yy'],c['xx']]):return
        key=(t,int(c['yy'][0]),int(c['xx'][0]),source)
        claims.setdefault(key,[]).append((target,track,role,c))
    for track,name,first in seeds:
        observations=[]
        chain=scored[(scored.track_id==track)&scored.frame.ge(first)&scored.physically_visible]
        for p in chain.itertuples(index=False):
            t=int(p.frame);source=int(p.candidate_owner)
            if source<=0 or source==name:continue
            # The observed core can overlap a small shifted outline even when
            # its rounded centre falls outside. Reuse the established component
            # locator and shared-soma refusal instead of a one-pixel condition.
            mask=established.soma._component(labels[t],source,p)
            if mask is None or established._strong_other(scored,t,track,p,mask):continue
            candidates=[c for c in components(t,source) if mask[c['yy'][0],c['xx'][0]]]
            if len(candidates)!=1:continue
            c=candidates[0]
            claim(t,c,source,name,track,'observed_predecessor_memory')
            observations.append(dict(frame=t+1,source_identity=source,expected_identity=name))
        extensions=[]
        audits.append(dict(anchor=name,raw_track=track,first_frame=first+1,
            minimum_observed_frames=minimum,maximum_blank_frames=gap,
            measured_speed_limit=limit,observations=observations,extensions=extensions,decision='discovered'))
    for key,items in claims.items():
        names={row[0] for row in items}
        if len(names)!=1:continue
        t,_,_,source=key;target,track,role,c=items[0]
        output[t,c['yy'],c['xx']]=target
        applications.append(dict(frame=t+1,source_identity=source,restored_identity=target,
            changed_pixels=len(c['yy']),raw_track=track,role=role))
    for row in audits:
        if row['decision']!='discovered':continue
        row['decision']='applied' if any(a['raw_track']==row['raw_track'] for a in applications) else 'unchanged'
    assert not np.any((output!=labels)&protected)
    assert np.array_equal(output>0,labels>0)
    return output,pd.DataFrame(audits,columns=['anchor','raw_track','first_frame','minimum_observed_frames','maximum_blank_frames','measured_speed_limit','observations','extensions','decision']),pd.DataFrame(applications,columns=['frame','source_identity','restored_identity','changed_pixels','raw_track','role'])
