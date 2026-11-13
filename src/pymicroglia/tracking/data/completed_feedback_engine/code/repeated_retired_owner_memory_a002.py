"""Keep a resident soma from repeatedly inheriting a retired neighbour's name.

A long independent early core and a long measured return anchor the same body.
The repeated invading name must have belonged to another measured detector which
has ended before the first invasion. Independent peaks retain their watershed
basins. Discovery receives aligned image arrays and raw detections only.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import resident_takeover as owner
import anchored_soma_owner_recovery_a002_snapshot as established
from coobserved_bookended_soma_memory_a001 import _runs
from overlapping_detector_soma_memory_a003 import _shared_body_basin
from shared_parent_soma_ownership_a002 import _core_intersects

STRUCTURE=np.ones((3,3),bool)
AUDIT_COLUMNS=['track','anchor','source','first_frame','last_frame','earlier_exclusive',
               'return_observations','retired_source_tracks','recording_step_limit_radii',
               'reference_radius','changed_pixels','decision']
APPLICATION_COLUMNS=['frame','source_identity','restored_identity','changed_pixels',
                     'raw_track','other_peaks_preserved','role']


def _reference(group):
    return (float(group.x.median()),float(group.y.median()),float(group.radius_px.median()))


def _overlaps_reference(p,reference):
    x,y,r=reference
    return math.hypot(float(p.x)-x,float(p.y)-y)<=float(p.radius_px)+r


def correct(labels,raw,points,protected):
    if labels.shape!=raw.shape or labels.shape!=protected.shape:
        raise ValueError('Aligned complete movies required')
    scored=owner.attach_owners(points,labels)
    visible=scored[scored.physically_visible.astype(bool)]
    strong=visible[visible.strong.astype(bool)&visible.state.eq('observed')]
    groups={int(tr):g.sort_values('frame') for tr,g in visible.groupby('track_id')}
    byframe={int(t):list(g.itertuples(index=False)) for t,g in strong.groupby('frame')}
    minimum=established._minimum(len(labels),established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    step=established._recording_step_limit(scored)
    sigma=float(strong.radius_px.median())/2 if len(strong) else 0
    parts={};claims={};audits=[]
    def component(p):
        t=int(p.frame);name=int(p.candidate_owner);key=(t,name)
        if key not in parts:parts[key]=ndi.label(labels[t]==name,STRUCTURE)[0]
        a=parts[key];y=round(float(p.y));x=round(float(p.x))
        if not (0<=y<a.shape[0] and 0<=x<a.shape[1]) or a[y,x]==0:return None
        return a==a[y,x]
    def independent_peaks(p,mask):
        return [q for q in byframe.get(int(p.frame),[]) if int(q.track_id)!=int(p.track_id)
                and _core_intersects(mask,q)
                and math.hypot(q.x-p.x,q.y-p.y)>max(q.radius_px,p.radius_px)]
    for track,g in groups.items():
        measured=g[g.strong.astype(bool)&g.state.eq('observed')&g.candidate_owner.gt(0)]
        if measured.empty:continue
        for name in sorted(set(measured.candidate_owner.astype(int))):
            runs=_runs(g)
            repeated={int(run.candidate_owner.iloc[0]) for run in runs
                      if int(run.candidate_owner.iloc[0])>0 and int(run.candidate_owner.iloc[0])!=name}
            for source in sorted(repeated):
                wrong=[run for run in runs if int(run.candidate_owner.iloc[0])==source]
                if len(wrong)<2:continue
                first=int(wrong[0].frame.min());last=int(wrong[-1].frame.max())
                before=measured[(measured.frame<first)&measured.candidate_owner.eq(name)]
                returns=[r for r in runs if int(r.candidate_owner.iloc[0])==name
                         and int(r.frame.min())>int(wrong[0].frame.max())
                         and int(r.frame.max())<int(wrong[-1].frame.min())]
                returns=[r[r.strong.astype(bool)&r.state.eq('observed')] for r in returns]
                returns=[r for r in returns if len(r)>=minimum]
                if not returns:continue
                later_ref=_reference(pd.concat(returns))
                exclusive=[]
                for p in before.itertuples(index=False):
                    if not _overlaps_reference(p,later_ref):continue
                    mask=component(p)
                    if mask is not None and not independent_peaks(p,mask):exclusive.append(p)
                if len(exclusive)<minimum:continue
                # A detector can relocate before this episode: both independent anatomy
                # and a later measured return must agree with the same stable core.
                earlier=pd.DataFrame([p._asdict() for p in exclusive])
                early_ref=_reference(earlier)
                later_ref=_reference(pd.concat(returns))
                if not all(_overlaps_reference(p,early_ref) and _overlaps_reference(p,later_ref)
                           for p in pd.concat([earlier,*returns]).itertuples(index=False)):continue
                origins=strong[(strong.frame<first)&strong.candidate_owner.eq(source)&strong.track_id.ne(track)]
                retired=[]
                for tr,h in origins.groupby('track_id'):
                    if len(h)<minimum:continue
                    if any(_overlaps_reference(p,early_ref) for p in h.itertuples(index=False)):continue
                    retired.append(int(tr))
                if not retired or any(int(groups[tr].frame.max())>=first for tr in retired):continue
                path=g[g.frame.between(first-1,last)]
                positions=path[['x','y']].to_numpy(float);radii=path.radius_px.to_numpy(float)
                adjacent=path.frame.diff().to_numpy()[1:]==1
                motion=np.linalg.norm(np.diff(positions,axis=0),axis=1)/((radii[:-1]+radii[1:])/2)
                if not adjacent.all() or np.any(motion>step):continue
                if not all(_overlaps_reference(p,early_ref) and _overlaps_reference(p,later_ref)
                           for p in path.itertuples(index=False)):continue
                for p in pd.concat(wrong).itertuples(index=False):
                    t=int(p.frame);mask=component(p);reason='supported';changed=0
                    other=[]
                    if mask is None:reason='missing_component'
                    elif not p.strong or p.state!='observed':reason='no_measured_core'
                    else:
                        # A separately measured resident with its own earlier
                        # tenure blocks reuse of that resident's identity.
                        holders=[q for q in byframe.get(t,[]) if int(q.candidate_owner)==name
                                 and not _overlaps_reference(q,early_ref)]
                        independent=False
                        for q in holders:
                            history=strong[(strong.track_id==int(q.track_id))&(strong.frame<first)
                                           &strong.candidate_owner.eq(name)]
                            if len(history)>=minimum:independent=True;break
                        if independent:reason='independent_resident_holder'
                        else:
                            other=independent_peaks(p,mask)
                            basin=_shared_body_basin(raw[t],mask,[p],other,sigma)
                            if basin is None:reason='ambiguous_raw_basin'
                            elif np.any(basin&protected[t]):reason='protected_basin'
                            else:
                                yy,xx=np.where(basin);changed=len(yy)
                                key=(t,int(p.candidate_owner),int(yy[0]),int(xx[0]))
                                claims.setdefault(key,[]).append((name,track,yy,xx,len(other)))
                    audits.append(dict(track=track,anchor=name,source=source,first_frame=t+1,
                        last_frame=t+1,earlier_exclusive=len(exclusive),return_observations=sum(map(len,returns)),
                        retired_source_tracks=','.join(map(str,sorted(retired))),recording_step_limit_radii=step,
                        reference_radius=early_ref[2],changed_pixels=changed,decision=reason))
    result=labels.copy();apps=[]
    for (t,source,_,_),contenders in claims.items():
        if len({c[0] for c in contenders})!=1:continue
        name,track,yy,xx,nother=min(contenders,key=lambda c:c[1])
        # Conflicting independently discovered resident claims are rejected.
        if np.any((result[t,yy,xx]!=labels[t,yy,xx])&(result[t,yy,xx]!=name)):continue
        result[t,yy,xx]=name
        apps.append(dict(frame=t+1,source_identity=source,restored_identity=name,
                         changed_pixels=len(yy),raw_track=track,other_peaks_preserved=nother,
                         role='repeated_retired_owner'))
    assert not np.any((result!=labels)&protected)
    assert np.array_equal(result>0,labels>0)
    return result,pd.DataFrame(audits,columns=AUDIT_COLUMNS),pd.DataFrame(apps,columns=APPLICATION_COLUMNS)
