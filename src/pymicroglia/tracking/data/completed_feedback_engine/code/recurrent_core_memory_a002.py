"""Recover recurrent soma ownership without inheriting a detector's whole path.

An old sustained exclusive core and a shorter independent return provide two
physical references. Both references must overlap an observed core (or its raw
intensity peak) at every proposed frame. Two sustained different names around
the return distinguish repeated loss of ownership from detector restarts alone.
Feedback and literal recording, object, frame or position selectors are absent.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import resident_takeover as owner
import anchored_soma_owner_recovery_a002_snapshot as established


def _overlaps(point, reference):
    return any(math.hypot(float(point.x)-x, float(point.y)-y) <= float(point.radius_px)+r for x,y,r in reference.get('cores',[(reference['x'],reference['y'],reference['radius'])]))


def _anchors(scored, labels, long_minimum, short_minimum, prepared_rows=None):
    anchors=[]
    strong=scored[scored.state.eq('observed') & scored.strong.astype(bool)]
    for track, group in strong.groupby('track_id'):
        group=group.sort_values('frame')
        runs=(group.candidate_owner.ne(group.candidate_owner.shift()) | group.frame.diff().ne(1)).cumsum()
        for _, run in group.groupby(runs):
            name=int(run.candidate_owner.iloc[0])
            if name<=0 or len(run)<short_minimum:continue
            center=run[['x','y']].median()
            ref=dict(track=int(track), name=name, first=int(run.frame.min()), last=int(run.frame.max()),
                     x=float(center.x), y=float(center.y), radius=float(run.radius_px.median()))
            if not all(_overlaps(p,ref) for p in run.itertuples(index=False)):continue
            exclusive=[]
            for p in run.itertuples(index=False):
                mask=established.soma._component(labels[int(p.frame)],name,p)
                if mask is not None and not established.soma._strong_other_somata(scored,int(p.frame),int(track),p,mask,prepared_rows=prepared_rows):
                    exclusive.append(p)
            if len(exclusive)<short_minimum:continue
            ref['cores']=[(float(p.x),float(p.y),float(p.radius_px)) for p in exclusive]
            ref['observations']=len(exclusive)
            ref['long']=len(exclusive)>=long_minimum
            anchors.append(ref)
    return anchors


def correct(labels,raw,points,protected):
    if labels.shape!=raw.shape or labels.shape!=protected.shape:
        raise ValueError('Aligned complete movies required')
    scored=owner.attach_owners(points,labels)
    long_minimum=established._minimum(len(labels),established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    short_minimum=established._minimum(len(labels),established.DEFAULTS['minimum_source_anchor_movie_fraction'])
    prepared_rows=established.soma.prepare_strong_somata_by_frame(scored)
    anchors=_anchors(scored,labels,long_minimum,short_minimum,prepared_rows)
    visible=scored[scored.physically_visible.astype(bool)]
    by_frame={int(t):list(g.itertuples(index=False)) for t,g in visible.groupby('frame')}
    pairs=[]
    for late in anchors:
        # Longer returns are already eligible for the established anchor layer.
        if late['long']:continue
        earlier=[a for a in anchors if a['long'] and a['name']==late['name']
                 and a['track']!=late['track'] and a['last']<late['first']
                 and math.hypot(a['x']-late['x'],a['y']-late['y'])<=a['radius']+late['radius']]
        if not earlier:continue
        # Equal-time independent histories are ambiguous; numerical track order
        # must never decide biological ancestry.
        first=min(a['first'] for a in earlier)
        earliest=[a for a in earlier if a['first']==first]
        if len(earliest)!=1:continue
        pairs.append((earliest[0],late))
    output=labels.copy();audits=[];claims={};applications=[]
    for early,late in pairs:
        name=early['name']; observations=[]; refusals={}
        for t in range(early['last']+1,len(labels)):
            choices={}
            for p in by_frame.get(t,[]):
                source=int(p.candidate_owner)
                if source<=0:continue
                core_match=_overlaps(p,early) and _overlaps(p,late)
                # A detector can follow a bright process instead of the soma.
                # An actual image peak in the same component can independently
                # retain membership; the outline's name is not that evidence.
                if not core_match and int(p.track_id)!=late['track']:continue
                mask=established.soma._component(labels[t],source,p)
                if mask is None:continue
                if not core_match:
                    yy,xx=np.where(mask)
                    value=raw[t,yy,xx];top=value==value.max()
                    peak=p._replace(x=float(xx[top].mean()),y=float(yy[top].mean()))
                    if not (_overlaps(peak,early) and _overlaps(peak,late)):continue
                if established.soma._strong_other_somata(scored,t,int(p.track_id),p,mask,prepared_rows=prepared_rows):
                    refusals['shared_core']=refusals.get('shared_core',0)+1;continue
                yy,xx=np.where(mask);key=(int(yy[0]),int(xx[0]),source)
                choices.setdefault(key,[]).append((p,yy,xx))
            # An already-correct fragment is evidence of the same remembered
            # name; it does not by itself make the wrong-name component ambiguous.
            wrong={key:value for key,value in choices.items() if key[2]!=name}
            if wrong:choices=wrong
            if len(choices)!=1:
                refusals['missing_or_ambiguous_body']=refusals.get('missing_or_ambiguous_body',0)+1;continue
            members=next(iter(choices.values()))
            # Point order cannot affect the selected component or strong status.
            p,yy,xx=members[0]
            is_strong=any(bool(q.strong) and q.state=='observed' for q,_,_ in members)
            observations.append((t,int(p.candidate_owner),yy,xx,is_strong,int(p.track_id),p))
        before={};after={}
        for t,source,yy,xx,strong,track,p in observations:
            if strong and source!=name:
                target=before if t<late['first'] else after if t>late['last'] else None
                if target is not None:target[source]=target.get(source,0)+1
        prior_names={k for k,v in before.items() if v>=short_minimum}
        later_names={k for k,v in after.items() if v>=short_minimum}
        qualifies=bool(prior_names and later_names and any(a!=b for a in prior_names for b in later_names))
        row=dict(anchor=name,early_track=early['track'],late_track=late['track'],
                 early_first_frame=early['first']+1,early_last_frame=early['last']+1,
                 return_first_frame=late['first']+1,return_last_frame=late['last']+1,
                 earlier_support=early['observations'],return_support=late['observations'],
                 intervening_names=str(sorted(prior_names)),later_names=str(sorted(later_names)),
                 refusals=str(refusals),decision='eligible' if qualifies else 'no_repeated_name_loss')
        audits.append(row)
        if not qualifies:continue
        for t,source,yy,xx,strong,track,p in observations:
            if source==name or np.any(protected[t,yy,xx]):continue
            # A separate observed holder remains a correct control until its own
            # ancestry is resolved; never merge it by shared name alone.
            holders=[q for q in by_frame.get(t,[]) if q.strong and q.state=='observed'
                     and int(q.candidate_owner)==name
                     and not (_overlaps(q,early) and _overlaps(q,late))]
            if holders:continue
            key=(t,int(yy[0]),int(xx[0]),source)
            claims.setdefault(key,[]).append((name,track,yy,xx))
    for key,items in claims.items():
        if len({item[0] for item in items})!=1:continue
        t,_,_,source=key;name,track,yy,xx=items[0]
        output[t,yy,xx]=name
        applications.append(dict(frame=t+1,source_identity=source,restored_identity=name,
                                 changed_pixels=len(yy),raw_track=track,role='recurrent_observed_core'))
    assert not np.any((output!=labels)&protected)
    assert np.array_equal(output>0,labels>0)
    return output,pd.DataFrame(audits,columns=['anchor','early_track','late_track','early_first_frame','early_last_frame','return_first_frame','return_last_frame','earlier_support','return_support','intervening_names','later_names','refusals','decision']),pd.DataFrame(applications,columns=['frame','source_identity','restored_identity','changed_pixels','raw_track','role'])
