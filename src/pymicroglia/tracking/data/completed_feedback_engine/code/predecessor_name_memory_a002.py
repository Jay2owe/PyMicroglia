"""Preserve physical name evidence across adjacent detector track restarts.

Linking uses the accepted mutual-nearest continuation rule and the recording's
measured motion envelope. Only chains whose combined earlier tenure qualifies
where the successor alone did not can propose new changes. Existing approved
pixels and all unrelated proposals are protected. No feedback is consumed.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

import anchored_soma_owner_recovery_a002_snapshot as established
import predecessor_suffix_area_continuity as suffix
import resident_takeover as owner


def preserve_predecessors(labels, points):
    scored = owner.attach_owners(points, labels)
    limit = established._recording_step_limit(scored)
    visible = scored[scored.physically_visible & scored.strong.astype(bool)]
    full = {int(tr):g.sort_values('frame') for tr,g in scored.groupby('track_id')}
    starts = {int(tr):int(g.frame.min()) for tr,g in full.items()}
    ends = {int(tr):int(g.frame.max()) for tr,g in full.items()}
    by_frame = {int(t):g for t,g in visible.groupby('frame')}
    edges = {}
    rows = []
    for tr, group in full.items():
        last = group.iloc[-1]
        if not last.physically_visible or not bool(last.strong): continue
        t = int(last.frame)
        new = by_frame.get(t+1, visible.iloc[:0])
        new = new[new.track_id.map(starts).eq(t+1)]
        if new.empty: continue
        d = np.hypot(new.x.to_numpy(float)-last.x,new.y.to_numpy(float)-last.y)
        radii = (new.radius_px.to_numpy(float)+float(last.radius_px))/2
        ratios = d/np.maximum(radii,np.finfo(float).eps)
        best = np.flatnonzero(ratios == ratios.min())
        if len(best) != 1 or ratios[best[0]] > limit: continue
        first = new.iloc[int(best[0])]
        prior = by_frame[t]
        back = np.hypot(prior.x.to_numpy(float)-first.x,prior.y.to_numpy(float)-first.y)
        choices = np.flatnonzero(back == back.min())
        if len(choices) != 1 or int(prior.iloc[int(choices[0])].track_id) != tr: continue
        # Equal competing distances cannot be resolved by integer track order.
        successor = int(first.track_id)
        if successor in edges: raise AssertionError('multiple predecessors')
        edges[successor] = tr
        rows.append(dict(predecessor_track=tr,successor_track=successor,
            last_predecessor_frame=t+1,first_successor_frame=t+2,
            endpoint_step_radii=float(ratios[best[0]]),recording_step_limit_radii=limit))
    def root(tr):
        while tr in edges: tr=edges[tr]
        return tr
    merged = points.copy()
    merged['track_id'] = merged.track_id.map(lambda tr:root(int(tr)))
    return merged, rows, scored


def correct(labels, raw, points, protected):
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError('aligned labels, raw and accepted pixels required')
    merged, links, original = preserve_predecessors(labels,points)
    scored = owner.attach_owners(merged,labels)
    minimum = established._minimum(len(labels),established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    gap = math.ceil(len(labels)*established.DEFAULTS['maximum_blank_movie_fraction'])
    proposals = owner.discover_proposals(scored,{
        'minimum_resident_run_frames':minimum,
        'minimum_takeover_run_frames':established._minimum(len(labels),established.DEFAULTS['minimum_shared_contact_movie_fraction']),
        'maximum_adjoining_blank_frames':gap,
        'maximum_endpoint_step_radii':established.DEFAULTS['maximum_endpoint_step_radii']})
    eligible=set()
    memory=[]
    for q in proposals[proposals.precohort_status.eq('eligible')].itertuples(index=False):
        first=int(q.first_frame); tr=int(q.track_id); name=int(q.resident_owner)
        new = scored[(scored.track_id==tr)&(scored.frame==first)]
        if len(new)!=1: continue
        p=new.iloc[0]
        old=original[(original.frame==first)&(original.x==p.x)&(original.y==p.y)]
        if len(old)!=1: continue
        oldtr=int(old.iloc[0].track_id)
        before=established._exclusive_earlier(original,labels,oldtr,name,first,gap)
        after=established._exclusive_earlier(scored,labels,tr,name,first,gap)
        if before < minimum <= after and oldtr != tr:
            eligible.add(tr)
            memory.append(dict(chain_track=tr,successor_track=oldtr,identity=name,
                first_changed_name_frame=first+1,successor_only_anchor_frames=before,
                physical_chain_anchor_frames=after,required_anchor_frames=minimum))
    # Reuse the isolated, auditable suffix transaction, not the rejected output.
    # Filter proposals through newly restored predecessor evidence only. Other
    # soma rows remain present as competing ownership evidence.
    proposed,audit,apps=suffix.correct(labels,raw,merged,protected)
    accepted_apps=apps[apps.raw_track.isin(eligible)].copy() if len(apps) else apps
    result=labels.copy()
    apply_rows=[]
    for row in accepted_apps.itertuples(index=False):
        t=int(row.frame)-1
        point=scored[(scored.track_id==int(row.raw_track))&(scored.frame==t)].iloc[0]
        mask=established.soma._component(labels[t],int(row.source_identity),point)
        if mask is None: raise AssertionError('applied component missing')
        result[t][mask]=int(row.restored_identity)
        apply_rows.append(row._asdict())
    if np.any((result!=labels)&protected) or not np.array_equal(result>0,labels>0):
        raise AssertionError('predecessor memory crossed accepted boundary')
    audit['new_predecessor_evidence']=audit.raw_track.isin(eligible)
    audit.loc[~audit.new_predecessor_evidence,'decision']='outside_new_boundary'
    audit.attrs['links']=links
    audit.attrs['memory']=memory
    return result,audit,pd.DataFrame(apply_rows,columns=apps.columns)
