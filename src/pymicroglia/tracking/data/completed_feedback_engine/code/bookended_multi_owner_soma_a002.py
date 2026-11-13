"""Resolve unstable outline aliases around a continuously measured soma.

One independently anchored soma has the same name before and after a sequence
of at least two foreign positive names. Its raw observation path must be
continuous within the recording's measured motion envelope. Correct supported
named basins throughout the interruption, leaving absent foreground unresolved.
No feedback, recording, identity, detector, frame or region selector is accepted.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import resident_takeover as owner
import anchored_soma_owner_recovery_a002_snapshot as established
from shared_parent_soma_ownership_a002 import _core_intersects
from successive_detector_soma_memory_a001 import _shared_body_basin

AUDIT=['anchor','predecessor','successor','first_frame','last_supported_frame',
       'earlier_strong_frames','return_strong_frames','foreign_names','motion_limit',
       'stop_reason','split_frames','unresolved_frames','collision_frames','changed_pixels',
       'coupled_root_track','coupled_branch']
APPLICATIONS=['frame','source_identity','restored_identity','changed_pixels','raw_track','role']
STRUCTURE=np.ones((3,3),bool)

def episodes(scored,frames):
    """Measured A-[several foreign names]-A intervals, without a duration cutoff."""
    minimum=established._minimum(frames,established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    returning=established._minimum(frames,established.soma.DEFAULTS['minimum_competitor_return_movie_fraction'])
    visible=scored[scored.physically_visible.astype(bool)]
    rows=[]
    for track,g in visible.groupby('track_id'):
        g=g.sort_values('frame')
        runs=owner._runs(g)
        for left in range(len(runs)-2):
            pre=runs[left];name=pre['owner']
            before=[p for p in pre['points'] if p.strong and p.state=='observed']
            if (name<=0 or len(before)<minimum or int(pre['points'][0].frame)!=0
                    or not (pre['points'][-1].strong and pre['points'][-1].state=='observed')):continue
            for right in range(left+2,len(runs)):
                post=runs[right]
                if post['owner']!=name:continue
                middle=runs[left+1:right]
                foreign={r['owner'] for r in middle if r['owner']>0 and r['owner']!=name}
                after=[p for p in post['points'] if p.strong and p.state=='observed']
                if len(foreign)>=2 and len(after)>=returning:
                    interval=[*pre['points'][-1:],*[p for r in middle for p in r['points']],*post['points'][:1]]
                    times=[int(p.frame) for p in interval]
                    if times==list(range(times[0],times[-1]+1)):
                        rows.append((int(track),name,before,after,interval,foreign))
                break
    return rows

def correct(labels,raw,points,protected):
    if labels.shape!=raw.shape or labels.shape!=protected.shape:
        raise ValueError('Aligned complete movies required')
    scored=owner.attach_owners(points,labels)
    strong=scored[scored.strong.astype(bool)&scored.state.eq('observed')]
    visible=scored[scored.physically_visible.astype(bool)]
    byframe={int(t):list(g.itertuples(index=False)) for t,g in visible.groupby('frame')}
    groups={int(tr):g.sort_values('frame') for tr,g in scored.groupby('track_id')}
    minimum=established._minimum(len(labels),established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    returning=established._minimum(len(labels),established.soma.DEFAULTS['minimum_competitor_return_movie_fraction'])
    source_minimum=established._minimum(len(labels),established.DEFAULTS['minimum_source_anchor_movie_fraction'])
    motion=established._recording_step_limit(scored)
    sigma=max(.5,float(strong.radius_px.median())/4)
    cache={}
    def body(t,name,p):
        key=(t,name)
        if key not in cache:cache[key]=ndi.label(labels[t]==name,STRUCTURE)[0]
        cc=cache[key];y,x=round(p.y),round(p.x)
        cid=int(cc[y,x])
        if not cid:
            r=float(p.radius_px)
            y0,y1=max(0,int(np.floor(p.y-r))),min(cc.shape[0],int(np.ceil(p.y+r+1)))
            x0,x1=max(0,int(np.floor(p.x-r))),min(cc.shape[1],int(np.ceil(p.x+r+1)))
            yy,xx=np.ogrid[y0:y1,x0:x1]
            near=cc[y0:y1,x0:x1][(yy-p.y)**2+(xx-p.x)**2<=r*r]
            ids,counts=np.unique(near[near>0],return_counts=True)
            if not len(ids):return None
            winners=ids[counts==counts.max()]
            if len(winners)!=1:return None
            cid=int(winners[0])
        return cc==cid
    ancestry={}
    def foreign(p,t,name,track):
        key=(int(p.track_id),t,name,track)
        if key not in ancestry:
            history=groups[int(p.track_id)]
            history=history[history.frame.lt(t)&history.strong.astype(bool)&history.state.eq('observed')
                            &history.candidate_owner.gt(0)&history.candidate_owner.ne(name)]
            counts={}
            for q in history.itertuples(index=False):
                mask=body(int(q.frame),int(q.candidate_owner),q)
                if mask is None or any(int(a.track_id)==track and _core_intersects(mask,a)
                    for a in byframe.get(int(q.frame),[])):continue
                identity=int(q.candidate_owner);counts[identity]=counts.get(identity,0)+1
            ancestry[key]=any(n>=source_minimum for n in counts.values())
        return ancestry[key]
    audits=[];claims=[]
    for track,name,before,after,interval,names in episodes(scored,len(labels)):
        record=dict(anchor=name,predecessor=track,successor=track,
            first_frame=int(interval[1].frame)+1,last_supported_frame=int(interval[-1].frame)+1,
            earlier_strong_frames=len(before),return_strong_frames=len(after),
            foreign_names=';'.join(map(str,sorted(names))),motion_limit=motion,
            stop_reason='supported_bookended_soma',split_frames=0,unresolved_frames=0,
            collision_frames=0,changed_pixels=0)
        exclusive=0
        for p in before:
            mask=body(int(p.frame),name,p)
            if mask is not None and not established._strong_other(scored,int(p.frame),track,p,mask):exclusive+=1
        if exclusive<minimum:
            record['stop_reason']='earlier_soma_not_independently_anchored';audits.append(record);continue
        return_support=sum(body(int(p.frame),name,p) is not None and
            not established._strong_other(scored,int(p.frame),track,p,body(int(p.frame),name,p)) for p in after)
        if return_support<returning:
            record['stop_reason']='returning_soma_not_independently_anchored';audits.append(record);continue
        steps=[np.hypot(a.x-b.x,a.y-b.y)/((a.radius_px+b.radius_px)/2) for a,b in zip(interval,interval[1:])]
        if any(v>motion for v in steps):
            record['stop_reason']='raw_soma_motion_outside_recording_envelope';audits.append(record);continue
        previous=body(int(interval[0].frame),name,interval[0])
        if previous is None:
            record['stop_reason']='missing_earlier_body';audits.append(record);continue
        for p in interval[1:-1]:
            t=int(p.frame);current=int(p.candidate_owner)
            if current<=0 or not (p.strong and p.state=='observed'):
                record['unresolved_frames']+=1;continue
            mask=body(t,current,p)
            if mask is None or not np.any(mask&previous):
                record['unresolved_frames']+=1;continue
            others=[q for q in byframe.get(t,[]) if int(q.track_id)!=track and q.strong and q.state=='observed'
                    and _core_intersects(mask,q) and np.hypot(q.x-p.x,q.y-p.y)>max(q.radius_px,p.radius_px)
                    and foreign(q,t,name,track)]
            basin=_shared_body_basin(raw[t],mask,[p],others,sigma)
            if basin is None or not basin.any() or not np.any(basin&previous):
                record['unresolved_frames']+=1;continue
            previous=basin
            record['split_frames']+=bool(others)
            change=basin&(labels[t]!=name)
            if np.any(change&protected[t]):
                record['unresolved_frames']+=1;continue
            claims.append((t,basin,name,track,record))
        audits.append(record)
    claims,families=complete_coupled_ancestry(claims,audits,labels,raw,scored,protected,body,byframe,groups,source_minimum,motion,sigma)
    out=labels.copy();apps=[];perframe={}
    for claim in claims:perframe.setdefault(claim[0],[]).append(claim)
    for t,patches in perframe.items():
        wants=np.zeros(labels.shape[1:],labels.dtype);ambiguous=np.zeros_like(wants,bool)
        for _,mask,name,track,record in patches:
            ambiguous|=mask&(wants>0)&(wants!=name)
            wants[mask&(wants==0)]=name
        wants[ambiguous]=0
        frame=labels[t].copy();frame[wants>0]=wants[wants>0]
        measured=owner.attach_owners(points[points.frame.eq(t)].assign(frame=0),frame[None])
        for name in np.unique(wants[wants>0]):
            holders=measured[measured.strong.astype(bool)&measured.state.eq('observed')&measured.candidate_owner.eq(name)]
            masks=[established.soma._component(frame,int(name),p) for p in holders.itertuples(index=False)]
            conflict=any(a is not None and b is not None and not np.any(a&b)
                and frozenset((int(holders.iloc[i].track_id),int(holders.iloc[j].track_id))) not in families.get(int(name),set())
                and (np.any((frame!=labels[t])&a) or np.any((frame!=labels[t])&b))
                for i,a in enumerate(masks) for j,b in enumerate(masks) if j>i)
            if conflict:
                frame[(wants==name)&(frame!=labels[t])]=labels[t][(wants==name)&(frame!=labels[t])]
                for _,_,target,_,record in patches:
                    if target==name:record['collision_frames']+=1
        out[t]=frame;remaining=frame!=labels[t]
        for _,mask,name,track,record in patches:
            selected=remaining&mask&(frame==name)
            for original in np.unique(labels[t][selected]):
                changed=selected&(labels[t]==original);count=int(changed.sum())
                if count:
                    apps.append(dict(frame=t+1,source_identity=int(original),restored_identity=name,
                        changed_pixels=count,raw_track=track,role='coupled_bookended_soma' if record.get('coupled_branch') else 'bookended_multi_owner_soma'))
                    record['changed_pixels']+=count;remaining[changed]=False
        assert not remaining.any()
    assert not np.any((out!=labels)&protected)
    assert np.array_equal(out>0,labels>0)
    return out,pd.DataFrame(audits,columns=AUDIT),pd.DataFrame(apps,columns=APPLICATIONS)


def complete_coupled_ancestry(claims,audits,labels,raw,scored,protected,body,byframe,groups,minimum,motion,sigma):
    """A branch repeatedly attached to an original soma cannot become its rival.

    Only existing holders that block a independently bookended correction are
    examined. Both raw attachment histories must agree on one recording-start
    ancestor. Every directly measured borrowed-name mask is proposed together;
    an unsupported or protected observation leaves that branch untouched.
    """
    families={};resolved=set();extra=[]
    wanted={(t,name,track) for t,_,name,track,_ in claims}
    for t,name,track in sorted(wanted):
        blockers=[p for p in byframe.get(t,[]) if int(p.track_id)!=track and p.strong
                  and p.state=='observed' and int(p.candidate_owner)==name]
        for block in blockers:
            branch=int(block.track_id)
            if (branch,name) in resolved:continue
            g=groups[branch]
            runs=owner._runs(g[g.physically_visible.astype(bool)])
            at=next((i for i,r in enumerate(runs) if r['owner']==name
                     and int(r['points'][0].frame)<=t<=int(r['points'][-1].frame)),None)
            if at is None:continue
            left=at-1
            while left>=0 and runs[left]['owner']==0:left-=1
            right=at+1
            while right<len(runs) and runs[right]['owner']==0:right+=1
            if left<0 or right>=len(runs):continue
            ancestor=int(runs[left]['owner'])
            if ancestor<=0 or ancestor==name or runs[right]['owner']!=ancestor:continue
            before=[p for p in runs[left]['points'] if p.strong and p.state=='observed']
            after=[p for p in runs[right]['points'] if p.strong and p.state=='observed']
            interval=[*runs[left]['points'][-1:],
                      *[p for r in runs[left+1:right] for p in r['points']],
                      *runs[right]['points'][:1]]
            times=[int(p.frame) for p in interval]
            if times!=list(range(times[0],times[-1]+1)):continue
            if any(np.hypot(a.x-b.x,a.y-b.y)/((a.radius_px+b.radius_px)/2)>motion
                   for a,b in zip(interval,interval[1:])):continue
            roots=[]
            for other,h in groups.items():
                if other==branch or int(h.frame.min())!=0:continue
                relevant=h[h.frame.le(int(after[-1].frame))]
                positive=relevant[relevant.physically_visible.astype(bool)&relevant.candidate_owner.gt(0)]
                if positive.empty or not positive.candidate_owner.eq(ancestor).all():continue
                lookup={int(p.frame):p for p in h.itertuples(index=False)}
                counts=[]
                for observations in (before,after):
                    n=0
                    for p in observations:
                        q=lookup.get(int(p.frame))
                        if q is None or not q.strong or q.state!='observed' or int(q.candidate_owner)!=ancestor:continue
                        mask=body(int(p.frame),ancestor,p)
                        if mask is not None and _core_intersects(mask,q):n+=1
                    counts.append(n)
                if min(counts)>=minimum:roots.append((other,counts))
            if len(roots)!=1:continue
            root,counts=roots[0]
            record=dict(anchor=ancestor,predecessor=branch,successor=branch,
                first_frame=int(runs[at]['points'][0].frame)+1,
                last_supported_frame=int(runs[at]['points'][-1].frame)+1,
                earlier_strong_frames=counts[0],return_strong_frames=counts[1],
                foreign_names=str(name),motion_limit=motion,stop_reason='coupled_attached_branch',
                split_frames=0,unresolved_frames=0,collision_frames=0,changed_pixels=0,
                coupled_root_track=root,coupled_branch=True)
            branch_claims=[]
            for p in runs[at]['points']:
                bt=int(p.frame)
                if not p.strong or p.state!='observed':break
                mask=body(bt,name,p)
                if mask is None:break
                unrelated=[q for q in byframe.get(bt,[]) if int(q.track_id) not in (branch,root)
                    and q.strong and q.state=='observed' and _core_intersects(mask,q)
                    and np.hypot(q.x-p.x,q.y-p.y)>max(q.radius_px,p.radius_px)]
                basin=_shared_body_basin(raw[bt],mask,[p],unrelated,sigma)
                if basin is None or not basin.any() or np.any(basin&protected[bt]):break
                branch_claims.append((bt,basin,ancestor,branch,record))
            if len(branch_claims)!=len(runs[at]['points']):continue
            resolved.add((branch,name));extra.extend(branch_claims);audits.append(record)
            families.setdefault(ancestor,set()).add(frozenset((branch,root)))
    return claims+extra,families
