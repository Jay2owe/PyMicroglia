"""Prevent a fresh name from replacing a continuing measured soma.

Only raw-core witnessed name births activate this operation. Temporal edges
use existing reciprocal overlap, bounded observed cores and mutually nearest
measured outlines separated by less than their own area-derived radius.
Feedback and dataset selectors are not accepted.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from optical_basin_soma_paths import physical_paths

COLUMNS=['node','identity','frame','seed_y','seed_x','source_identity','changed_pixels','root','birth_frame','decision']

def restore_measured_birth_lineage(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels,established_birth_only=False,core_supported_roots=False,require_core_turnover=False):
    labels=current_labels;out=labels.copy();empty=pd.DataFrame(columns=COLUMNS)
    if not labels.shape==observed.shape==proposals.shape==raw.shape==original_labels.shape:raise ValueError('Aligned complete movies required')
    nodes,records,paths,_=physical_paths(labels,observed);N=len(records)
    path_of=np.zeros(N,np.int32)
    for path,members in paths.items():path_of[members]=path
    parents=np.arange(N);incoming=[set() for _ in range(N)];edges=[]
    def find(n):
        while parents[n]!=n:parents[n]=parents[parents[n]];n=int(parents[n])
        return n
    def join(a,b):
        if a==b:return
        if records[a]['frame']>records[b]['frame']:a,b=b,a
        if records[a]['frame']!=records[b]['frame']:incoming[b].add(a)
        x,y=find(a),find(b)
        if x!=y:parents[y]=x
        edges.append((a,b))
    for members in paths.values():
        for a,b in zip(members,members[1:]):join(a,b)
    pts=points[points.strong.astype(bool)&points.state.eq('observed')].copy()
    if 'physically_visible' in pts:pts=pts[pts.physically_visible.astype(bool)]
    tt=pts.frame.to_numpy(int);yy=np.floor(pts.y.to_numpy()+.5).astype(int);xx=np.floor(pts.x.to_numpy()+.5).astype(int)
    valid=(tt>=0)&(tt<len(labels))&(yy>=0)&(yy<labels.shape[1])&(xx>=0)&(xx<labels.shape[2]);pts=pts[valid].copy();tt=tt[valid];yy=yy[valid];xx=xx[valid]
    pts['node']=nodes[tt,yy,xx];pts['native']=proposals[tt,yy,xx];pts['measured']=observed[tt,yy,xx];pts['owner']=labels[tt,yy,xx]
    births=[];first_native={}
    for name in np.unique(proposals):
        if name:first_native[int(name)]=int(np.flatnonzero(np.any(proposals==name,axis=(1,2)))[0])
    for track,g in pts.groupby('track_id'):
        previous=None
        for q in g.sort_values('frame').itertuples(index=False):
            if previous is not None:
                dt=int(q.frame)-int(previous.frame)
                if dt>0 and np.hypot(q.x-previous.x,q.y-previous.y)<=dt*(q.radius_px+previous.radius_px):
                    if q.node and previous.node:
                        join(int(previous.node),int(q.node))
                        if q.native and previous.native and q.native!=previous.native and q.measured==q.native and previous.measured==previous.native and first_native[int(q.native)]==int(q.frame):
                            births.append((int(previous.node),int(q.node),int(previous.native),int(q.native),int(q.frame)+1))
            previous=q
    if not births:return out,empty
    if established_birth_only:
        # A single first glimpse is not established ownership. Require the
        # predecessor's own physical path to have repeated measurements and
        # more than one measured core history before a fresh alias appears.
        births=[b for b in births if
            pts[pts.node.map(lambda n:path_of[n]).eq(path_of[b[0]])&pts.frame.lt(b[4]-1)&pts.measured.eq(b[2])].track_id.nunique()>1
            and pts[pts.node.map(lambda n:path_of[n]).eq(path_of[b[0]])&pts.frame.lt(b[4]-1)&pts.measured.eq(b[2])].frame.nunique()>1]
        if not births:return out,empty
    if require_core_turnover:
        # Stable soma ownership must outlast one individual detected maximum.
        # A completed core and its continuing successor provide that evidence
        # without a recording-specific duration or fitted frame cutoff.
        last=pts.groupby('track_id').frame.max().to_dict()
        valid_births=[]
        for b in births:
            history=pts[pts.node.map(lambda n:path_of[n]).eq(path_of[b[0]])&pts.frame.lt(b[4]-1)&pts.measured.eq(b[2])]
            if any(last[int(tr)]<b[4]-2 for tr in history.track_id.unique()):valid_births.append(b)
        births=valid_births
        if not births:return out,empty
    # Link only disjoint measured shapes. Reciprocal unique nearest edges
    # prevent a vanished outline from selecting among competing neighbours.
    boxes={};masks={};areas=np.array([0]+[q['proposed_pixels'] for q in records[1:]])
    for t,frame in enumerate(nodes):
        for n,box in enumerate(ndi.find_objects(frame),1):
            if box is not None:boxes[n]=box;masks[n]=frame[box]==n
    per_frame={t:[q['node'] for q in records[1:] if q['frame']==t+1 and q['measured_pixels']==q['proposed_pixels']] for t in range(len(labels))}
    for t in range(1,len(labels)):
        scores=[]
        for a in per_frame[t-1]:
            ba=boxes[a];ra=np.sqrt(areas[a]/np.pi)
            for b in per_frame[t]:
                if find(a)==find(b):continue
                bb=boxes[b];radius=min(ra,np.sqrt(areas[b]/np.pi))
                dy=max(0,ba[0].start-bb[0].stop+1,bb[0].start-ba[0].stop+1);dx=max(0,ba[1].start-bb[1].stop+1,bb[1].start-ba[1].stop+1)
                if np.hypot(dx,dy)>radius:continue
                y0=min(ba[0].start,bb[0].start);y1=max(ba[0].stop,bb[0].stop);x0=min(ba[1].start,bb[1].start);x1=max(ba[1].stop,bb[1].stop)
                left=nodes[t-1,y0:y1,x0:x1]==a;right=nodes[t,y0:y1,x0:x1]==b
                gap=float(ndi.distance_transform_edt(~left)[right].min())
                if 0<gap<=radius:scores.append((a,b,gap/radius))
        for a,b,score in scores:
            aa=[r for r in scores if r[0]==a];bb=[r for r in scores if r[1]==b]
            if sum(r[2]<=score for r in aa)==sum(r[2]<=score for r in bb)==1:join(a,b)
    # Existing same-name fragments are one current ownership group. Including
    # them cannot add a cell; independent origins still veto the proposal.
    for t in range(len(labels)):
        by_owner={}
        for q in records[1:]:
            if q['frame']==t+1:by_owner.setdefault(q['identity'],[]).append(q['node'])
        for members in by_owner.values():
            for a,b in zip(members,members[1:]):join(a,b)
    groups={}
    for n in range(1,N):groups.setdefault(find(n),[]).append(n)
    audit=[]
    for group,members in groups.items():
        anchors=[b for b in births if find(b[0])==group]
        if not anchors:continue
        roots=[n for n in members if not incoming[n]]
        if core_supported_roots:
            supported=[];invalid=False
            for n in roots:
                own=pts[pts.node.isin(paths[path_of[n]])]
                if len(own):supported.append(n);continue
                t=records[n]['frame']-1;piece=nodes[t]==n
                peers=[q for q in members if records[q]['frame']==t+1 and q!=n]
                if not peers:invalid=True;break
                core=pts[pts.frame.eq(t)&pts.node.isin(peers)]
                if core.empty:invalid=True;break
                body=np.isin(nodes[t],peers);gap=float(ndi.distance_transform_edt(~body)[piece].min())
                y,x=np.argwhere(piece).mean(axis=0);q=core.iloc[np.argmin((core.x-x)**2+(core.y-y)**2)]
                if gap>float(q.radius_px):invalid=True;break
            if invalid:continue
            roots=supported
        names={records[n]['identity'] for n in roots}
        if len(names)!=1:continue
        name=next(iter(names))
        if not any(b[2]==name for b in anchors):continue
        measured_seed=pts[pts.node.isin(members)&pts.measured.eq(name)]
        if measured_seed.frame.nunique()<2:continue
        target_frames={records[n]['frame'] for n in members};bad=False;changes=[]
        for t in target_frames:
            ids=[n for n in members if records[n]['frame']==t];mask=np.isin(nodes[t-1],ids)
            if np.any((labels[t-1]==name)&~mask):bad=True;break
            change=mask&(labels[t-1]!=name)
            if np.any(change&(labels[t-1]!=original_labels[t-1])):bad=True;break
            if change.any():changes.extend(n for n in ids if records[n]['identity']!=name)
        if bad or not changes:continue
        for n in changes:
            q=records[n];t=q['frame']-1;out[t][nodes[t]==n]=name
            audit.append(dict(node=n,identity=name,frame=t+1,seed_y=q['seed_y'],seed_x=q['seed_x'],source_identity=q['identity'],changed_pixels=q['proposed_pixels'],root=group,birth_frame=min(b[4] for b in anchors),decision='fresh_name_inside_continuing_measured_lineage'))
    return out,pd.DataFrame(audit,columns=COLUMNS)
