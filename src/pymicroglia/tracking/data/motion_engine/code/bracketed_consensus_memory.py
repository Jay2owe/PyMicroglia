"""Preserve a measured motion bracket when future label counts erase its name.

Names are evidence only when the measured body exists on both sides of an
initial raw-core observation and its interpolated body contains that core.
The reserved name follows that core and reciprocal physical overlap, including
new cores born on the same body. No feedback or literal selectors are accepted.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from optical_basin_soma_paths import physical_paths

COLUMNS=['track','identity','invading_identity','anchor_frame','left_frame','right_frame',
         'bracket_distance_radii','path','node','frame','source_identity','seed_y','seed_x',
         'changed_pixels','decision']

def restore_bracketed_consensus_memory(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels,retained_source_only=False,independent_core_history=False,returned_same_name_fragments=False,preserve_label_partition=False,nearby_same_name_fragments=False):
    if not (observed.shape==proposals.shape==raw.shape==current_labels.shape==original_labels.shape):
        raise ValueError('Aligned historical measurements and approved complete movie required')
    result=current_labels.copy();empty=pd.DataFrame(columns=COLUMNS)
    pts=points[points.strong.astype(bool)&points.state.eq('observed')].copy()
    if 'physically_visible' in pts:pts=pts[pts.physically_visible.astype(bool)]
    elif 'visibility' in pts:pts=pts[pts.visibility.gt(0)]
    yy=np.floor(pts.y.to_numpy()+.5).astype(int);xx=np.floor(pts.x.to_numpy()+.5).astype(int);tt=pts.frame.to_numpy(int)
    valid=(tt>=0)&(tt<len(raw))&(yy>=0)&(yy<raw.shape[1])&(xx>=0)&(xx<raw.shape[2])
    pts=pts[valid].copy();yy=yy[valid];xx=xx[valid];tt=tt[valid]
    pts['native']=proposals[tt,yy,xx];pts['current']=current_labels[tt,yy,xx];pts['measured']=observed[tt,yy,xx]
    anchors=[];time_cache={};shape_cache={}
    def geometry(name,t):
        key=(name,t)
        if key not in shape_cache:
            mask=observed[t]==name;count=ndi.label(mask,np.ones((3,3),bool))[1]
            y,x=np.nonzero(mask)
            shape_cache[key]=None if count!=1 else np.array([y.mean(),x.mean(),np.sqrt(len(y)/np.pi)])
        return shape_cache[key]
    for track,g in pts.groupby('track_id'):
        g=g[g.native.gt(0)].sort_values('frame')
        if g.empty:continue
        q=g.iloc[0];name=int(q.native);other=int(q.current);t=int(q.frame)
        if not other or name==other or int(q.measured)!=name:continue
        # Future numerical support must actually favour the invading name;
        # a later scientific identity correction is not this mechanism.
        counts=g.groupby('native').size();winners=counts[counts.eq(counts.max())].index
        if len(winners)!=1 or int(winners[0])!=other:continue
        if name not in time_cache:time_cache[name]=np.flatnonzero(np.any(observed==name,axis=(1,2)))
        times=time_cache[name];left=times[times<t];right=times[times>t]
        if not len(left) or not len(right):continue
        a,b=int(left[-1]),int(right[0]);ga,gb=geometry(name,a),geometry(name,b)
        if ga is None or gb is None:continue
        fraction=(t-a)/(b-a);pred=(1-fraction)*ga+fraction*gb
        distance=float(np.hypot(q.y-pred[0],q.x-pred[1])/pred[2])
        if distance>1:continue
        if retained_source_only:
            # The incoming name has a different previously measured body,
            # still independently observed in that territory at the transfer.
            # Its historical absence alone cannot establish another owner.
            if other not in time_cache:time_cache[other]=np.flatnonzero(np.any(observed==other,axis=(1,2)))
            older=time_cache[other][time_cache[other]<t]
            if not len(older):continue
            source_t=int(older[-1]);source=observed[source_t]==other
            if ndi.label(source,np.ones((3,3),bool))[1]!=1:continue
            cc,_=ndi.label(observed[t]==name,np.ones((3,3),bool))
            component=int(cc[int(q.y+.5),int(q.x+.5)]);target=cc==component
            if not component or np.any(source&target):continue
            owners={int(r.track_id) for r in pts[pts.frame.eq(source_t)].itertuples(index=False)
                    if source[int(r.y+.5),int(r.x+.5)]}
            surviving=pts[pts.frame.eq(t)&pts.track_id.isin(owners)&pts.track_id.ne(track)]
            if not any(source[int(r.y+.5),int(r.x+.5)] for r in surviving.itertuples(index=False)):continue
        anchors.append(dict(track=int(track),identity=name,invading_identity=other,anchor_frame=t+1,left_frame=a+1,right_frame=b+1,bracket_distance_radii=distance))
    if not anchors:return result,empty
    nodes,records,groups,_=physical_paths(current_labels,observed)
    lookup=np.zeros(len(records),np.int32)
    for path,ns in groups.items():lookup[ns]=path
    pts['path']=lookup[nodes[tt,yy,xx]]
    tracks={int(tr):g.sort_values('frame') for tr,g in pts.groupby('track_id')}
    continuations={};first_paths={}
    for tr,g in tracks.items():
        positive=g[g.path.gt(0)]
        if positive.empty:continue
        first=positive.iloc[0];first_paths[tr]=int(first.path);selected=[];previous=None
        for q in g[g.frame.ge(first.frame)].itertuples(index=False):
            if previous is not None:
                dt=int(q.frame)-int(previous.frame)
                if dt<=0 or np.hypot(q.x-previous.x,q.y-previous.y)>dt*(q.radius_px+previous.radius_px):break
            if q.path:selected.append(int(q.path))
            previous=q
        continuations[tr]=set(selected)
    plans=[]
    for anchor in anchors:
        tr=anchor['track'];name=anchor['identity'];start=anchor['anchor_frame']
        selected=set(continuations.get(tr,()))
        while True:
            enlarged=selected|set().union(*(continuations[t] for t,p in first_paths.items() if p in selected))
            if returned_same_name_fragments:
                # A temporary fragment that later returns to this body has
                # no independent owner merely because its outline detached.
                # Require birth within this episode and no different native
                # or current named history. Established earlier bodies stay
                # independent even when an outline tracker recycles a name.
                for core_track,paths in continuations.items():
                    if not paths.intersection(selected):continue
                    history=tracks[core_track];named=history[history.current.gt(0)]
                    if named.empty or int(named.iloc[0].frame)+1<start:continue
                    if not set(map(int,named.current)).issubset({name,anchor['invading_identity']}):continue
                    if not set(map(int,history.loc[history.native.gt(0),'native'])).issubset({name,anchor['invading_identity']}):continue
                    enlarged|=paths
            if nearby_same_name_fragments:
                # Segmented outlines may have disconnected edge pieces. A
                # same-name piece within the measured soma's own radius is
                # part of the existing ownership partition, unless it has an
                # independently established core or another named history.
                for path, members in groups.items():
                    if path in enlarged:continue
                    if any(records[n]['frame']<start or records[n]['identity'] not in {name,anchor['invading_identity']} for n in members):continue
                    attached=True
                    for n in members:
                        t=records[n]['frame']-1;owner=records[n]['identity'];piece=nodes[t]==n
                        peers=[q for p in selected for q in groups[p] if records[q]['frame']==t+1 and records[q]['identity']==owner]
                        if not peers:attached=False;break
                        body=np.isin(nodes[t],peers)
                        cores=pts[pts.frame.eq(t)&pts.path.isin(selected)]
                        if cores.empty:attached=False;break
                        # Compare the actual edge gap with the radius of the
                        # nearest supported soma, not a fixed pixel distance.
                        gap=float(ndi.distance_transform_edt(~body)[piece].min())
                        py,px=np.argwhere(piece).mean(axis=0)
                        core=cores.iloc[np.argmin((cores.x-px)**2+(cores.y-py)**2)]
                        if gap>float(core.radius_px):attached=False;break
                        native_names=set(map(int,np.unique(proposals[t][piece])))-{0}
                        if not native_names.issubset({name,anchor['invading_identity']}):attached=False;break
                        independent=pts[pts.frame.eq(t)&pts.path.eq(path)]
                        if any(first_paths.get(int(q.track_id)) not in selected for q in independent.itertuples(index=False)):attached=False;break
                    if attached:enlarged.add(path)
            if enlarged==selected:break
            selected=enlarged
        ns=[n for p in selected for n in groups[p] if records[n]['frame']>=start]
        frames=[records[n]['frame'] for n in ns]
        if not ns:continue
        if not preserve_label_partition and len(frames)!=len(set(frames)):continue
        frame_masks={}
        if preserve_label_partition:
            invalid=False
            for f in set(frames):
                members=[n for n in ns if records[n]['frame']==f]
                owners={records[n]['identity'] for n in members}
                if len(owners)!=1:invalid=True;break
                mask=np.isin(nodes[f-1],members);owner=next(iter(owners));frame_masks[f-1]=mask
                if np.any((current_labels[f-1]==owner)&~mask):invalid=True;break
            if invalid:continue
        bad=False;changes=[]
        for n in ns:
            q=records[n];t=q['frame']-1;body=nodes[t]==n
            if np.any((current_labels[t]==name)&~frame_masks.get(t,body)):bad=True;break
            change=body&(current_labels[t]!=name)
            if np.any(change&(current_labels[t]!=original_labels[t])):bad=True;break
            # Distinct, separated measured cores inside a merged outline
            # require partitioning; this correction does not rename a merge.
            inside=pts[pts.frame.eq(t)].loc[lambda g:g.path.eq(lookup[n])]
            core=list(inside.itertuples(index=False))
            separated=any(np.hypot(a.x-b.x,a.y-b.y)>a.radius_px+b.radius_px
                and (not independent_core_history or first_paths.get(int(a.track_id)) not in selected or first_paths.get(int(b.track_id)) not in selected)
                for i,a in enumerate(core) for b in core[i+1:])
            if separated:bad=True;break
            if change.any():changes.append((n,int(change.sum())))
        if not bad and changes:plans.append((anchor,changes))
    claims={}
    for a,changes in plans:
        for n,_ in changes:claims.setdefault(n,set()).add(a['identity'])
    audit=[];done=set()
    for a,changes in plans:
        if any(len(claims[n])!=1 for n,_ in changes):continue
        for n,count in changes:
            if n in done:continue
            done.add(n);q=records[n];t=q['frame']-1;result[t][nodes[t]==n]=a['identity']
            audit.append(dict(a,path=int(lookup[n]),node=n,frame=t+1,source_identity=q['identity'],seed_y=q['seed_y'],seed_x=q['seed_x'],changed_pixels=count,decision='measured_motion_bracket_retains_core_ownership'))
    return result,pd.DataFrame(audit,columns=COLUMNS)
