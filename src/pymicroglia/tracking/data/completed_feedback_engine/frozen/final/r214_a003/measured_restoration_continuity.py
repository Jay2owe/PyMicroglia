"""Continue an established ownership restoration along a measured soma.

No recording, name, frame, location or feedback selector is used. The original
ownership operation still decides restorations and resolves its proposals.
"""
from contextlib import contextmanager
import numpy as np
from scipy import ndimage as ndi


def geometry_by_frame(points):
    p=points[points.state.eq('observed') & points.strong.astype(bool)].copy()
    if 'physically_visible' in p:
        p=p[p.physically_visible.astype(bool)]
    values=p[['frame','y','x','radius_px']].to_numpy(float)
    p=p[np.isfinite(values).all(axis=1)&(values[:,3]>0)]
    return {int(f):g[['y','x','radius_px']].to_numpy(float) for f,g in p.groupby('frame')}


class RestoredSomaContinuity:
    def __init__(self, geometry, events):
        self.geometry=geometry
        self.events=events
        self.seeds=[]

    def cores(self,frame):
        return self.geometry.get(frame,np.empty((0,3)))

    @staticmethod
    def pixels(points,shape):
        y,x=np.floor(points[:,:2]+.5).astype(int).T
        valid=(y>=0)&(y<shape[0])&(x>=0)&(x<shape[1])
        return y,x,valid

    def record(self,before,after,frame,actions):
        # Seed only actual accepted red-growth restorations from this operation.
        # A numerical name's mere presence or first appearance proves nothing.
        owners={int(a['assigned_identity']) for a in actions
                if a.get('action')=='connected_red_growth' and a.get('changed_pixels',0)>0}
        cores=self.cores(frame)
        y,x,valid=self.pixels(cores,after.shape)
        for owner in owners:
            parts,_=ndi.label(after==owner,np.ones((3,3),bool))
            changed=(before!=after)&(after==owner)
            for part in np.unique(parts[changed]):
                if not part:continue
                hit=np.flatnonzero(valid & (parts[np.clip(y,0,after.shape[0]-1),np.clip(x,0,after.shape[1]-1)]==part))
                if len(hit)!=1:continue
                core=cores[hit[0]].copy()
                if any(s['owner']==owner and s['frame']==frame and np.array_equal(s['core'],core) for s in self.seeds):
                    continue
                self.seeds.append(dict(owner=owner,frame=frame,core=core,origin_frame=frame,origin_core=core.copy()))

    def propose(self,previous,current,frame,existing):
        source=self.cores(frame-1);target=self.cores(frame)
        if not len(source) or not len(target):
            self.seeds=[]
            return existing
        costs=((source[:,None,:2]-target[None,:,:2])**2).sum(axis=2)/(source[:,None,2]+target[None,:,2])**2
        row_min=costs.min(axis=1);col_min=costs.min(axis=0)
        near_source=np.isclose(costs,row_min[:,None],rtol=16*np.finfo(float).eps,atol=0.)
        near_target=np.isclose(costs,col_min[None,:],rtol=16*np.finfo(float).eps,atol=0.)
        ty,tx,valid=self.pixels(target,current.shape)
        updated=[];proposals=[];parts_cache={}
        for seed in self.seeds:
            if seed['frame']!=frame-1:continue
            indices=np.flatnonzero(np.all(source==seed['core'],axis=1))
            if len(indices)!=1:continue
            i=int(indices[0]);matches=np.flatnonzero(near_source[i])
            if len(matches)!=1:continue
            j=int(matches[0])
            if near_target[:,j].sum()!=1 or not near_target[i,j]:continue
            if costs[i,j]>1 and not np.isclose(costs[i,j],1.,rtol=16*np.finfo(float).eps,atol=0.):continue
            owner=int(seed['owner'])
            updated.append({**seed,'frame':frame,'core':target[j].copy()})
            audit=dict(frame=frame+1,owner=owner,origin_frame=seed['origin_frame']+1,origin_core=seed['origin_core'].tolist(),source_core=source[i].tolist(),target_core=target[j].tolist(),normalized_squared_displacement=float(costs[i,j]))
            if not valid[j]:continue
            old=int(current[ty[j],tx[j]])
            if not old:
                self.events.append({**audit,'decision':'measured_soma_without_named_outline'})
                continue
            if old==owner:
                continue
            if old not in parts_cache:
                parts_cache[old]=ndi.label(current==old,np.ones((3,3),bool))[0]
            parts=parts_cache[old];number=int(parts[ty[j],tx[j]]);body=parts==number
            inside=np.zeros(len(target),bool);inside[valid]=body[ty[valid],tx[valid]]
            if inside.sum()!=1:
                self.events.append({**audit,'decision':'shared_measured_outline_held'})
                continue
            holders=np.zeros(len(target),bool);holders[valid]=current[ty[valid],tx[valid]]==owner
            if np.any(holders & ~inside):
                self.events.append({**audit,'decision':'independent_measured_name_holder_preserved'})
                continue
            if any(np.any(body[tuple(p['pixels'].T)]) for p in existing):
                self.events.append({**audit,'decision':'existing_ownership_proposal_preserved'})
                continue
            proposals.append(dict(pixels=np.column_stack(np.nonzero(body)),owner=owner,
                original_identity=old,action='measured_soma_restoration_continued',
                tenure_mode='measured_soma_continuation',distance_px=float(np.linalg.norm(source[i,:2]-target[j,:2])),
                _component_key=(old,number),_audit=audit))
        self.seeds=updated
        # Simultaneous rival restoration histories are ambiguous; never break
        # such a tie by name, insertion order, or current area.
        choices={}
        for p in proposals:choices.setdefault(p['_component_key'],set()).add(p['owner'])
        result=list(existing);seen=set()
        for p in proposals:
            key=p.pop('_component_key');audit=p.pop('_audit')
            if len(choices[key])!=1:
                self.events.append({**audit,'decision':'competing_restoration_histories_held'})
                continue
            if key in seen:continue
            seen.add(key);result.append(p)
            self.events.append({**audit,'decision':'measured_continuation_proposed','pixels':len(p['pixels'])})
        return result


@contextmanager
def measured_restoration_continuity(ownership,points,events=None,*,source_offset=2):
    if events is None:events=[]
    continuity=RestoredSomaContinuity(geometry_by_frame(points),events)
    original_overlap=ownership.overlap_proposals
    original_apply=ownership.apply_proposals
    frame=0
    def overlap(previous,current,lag,cfg,lookup):
        nonlocal frame
        frame+=1
        existing=original_overlap(previous,current,lag,cfg,lookup)
        return continuity.propose(previous,current,frame,existing)
    def apply(current,proposals,method,cfg,source_frame):
        result,actions=original_apply(current,proposals,method,cfg,source_frame)
        if method=='R':
            f=int(source_frame)-int(source_offset)-1
            if f!=frame:raise ValueError('Ownership source frames do not align with measured soma frames')
            continuity.record(current,result,f,actions)
        return result,actions
    ownership.overlap_proposals=overlap;ownership.apply_proposals=apply
    try:yield events
    finally:
        ownership.overlap_proposals=original_overlap
        ownership.apply_proposals=original_apply
