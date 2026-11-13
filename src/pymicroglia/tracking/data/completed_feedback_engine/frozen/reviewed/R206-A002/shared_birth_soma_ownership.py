"""Keep an independently measured soma when a shared outline gets a new name.

Motion-object memberships may be copied to both sides of a split. They do not
select an owner here. Repeated observed raw cores, their positions and the
complete reciprocal outline paths supply the ownership evidence instead.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from optical_basin_soma_paths import physical_paths

COLUMNS=['node','identity','frame','seed_y','seed_x','source_identity',
         'changed_pixels','contact_frame','core_track','decision']


def preserve_shared_birth_owners(observed,proposals,raw,points,params=None,*,
                                proposal_history=None,current_labels,original_labels):
    labels=np.asarray(current_labels)
    if not labels.shape==observed.shape==proposals.shape==raw.shape==original_labels.shape:
        raise ValueError('Aligned complete movies required')
    nodes,records,paths,_=physical_paths(labels,observed)
    path_of=np.zeros(len(records),np.int32)
    for path,ns in paths.items():path_of[ns]=path
    pts=points[points.strong.astype(bool)&points.state.eq('observed')].copy()
    if 'physically_visible' in pts:pts=pts[pts.physically_visible.astype(bool)]
    t=pts.frame.to_numpy(int);y=np.floor(pts.y+.5).astype(int);x=np.floor(pts.x+.5).astype(int)
    valid=(t>=0)&(t<len(labels))&(y>=0)&(y<labels.shape[1])&(x>=0)&(x<labels.shape[2])
    pts=pts[valid].copy();t=t[valid];y=y[valid];x=x[valid]
    pts['node']=nodes[t,y,x]
    # Motion outlines can have an empty centre. Associate such an observed raw
    # core only with its uniquely nearest outline within its measured radius.
    for index,p in pts[pts.node.eq(0)].iterrows():
        radius=float(p.radius_px);f=int(p.frame)
        y0=max(0,int(np.floor(p.y-radius)));y1=min(labels.shape[1],int(np.ceil(p.y+radius))+1)
        x0=max(0,int(np.floor(p.x-radius)));x1=min(labels.shape[2],int(np.ceil(p.x+radius))+1)
        field=nodes[f,y0:y1,x0:x1];yy,xx=np.indices(field.shape)
        distance=(yy+y0-p.y)**2+(xx+x0-p.x)**2
        hit=(field>0)&(distance<=radius**2)
        if not hit.any():continue
        nearest=distance[hit].min();choices=np.unique(field[hit&(distance==nearest)])
        if len(choices)==1:pts.loc[index,'node']=int(choices[0])
    pts['path']=path_of[pts.node.to_numpy(int)]
    pts['owner']=[records[int(n)]['identity'] if n else 0 for n in pts.node]
    by_track={int(k):g.sort_values('frame') for k,g in pts.groupby('track_id')}
    by_node={int(k):g for k,g in pts[pts.node.ne(0)].groupby('node')}
    births={};by_name={}
    for q in records[1:]:
        births[q['identity']]=min(births.get(q['identity'],q['frame']),q['frame'])
        by_name.setdefault((q['frame'],q['identity']),set()).add(q['node'])
    pixels={}
    def indices(n):
        if n not in pixels:
            q=records[n];pixels[n]=np.flatnonzero(nodes[q['frame']-1].ravel()==n)
        return pixels[n]
    def continuous(g):
        a=g[['frame','x','y','radius_px']].to_numpy(float)
        return all(np.linalg.norm(q[1:3]-p[1:3])<=(q[0]-p[0])*(q[3]+p[3])
                   for p,q in zip(a,a[1:]))
    def row(n,owner,contact,track,decision,changed=0):
        q=records[n]
        return dict(node=n,identity=owner,frame=q['frame'],seed_y=q['seed_y'],
                    seed_x=q['seed_x'],source_identity=q['identity'],
                    changed_pixels=changed,contact_frame=contact+1,
                    core_track=track,decision=decision)
    plans=[];audit=[]
    for track,g in by_track.items():
        measured=g[g.node.ne(0)]
        if measured.empty or not continuous(g):continue
        for p in measured.itertuples():
            n=int(p.node);f=int(p.frame);new=int(p.owner);path=int(p.path)
            if births[new]!=f+1:continue
            peers=by_node[n]
            if peers.track_id.nunique()<2:continue
            prefix=measured[measured.frame.lt(f)]
            if prefix.empty:continue
            old=int(prefix.iloc[-1].owner)
            if old==new or not old:continue
            prefix=prefix[prefix.owner.eq(old)&prefix.path.eq(path)]
            if prefix.empty:continue
            independent=prefix[prefix.node.map(lambda k:by_node[int(k)].track_id.nunique()==1)]
            if independent.frame.nunique()<2:continue
            # This boundary covers a new name born inside a pre-existing soma's
            # shared outline. It does not reinterpret an established neighbour.
            competing=[]
            for other in peers.track_id.unique():
                if int(other)==track:continue
                h=by_track[int(other)]
                earlier=h[h.frame.lt(f)&h.owner.ne(0)]
                if earlier.frame.nunique()>=2:competing.append(int(other))
            if competing:continue
            cohort=set();shared={}
            for q in measured[measured.frame.ge(f)&measured.path.eq(path)].itertuples():
                other=by_node[int(q.node)]
                other=other[other.track_id.ne(track)].copy()
                if other.empty:continue
                separation=np.hypot(other.x-q.x,other.y-q.y)
                other=other[separation>(other.radius_px+q.radius_px)]
                if len(other):
                    cohort.update(map(int,other.track_id));shared[int(q.node)]=(q,other)
            if not cohort:continue
            # The other measured somata must actually separate into independently
            # outlined bodies. A second peak that never separates is not a cell.
            children=set()
            for other in cohort:
                h=by_track[other]
                if not continuous(h):continue
                tails=h[h.frame.ge(f)&h.node.ne(0)&h.path.ne(path)]
                children.update(map(int,tails.path))
            if not children:continue
            chosen={int(k) for k in paths[path] if records[k]['frame']>=f+1}
            for child in children:
                chosen.update(int(k) for k in paths[child] if records[k]['frame']>=f+1)
            assignment={k:np.full(len(indices(k)),old if path_of[k]==path else new,labels.dtype)
                        for k in chosen}
            failed=None
            for k,(main,others) in shared.items():
                frame=records[k]['frame']-1;idx=indices(k);yy,xx=np.unravel_index(idx,labels.shape[1:])
                y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
                body=nodes[frame,y0:y1,x0:x1]==k;markers=np.zeros(body.shape,np.int32)
                for marker,cores in [(1,[main]),(2,list(others.itertuples()))]:
                    for core in cores:
                        distances=(yy-core.y)**2+(xx-core.x)**2;minimum=distances.min()
                        if minimum>core.radius_px**2:failed='raw_core_outside_own_radius';break
                        at=np.flatnonzero(distances==minimum)
                        if len(at)!=1:failed='ambiguous_partition_seed';break
                        j=int(at[0]);cy,cx=int(yy[j]-y0),int(xx[j]-x0)
                        if markers[cy,cx] not in (0,marker):failed='coincident_partition_seeds';break
                        markers[cy,cx]=marker
                    if failed:break
                if failed:break
                partition=watershed(-raw[frame,y0:y1,x0:x1].astype(float),markers,mask=body,connectivity=2)
                if not np.all(partition[body]>0):failed='unpartitioned_shared_body';break
                assignment[k]=np.where(partition[body]==1,old,new).astype(labels.dtype)
            if failed:
                audit.append(row(n,old,f,track,failed));continue
            # Reserve each physical owner through the complete proposal. Resolve
            # borrowed holders only from their own independent measured history.
            wanted={(records[k]['frame'],int(owner)) for k,values in assignment.items() for owner in np.unique(values)}
            outsiders=set().union(*(by_name.get(key,set())-set(assignment) for key in wanted))
            for k in sorted(outsiders):
                q=records[k];cores=by_node.get(k)
                if cores is None:
                    # A core-free nearby remnant can belong to a selected branch;
                    # require one nearest owner within that branch's area radius.
                    t=q['frame']-1;mask=nodes[t]==k;distance=ndi.distance_transform_edt(~mask);choices=[]
                    for s,values in assignment.items():
                        if records[s]['frame']!=q['frame']:continue
                        si=indices(s);owners=np.unique(values)
                        for owner in owners:
                            sub=si[values==owner];gap=float(distance.ravel()[sub].min())
                            scale=np.sqrt(len(sub)/np.pi)
                            if gap<=scale:choices.append((gap/scale,int(owner)))
                    if not choices:failed='unresolved_core_free_holder';break
                    best=min(x[0] for x in choices);owners={v for d,v in choices if d==best}
                    if len(owners)!=1:failed='ambiguous_fragment_owner';break
                    assignment[k]=np.full(len(indices(k)),owners.pop(),labels.dtype)
                    continue
                alternatives=set()
                for other in cores.track_id.unique():
                    h=by_track[int(other)]
                    if not continuous(h):continue
                    h=h[h.node.ne(0)&~h.owner.isin([old,new])]
                    for candidate in h.itertuples():
                        if by_node[int(candidate.node)].track_id.nunique()==1:
                            alternatives.add(int(candidate.owner))
                if len(alternatives)!=1:failed='independent_holder_has_no_unique_own_name';break
                owner=next(iter(alternatives))
                if by_name.get((q['frame'],owner),set())-set(assignment)-{k}:
                    failed='alternative_name_is_occupied';break
                assignment[k]=np.full(len(indices(k)),owner,labels.dtype)
            if not failed:
                for k,values in assignment.items():
                    t=records[k]['frame']-1;idx=indices(k);before=labels[t].ravel()[idx]
                    if np.any((values!=before)&(before!=original_labels[t].ravel()[idx])):
                        failed='protected_prior_revision';break
            if failed:audit.append(row(n,old,f,track,failed));continue
            plans.append((f,track,assignment));break
    claims={}
    for i,(_,_,assignment) in enumerate(plans):
        for n,values in assignment.items():claims.setdefault(n,[]).append((i,values))
    rejected=set()
    for entries in claims.values():
        if any(not np.array_equal(entries[0][1],v) for _,v in entries[1:]):rejected.update(i for i,_ in entries)
    result=labels.copy()
    for i,(f,track,assignment) in enumerate(plans):
        if i in rejected:continue
        for n,values in assignment.items():
            q=records[n];t=q['frame']-1;idx=indices(n);before=labels[t].ravel()[idx]
            result[t].ravel()[idx]=values
            for owner in np.unique(values):
                changed=int(np.count_nonzero((values==owner)&(before!=owner)))
                if changed:audit.append(row(n,int(owner),f,track,'measured_soma_survives_shared_birth',changed))
    return result,pd.DataFrame(audit,columns=COLUMNS)
