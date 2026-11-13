"""Conserve a measured body's name through contact and remote recovery theft.

A remote pre-existing unnamed core must not consume the name of a body entering
a neighbour. A later fresh branch identifies the name released by that contact.
Exchange those names together; leave ambiguous shared outlines unchanged.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from optical_basin_soma_paths import physical_paths

COLUMNS=['node','identity','frame','seed_y','seed_x','source_identity','changed_pixels','contact_frame','release_frame','decision']

def reserve_contact_identity(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels,core_witnessed_exits=False):
    labels=current_labels; result=labels.copy(); empty=pd.DataFrame(columns=COLUMNS)
    if not labels.shape==observed.shape==proposals.shape==raw.shape==original_labels.shape:raise ValueError('Aligned movies required')
    nodes,records,paths,_=physical_paths(labels,observed)
    path_of=np.zeros(len(records),np.int32)
    for path,ns in paths.items():path_of[ns]=path
    pts=points[points.strong.astype(bool)&points.state.eq('observed')].copy()
    if 'physically_visible' in pts:pts=pts[pts.physically_visible.astype(bool)]
    t=pts.frame.to_numpy(int);y=np.floor(pts.y+.5).astype(int);x=np.floor(pts.x+.5).astype(int)
    good=(t>=0)&(t<len(labels))&(y>=0)&(y<labels.shape[1])&(x>=0)&(x<labels.shape[2]);pts=pts[good].copy();t=t[good];y=y[good];x=x[good]
    pts['node']=nodes[t,y,x];pts['owner']=labels[t,y,x];pts['path']=path_of[pts.node.to_numpy(int)]
    frame_nodes={f:[] for f in range(len(labels))};by_name={};boxes={};centers={};radius={}
    for q in records[1:]:
        n=q['node'];f=q['frame']-1;frame_nodes[f].append(n);by_name.setdefault((f,q['identity']),[]).append(n)
    for f,frame in enumerate(nodes):
        for n,box in enumerate(ndi.find_objects(frame),1):
            if box is None:continue
            boxes[n]=box;yy,xx=np.nonzero(frame[box]==n);centers[n]=np.array([yy.mean()+box[0].start,xx.mean()+box[1].start]);radius[n]=np.sqrt(len(yy)/np.pi)
    def gap(a,b):
        ba,bb=boxes[a],boxes[b];ta=records[a]['frame']-1;tb=records[b]['frame']-1
        y0=min(ba[0].start,bb[0].start);y1=max(ba[0].stop,bb[0].stop);x0=min(ba[1].start,bb[1].start);x1=max(ba[1].stop,bb[1].stop)
        ma=nodes[ta,y0:y1,x0:x1]==a;mb=nodes[tb,y0:y1,x0:x1]==b
        return float(ndi.distance_transform_edt(~ma)[mb].min())
    first={name:min(f for (f,k) in by_name if k==name) for name in {k for f,k in by_name}}
    plans=[]
    for f in range(1,len(labels)):
        for a in frame_nodes[f-1]:
            name=records[a]['identity'];remote=by_name.get((f,name),[])
            if len(remote)!=1 or len(by_name[(f-1,name)])!=1:continue
            remote=remote[0]
            if np.linalg.norm(centers[a]-centers[remote])<=radius[a]+radius[remote]:continue
            if gap(a,remote)<=radius[a]+radius[remote]:continue
            old_history=pts[pts.path.eq(path_of[a])&pts.frame.lt(f)&pts.owner.eq(name)]
            if old_history.frame.nunique()<2:continue
            remote_points=pts[pts.node.eq(remote)]
            if remote_points.track_id.nunique()!=1:continue
            track=int(remote_points.iloc[0].track_id);history=pts[pts.track_id.eq(track)];prefix=history[history.frame.lt(f)]
            if prefix.frame.nunique()<2 or prefix.owner.ne(0).any():continue
            # The new holder existed independently before this name arrived.
            # Its strong-core positions must continue without an unbounded jump.
            ordered=history.sort_values('frame');arr=ordered[['frame','x','y','radius_px']].to_numpy()
            if any(np.linalg.norm(b[1:3]-a0[1:3])>(b[0]-a0[0])*(b[3]+a0[3]) for a0,b in zip(arr,arr[1:])):continue
            neighbours=[]
            for b in frame_nodes[f]:
                other=records[b]['identity'];previous=by_name.get((f-1,other),[])
                if other==name or len(previous)!=1 or len(by_name[(f,other)])!=1:continue
                old=previous[0]
                if path_of[old]!=path_of[b] or not len(pts[pts.node.eq(b)]):continue
                ba,bb=boxes[a],boxes[b]
                dy=max(0,ba[0].start-bb[0].stop+1,bb[0].start-ba[0].stop+1);dx=max(0,ba[1].start-bb[1].stop+1,bb[1].start-ba[1].stop+1)
                if np.hypot(dx,dy)>radius[a]+radius[b]:continue
                distance=gap(a,b)/(radius[a]+radius[b])
                if distance>1:continue
                combined=records[a]['proposed_pixels']+records[old]['proposed_pixels'];size=records[b]['proposed_pixels']
                if abs(size-combined)>=abs(size-records[old]['proposed_pixels']):continue
                neighbours.append((distance,b,old))
            if not neighbours:continue
            neighbours.sort()
            if len(neighbours)>1 and neighbours[0][0]==neighbours[1][0]:continue
            _,merged,old_other=neighbours[0];other=records[merged]['identity'];chain=paths[path_of[merged]]
            releases=[]
            for b in chain:
                at=records[b]['frame']-1
                if at<f:continue
                previous=by_name.get((at-1,other),[])
                if len(previous)!=1:continue
                prior=previous[0]
                for c in frame_nodes[at]:
                    new=records[c]['identity']
                    if new in {name,other} or first[new]!=at:continue
                    if path_of[c]==path_of[b] or gap(prior,c)!=0:continue
                    if not len(pts[pts.node.eq(c)]):continue
                    # The two exit bodies preserve the entering side ordering.
                    if not core_witnessed_exits and np.dot(centers[a]-centers[old_other],centers[c]-centers[b])<=0:continue
                    size=records[prior]['proposed_pixels'];combined=records[b]['proposed_pixels']+records[c]['proposed_pixels']
                    if abs(size-combined)>=abs(size-records[b]['proposed_pixels']):continue
                    releases.append((at,c,b))
            if not releases:continue
            at,c,b=min(releases);same_time=[r for r in releases if r[0]==at]
            if len(same_time)!=1:continue
            new=records[c]['identity'];child=paths[path_of[c]]
            if any(records[n]['identity']!=new for n in child):continue
            retained=[];child_target=name
            if core_witnessed_exits:
                from contact_core_continuity import witnessed_exit,continue_measured_path
                exit_node=witnessed_exit(a,old_other,merged,b,c,f,at,pts,centers,radius,records,path_of)
                if exit_node is None:continue
                if exit_node==b:
                    child_target=other
                    retained=[n for n in continue_measured_path(path_of[b],paths,records,radius,gap,pts)
                              if records[n]['frame']-1>=at]
            # Retain complete old ownership groups and all earlier revisions.
            changes=[];bad=False
            for n in child:
                q=records[n];t=q['frame']-1
                if len(by_name[(t,new)])!=1:bad=True;break
                changes.append((n,child_target))
            if bad:continue
            satellites={};child_at={records[n]['frame']:n for n in child}
            for n in retained:
                q=records[n];t=q['frame']-1
                for peer in by_name[(t,q['identity'])]:
                    if peer==n or peer in retained:continue
                    # Keep a close disconnected outline fragment with its
                    # soma, without absorbing an independently measured core.
                    if len(pts[pts.node.eq(peer)]):bad=True;break
                    anchors=[(gap(n,peer)/radius[n],name)]
                    companion=child_at.get(q['frame'])
                    if companion is not None:
                        anchors.append((gap(companion,peer)/radius[companion],child_target))
                    anchors.sort()
                    if anchors[0][0]>1 or (len(anchors)>1 and anchors[0][0]==anchors[1][0]):
                        bad=True;break
                    satellites[peer]=anchors[0][1]
                if bad:break
                changes.append((n,name))
            if bad:continue
            retained.extend(n for n in satellites if n not in retained)
            changes.extend(satellites.items())
            for (t,k),nn in by_name.items():
                if k!=name or t<f:continue
                if len(nn)!=1:bad=True;break
                n=nn[0]
                if t<ordered.frame.min() or t>ordered.frame.max():bad=True;break
                cy=np.interp(t,arr[:,0],arr[:,2]);cx=np.interp(t,arr[:,0],arr[:,1]);cr=np.interp(t,arr[:,0],arr[:,3])
                if np.linalg.norm(centers[n]-[cy,cx])>cr+radius[n]:bad=True;break
                changes.append((n,new))
            if bad:continue
            for n,target in changes:
                q=records[n];t=q['frame']-1;mask=nodes[t]==n
                if n not in retained and np.any(mask&(labels[t]!=original_labels[t])):bad=True;break
            if not bad and core_witnessed_exits:
                # Account for both displaced owners: no recipient can keep an
                # additional holder outside this coupled physical correction.
                selected={n for n,_ in changes}
                for n,target in changes:
                    t=records[n]['frame']-1
                    if any(k not in selected for k in by_name.get((t,target),[])):
                        bad=True;break
            if not bad:plans.append((f,at,changes))
    claims={}
    for f,at,changes in plans:
        for n,target in changes:claims.setdefault(n,set()).add(target)
    audit=[];done=set()
    for f,at,changes in plans:
        if any(len(claims[n])!=1 for n,target in changes):continue
        for n,target in changes:
            if n in done:continue
            done.add(n);q=records[n];t=q['frame']-1;result[t][nodes[t]==n]=target
            audit.append(dict(node=n,identity=target,frame=t+1,seed_y=q['seed_y'],seed_x=q['seed_x'],source_identity=q['identity'],changed_pixels=q['proposed_pixels'] if q['identity']!=target else 0,contact_frame=f+1,release_frame=at+1,decision='measured_core_reserves_both_contact_owners' if core_witnessed_exits else 'reserve_entering_name_through_contact_and_remote_recovery'))
    return result,pd.DataFrame(audit,columns=COLUMNS)


def reserve_contact_identity_by_core(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels):
    """Use a measured continuous core instead of the rejected side-order premise."""
    return reserve_contact_identity(observed,proposals,raw,points,params,proposal_history=proposal_history,current_labels=current_labels,original_labels=original_labels,core_witnessed_exits=True)
