"""Keep both measured owners when a shared outline inherits the wrong name.

Eligibility is geometric: two overlapping predecessors explain a shared area
better than either alone; measured cores witness both separate outgoing paths.
No recording, feedback, identity value or image region selects a correction.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from optical_basin_soma_paths import physical_paths

COLUMNS=['node','identity','frame','seed_y','seed_x','source_identity','changed_pixels','contact_frame','release_frame','decision']


def preserve_precontact_owners(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels):
    labels=current_labels
    if not labels.shape==observed.shape==proposals.shape==raw.shape==original_labels.shape:
        raise ValueError('Aligned full movies required')
    nodes,records,paths,_=physical_paths(labels,observed)
    path_of=np.zeros(len(records),np.int32)
    for path,ns in paths.items():path_of[ns]=path
    pts=points[points.strong.astype(bool)&points.state.eq('observed')].copy()
    if 'physically_visible' in pts:pts=pts[pts.physically_visible.astype(bool)]
    t=pts.frame.to_numpy(int);y=np.floor(pts.y+.5).astype(int);x=np.floor(pts.x+.5).astype(int)
    valid=(t>=0)&(t<len(labels))&(y>=0)&(y<labels.shape[1])&(x>=0)&(x<labels.shape[2])
    pts=pts[valid].copy();t=t[valid];y=y[valid];x=x[valid]
    pts['node']=nodes[t,y,x];pts['owner']=labels[t,y,x];pts['path']=path_of[pts.node.to_numpy(int)]
    by_track={int(k):q.sort_values('frame') for k,q in pts.groupby('track_id')}
    by_name={}
    for q in records[1:]:by_name.setdefault((q['frame'],q['identity']),set()).add(q['node'])
    def witnessed_tracks(n,f):
        q=pts[pts.path.eq(path_of[n])&pts.owner.eq(records[n]['identity'])&pts.frame.lt(f)]
        eligible=[]
        for track,h in q.groupby('track_id'):
            if h.frame.nunique()<2:continue
            full=by_track[int(track)];prefix=full[full.frame.lt(f)]
            if int(prefix.iloc[-1].path)!=path_of[n] or int(prefix.iloc[-1].owner)!=records[n]['identity']:continue
            # An expired core is not a competing witness to the contact exit.
            # Require an actually observed segmented body on the far side.
            if not len(full[full.frame.ge(f)&full.node.ne(0)]):continue
            eligible.append(int(track))
        return eligible
    plans=[]
    for f in range(1,len(labels)):
        hit=(nodes[f-1]>0)&(nodes[f]>0)
        pairs=np.unique(np.stack((nodes[f-1][hit],nodes[f][hit]),axis=1),axis=0)
        incoming={}
        for a,b in pairs:incoming.setdefault(int(b),[]).append(int(a))
        for merged,old in incoming.items():
            # Incidental edge overlap is not entry into a contact when that
            # neighbour retains its own independent reciprocal successor.
            old=[n for n in old if path_of[n]==path_of[merged] or
                 not any(records[k]['frame']==f+1 for k in paths[path_of[n]])]
            if len(old)!=2:continue
            retained=[n for n in old if path_of[n]==path_of[merged]]
            if len(retained)!=1:continue
            other=retained[0];donor=next(n for n in old if n!=other)
            name=records[donor]['identity'];other_name=records[other]['identity']
            if name==other_name or records[merged]['identity']!=name:continue
            area=records[merged]['proposed_pixels'];sizes=[records[n]['proposed_pixels'] for n in old]
            if abs(area-sum(sizes))>=min(abs(area-s) for s in sizes):continue
            tracks=[witnessed_tracks(n,f) for n in [donor,other]]
            if any(len(q)!=1 for q in tracks) or tracks[0]==tracks[1]:continue
            donor_track,other_track=[by_track[q[0]] for q in tracks]
            exits=donor_track[donor_track.frame.ge(f)&donor_track.node.ne(0)&donor_track.path.ne(path_of[merged])]
            if exits.empty:continue
            exit_row=exits.iloc[0];child_path=int(exit_row.path);child=paths[child_path]
            release=records[child[0]]['frame']-1
            if release<f or release==0:continue
            # The outgoing child must originate in this shared outline.
            first=child[0];before=nodes[release-1][nodes[release]==first]
            preceding=set(map(int,np.unique(before[before>0])))
            if preceding:
                if len(preceding)!=1:continue
                prior=next(iter(preceding))
                if path_of[prior]!=path_of[merged]:continue
            else:
                # A moving soma may separate without pixel overlap. Its
                # measured track still supplies the link; bound the outline
                # gap by the two observed body sizes, not a pixel cutoff.
                prior_nodes=[n for n in paths[path_of[merged]] if records[n]['frame']==release]
                if len(prior_nodes)!=1:continue
                prior=prior_nodes[0];a=nodes[release-1]==prior;b=nodes[release]==first
                yy,xx=np.nonzero(a|b);box=(slice(yy.min(),yy.max()+1),slice(xx.min(),xx.max()+1))
                distance=float(ndi.distance_transform_edt(~a[box])[b[box]].min())
                if distance>np.sqrt(a.sum()/np.pi)+np.sqrt(b.sum()/np.pi):continue
            separation=int(exit_row.frame)
            second=other_track[other_track.frame.ge(separation)&other_track.node.ne(0)]
            if second.empty or int(second.iloc[0].path)!=path_of[merged]:continue
            witnessed=True
            for history,tail in [(donor_track,exit_row),(other_track,second.iloc[0])]:
                prefix=history[history.frame.lt(f)].iloc[-1]
                interval=history[history.frame.between(prefix.frame,tail.frame)]
                if interval.path[interval.node.ne(0)].isin([path_of[donor],path_of[merged],child_path]).all()==False:
                    witnessed=False;break
                rows=interval[['frame','x','y','radius_px']].to_numpy()
                if any(np.linalg.norm(b[1:3]-a[1:3])>(b[0]-a[0])*(a[3]+b[3]) for a,b in zip(rows,rows[1:])):
                    witnessed=False;break
            if not witnessed:continue
            # A small detached prefix can precede physical separation. Keep
            # the still-shared outline unchanged until the entering soma is
            # actually measured in the independent outgoing body.
            retained_tail=[n for n in paths[path_of[merged]] if records[n]['frame']-1>=separation]
            child=[n for n in child if records[n]['frame']-1>=separation]
            # A later re-merger of the same pair ends independent ownership
            # evidence. A single-contact correction cannot carry one owner's
            # name over the combined body and silently erase its neighbour.
            repeated=False
            for left,right in [(child,retained_tail),(retained_tail,child)]:
                end=left[-1];at=records[end]['frame'];following=[n for n in right if records[n]['frame']==at+1]
                if following and np.any((nodes[at-1]==end)&(nodes[at]==following[0])):
                    repeated=True;break
            if repeated:continue
            changes=[(n,name) for n in child]+[(n,other_name) for n in retained_tail]
            selected={n for n,_ in changes}
            outsiders=set().union(*(by_name.get((records[n]['frame'],target),set())-selected for n,target in changes))
            # Resolve a displaced temporary holder only from its own observed
            # continuation, or an unmeasured adjacent fragment of a measured
            # body. No new identity is invented to free a contested name.
            failed=False;extra={};released={records[n]['identity'] for n,target in changes if records[n]['identity']!=target}-{name,other_name}
            for n in outsiders:
                q=records[n];t=q['frame']-1;core=pts[pts.node.eq(n)]
                chain=paths[path_of[n]];alternatives={records[k]['identity'] for k in chain}-{name,other_name}
                if len(core) and len(alternatives)==1:
                    target=next(iter(alternatives))
                    if target not in released or by_name.get((t+1,target),set())-selected:
                        failed=True;break
                    extra[n]=target
                elif not len(core) and not alternatives:
                    mask=nodes[t]==n;dist=ndi.distance_transform_edt(~mask);choices=[]
                    for k in np.unique(pts.loc[pts.frame.eq(t),'node']):
                        k=int(k)
                        if not k or k in selected or k in outsiders:continue
                        body=nodes[t]==k;score=float(dist[body].min()/np.sqrt(body.sum()/np.pi))
                        if score<=1:choices.append((score,k))
                    choices.sort()
                    if not choices or (len(choices)>1 and choices[0][0]==choices[1][0]):failed=True;break
                    extra[n]=records[choices[0][1]]['identity']
                else:failed=True;break
            if failed:continue
            changes.extend(extra.items());selected.update(extra)
            # The only allowed additional same-name holder is the measured
            # parent of an adjacent core-free fragment just qualified above.
            if any(by_name.get((records[n]['frame'],target),set())-selected for n,target in changes if n not in extra):continue
            plans.append((f,separation,changes))
    claims={}
    for _,_,changes in plans:
        for n,target in changes:claims.setdefault(n,set()).add(target)
    result=labels.copy();audit=[];done=set()
    for contact,release,changes in plans:
        if any(len(claims[n])!=1 for n,_ in changes):continue
        for n,target in changes:
            if n in done:continue
            done.add(n);q=records[n];t=q['frame']-1
            result[t][nodes[t]==n]=target
            audit.append(dict(node=n,identity=target,frame=t+1,seed_y=q['seed_y'],seed_x=q['seed_x'],source_identity=q['identity'],changed_pixels=q['proposed_pixels'] if target!=q['identity'] else 0,contact_frame=contact+1,release_frame=release+1,decision='two_measured_owners_survive_shared_outline'))
    return result,pd.DataFrame(audit,columns=COLUMNS)
