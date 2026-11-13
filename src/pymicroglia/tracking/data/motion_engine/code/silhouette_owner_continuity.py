"""Retain measured cell ownership across detector-fragment name excursions.

The independent evidence unit is a reciprocal silhouette path, not the lifetime
of one bright peak. Two longer flanks must agree; separate biological holders
and prior approved changes veto a proposal. No feedback or literal selector is
used to construct, qualify or apply a proposal.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from optical_basin_soma_paths import physical_paths

COLUMNS=['node','identity','frame','seed_y','seed_x','source_identity','changed_pixels','episode_first','episode_last','decision']


def preserve_silhouette_owners(observed,proposals,raw,points,params=None,*,
                              proposal_history=None,current_labels,original_labels):
    labels=np.asarray(current_labels)
    if not labels.shape==observed.shape==proposals.shape==raw.shape==original_labels.shape:
        raise ValueError('Aligned complete movies required')
    nodes,records,paths,_=physical_paths(labels,observed)
    path_of=np.zeros(len(records),np.int32)
    for path,members in paths.items():path_of[members]=path
    boxes={}
    for f,field in enumerate(nodes):
        for n,box in enumerate(ndi.find_objects(field),1):
            if box is not None:boxes[n]=box
    pixels={}
    def indices(n):
        if n not in pixels:
            f=records[n]['frame']-1;box=boxes[n];yy,xx=np.nonzero(nodes[f][box]==n)
            pixels[n]=(yy+box[0].start)*labels.shape[2]+xx+box[1].start
        return pixels[n]
    pts=points[points.state.eq('observed')&points.strong.astype(bool)].copy()
    if 'physically_visible' in pts:pts=pts[pts.physically_visible.astype(bool)]
    t=pts.frame.to_numpy(int);y=np.floor(pts.y+.5).to_numpy(int);x=np.floor(pts.x+.5).to_numpy(int)
    good=(t>=0)&(t<len(labels))&(y>=0)&(y<labels.shape[1])&(x>=0)&(x<labels.shape[2])
    pts=pts[good].copy();t=t[good];y=y[good];x=x[good]
    pts['node']=nodes[t,y,x];pts['path']=path_of[pts.node.to_numpy(int)]
    by_track={int(k):g.sort_values('frame') for k,g in pts.groupby('track_id')}
    by_node={int(k):g for k,g in pts[pts.node.ne(0)].groupby('node')}
    by_name={}
    for q in records[1:]:by_name.setdefault((q['frame'],q['identity']),set()).add(q['node'])
    def runs(members):
        out=[]
        for n in sorted(members,key=lambda k:records[k]['frame']):
            name=records[n]['identity']
            if not out or out[-1][0]!=name:out.append((name,[]))
            out[-1][1].append(n)
        return out
    # Join only mutually unique nearby endpoints with a repeated common owner.
    # The gap cannot exceed the support on either side; no frame cutoff is fit.
    details={}
    for path,members in paths.items():
        rs=runs(members);support={}
        for name,ns in rs:
            measured=sum(n in by_node for n in ns)
            if measured>1:support[name]=max(support.get(name,0),measured)
        if support:
            ends=[]
            for n in (members[0],members[-1]):
                yy,xx=np.unravel_index(indices(n),labels.shape[1:]);ends.append((float(xx.mean()),float(yy.mean()),np.sqrt(len(xx)/np.pi)))
            details[path]=(members,support,ends)
    options=[];starts={}
    for p,(ns,sup,ends) in details.items():
        for name in sup:starts.setdefault(name,[]).append(p)
    for a,(an,asp,ae) in details.items():
        for b in set(q for name in asp for q in starts[name]):
            if a==b:continue
            bn,bsp,be=details[b];gap=records[bn[0]]['frame']-records[an[-1]]['frame']
            common=set(asp)&set(bsp)
            if gap<=1 or len(common)!=1:continue
            name=next(iter(common))
            if gap>min(asp[name],bsp[name]):continue
            distance=np.hypot(ae[1][0]-be[0][0],ae[1][1]-be[0][1])/(ae[1][2]+be[0][2])
            if distance<=gap:options.append((a,b,float(distance)))
    outgoing={};incoming={}
    for a,b,d in options:outgoing.setdefault(a,[]).append((d,b));incoming.setdefault(b,[]).append((d,a))
    next_path={};previous={}
    for a,b,d in options:
        left=outgoing[a];right=incoming[b]
        if d!=min(q[0] for q in left) or d!=min(q[0] for q in right):continue
        if sum(q[0]==d for q in left)!=1 or sum(q[0]==d for q in right)!=1:continue
        next_path[a]=b;previous[b]=a
    groups=[]
    for path in paths:
        if path in previous:continue
        chain=[];p=path
        while True:
            chain.extend(paths[p])
            if p not in next_path:break
            p=next_path[p]
        groups.append(chain)
    # Recover isolated measured silhouettes inside an already justified gap.
    # A short fragment must reach both endpoints at an area-normalised speed,
    # have no independent measured ancestry, and be unique at each frame.
    coordinates={}
    def geometry(n):
        if n not in coordinates:
            yy,xx=np.unravel_index(indices(n),labels.shape[1:])
            coordinates[n]=np.column_stack((yy,xx))
        return coordinates[n]
    def reachable(a,b):
        elapsed=abs(records[a]['frame']-records[b]['frame'])
        scale=np.sqrt(len(indices(a))/np.pi)+np.sqrt(len(indices(b))/np.pi)
        distance=cKDTree(geometry(a)).query(geometry(b),k=1)[0].min()
        return distance/scale<=elapsed
    for members in groups:
        ordered=sorted(members,key=lambda n:records[n]['frame'])
        lineage={int(path_of[n]) for n in members};extras=[]
        for left,right in zip(ordered,ordered[1:]):
            first,last=records[left]['frame'],records[right]['frame']
            if last-first<=1:continue
            candidates=[]
            for path,ns in paths.items():
                if path in lineage:continue
                if not first<records[ns[0]]['frame']<=records[ns[-1]]['frame']<last:continue
                if not any(n in by_node for n in ns):continue
                if not all(reachable(left,n) and reachable(n,right) for n in ns):continue
                independent=False
                for n in ns:
                    for tid in by_node.get(n,pd.DataFrame(columns=['track_id'])).track_id.unique():
                        h=by_track[int(tid)]
                        outside=h[h.node.ne(0)&~h.path.isin(lineage|{path})]
                        if len(outside):independent=True;break
                    if independent:break
                if not independent:candidates.append(ns)
            occupied={}
            for ns in candidates:
                for n in ns:occupied.setdefault(records[n]['frame'],[]).append(n)
            for ns in candidates:
                if all(len(occupied[records[n]['frame']])==1 for n in ns):extras.extend(ns)
        members.extend(extras)
    plans=[];audit=[];supported_lineages={};branch_decisions={}
    for members in groups:
        rs=runs(members);lineage={int(path_of[n]) for n in members}
        for left,(owner,prefix) in enumerate(rs):
            right=next((i for i in range(left+1,len(rs)) if rs[i][0]==owner),None)
            if right is None or right==left+1:continue
            middle=[n for _,ns in rs[left+1:right] for n in ns];suffix=rs[right][1]
            if min(len(prefix),len(suffix))<=len(middle):continue
            if sum(n in by_node for n in prefix)<2 or sum(n in by_node for n in suffix)<2:continue
            if not any(n in by_node for n in middle):continue
            first=records[middle[0]]['frame'];last=records[middle[-1]]['frame']
            wanted=set(middle);failed=None
            # An independently arriving/leaving measured body cannot be erased
            # merely because overlap followed the wrong side of a contact.
            for n in middle:
                for tid in by_node.get(n,pd.DataFrame(columns=['track_id'])).track_id.unique():
                    h=by_track[int(tid)];outside=h[h.node.ne(0)&~h.path.isin(lineage)]
                    if (h.frame<first-1).any() and (h.frame>last-1).any():
                        failed='continuous_detector_outside_fragment_boundary';break
                    if len(outside[(outside.frame<first-1)|(outside.frame>last-1)]):
                        failed='independent_entering_or_leaving_body';break
                if failed:break
            if failed:continue
            for n in middle:
                q=records[n];f=q['frame']-1
                if np.any(labels[f].ravel()[indices(n)]!=original_labels[f].ravel()[indices(n)]):
                    failed='protected_prior_revision';break
                for holder in by_name.get((q['frame'],owner),set())-wanted:
                    if path_of[holder] in lineage:continue
                    cores=by_node.get(holder)
                    if cores is None:
                        # A retained outline remnant must actually touch the
                        # continuing body; proximity alone is not identity.
                        mask=np.zeros(labels.shape[1:],bool);mask.ravel()[indices(holder)]=True
                        if not np.any(ndi.binary_dilation(mask,np.ones((3,3),bool)).ravel()[indices(n)]):
                            failed='separate_unmeasured_owner_holder';break
                    else:
                        for tid in cores.track_id.unique():
                            h=by_track[int(tid)];prior=h[h.frame<first-1]
                            prior=prior[prior.node.ne(0)]
                            if prior.empty or not prior.path.isin(lineage).all():
                                failed='independent_owner_holder';break
                    if failed:break
                if failed:break
            if failed:continue
            plans.append((owner,first,last,middle))
            supported_lineages[id(middle)]=set(lineage)
            # Already-correct same-owner remnants with measured ancestry in
            # this cell are supporting observations, not separate competitors.
            for n in middle:
                for holder in by_name.get((records[n]['frame'],owner),set()):
                    hp=int(path_of[holder])
                    if all(records[k]['identity']==owner for k in paths[hp]):
                        supported_lineages[id(middle)].add(hp)
    initial_claims={}
    for owner,_,_,members in plans:
        for n in members:initial_claims.setdefault(n,set()).add(owner)
    plans=[p for p in plans if all(len(initial_claims[n])==1 for n in p[3])]
    extensions=[];seen=set()
    for owner,_,_,members in plans:
        lineage=supported_lineages[id(members)];key=(owner,frozenset(lineage))
        if key in seen:continue
        seen.add(key)
        first=min(records[n]['frame'] for p in lineage for n in paths[p])
        last=max(records[n]['frame'] for p in lineage for n in paths[p])
        for path,ns in paths.items():
            if path in lineage or any(n in by_node for n in ns):continue
            start=records[ns[0]]['frame'];end=records[ns[-1]]['frame']
            if not first<start<=end<last:continue
            # A raw-core-free silhouette may shed from, or return to, only
            # this demonstrated cell. An overlap with another path vetoes it.
            adjacent=set()
            for n in ns:
                t=records[n]['frame']-1
                for other in (t-1,t+1):
                    if 0<=other<len(nodes):
                        adjacent.update(int(path_of[k]) for k in np.unique(nodes[other].ravel()[indices(n)]) if k)
            adjacent.discard(path)
            if not adjacent or not adjacent.issubset(lineage):continue
            if any(np.any(labels[records[n]['frame']-1].ravel()[indices(n)]!=original_labels[records[n]['frame']-1].ravel()[indices(n)]) for n in ns):continue
            changed=[n for n in ns if records[n]['identity']!=owner]
            if not changed:continue
            extensions.append((owner,start,end,changed))
            for n in changed:branch_decisions[n]='unmeasured_outline_branch_has_one_supported_parent'
    plans.extend(extensions)
    claims={}
    for owner,_,_,members in plans:
        for n in members:claims.setdefault(n,set()).add(owner)
    result=labels.copy();done=set()
    for owner,first,last,members in plans:
        if any(len(claims[n])!=1 for n in members):continue
        for n in members:
            if n in done:continue
            done.add(n);q=records[n];f=q['frame']-1;idx=indices(n)
            result[f].ravel()[idx]=owner
            audit.append(dict(node=n,identity=owner,frame=f+1,seed_y=q['seed_y'],seed_x=q['seed_x'],source_identity=q['identity'],changed_pixels=len(idx),episode_first=first,episode_last=last,decision=branch_decisions.get(n,'two_flanks_support_continuing_silhouette_owner')))
    return result,pd.DataFrame(audit,columns=COLUMNS)
