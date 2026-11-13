"""Keep two observed somata distinct through an otherwise single-owner outline.

Current independently outlined bodies certify the names on both sides of a
contact. A detector restart can be bridged by reciprocal unique geometry, after
reserving every independently continuing core. No feedback enters discovery.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.segmentation import watershed

COLUMNS=['decision','contact_first','contact_last','frame','target','resident',
         'core_chain','resident_chain','changed_pixels','restart_links']


def preserve_reconnected_contact_somas(observed,proposals,raw,points,params=None,*,
                                      proposal_history=None,current_labels,original_labels):
    labels=np.asarray(current_labels)
    if not labels.shape==raw.shape==observed.shape==proposals.shape==original_labels.shape:
        raise ValueError('Aligned complete movies required')
    nodes=np.zeros(labels.shape,np.int32);areas=[0];names=[0]
    for f,plane in enumerate(labels):
        parts=label(plane,background=0,connectivity=2)
        offset=len(areas)-1
        nodes[f]=np.where(parts,parts+offset,0)
        for i,box in enumerate(ndi.find_objects(parts),1):
            hit=parts[box]==i
            areas.append(int(hit.sum()));names.append(int(plane[box][hit][0]))
    areas=np.asarray(areas);names=np.asarray(names)
    pts=points[points.strong.astype(bool)&points.state.eq('observed')].copy()
    if 'physically_visible' in pts:pts=pts[pts.physically_visible.astype(bool)]
    f=pts.frame.to_numpy(int);y=np.floor(pts.y+.5).astype(int);x=np.floor(pts.x+.5).astype(int)
    valid=(f>=0)&(f<len(labels))&(y>=0)&(y<labels.shape[1])&(x>=0)&(x<labels.shape[2])
    pts=pts[valid].copy();f=f[valid];y=y[valid];x=x[valid]
    pts['node']=nodes[f,y,x];pts['owner']=names[pts.node.to_numpy(int)]
    groups={int(k):g.sort_values('frame') for k,g in pts.groupby('track_id')}
    counts=pts[pts.node.ne(0)].groupby('node').track_id.nunique().to_dict()
    def independent(g):
        return g[g.node.ne(0)&g.node.map(counts).eq(1)]
    def pair_distance(a,b):
        scale=np.sqrt(areas[int(a.node)]/np.pi)+float(b.radius_px)
        return float(np.hypot(a.x-b.x,a.y-b.y)/scale) if scale>0 else np.inf
    # Bridge only ended and newly observed core fragments; an ongoing core is
    # reserved, so a nearby neighbour cannot be stolen by nearest distance.
    ends={};starts={}
    for track,g in groups.items():
        if g.frame.duplicated().any():continue
        end=g.iloc[-1];start=g.iloc[0]
        if end.node and counts[int(end.node)]==1:ends.setdefault(int(end.frame),[]).append((track,end))
        if start.node and counts[int(start.node)]==2:starts.setdefault(int(start.frame),[]).append((track,start))
    candidates=[]
    for f,heads in starts.items():
        for dest,head in heads:
            for source,tail in ends.get(f-1,[]):
                if source==dest:continue
                prefix=independent(groups[source]);suffix=independent(groups[dest])
                if len(prefix)<2 or len(suffix)<2:continue
                owner=int(tail.owner)
                if not prefix.iloc[-2:].owner.eq(owner).all() or not suffix.iloc[:2].owner.eq(owner).all():continue
                if int(head.owner)==owner:continue
                # The separated source must actually enter this connected body.
                if not np.any((nodes[f-1]==int(tail.node))&(nodes[f]==int(head.node))):continue
                score=pair_distance(tail,head)
                if score<=1:candidates.append((source,dest,score))
    links=[]
    for source,dest,score in candidates:
        forward=[d for s,d,v in candidates if s==source and v<=score]
        backward=[s for s,d,v in candidates if d==dest and v<=score]
        if len(forward)==len(backward)==1:links.append((source,dest,score))
    parent={k:k for k in groups}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    for a,b,_ in links:parent[find(b)]=find(a)
    pts['chain']=[find(int(k)) for k in pts.track_id]
    chains={int(k):g.sort_values('frame') for k,g in pts.groupby('chain')}
    by_node={int(k):g for k,g in pts[pts.node.ne(0)].groupby('node')}
    name_frames={(f,int(k)) for f,plane in enumerate(labels) for k in np.unique(plane) if k}
    proposals_by_node={};audit=[]
    for chain,g in chains.items():
        if g.frame.duplicated().any():continue
        solo=independent(g)
        rows=list(solo.itertuples())
        for i,(left,right) in enumerate(zip(rows,rows[1:])):
            start=int(left.frame)+1;stop=int(right.frame)
            target=int(left.owner)
            if start>=stop or target!=int(right.owner):continue
            # Two separate observations on either side establish moving bodies,
            # rather than a single displaced point or a transient projection.
            if i<1 or i+2>=len(rows):continue
            if int(rows[i-1].owner)!=target or int(rows[i+2].owner)!=target:continue
            interval=g[g.frame.between(start,stop-1)]
            if len(interval)!=stop-start or interval.node.eq(0).any():continue
            if not interval.owner.eq(int(interval.iloc[0].owner)).all():continue
            resident=int(interval.iloc[0].owner)
            if resident==target or not resident:continue
            other_chains=set();shared=[];reason=None
            for p in interval.itertuples():
                peers=by_node[int(p.node)]
                if peers.chain.nunique()!=2 or len(peers)!=2:reason='not_two_independent_core_paths';break
                other=peers[peers.chain.ne(chain)].iloc[0]
                if np.hypot(p.x-other.x,p.y-other.y)<=p.radius_px+other.radius_px:
                    reason='raw_cores_not_spatially_distinct';break
                other_chains.add(int(other.chain));shared.append((p,other))
            if reason or len(other_chains)!=1:continue
            resident_chain=other_chains.pop();h=chains[resident_chain]
            resident_solo=independent(h)
            prior=resident_solo[resident_solo.frame.lt(start)].tail(2)
            later=resident_solo[resident_solo.frame.ge(stop)].head(2)
            if len(prior)<2 or len(later)<2 or not prior.owner.eq(resident).all() or not later.owner.eq(resident).all():continue
            # Require the two actual paths at both contact boundaries. Neither
            # name can be borrowed from a far-away or later unrelated body.
            if int(prior.iloc[-1].frame)!=start-1 or int(later.iloc[0].frame)!=stop:continue
            if any((f,target) in name_frames for f in range(start,stop)):continue
            local=[]
            for p,other in shared:
                f=int(p.frame);n=int(p.node);yy,xx=np.nonzero(nodes[f]==n)
                y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
                body=nodes[f,y0:y1,x0:x1]==n;seeds=np.zeros(body.shape,np.int32)
                for marker,core in [(1,p),(2,other)]:
                    distances=(yy-core.y)**2+(xx-core.x)**2;at=np.flatnonzero(distances==distances.min())
                    if len(at)!=1 or distances[at[0]]>core.radius_px**2:reason='ambiguous_or_distant_raw_core';break
                    j=int(at[0]);sy,sx=int(yy[j]-y0),int(xx[j]-x0)
                    if seeds[sy,sx]:reason='coincident_raw_cores';break
                    seeds[sy,sx]=marker
                if reason:break
                partition=watershed(-raw[f,y0:y1,x0:x1].astype(float),seeds,mask=body,connectivity=2)
                if np.any(partition[body]==0):reason='unpartitioned_foreground';break
                changed=(partition==1)&body
                if np.any(changed&(labels[f,y0:y1,x0:x1]!=original_labels[f,y0:y1,x0:x1])):
                    reason='protected_prior_revision';break
                indices=np.ravel_multi_index((yy,xx),labels.shape[1:])
                values=np.where(partition[body]==1,target,resident).astype(labels.dtype)
                local.append((f,n,indices,values))
            entry=dict(contact_first=start+1,contact_last=stop,frame=start+1,target=target,resident=resident,
                       core_chain=chain,resident_chain=resident_chain,changed_pixels=0,
                       restart_links=sum(find(a)==chain for a,_,_ in links))
            if reason:audit.append(dict(entry,decision=reason));continue
            plan=len(audit);audit.append(dict(entry,decision='proposed_observed_contact_partition'))
            for f,n,indices,values in local:proposals_by_node.setdefault((f,n),[]).append((plan,indices,values))
    conflicts=set()
    for entries in proposals_by_node.values():
        if any(not np.array_equal(entries[0][2],q[2]) for q in entries[1:]):conflicts.update(q[0] for q in entries)
    result=labels.copy()
    for (f,n),entries in proposals_by_node.items():
        valid=[q for q in entries if q[0] not in conflicts]
        if not valid:continue
        plan,indices,values=valid[0]
        changed=int(np.count_nonzero(labels[f].ravel()[indices]!=values))
        result[f].ravel()[indices]=values
        audit[plan]['changed_pixels']+=changed
    for i,row in enumerate(audit):
        if row['decision']=='proposed_observed_contact_partition':
            row['decision']='conflicting_owner_proposals' if i in conflicts else ('observed_contact_owners_preserved' if row['changed_pixels'] else 'duplicate_compatible_proposal')
    return result,pd.DataFrame(audit,columns=COLUMNS)
