"""Keep measured cell ownership when an observed peak is replaced in contact.

The existing point tracks are evidence and remain unchanged. A new strong
detector may inherit a lost detector's established owner only through unique
measured footprint overlap, while another body continues independently. Weak
positions cannot carry ownership farther than the last observed soma's size.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from optical_basin_soma_paths import physical_paths, source_catchment
from remove_optical_core_adjacent_pseudosomata import _optical_core_union

COLUMNS=['node','identity','frame','seed_y','seed_x','source_identity',
         'changed_pixels','source_track','successor_track','decision']


def preserve_core_replacement_owners(observed,proposals,raw,points,params=None,*,
                                    proposal_history=None,current_labels,original_labels):
    labels=np.asarray(current_labels);result=labels.copy()
    if not labels.shape==raw.shape==observed.shape==proposals.shape==original_labels.shape:
        raise ValueError('Aligned complete movies required')
    nodes,records,_,_=physical_paths(labels,observed)
    pts=points.copy()
    pts['node']=0
    for index,p in pts.iterrows():
        f=int(p.frame);y=int(np.floor(p.y+.5));x=int(np.floor(p.x+.5))
        if not (0<=f<len(labels) and 0<=y<labels.shape[1] and 0<=x<labels.shape[2]):continue
        n=int(nodes[f,y,x])
        if not n:
            radius=float(p.radius_px)
            y0=max(0,int(np.floor(p.y-radius)));y1=min(labels.shape[1],int(np.ceil(p.y+radius))+1)
            x0=max(0,int(np.floor(p.x-radius)));x1=min(labels.shape[2],int(np.ceil(p.x+radius))+1)
            field=nodes[f,y0:y1,x0:x1];yy,xx=np.indices(field.shape)
            distance=(yy+y0-p.y)**2+(xx+x0-p.x)**2
            hit=(field>0)&(distance<=radius**2)
            if hit.any():
                minimum=distance[hit].min();choices=np.unique(field[hit&(distance==minimum)])
                if len(choices)==1:n=int(choices[0])
        pts.loc[index,'node']=n
    pts['owner']=[records[int(n)]['identity'] if n else 0 for n in pts.node]
    actual=pts[pts.state.eq('observed')].copy()
    strong=actual[actual.strong.astype(bool)]
    by_track={int(k):g.sort_values('frame') for k,g in pts.groupby('track_id')}
    observations={int(k):g.sort_values('frame') for k,g in actual.groupby('track_id')}
    strong_tracks={int(k):g.sort_values('frame') for k,g in strong.groupby('track_id')}
    by_node={int(k):g for k,g in strong[strong.node.gt(0)].groupby('node')}
    first={k:int(g.frame.min()) for k,g in observations.items()}
    last_strong={k:int(g.frame.max()) for k,g in strong_tracks.items()}
    boxes={};flat={}
    for f,frame in enumerate(nodes):
        for n,box in enumerate(ndi.find_objects(frame),1):
            if box is not None:boxes[n]=box

    def pixels(n):
        if n not in flat:
            f=records[n]['frame']-1;flat[n]=np.flatnonzero(nodes[f].ravel()==n)
        return flat[n]

    basin_cache={};alias_parent={}
    def alias_root(track):
        while track in alias_parent:track=alias_parent[track]
        return track
    def basin(p,*,measured_core=False):
        key=(int(p.track_id),int(p.frame),measured_core);n=int(p.node)
        if not n:return None
        if key in basin_cache:return basin_cache[key]
        f=int(p.frame);box=boxes[n];body=nodes[f][box]==n
        yy,xx=np.nonzero(body);gy=yy+box[0].start;gx=xx+box[1].start
        distance=(gy-p.y)**2+(gx-p.x)**2
        minimum=distance.min();at=np.flatnonzero(distance==minimum)
        if minimum>float(p.radius_px)**2 or (not measured_core and len(at)!=1):return None
        markers=np.zeros(body.shape,np.int32)
        if measured_core:
            support=distance<=float(p.radius_px)**2
            markers[yy[support],xx[support]]=1
        else:
            j=int(at[0]);markers[yy[j],xx[j]]=1
        peers=by_node.get(n,strong.iloc[:0]);peers=peers[peers.track_id.ne(p.track_id)]
        for q in peers.itertuples():
            same_soma=alias_root(int(q.track_id))==alias_root(int(p.track_id))
            # Overlapping measured core disks do not prove separate bodies.
            if not same_soma and np.hypot(q.x-p.x,q.y-p.y)<=q.radius_px+p.radius_px:return None
            d=(gy-q.y)**2+(gx-q.x)**2;hits=np.flatnonzero(d==d.min())
            if (not measured_core and len(hits)!=1) or d.min()>q.radius_px**2:return None
            if measured_core:
                support=d<=q.radius_px**2
                if not same_soma and np.any(markers[yy[support],xx[support]]==1):return None
                markers[yy[support],xx[support]]=1 if same_soma else 2
            else:
                k=int(hits[0])
                if markers[yy[k],xx[k]]==1 and not same_soma:return None
                markers[yy[k],xx[k]]=1 if same_soma else 2
        regions=watershed(-raw[f][box].astype(float),markers,mask=body,connectivity=2)
        selected=regions[yy,xx]==1
        idx=np.ravel_multi_index((gy[selected],gx[selected]),labels.shape[1:])
        answer=(n,idx)
        basin_cache[key]=answer
        return answer

    optical_core=None;optical_cache={}
    def independent_optical_source(p):
        """Reuse accepted optical catchments at this measured silhouette's scale."""
        nonlocal optical_core
        key=(int(p.frame),int(p.node))
        if key in optical_cache:return optical_cache[key]
        if optical_core is None:
            yy,xx=np.nonzero(np.any(raw!=0,axis=0))
            if not len(yy):optical_core=(None,None)
            else:
                box=(slice(int(yy.min()),int(yy.max())+1),slice(int(xx.min()),int(xx.max())+1))
                optical_core=(box,_optical_core_union(raw[(slice(None),)+box]))
        box,core=optical_core
        if core is None or not core.any():return True
        f=int(p.frame);n=int(p.node);body=nodes[f][box]==n
        catch=source_catchment(raw[f][box],core,int(body.sum()))
        independent=np.count_nonzero(catch&body)*2<=np.count_nonzero(body)
        optical_cache[key]=independent
        return independent

    root_cache={}
    def root_owner(track):
        if track in root_cache:return root_cache[track]
        g=strong_tracks.get(track)
        if g is None:return 0
        independent=g[g.node.map(lambda n:n>0 and len(by_node.get(int(n),()))==1)]
        names=set(map(int,independent.owner))-{0}
        # Two observations supply temporal evidence; one peak alone does not.
        owner=next(iter(names)) if len(names)==1 and independent.frame.nunique()>=2 else 0
        if owner:
            final=g.iloc[-1]
            if int(final.owner)!=owner:owner=0
            else:
                field=nodes[int(final.frame)];held={q['node'] for q in records[1:]
                    if q['frame']==int(final.frame)+1 and q['identity']==owner}
                if held!={int(final.node)}:owner=0
        root_cache[track]=owner
        return owner

    owner_of={};edges={};used=set();audit=[]
    def coincident_replacement(source,target,start):
        """A measured detector overlap, with no independent coexistence."""
        if source not in owner_of:return False
        future=strong_tracks[source];future=future[future.frame.ge(start)]
        if future.empty:return False
        successor=strong_tracks[target].set_index('frame',drop=False)
        for p in future.itertuples():
            if int(p.frame) not in successor.index:return False
            q=successor.loc[int(p.frame)]
            if not int(p.node) or int(p.node)!=int(q.node):return False
            if np.hypot(p.x-q.x,p.y-q.y)>p.radius_px+q.radius_px:return False
        later=successor[successor.frame.gt(int(future.frame.max()))]
        independent=later[later.node.map(lambda n:n>0 and len(by_node.get(int(n),()))==1)]
        return independent.frame.nunique()>=2
    births={}
    for track,g in observations.items():
        p=g.iloc[0]
        if bool(p.strong) and int(p.node)>0:births.setdefault(int(p.frame),[]).append(track)
    endings={}
    for track,g in observations.items():
        for f in g.frame.astype(int):endings.setdefault(f+1,[]).append(track)
    for f,targets in sorted(births.items()):
        alternatives=[]
        for target in targets:
            p=observations[target].iloc[0];n=int(p.node)
            peers=by_node.get(n,strong.iloc[:0])
            continuing=[q for q in peers.itertuples() if int(q.track_id)!=target
                and first[int(q.track_id)]<f and last_strong.get(int(q.track_id),-1)>f
                and np.hypot(q.x-p.x,q.y-p.y)>q.radius_px+p.radius_px]
            for source in endings.get(f,[]):
                if source==target or source in used or source not in last_strong:continue
                # Avoid constructing root evidence for ineligible isolated births.
                if not continuing and source not in owner_of:continue
                coincident=last_strong[source]>=f and coincident_replacement(source,target,f)
                if last_strong[source]>=f and not coincident:continue
                obs=observations[source]
                if obs.frame.eq(f).any() and not coincident:continue
                previous=obs[obs.frame.eq(f-1)].iloc[-1]
                owner=owner_of.get(source,0) or root_owner(source)
                if not owner or owner==int(p.owner):continue
                prior=strong_tracks[source]
                if coincident:prior=prior[prior.frame.lt(f)]
                if prior.empty:continue
                last=prior.iloc[-1]
                inherited=source in owner_of
                # After a qualified contact reset, a returning independent soma
                # can continue that owner. The intervening weak predictor is not
                # a measured displacement of the soma.
                independent_return=inherited and len(peers)==1
                if not continuing and not independent_return and not coincident:continue
                # The weak interval cannot drift outside the measured soma scale.
                if np.hypot(previous.x-last.x,previous.y-last.y)>previous.radius_px+last.radius_px:continue
                region=basin(last)
                if region is None:continue
                _,idx=region;yy,xx=np.unravel_index(idx,labels.shape[1:])
                if independent_return:
                    target_region=basin(p)
                    if target_region is None or not np.intersect1d(idx,target_region[1],assume_unique=True).size:continue
                    if np.hypot(p.x-last.x,p.y-last.y)>p.radius_px+last.radius_px:continue
                else:
                    dy=previous.y-last.y;dx=previous.x-last.x
                    gap2=np.min((yy+dy-p.y)**2+(xx+dx-p.x)**2)
                    if gap2>p.radius_px**2:continue
                    # A dimmer sample must not shrink the established soma abruptly.
                    scale=max(float(previous.radius_px),float(last.radius_px))
                    if np.hypot(p.x-previous.x,p.y-previous.y)>p.radius_px+scale:continue
                # Optical-source basins cannot establish an inherited cell owner.
                if not inherited and not independent_optical_source(last):continue
                alternatives.append((source,target,owner,coincident))
        for source,target,owner,coincident in alternatives:
            if sum(s==source for s,_,_,_ in alternatives)!=1 or sum(t==target for _,t,_,_ in alternatives)!=1:continue
            owner_of[target]=owner;edges[target]=(source,f);used.add(source)
            if coincident:
                alias_parent[alias_root(target)]=alias_root(source)
                basin_cache.clear()
    if not edges:return result,pd.DataFrame(columns=COLUMNS)
    end={source:start for target,(source,start) in edges.items()}
    plans=[];release_plans={};coupled=[]
    def recover_displaced_holder(f,node,wanted,held):
        """Resolve a naming collision only with another measured owner's proof."""
        holder_nodes=set(map(int,np.unique(nodes[f].ravel()[held])))-{node,0}
        if len(holder_nodes)!=1:return None
        holder_node=holder_nodes.pop();cores=by_node.get(holder_node,strong.iloc[:0])
        if len(cores)!=1:return None
        point=cores.iloc[0];track=int(point.track_id);g=strong_tracks[track]
        first_wrong=int(g[g.owner.eq(wanted)].frame.min())
        prior=g[g.frame.lt(first_wrong)&g.node.gt(0)]
        prior=prior[prior.node.map(lambda n:len(by_node.get(int(n),()))==1)]
        names=set(map(int,prior.owner))-{0}
        if len(names)!=1 or prior.frame.nunique()<2:return None
        alternate=names.pop()
        if alternate==wanted or np.any(labels[f]==alternate):return None
        if not set(map(int,g.owner)).issubset({0,wanted,alternate}):return None
        anchor=prior.iloc[-1]
        interval=g[g.frame.ge(int(anchor.frame))&g.frame.le(f)]
        # Measured continuity across a segmentation gap, not extrapolation.
        if not np.array_equal(interval.frame.to_numpy(int),np.arange(int(anchor.frame),f+1)):return None
        if not interval[interval.frame.lt(first_wrong)].owner.eq(0).any():return None
        samples=list(interval.itertuples())
        if any(np.hypot(b.x-a.x,b.y-a.y)>a.radius_px+b.radius_px for a,b in zip(samples,samples[1:])):return None
        idx=pixels(holder_node)
        if not np.intersect1d(pixels(int(anchor.node)),idx,assume_unique=True).size:return None
        if np.any(labels[f].ravel()[idx]!=original_labels[f].ravel()[idx]):return None
        if not independent_optical_source(anchor):return None
        base=dict(node=holder_node,identity=alternate,frame=f+1,seed_y=int(np.floor(point.y+.5)),seed_x=int(np.floor(point.x+.5)),source_identity=wanted,changed_pixels=0,source_track=track,successor_track=track,decision='measured_holder_keeps_its_owner_across_segmentation_gap')
        return (f,holder_node,idx,alternate,base)
    for track,(source,start) in edges.items():
        owner=owner_of[track];g=by_track[track]
        g=g[g.frame.ge(start)&g.frame.lt(end.get(track,len(labels)))&g.state.isin(['observed','latent_visible'])]
        for p in g.itertuples():
            f=int(p.frame);n=int(p.node)
            base=dict(node=n,identity=owner,frame=f+1,seed_y=int(np.floor(p.y+.5)),seed_x=int(np.floor(p.x+.5)),
                source_identity=int(p.owner),changed_pixels=0,source_track=source,successor_track=track)
            # Once ownership is qualified, its measured core supplies the seed.
            # Keep the preceding point-based discovery inputs unchanged; whole
            # soma support only changes this ownership partitioning operation.
            region=basin(p,measured_core=True)
            if region is None:audit.append(dict(base,decision='no_independent_measured_basin'));continue
            _,idx=region;before=labels[f].ravel()[idx];change=before!=owner
            if not change.any():continue
            if np.any(change&(before!=original_labels[f].ravel()[idx])):
                audit.append(dict(base,decision='protected_prior_revision'));continue
            held=np.flatnonzero(labels[f].ravel()==owner)
            release_index=None
            if len(held) and np.any(nodes[f].ravel()[held]!=n):
                release=recover_displaced_holder(f,n,owner,held)
                if release is None:
                    audit.append(dict(base,decision='independent_existing_owner_holder'));continue
                key=(release[0],release[1],release[3])
                if key not in release_plans:
                    release_plans[key]=len(plans);plans.append(release)
                release_index=release_plans[key]
            main_index=len(plans)
            plans.append((f,n,idx[change],owner,base))
            if release_index is not None:coupled.append({release_index,main_index})
    # Reject competing pixel claims and separate simultaneous bodies with one name.
    bad=set()
    for i,(f,n,idx,owner,_) in enumerate(plans):
        for j in range(i):
            g,m,other,target,_=plans[j]
            if f!=g:continue
            if owner==target and n!=m:bad.update([i,j])
            elif owner!=target and np.intersect1d(idx,other,assume_unique=True).size:bad.update([i,j])
    # Freeing an incorrectly held name and assigning its established body are
    # one transaction. Any conflict rejects both sides, including shared releases.
    previous=-1
    while previous!=len(bad):
        previous=len(bad)
        for group in coupled:
            if bad&group:bad.update(group)
    for i,(f,n,idx,owner,base) in enumerate(plans):
        if i in bad:audit.append(dict(base,decision='conflicting_body_or_pixel_claim'));continue
        result[f].ravel()[idx]=owner
        audit.append(dict(base,changed_pixels=len(idx),decision=base.get('decision','owner_follows_unique_observed_core_replacement')))
    return result,pd.DataFrame(audit,columns=COLUMNS)
