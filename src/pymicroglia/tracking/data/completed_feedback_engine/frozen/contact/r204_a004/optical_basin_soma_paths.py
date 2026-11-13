"""Qualify apparent somata against the catchments of accepted optical sources.

An optical source can supply both the detector blob and the tracker peak. Those
are not independent cell evidence. A separate bright island at the silhouette's
own area scale, in ANY observed frame, protects the entire reciprocal path.
No feedback, names, coordinates, recording selection or fitted cutoff is used.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import area_opening, max_tree
from skimage.segmentation import watershed
from remove_optical_core_adjacent_pseudosomata import _optical_core_union
from reciprocal_soma_overlap import reciprocal_overlap_pairs

COLUMNS = ['node','identity','frame','seed_y','seed_x','proposed_pixels',
           'measured_pixels','path','path_first_frame','path_last_frame',
           'path_node_count','withdrawn_pixels','source_fraction',
           'touches_another_outline','decision']


def physical_paths(proposals, observed):
    """The accepted reciprocal-best overlap graph, without identity linking."""
    nodes=np.zeros(proposals.shape,np.int32);records=[None];parents=[0];blocked=set()
    for t,frame in enumerate(proposals):
        components=label(frame,background=0,connectivity=2)
        offset=len(records)-1
        nodes[t]=np.where(components>0,components+offset,0)
        for index,box in enumerate(ndi.find_objects(components),1):
            body=components[box]==index;yy,xx=np.nonzero(body);node=offset+index
            records.append(dict(node=node,identity=int(frame[box][body][0]),frame=t+1,
                seed_y=int(yy[0]+box[0].start),seed_x=int(xx[0]+box[1].start),
                proposed_pixels=int(body.sum()),measured_pixels=int(np.count_nonzero(observed[t][box][body]))))
            parents.append(node)
            if box[0].start==0 or box[1].start==0 or box[0].stop==frame.shape[0] or box[1].stop==frame.shape[1]:blocked.add(node)
        field=nodes[t];padded=np.pad(field,1)
        for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
            other=padded[1+dy:1+dy+field.shape[0],1+dx:1+dx+field.shape[1]]
            hit=(field>0)&(other>0)&(field!=other)
            blocked.update(map(int,np.unique(field[hit])));blocked.update(map(int,np.unique(other[hit])))
    def find(a):
        while parents[a]!=a:parents[a]=parents[parents[a]];a=parents[a]
        return a
    areas=np.array([0]+[q['proposed_pixels'] for q in records[1:]])
    for t in range(1,len(nodes)):
        shared=(nodes[t-1]>0)&(nodes[t]>0)
        pairs,counts=np.unique(np.column_stack((nodes[t-1][shared],nodes[t][shared])),axis=0,return_counts=True)
        for a,b in reciprocal_overlap_pairs(pairs,counts,areas):
            ra,rb=find(int(a)),find(int(b))
            if ra!=rb:parents[rb]=ra
    groups={}
    for node in range(1,len(records)):groups.setdefault(find(node),[]).append(node)
    return nodes,records,groups,blocked


def source_catchment(surface,core,area,tree=None):
    """Remove maxima smaller than a measured silhouette; identify source basins."""
    if tree is None:tree=max_tree(surface,connectivity=2)
    opened=area_opening(surface,area_threshold=int(area),connectivity=2,parent=tree[0],tree_traverser=tree[1])
    basins=watershed(-opened.astype(float),connectivity=2)
    return np.isin(basins,np.unique(basins[core]))


def constrain_optical_basin_soma_paths(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels=None):
    if current_labels is None:raise ValueError('An accepted complete parent is required')
    if not (observed.shape==proposals.shape==raw.shape==current_labels.shape):raise ValueError('Aligned full movies required')
    result=current_labels.copy();empty=pd.DataFrame(columns=COLUMNS)
    # Crop only empty acquisition padding, so translated input has the same
    # max-tree boundaries and watershed ties. Native acquisition is untouched.
    yy,xx=np.nonzero(np.any(raw!=0,axis=0))
    if not len(yy):return result,empty
    y0,y1,x0,x1=int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1
    movie=raw[:,y0:y1,x0:x1];labels=current_labels[:,y0:y1,x0:x1]
    core=_optical_core_union(movie)
    if not core.any():return result,empty
    obs=observed[:,y0:y1,x0:x1];saved=proposals[:,y0:y1,x0:x1]
    nodes,records,groups,blocked=physical_paths(labels,obs)
    median=np.median(movie,axis=0);tree=max_tree(median,connectivity=2);cache={};eligible={}
    # A median prefilter only avoids work: the complete actual path still has
    # to pass in every frame. Transient real cells cannot be removed merely
    # because their median image disappears.
    for path,members in groups.items():
        if any(n in blocked for n in members):continue
        rec=[records[n] for n in members];area=int(np.median([q['proposed_pixels'] for q in rec]))
        if area not in cache:cache[area]=source_catchment(median,core,area,tree)
        source=cache[area];passed=True
        for q in rec:
            t=q['frame']-1;body=nodes[t]==q['node']
            if np.any(body&(labels[t]!=saved[t])) or np.count_nonzero(source&body)*2<=q['proposed_pixels']:
                passed=False;break
        if passed:eligible[path]=members
    del cache,tree
    fractions={}
    for t,frame in enumerate(movie):
        active=[(path,n) for path,members in eligible.items() for n in members if records[n]['frame']==t+1]
        if not active:continue
        tree=max_tree(frame,connectivity=2);cache={};keep=set()
        for path,node in active:
            q=records[node];area=q['proposed_pixels']
            if area not in cache:cache[area]=source_catchment(frame,core,area,tree)
            count=int(np.count_nonzero(cache[area]&(nodes[t]==node)))
            fractions[node]=count/area
            if count*2<=area:keep.add(path)
        for path in keep:del eligible[path]
    audit=[];lookup=np.zeros(len(records),bool)
    for path,members in eligible.items():
        rec=[records[n] for n in members];first=min(q['frame'] for q in rec);last=max(q['frame'] for q in rec)
        for q in rec:
            lookup[q['node']]=True
            audit.append(dict(q,seed_y=q['seed_y']+y0,seed_x=q['seed_x']+x0,path=path,
                path_first_frame=first,path_last_frame=last,path_node_count=len(members),
                withdrawn_pixels=q['proposed_pixels'],source_fraction=fractions[q['node']],
                touches_another_outline=False,decision='apparent_soma_belongs_to_optical_source_at_its_own_area_scale'))
    result[:,y0:y1,x0:x1][lookup[nodes]]=0
    return result,pd.DataFrame(audit,columns=COLUMNS)
