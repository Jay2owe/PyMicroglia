"""Do not invent transit between two independently occupied endpoint somata.

This constrains fixed additive recovery proposals. Both disjoint measured
endpoint territories must contain independently qualified raw observations in
EVERY frame across the gap. Direct foreground, actual somata at the destination,
measured neighbours, and later accepted revisions are preserved by the caller.
No recording, identity, frame, feedback or coordinate is a rule selector.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from recovery_temporal_soma_support import temporally_supported_points
from inferred_raw_soma_evidence import has_raw_soma

COLUMNS=['identity','frame','first_gap_frame','last_gap_frame','withdrawn_pixels',
         'left_frame','right_frame','left_source_similarity','left_moved_similarity',
         'right_source_similarity','right_moved_similarity','decision']

def constrain_coexisting_endpoints(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels):
    if not (observed.shape==proposals.shape==raw.shape==current_labels.shape):
        raise ValueError('Aligned complete movies required')
    if np.any((observed>0)&(observed!=proposals)):
        raise ValueError('Fixed additive recovery proposals required')
    # Crop acquisition padding exactly as the accepted spatial evidence stage.
    yy,xx=np.nonzero(np.any(raw!=0,axis=0))
    if not len(yy):return current_labels.copy(),pd.DataFrame(columns=COLUMNS)
    y0,y1,x0,x1=int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1
    local=points.copy();local.y-=y0;local.x-=x0
    valid=temporally_supported_points(raw[:,y0:y1,x0:x1],local,centre_connected=True,internal_peak=True,peak_quantisation=True)
    valid=valid[valid.strong.astype(bool)&valid.state.eq('observed')].copy()
    if 'physically_visible' in valid:valid=valid[valid.physically_visible.astype(bool)]
    elif 'visibility' in valid:valid=valid[valid.visibility.gt(0)]
    valid.y+=y0;valid.x+=x0
    centres={}
    for q in valid.itertuples(index=False):
        t=int(q.frame);y=int(np.floor(q.y+.5));x=int(np.floor(q.x+.5))
        if 0<=t<len(raw) and 0<=y<raw.shape[1] and 0<=x<raw.shape[2]:
            centres.setdefault(t,[]).append((y,x))
    measured={};inferred={}
    for t,frame in enumerate(observed):
        for name in np.unique(frame):
            if name:measured.setdefault(int(name),[]).append(t)
        added=(frame==0)&(proposals[t]>0)&(current_labels[t]==proposals[t])
        for name in np.unique(proposals[t][added]):inferred.setdefault(int(name),set()).add(t)
    result=current_labels.copy();audit=[];contact={};eight=np.ones((3,3),bool)
    for name,frames in inferred.items():
        actual=measured.get(name,[])
        for left,right in zip(actual[:-1],actual[1:]):
            gap=sorted(frames.intersection(range(left+1,right)))
            if not gap:continue
            sources=[observed[left]==name,observed[right]==name]
            if np.any(sources[0]&sources[1]):continue
            # A source must be a single measured body, not a union of aliases.
            if any(ndi.label(s,eight)[1]!=1 for s in sources):continue
            if not all(any(s[y,x] for y,x in centres.get(t,[]))
                       for s in sources for t in range(left,right+1)):continue
            templates=[]
            for anchor,s in zip((left,right),sources):
                sy,sx=np.nonzero(s);ref=raw[anchor,sy,sx].astype(float)
                templates.append((sy,sx,ref,float(np.dot(ref,ref))))
            for t in gap:
                target=(proposals[t]==name)&(observed[t]==0)&(current_labels[t]==name)
                if not target.any() or any(np.any(target&s) for s in sources):continue
                if t not in contact:contact[t]=ndi.binary_dilation(observed[t]>0,eight)
                if np.any(target&contact[t]):continue
                # A genuine third soma is an ownership problem, not empty
                # foreground. Test each proposed component in its own box.
                components,count=ndi.label(target,eight)
                boxes=ndi.find_objects(components)
                if any(has_raw_soma(components[box]==i,raw[t][box])
                       for i,box in enumerate(boxes,1)):continue
                if any(target[y,x] for y,x in centres.get(t,[])):continue
                ty,tx=np.nonzero(target);scores=[]
                for sy,sx,ref,energy in templates:
                    shift=np.floor([ty.mean()-sy.mean(),tx.mean()-sx.mean()]+np.array([.5,.5])).astype(int)
                    dy,dx=sy+shift[0],sx+shift[1]
                    if energy<=0 or dy.min()<0 or dx.min()<0 or dy.max()>=raw.shape[1] or dx.max()>=raw.shape[2]:break
                    def similarity(v):return float(np.dot(ref,v)/max(np.sqrt(energy*np.dot(v,v)),1.))
                    held=similarity(raw[t,sy,sx].astype(float));moved=similarity(raw[t,dy,dx].astype(float))
                    if held<=moved:break
                    scores.append((held,moved))
                if len(scores)!=2:continue
                result[t][target]=0
                audit.append(dict(identity=name,frame=t+1,first_gap_frame=left+2,last_gap_frame=right,
                    withdrawn_pixels=int(target.sum()),left_frame=left+1,right_frame=right+1,
                    left_source_similarity=scores[0][0],left_moved_similarity=scores[0][1],
                    right_source_similarity=scores[1][0],right_moved_similarity=scores[1][1],
                    decision='two_coexisting_sources_cannot_certify_invented_transit'))
    return result,pd.DataFrame(audit,columns=COLUMNS)
