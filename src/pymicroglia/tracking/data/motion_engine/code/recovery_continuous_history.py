"""Prevent inferred movement contradicted by an independently resident source.

Two consecutive available measured observations establish source appearance;
every intervening frame must independently witness the source. The far
side of the gap may have a wrong identity assignment and is not an appearance
reference. Certification stops when the independent source evidence is lost.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from observed_soma_reservation import measured_cores_by_frame


def constrain_continuous_history(observed, proposals, raw, points, params=None):
    if observed.shape != proposals.shape or observed.shape != raw.shape:
        raise ValueError('Movie shapes differ')
    if np.any((observed>0)&(observed!=proposals)):
        raise ValueError('Expected additive non-carving proposals')
    quantized=points.copy();quantized['x']=np.floor(quantized.x+.5);quantized['y']=np.floor(quantized.y+.5)
    cores=measured_cores_by_frame(quantized)
    measured={};inferred={}
    for t,frame in enumerate(observed):
        for name in np.unique(frame):
            if name:measured.setdefault(int(name),[]).append(t)
        added=(frame==0)&(proposals[t]>0)
        for name in np.unique(proposals[t][added]):inferred.setdefault(int(name),set()).add(t)
    output=proposals.copy();audit=[];contact={}
    def inside(mask,y,x):return 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]
    for name,frames in inferred.items():
        actual=measured.get(name,[])
        for index,(left,right) in enumerate(zip(actual[:-1],actual[1:])):
            gap=sorted(frames.intersection(range(left+1,right)))
            if not gap:continue
            witnesses=[]
            # Consecutive available measured frames provide the appearance reference.
            # A guess from an earlier gap never supplies either observation.
            if index>0:
                witnesses.append((actual[index-1],left,range(left+1,right),'forward'))
            if index+2<len(actual):
                witnesses.append((right,actual[index+2],range(right-1,left,-1),'backward'))
            decisions={}
            for first,second,scan,direction in witnesses:
                sources=[observed[first]==name,observed[second]==name]
                if not np.any(sources[0]&sources[1]):continue
                envelope=sources[0]|sources[1]
                if not all(any(inside(mask,y,x) for _,y,x in cores.get(t,[])) for t,mask in zip((first,second),sources)):continue
                if not all(any(inside(envelope,y,x) for _,y,x in cores.get(t,[])) for t in range(first,second+1)):continue
                templates=[]
                for anchor,mask in zip((first,second),sources):
                    yy,xx=np.nonzero(mask);ref=raw[anchor,yy,xx].astype(float)
                    templates.append((yy,xx,ref,float(np.dot(ref,ref))))
                for t in scan:
                    if not any(inside(envelope,y,x) for _,y,x in cores.get(t,[])):break
                    if t not in frames or t in decisions:continue
                    target=(proposals[t]==name)&(observed[t]==0)
                    if np.any(target&envelope):continue
                    if t not in contact:contact[t]=ndi.binary_dilation(observed[t]>0,np.ones((3,3),bool))
                    if np.any(target&contact[t]):continue
                    ty,tx=np.nonzero(target);scores=[]
                    if not len(ty):continue
                    for yy,xx,ref,energy in templates:
                        shift=np.floor(np.array([ty.mean()-yy.mean(),tx.mean()-xx.mean()])+.5).astype(int)
                        dy,dx=yy+shift[0],xx+shift[1]
                        if energy<=0 or min(dy.min(),dx.min())<0 or dy.max()>=raw.shape[1] or dx.max()>=raw.shape[2]:break
                        def similarity(v):
                            denominator=np.sqrt(energy*np.dot(v,v))
                            return float(np.dot(ref,v)/denominator) if denominator>0 else 0.
                        held=similarity(raw[t,yy,xx].astype(float));moved=similarity(raw[t,dy,dx].astype(float))
                        if held<=moved:break
                        scores.append((held,moved))
                    if len(scores)!=2:continue
                    decisions[t]=dict(identity=name,frame=t+1,first_gap_frame=left+2,last_gap_frame=right,
                        first_measured_reference=first+1,second_measured_reference=second+1,direction=direction,
                        occupancy_evidence='continuous_source_between_and_since_measured_observations',withdrawn_pixels=int(target.sum()),
                        source_similarity_left=scores[0][0],moved_similarity_left=scores[0][1],
                        source_similarity_right=scores[1][0],moved_similarity_right=scores[1][1],
                        decision='measured_source_history_contradicts_inferred_departure')
            for t,record in sorted(decisions.items()):
                output[t][(proposals[t]==name)&(observed[t]==0)]=0;audit.append(record)
    return output,pd.DataFrame(audit)
