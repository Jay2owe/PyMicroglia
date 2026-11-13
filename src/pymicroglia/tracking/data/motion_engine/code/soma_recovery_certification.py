"""Accept gap-recovery hypotheses only with independent evidence of a cell.

The original recovery remains a hypothesis generator. Its inferred pixels
cannot establish their own soma evidence or open extra recovery claims when
another claim is refused. Every decision uses the same unmodified proposals.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from observed_soma_reservation import measured_cores_by_frame
from soma_interior_evidence import usual_measured_interior


def certify_recoveries(observed,proposals,raw,points,params):
    if observed.shape!=proposals.shape or observed.shape!=raw.shape:
        raise ValueError('Observed, proposed and raw movies must have identical shape')
    if np.any((observed>0)&(proposals!=observed)):
        raise ValueError('Soma certification expects non-carving gap recovery')
    quantized=points.copy()
    quantized['x']=np.floor(quantized.x+0.5)
    quantized['y']=np.floor(quantized.y+0.5)
    centres=measured_cores_by_frame(quantized)
    memory=int(params.get('descriptor_memory',9))
    intensity=float(params.get('minimum_intensity',120.0))
    measured={};hypothesised={};body_evidence={};gap_witness={};attached={}
    for t,frame in enumerate(observed):
        for name in np.unique(frame):
            if name:measured.setdefault(int(name),[]).append(t)
        inside=frame[1:-1,1:-1]
        bright=raw[t]>=intensity
        interior=((inside>0)&(inside==frame[:-2,1:-1])&(inside==frame[2:,1:-1])
                  &(inside==frame[1:-1,:-2])&(inside==frame[1:-1,2:])
                  &bright[1:-1,1:-1]&bright[:-2,1:-1]&bright[2:,1:-1]
                  &bright[1:-1,:-2]&bright[1:-1,2:])
        supported=set(map(int,np.unique(inside[interior])))
        witnesses=set()
        for track,y,x in centres.get(t,[]):
            if not (0<=y<frame.shape[0] and 0<=x<frame.shape[1]):continue
            source=int(frame[y,x]);proposed=int(proposals[t,y,x])
            if source:supported.add(source)
            elif proposed:witnesses.add(proposed)
        body_evidence[t]=supported;gap_witness[t]=witnesses
        # A fragment touching an independently supported observed body is an
        # ownership question, not an unsupported detached speckle. Keep it.
        body=np.isin(frame,list(supported)) if supported else np.zeros(frame.shape,bool)
        contact=ndi.binary_dilation(body,np.ones((3,3),bool),border_value=0)
        attached[t]=set(map(int,np.unique(proposals[t][contact&(frame==0)])))-{0}
        for name in np.unique(proposals[t][(frame==0)&(proposals[t]>0)]):
            hypothesised.setdefault(int(name),set()).add(t)
    output=proposals.copy();audit=[]
    for name,claimed_frames in hypothesised.items():
        actual=measured.get(name,[])
        boundaries=[-1]+actual+[len(observed)]
        for j,(left,right) in enumerate(zip(boundaries[:-1],boundaries[1:])):
            frames=sorted(claimed_frames.intersection(range(left+1,right)))
            if not frames:continue
            before=actual[max(0,j-memory):j]
            after=actual[j:j+memory]
            left_votes=[int(name in body_evidence[t]) for t in before]
            right_votes=[int(name in body_evidence[t]) for t in after]
            established=(usual_measured_interior(left_votes)>0 or usual_measured_interior(right_votes)>0)
            witnessed=any(name in gap_witness[t] for t in frames)
            connected=any(name in attached[t] for t in frames)
            supported=established or witnessed or connected
            pixels=0
            if not supported:
                for t in frames:
                    rejected=(proposals[t]==name)&(observed[t]==0)
                    pixels+=int(rejected.sum());output[t][rejected]=0
            audit.append(dict(identity=name,first_gap_frame=left+2,last_gap_frame=right,
                inferred_frames=len(frames),first_inferred_frame=frames[0]+1,last_inferred_frame=frames[-1]+1,
                measured_before_frames=len(before),measured_after_frames=len(after),
                measured_before_support=sum(left_votes),measured_after_support=sum(right_votes),
                established_soma=established,independent_soma_in_gap=witnessed,connected_to_observed_body=connected,
                decision='supported_recovery_retained' if supported else 'unsupported_detached_recovery_withdrawn',
                withdrawn_pixels=pixels))
    assert not np.any((output!=proposals)&(observed>0))
    assert not np.any((output!=proposals)&(output>0))
    return output,pd.DataFrame(audit)
