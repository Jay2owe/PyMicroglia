"""Preserve independent source witnesses across simultaneous recovery decisions."""
import numpy as np
import pandas as pd
from observed_soma_reservation import measured_cores_by_frame
from recovery_measured_history import constrain_measured_history


def constrain_consistent_history(observed,proposals,raw,points,params=None):
    candidate,audit=constrain_measured_history(observed,proposals,raw,points,params)
    if audit.empty:return candidate,audit
    quantized=points.copy();quantized['x']=np.floor(quantized.x+.5);quantized['y']=np.floor(quantized.y+.5)
    cores=measured_cores_by_frame(quantized);protected={}
    # Freeze every independently witnessed source before applying withdrawals.
    # Evidence that a body is present stays evidence even if its associated
    # correction is withheld. No processing-order or label-order winner exists.
    for q in audit.itertuples(index=False):
        first=int(q.first_measured_reference)-1;second=int(q.second_measured_reference)-1;t=int(q.frame)-1
        envelope=(observed[first]==q.identity)|(observed[second]==q.identity)
        span=range(second,t+1) if q.direction=='forward' else range(t,first+1)
        for frame in span:
            for _,y,x in cores.get(frame,[]):
                if 0<=y<envelope.shape[0] and 0<=x<envelope.shape[1] and envelope[y,x]:
                    protected.setdefault(frame,set()).add((y,x))
    rows=[]
    for q in audit.to_dict('records'):
        t=int(q['frame'])-1;name=int(q['identity']);target=(proposals[t]==name)&(observed[t]==0)
        conflicts=sum(bool(target[y,x]) for y,x in protected.get(t,set()))
        q['independent_source_conflicts']=conflicts
        if conflicts:
            candidate[t][target]=proposals[t][target]
            q['proposed_withdrawn_pixels']=q['withdrawn_pixels'];q['withdrawn_pixels']=0
            q['decision']='preserved_independent_source_used_by_another_proof'
        rows.append(q)
    return candidate,pd.DataFrame(rows)
