"""Reuse each transition's occupied coordinates for bounds and motion means."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np

SOURCE='34603089a862624829d01040ba36c00af126b76b787d9fa3abc0217af1b90658'


def evidence(old,new,lag_frame,module):
    old=np.asarray(old);new=np.asarray(new);lag_frame=np.asarray(lag_frame)
    before=np.column_stack(np.nonzero(old));after=np.column_stack(np.nonzero(new))
    def bounds(points):
        if not len(points):return 0,0,0,0
        low=points.min(axis=0);high=points.max(axis=0)+1
        return int(low[0]),int(high[0]),int(low[1]),int(high[1])
    support=module.motion_support(old,new,lag_frame,bounds(before),bounds(after))
    blue=before[lag_frame[before[:,0],before[:,1]]<=-1.5]
    red=after[lag_frame[after[:,0],after[:,1]]>=1.5]
    displacement=np.mean(after,axis=0)-np.mean(before,axis=0)
    cosine=float('nan')
    if len(blue)>=8 and len(red)>=8:
        motion_vector=red.mean(axis=0)-blue.mean(axis=0)
        denominator=float(np.linalg.norm(motion_vector)*np.linalg.norm(displacement))
        if denominator>1e-12:cosine=float(np.dot(motion_vector,displacement)/denominator)
    return {**support,'direction_cosine':cosine}


@contextmanager
def install(module):
    stats=dict(supported=False,calls=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE:
        yield stats;return
    original=module._transition_evidence
    def transition(old,new,lag_frame):
        if old.ndim!=2 or new.shape!=old.shape or lag_frame.shape!=old.shape or old.dtype!=np.bool_ or new.dtype!=np.bool_:
            return original(old,new,lag_frame)
        stats['calls']+=1
        return evidence(old,new,lag_frame,module)
    module._transition_evidence=transition;stats['supported']=True
    try:yield stats
    finally:module._transition_evidence=original
