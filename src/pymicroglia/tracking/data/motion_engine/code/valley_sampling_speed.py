"""Vectorize the unchanged rounded line samples used by contact evidence."""
from contextlib import contextmanager
from functools import lru_cache
import numpy as np


@contextmanager
def vectorized_valleys(module):
    original=module._valley_ratio
    @lru_cache(maxsize=256)
    def weights(samples):
        weight=np.linspace(0.,1.,samples)
        previous=1-weight
        weight.setflags(write=False);previous.setflags(write=False)
        return weight,previous
    def ratio(frame,left,right):
        distance=float(np.hypot(left[0]-right[0],left[1]-right[1]))
        samples=max(3,int(np.ceil(distance))+1)
        weight,previous=weights(samples)
        x=np.rint(previous*left[0]+weight*right[0])
        y=np.rint(previous*left[1]+weight*right[1])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return original(frame,left,right)
        x=np.clip(x,0,frame.shape[1]-1).astype(np.intp)
        y=np.clip(y,0,frame.shape[0]-1).astype(np.intp)
        values=frame[y,x].astype(float)
        if not np.isfinite(values).all():return original(frame,left,right)
        endpoint=max(min(float(values[0]),float(values[-1])),1.)
        interior=values[1:-1] if len(values)>2 else values
        return float(np.clip(float(np.min(interior))/endpoint,0.,2.))
    module._valley_ratio=ratio
    try:yield
    finally:module._valley_ratio=original
