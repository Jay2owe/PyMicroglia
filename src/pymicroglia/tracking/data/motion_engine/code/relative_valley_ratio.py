"""Experimental gain-relative contact contrast; not enabled in production."""
from contextlib import contextmanager
import numpy as np


@contextmanager
def relative_valleys(module):
    original=module._valley_ratio
    def ratio(frame,left,right):
        distance=float(np.hypot(left[0]-right[0],left[1]-right[1]))
        weight=np.linspace(0.,1.,max(3,int(np.ceil(distance))+1))
        x=np.rint((1-weight)*left[0]+weight*right[0]);y=np.rint((1-weight)*left[1]+weight*right[1])
        if not np.isfinite(x).all() or not np.isfinite(y).all():return original(frame,left,right)
        x=np.clip(x,0,frame.shape[1]-1).astype(np.intp);y=np.clip(y,0,frame.shape[0]-1).astype(np.intp)
        values=frame[y,x].astype(float)
        if not np.isfinite(values).all():return original(frame,left,right)
        endpoint=min(float(values[0]),float(values[-1]));interior=float(np.min(values[1:-1]))
        if endpoint>0:return float(np.clip(interior/endpoint,0.,2.))
        return 2. if interior>0 else 0.
    ratio._motion_relative_valley=True
    module._valley_ratio=ratio
    try:yield
    finally:module._valley_ratio=original
