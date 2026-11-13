"""Reuse exact local motion maxima during the existing immutable-frame linker.

A link's support is the largest 3 x 3 maximum of the motion plane at seven
points along it. For real coordinates the points are found with Python floats:
the same float64 products and sums as seven-element NumPy arrays of
np.linspace's weights and their complements, Python's round is NumPy's rint
(half to even), and the bounds tests and gathered maxima are the same, in the
same order. Their maximum is taken in Python; only a zero result is taken again
with np.max, the one case (signed zeros) where equal values can differ in bits.
Other coordinates use the NumPy arrays.
"""
from contextlib import contextmanager
import math
from numbers import Integral
import numpy as np
from scipy import ndimage as ndi

EXACT=1<<53


def _real(value):
    """value as the exact float64 NumPy would use, or None to keep NumPy."""
    if isinstance(value,(float,np.float64,np.float32,np.float16)):return float(value)
    if isinstance(value,Integral) and not isinstance(value,(bool,np.bool_)) and -EXACT<=int(value)<=EXACT:
        return float(int(value))
    return None


@contextmanager
def cached_motion_samples(module):
    original=module._motion_support_plane
    cache={}
    weights=np.linspace(0.,1.,7)
    previous_weights=1-weights
    pairs=list(zip(previous_weights.tolist(),weights.tolist()))
    def support(plane,x0,y0,x1,y1):
        if plane is None:return original(plane,x0,y0,x1,y1)
        if plane.dtype.kind not in 'biuf' or (plane.dtype.kind=='f' and plane.dtype.itemsize not in (4,8)):
            return original(plane,x0,y0,x1,y1)
        if cache.get('plane') is not plane:
            finite=bool(np.isfinite(plane).all())
            cache.clear();cache.update(plane=plane,finite=finite)
            if finite:
                fill=np.iinfo(plane.dtype).min if plane.dtype.kind in 'iu' else (-np.inf if plane.dtype.kind=='f' else 0)
                padded=np.pad(plane,1,mode='constant',constant_values=fill)
                cache['maxima']=ndi.maximum_filter(padded,size=3,mode='constant',cval=fill)
        coordinates=(_real(x0),_real(y0),_real(x1),_real(y1))
        if not cache['finite'] or None in coordinates:return numpy_support(plane,x0,y0,x1,y1)
        fx0,fy0,fx1,fy1=coordinates
        if not (math.isfinite(fx0) and math.isfinite(fy0) and math.isfinite(fx1) and math.isfinite(fy1)):
            return original(plane,x0,y0,x1,y1)
        height,width=plane.shape[0],plane.shape[1]
        maxima=cache['maxima']
        found=[]
        for before,after in pairs:
            x=round(before*fx0+after*fx1)
            y=round(before*fy0+after*fy1)
            if -1<=x<=width and -1<=y<=height:found.append(maxima[y+1,x+1])
        if not found:value=0.
        else:
            value=float(max(found))
            if value==0:value=float(np.max(np.array(found)))
        value=value/65535.
        return value if 0.<=value<=1. else (0. if value<0. else 1.)
    def numpy_support(plane,x0,y0,x1,y1):
        if not cache['finite'] or not np.isfinite([x0,y0,x1,y1]).all():return original(plane,x0,y0,x1,y1)
        x=np.rint(previous_weights*x0+weights*x1)
        y=np.rint(previous_weights*y0+weights*y1)
        valid=(x>=-1)&(x<=plane.shape[1])&(y>=-1)&(y<=plane.shape[0])
        if not valid.any():value=0.
        else:value=float(np.max(cache['maxima'][y[valid].astype(np.intp)+1,x[valid].astype(np.intp)+1]))
        return float(np.clip(value/65535.,0.,1.))
    module._motion_support_plane=support
    try:yield
    finally:module._motion_support_plane=original
