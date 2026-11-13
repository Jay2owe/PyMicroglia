"""Bounded, content-verified reuse of normalized cell-appearance patches."""
from collections import OrderedDict
from contextlib import contextmanager


@contextmanager
def cached_patches(module, capacity=4096):
    original=module._patch
    cache=OrderedDict();statistics=dict(hits=0,misses=0)
    def patch(frame,x,y,radius=4):
        cy,cx=int(round(y)),int(round(x))
        if capacity<=0 or radius<0 or not (0<=cy<frame.shape[0] and 0<=cx<frame.shape[1]):
            return original(frame,x,y,radius)
        y0,y1=max(0,cy-radius),min(frame.shape[0],cy+radius+1)
        x0,x1=max(0,cx-radius),min(frame.shape[1],cx+radius+1)
        region=frame[y0:y1,x0:x1]
        key=(radius,cy-y0,cx-x0,region.shape,region.dtype.str,region.tobytes())
        if key in cache:
            statistics['hits']+=1;cache.move_to_end(key)
            return cache[key].copy()
        statistics['misses']+=1
        result=original(frame,x,y,radius)
        cache[key]=result.copy()
        if len(cache)>capacity:cache.popitem(last=False)
        return result
    module._patch=patch
    try:yield statistics
    finally:module._patch=original
