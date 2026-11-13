"""Exact review pixels with per-render fonts and bounded locator morphology."""
from contextlib import contextmanager
from functools import lru_cache
import numpy as np
from scipy import ndimage


def marked_changes_local(field,renderer):
    mask=renderer.change_mask(field);out=field.copy();yy,xx=np.nonzero(mask)
    if not len(yy):return out
    y0=max(0,int(yy.min())-8);y1=min(field.shape[0],int(yy.max())+9)
    x0=max(0,int(xx.min())-8);x1=min(field.shape[1],int(xx.max())+9)
    groups,_=ndimage.label(ndimage.binary_dilation(mask[y0:y1,x0:x1],iterations=8))
    for box in ndimage.find_objects(groups):
        if box is None:continue
        ys,xs=box;a,b=ys.start+y0,ys.stop+y0;c,d=xs.start+x0,xs.stop+x0
        out[a:min(a+2,b),c:d]=renderer.YELLOW;out[max(a,b-2):b,c:d]=renderer.YELLOW
        out[a:b,c:min(c+2,d)]=renderer.YELLOW;out[a:b,max(c,d-2):d]=renderer.YELLOW
    out[mask]=renderer.RED
    return out


@contextmanager
def speed_review_render(renderer):
    original_font,original_mark=renderer.font,renderer.marked_changes
    cached=lru_cache(maxsize=None)(original_font)
    canonical_font=getattr(original_font,'__module__',None)==renderer.__name__
    canonical_mark=getattr(original_mark,'__module__',None)==renderer.__name__ and getattr(original_mark,'__name__',None)=='marked_changes'
    if canonical_font:renderer.font=cached
    if canonical_mark:renderer.marked_changes=lambda field:marked_changes_local(field,renderer)
    try:yield cached
    finally:
        renderer.font,renderer.marked_changes=original_font,original_mark
        cached.cache_clear()
