"""Lazy boolean masks and frame-buffered review IO for the frozen engine.

Every mask is the same equality test against its immutable observation image.
This avoids retaining a full camera-sized boolean image for every cell at every
timepoint. Display buffering retains the global contrast and temporal edge halo.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import sys, tempfile, threading, weakref
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

class LabelMask(NDArrayOperatorsMixin):
    __array_priority__=1000
    def __init__(self,labels,identity):
        self.labels=labels;self.identity=int(identity)
    @property
    def shape(self):return self.labels.shape
    @property
    def dtype(self):return np.dtype(bool)
    @property
    def ndim(self):return self.labels.ndim
    @property
    def size(self):return self.labels.size
    def __len__(self):return len(self.labels)
    def __array__(self,dtype=None,copy=None):
        value=self.labels==self.identity
        return value if dtype is None else value.astype(dtype,copy=False)
    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        converted=tuple(np.asarray(v) if isinstance(v,LabelMask) else v for v in inputs)
        if kwargs.get('out') is not None:
            kwargs['out']=tuple(np.asarray(v) if isinstance(v,LabelMask) else v for v in kwargs['out'])
        return getattr(ufunc,method)(*converted,**kwargs)
    def __getitem__(self,key):return self.labels[key]==self.identity
    def __getattr__(self,name):
        if name.startswith('__'):raise AttributeError(name)
        return getattr(np.asarray(self),name)

class ObservationLabels(np.ndarray):
    """Let the unchanged reader retain equality views instead of full masks."""
    def __eq__(self,identity):
        if isinstance(identity,(int,np.integer)):
            return LabelMask(self.view(np.ndarray),identity)
        return super().__eq__(identity)

def observation_reader(original,copy_labels=None):
    def frame(labels,raw,*args,**kwargs):
        # The original Boolean masks are snapshots, even when callers later
        # relabel their working image. Share one immutable integer snapshot
        # between the cells instead of retaining one full Boolean image each.
        snapshot=np.array(labels,copy=True) if copy_labels is None else copy_labels(labels)
        snapshot.flags.writeable=False
        result=original(snapshot.view(ObservationLabels),raw,*args,**kwargs)
        for identity,row in result.items():row['mask']=LabelMask(snapshot,identity)
        return result
    return frame

def overlay_renderer(common,scratch):
    from .display_scale import positive_display_high
    original=common.outline_overlay
    original_display=common.display_raw
    lock=threading.RLock()
    def render(raw,labels,thick=2,inferred=None,unresolved=None):
        high=positive_display_high(raw)
        handle=tempfile.NamedTemporaryFile(prefix='rgb-',suffix='.dat',dir=scratch,delete=False)
        path=Path(handle.name);handle.close()
        output=np.memmap(path,mode='w+',dtype=np.uint8,shape=(*raw.shape,3))
        def cleanup(mapping,filename):
            mapping.close();filename.unlink(missing_ok=True)
        weakref.finalize(output,cleanup,output._mmap,path)
        def display(chunk):
            grey=np.clip(chunk.astype(np.float32)/max(high,1.),0,1)**.7
            return np.repeat((grey[...,None]*180).astype(np.uint8),3,axis=-1)
        halo=max(int(thick)-1,1)
        with lock:
            common.display_raw=display
            try:
                for t in range(len(raw)):
                    lo,hi=max(0,t-halo),min(len(raw),t+halo+1)
                    chunk=original(raw[lo:hi],labels[lo:hi],thick=thick,
                        inferred=None if inferred is None else inferred[lo:hi],
                        unresolved=None if unresolved is None else unresolved[lo:hi])
                    output[t]=chunk[t-lo]
            finally:common.display_raw=original_display
        return output
    return render

@contextmanager
def install(common,code_root,scratch,*,array_pool=None):
    """Replace storage adapters across engine aliases, then restore every alias."""
    import importlib
    tracking=importlib.import_module('tracking')
    old_observation=tracking.frame_observations
    old_overlay=common.outline_overlay
    copy_labels=None if array_pool is None else lambda labels:array_pool.array(labels,copy=True,force=True)
    replacements={old_observation:observation_reader(old_observation,copy_labels),
                  old_overlay:overlay_renderer(common,scratch)}
    aliases=[]
    for module in list(sys.modules.values()):
        location=getattr(module,'__file__',None)
        if location is None:continue
        try:inside=Path(location).resolve().is_relative_to(code_root.resolve())
        except (ValueError,OSError):inside=False
        if not inside:continue
        for name in ('frame_observations','outline_overlay'):
            old=getattr(module,name,None)
            if old in replacements:
                aliases.append((module,name,old));setattr(module,name,replacements[old])
    try:yield
    finally:
        for module,name,old in reversed(aliases):setattr(module,name,old)
