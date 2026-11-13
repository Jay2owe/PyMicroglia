"""Bounded, content-verified reuse of repeated Gaussian contrast images."""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
from threading import RLock
import numpy as np


@contextmanager
def cached_evidence_images(module, maximum_bytes=128*1024*1024):
    original=module.evidence_image;cache=OrderedDict();lock=RLock()
    statistics=dict(hits=0,misses=0,bytes=0,evictions=0)
    def evidence(frame,params):
        array=np.ascontiguousarray(frame)
        core=float(params.get('core_sigma_px',1.2));background=float(params.get('background_sigma_px',8.))
        if array.dtype.hasobject or not np.isfinite([core,background]).all():return original(frame,params)
        key=(array.shape,array.dtype.str,core,background,hashlib.sha256(memoryview(array).cast('B')).digest())
        with lock:
            if key in cache:
                statistics['hits']+=1;cache.move_to_end(key);return cache[key].copy()
            statistics['misses']+=1
        result=original(frame,params)
        if result.nbytes<=maximum_bytes:
            stored=result.copy()
            with lock:
                if key not in cache:
                    while cache and statistics['bytes']+stored.nbytes>maximum_bytes:
                        _,removed=cache.popitem(last=False);statistics['bytes']-=removed.nbytes;statistics['evictions']+=1
                    cache[key]=stored;statistics['bytes']+=stored.nbytes
        return result
    module.evidence_image=evidence
    try:yield statistics
    finally:module.evidence_image=original
