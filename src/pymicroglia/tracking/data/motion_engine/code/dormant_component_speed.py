"""Per-detection reuse of components from immutable label frames.

The source detector does not mutate labels. Its later reconciler does; this
cache ends before reconciliation and is never shared between detection calls.
Masks are compressed within occupied bounds and reconstructed independently.
The byte budget governs execution memory only, never scientific eligibility.
"""
from collections import OrderedDict
import functools
import sys
import types
import numpy as np
from array_bounds import occupied_bounds


class ComponentCache:
    def __init__(self, original, labels, budget_bytes=32*1024*1024):
        self.original=original
        self.labels=labels  # Keep the known immutable frame storage alive.
        self.frames={self.token(frame) for frame in labels}
        self.entries=OrderedDict();self.bytes=0;self.budget=budget_bytes
        self.calls=0;self.misses=0;self.hits=0;self.evictions=0;self.peak_bytes=0

    @staticmethod
    def token(frame):
        return (int(frame.ctypes.data),frame.shape,frame.strides,frame.dtype.str)

    def __call__(self, frame, identity, minimum_area=8):
        self.calls+=1
        if type(frame) is not np.ndarray or frame.ndim!=2 or self.token(frame) not in self.frames:
            self.misses+=1
            return self.original(frame,identity,minimum_area)
        key=(self.token(frame),identity,minimum_area)
        if key in self.entries:
            self.hits+=1;items,cost=self.entries[key];self.entries.move_to_end(key)
            result=[]
            for values,bounds,packed in items:
                mask=np.zeros(frame.shape,dtype=bool)
                if bounds is not None:
                    y0,y1,x0,x1=bounds
                    mask[y0:y1,x0:x1]=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),count=(y1-y0)*(x1-x0)).reshape(y1-y0,x1-x0)
                result.append({name:(mask if name=='mask' else value) for name,value in values})
            return result
        self.misses+=1
        result=self.original(frame,identity,minimum_area)
        items=[];cost=512  # Conservative container/key overhead allowance.
        for item in result:
            mask=item['mask'];bounds=occupied_bounds(mask)
            packed=b''
            if bounds is not None:
                y0,y1,x0,x1=bounds
                packed=np.packbits(mask[y0:y1,x0:x1]).tobytes()
            values=tuple((name,None if name=='mask' else value) for name,value in item.items())
            items.append((values,bounds,packed))
            cost+=1024+sys.getsizeof(packed)+sum(sys.getsizeof(value) for _,value in values)
        if cost<=self.budget:
            while self.entries and self.bytes+cost>self.budget:
                _,(_,old_cost)=self.entries.popitem(last=False);self.bytes-=old_cost;self.evictions+=1
            self.entries[key]=(items,cost);self.bytes+=cost;self.peak_bytes=max(self.peak_bytes,self.bytes)
        return result


def make_detect(module, budget_bytes=32*1024*1024, statistics=None):
    if (getattr(module,"CACHES_OWN_COMPONENTS",False)
            and budget_bytes==32*1024*1024 and statistics is None):
        return module.detect  # It keeps its own per-call component cache.
    original=module.detect
    original_components=module.components
    original_namespace=dict(module.__dict__)
    functions={name:getattr(module,name) for name in ('largest','history','persistence','detect')}
    @functools.wraps(original)
    def detect(labels, *args, **kwargs):
        if type(labels) is not np.ndarray or labels.ndim!=3:
            return original(labels,*args,**kwargs)
        cache=ComponentCache(original_components,labels,budget_bytes)
        namespace=dict(original_namespace);namespace['components']=cache
        # Clone only these source functions so their calls resolve to the private
        # cache. No module-global mutation or cache reaches another worker.
        for name,fn in functions.items():
            namespace[name]=types.FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
        try:return namespace['detect'](labels,*args,**kwargs)
        finally:
            if statistics is not None:
                statistics.append(dict(calls=cache.calls,misses=cache.misses,hits=cache.hits,
                    evictions=cache.evictions,peak_accounted_bytes=cache.peak_bytes,
                    entries=len(cache.entries),budget_bytes=cache.budget))
            # Cloned functions refer back to their namespace. Break that cycle
            # explicitly so cached masks and frame references end with the call.
            cache.entries.clear()
            cache.frames.clear()
            cache.labels=None
            cache.bytes=0
            namespace.clear()
    return detect
