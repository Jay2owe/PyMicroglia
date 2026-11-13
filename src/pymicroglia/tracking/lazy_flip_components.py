"""Build immutable application maps only for frames used by accepted proposals."""
from contextlib import contextmanager
import hashlib
import inspect
import threading

SOURCE_SHA256='719b53ecb5b47a17e6d75d29bb518a069b5721b135fdd6c7f57d5b5186f67175'


@contextmanager
def install(module,scratch):
    # Standalone benchmark import and installed package import share this source.
    if __package__:
        from .frame_component_reuse import FrameCache
    else:
        from frame_component_reuse import FrameCache
    original=module.apply
    source=inspect.getsource(original).rstrip()
    counters=dict(component_hits=0,component_misses=0,uncached_calls=0,
        reserved_disk_bytes=0,cached_vector_bytes=0,peak_live_maps=0,source_supported=False)
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA256:
        yield counters
        return
    counters['source_supported']=True
    block='''    component_maps = [
        component_accounting.label_value_components(frame)[0]
        for frame in labels]'''
    assert source.count(block)==1
    source=source.replace(block,'    component_maps = _application_maps()')
    local=threading.local()
    class Maps:
        def __init__(self,cache):self.cache=cache
        def __getitem__(self,index):
            return self.cache.get(self.cache.labels[index],module.component_accounting.label_value_components)[0]
    namespace=dict(original.__globals__)
    namespace['_application_maps']=lambda: Maps(local.cache)
    exec(compile(source,__file__,'exec'),namespace)
    replacement=namespace[original.__name__]
    def apply(labels,*args,**kwargs):
        previous=getattr(local,'cache',None)
        # Application keeps at most one live map, as before.
        cache=FrameCache(labels,scratch,counters,open_maps=0)
        local.cache=cache
        try:return replacement(labels,*args,**kwargs)
        finally:
            local.cache=previous
            cache.close()
    module.apply=apply
    try:yield counters
    finally:module.apply=original
