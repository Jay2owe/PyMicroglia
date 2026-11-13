"""Reuse immutable frame components only during one identity discovery call.

Keep small owner/area vectors in RAM and component maps in one disposable file.
Each returned map owns its mapping; the cache retains metadata, not live maps.
No thresholds, candidates, physical links or iteration order change.
"""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import threading
import weakref
import numpy as np

# Recently read maps stay open; each is read-only and released on close.
OPEN_MAPS = 32


def key(frame):
    return (int(frame.__array_interface__['data'][0]), frame.shape,
            frame.strides, frame.dtype.str)


class MappingLease:
    pass


class CachedFrame(np.memmap):
    def __array_finalize__(self, source):
        super().__array_finalize__(source)
        self._lease = getattr(source, '_lease', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Derived masks are independent ndarrays, not mapped aliases. Keeping
        # the subclass would retain maps and change downstream fast-path dispatch.
        plain = tuple(np.asarray(v) if isinstance(v, CachedFrame) else v for v in inputs)
        output = kwargs.get('out')
        if output is not None:
            kwargs['out'] = tuple(np.asarray(v) if isinstance(v, CachedFrame) else v for v in output)
        result = getattr(ufunc, method)(*plain, **kwargs)
        if output is None:
            return result
        if ufunc.nout == 1:
            return result if output[0] is None else output[0]
        return tuple(value if original is None else original for value, original in zip(result, output))


class FrameCache:
    def __init__(self, labels, scratch, counters, open_maps=OPEN_MAPS):
        self.labels = labels  # Keep backing addresses valid throughout discovery.
        self.frames = {key(frame): index for index, frame in enumerate(labels)}
        self.rows = {}
        self.path = None
        self.handle = None
        self.open = OrderedDict()
        self.open_maps = int(open_maps)
        self.file_released = False
        self.scratch = Path(scratch)
        self.counters = counters
        self.closed = False
        self.live_maps = 0

    def mapped(self, index, dtype, shape, mode):
        size = int(np.prod(shape)) * dtype.itemsize
        # Map through the one open handle; resolving and reopening the path
        # for every frame cost more than reading the frame.
        plain = np.memmap(self.handle, dtype=dtype, mode=mode, offset=index * size, shape=shape)
        lease = MappingLease()
        self.live_maps += 1
        self.counters['peak_live_maps'] = max(self.counters['peak_live_maps'], self.live_maps)
        def release(mapping):
            if not mapping.closed:
                mapping.close()
            self.live_maps -= 1
            if self.closed and self.live_maps == 0:
                self.release_file()
        lease.finalizer = weakref.finalize(lease, release, plain._mmap)
        plain._lease = lease
        return plain.view(CachedFrame)

    def get(self, frame, original):
        index = self.frames.get(key(frame))
        if index is None:
            self.counters['uncached_calls'] += 1
            return original(frame)
        if index in self.rows:
            self.counters['component_hits'] += 1
            dtype, shape, owners, areas = self.rows[index]
            mapped = self.open.get(index)
            if mapped is None:
                mapped = self.remember(index, self.mapped(index, dtype, shape, 'r'))
            else:
                self.open.move_to_end(index)
            return mapped, owners, areas
        self.counters['component_misses'] += 1
        components, owners, areas = original(frame)
        shape, dtype = components.shape, components.dtype
        size = components.nbytes
        if self.path is None:
            total = len(self.labels) * size
            self.scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.scratch).free < total + 2 * 1024**3:
                raise OSError('Insufficient space for immutable component cache')
            with tempfile.NamedTemporaryFile(prefix='frame-components-', suffix='.dat',
                    dir=self.scratch, delete=False) as handle:
                self.path = Path(handle.name)
                handle.truncate(total)
            self.handle = open(self.path, 'r+b')
            self.counters['reserved_disk_bytes'] += total
            self.layout = shape, dtype
        if (shape, dtype) != self.layout:
            # A different representation must not reinterpret the packed file.
            self.counters['uncached_calls'] += 1
            return components, owners, areas
        mapped = self.mapped(index, dtype, shape, 'r+')
        np.copyto(mapped, components)
        mapped.flags.writeable = False
        owners = np.array(owners, copy=True)
        areas = np.array(areas, copy=True)
        owners.flags.writeable = areas.flags.writeable = False
        self.rows[index] = (dtype, shape, owners, areas)
        self.remember(index, mapped)
        self.counters['cached_vector_bytes'] += owners.nbytes + areas.nbytes
        return mapped, owners, areas

    def remember(self, index, mapped):
        if not self.open_maps:
            return mapped
        self.open[index] = mapped
        if len(self.open) > self.open_maps:
            self.open.popitem(last=False)
        return mapped

    def close(self):
        self.closed = True
        self.open.clear()
        self.rows.clear()
        self.frames.clear()
        self.labels = None
        if self.path is not None and self.live_maps == 0:
            self.release_file()

    def release_file(self):
        # Clearing the open maps may already have released the last one.
        if self.file_released:
            return
        self.file_released = True
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.path.unlink()


@contextmanager
def install(module, scratch):
    """Install around execution; the cache exists only inside discover()."""
    local = threading.local()
    accounting = module.component_accounting
    old_components = accounting.label_value_components
    old_discover = module.discover
    counters = dict(component_hits=0, component_misses=0, uncached_calls=0,
                    reserved_disk_bytes=0, cached_vector_bytes=0, peak_live_maps=0)

    def components(frame):
        cache = getattr(local, 'cache', None)
        return old_components(frame) if cache is None else cache.get(frame, old_components)

    def discover(labels, *args, **kwargs):
        previous = getattr(local, 'cache', None)
        cache = FrameCache(labels, scratch, counters)
        local.cache = cache
        try:
            return old_discover(labels, *args, **kwargs)
        finally:
            local.cache = previous
            cache.close()

    accounting.label_value_components = components
    module.discover = discover
    try:
        yield counters
    finally:
        accounting.label_value_components = old_components
        module.discover = old_discover
