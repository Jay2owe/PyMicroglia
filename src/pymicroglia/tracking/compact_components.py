"""Store exact read-only component maps as their nonzero bounding rectangles.

The original labeller still determines every component value and count. Only
retention changes: cache small crops and materialize full frames when requested.
Install only around audited readers of label maps, never general mutable users.
"""
from contextlib import contextmanager
import weakref
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class CompactLabels(NDArrayOperatorsMixin):
    __array_priority__ = 1000

    def __init__(self, values, counters):
        self.shape, self.dtype, self.ndim = values.shape, values.dtype, 2
        self.size = values.size
        self.nbytes = values.nbytes
        rows = np.flatnonzero(np.any(values, axis=1))
        columns = np.flatnonzero(np.any(values, axis=0))
        if len(rows):
            self.y0, self.y1 = int(rows[0]), int(rows[-1])+1
            self.x0, self.x1 = int(columns[0]), int(columns[-1])+1
        else:
            self.y0 = self.y1 = self.x0 = self.x1 = 0
        self.region = (slice(self.y0,self.y1), slice(self.x0,self.x1))
        self.crop = np.ndarray((self.y1-self.y0,self.x1-self.x0), dtype=self.dtype)
        np.copyto(self.crop, values[self.region])
        self.crop.flags.writeable = False
        size = self.crop.nbytes
        counters['maps'] += 1
        counters['live_dense_equivalent_bytes'] += values.nbytes
        counters['live_crop_bytes'] += size
        counters['peak_dense_equivalent_bytes'] = max(counters['peak_dense_equivalent_bytes'],counters['live_dense_equivalent_bytes'])
        counters['peak_crop_bytes'] = max(counters['peak_crop_bytes'],counters['live_crop_bytes'])
        def release(dense, compact):
            counters['live_dense_equivalent_bytes'] -= dense
            counters['live_crop_bytes'] -= compact
        self._release = weakref.finalize(self,release,values.nbytes,size)

    @classmethod
    def from_crop(cls, shape, region, crop, counters):
        """The same map as CompactLabels(full) when region is full's nonzero rectangle holding crop."""
        self = cls.__new__(cls)
        dtype = crop.dtype
        self.shape, self.dtype, self.ndim = tuple(shape), dtype, 2
        self.size = int(shape[0]) * int(shape[1])
        self.nbytes = self.size * dtype.itemsize
        self.y0, self.y1 = region[0].start, region[0].stop
        self.x0, self.x1 = region[1].start, region[1].stop
        self.region = (slice(self.y0,self.y1), slice(self.x0,self.x1))
        self.crop = np.ndarray((self.y1-self.y0,self.x1-self.x0), dtype=dtype)
        np.copyto(self.crop, crop)
        self.crop.flags.writeable = False
        size = self.crop.nbytes
        counters['maps'] += 1
        counters['live_dense_equivalent_bytes'] += self.nbytes
        counters['live_crop_bytes'] += size
        counters['peak_dense_equivalent_bytes'] = max(counters['peak_dense_equivalent_bytes'],counters['live_dense_equivalent_bytes'])
        counters['peak_crop_bytes'] = max(counters['peak_crop_bytes'],counters['live_crop_bytes'])
        def release(dense, compact):
            counters['live_dense_equivalent_bytes'] -= dense
            counters['live_crop_bytes'] -= compact
        self._release = weakref.finalize(self,release,self.nbytes,size)
        return self

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError('A compact label map needs materialization')
        result = np.ndarray(self.shape, dtype=self.dtype)
        result.fill(0)
        result[self.region] = self.crop
        return result if dtype is None else result.astype(dtype,copy=False)

    def __getitem__(self, index):
        if (isinstance(index,tuple) and len(index)==2
                and all(isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_)) for v in index)):
            y,x = map(int,index)
            if y < 0: y += self.shape[0]
            if x < 0: x += self.shape[1]
            if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
                raise IndexError('component index outside image')
            return self.crop[y-self.y0,x-self.x0] if self.y0 <= y < self.y1 and self.x0 <= x < self.x1 else self.dtype.type(0)
        return np.asarray(self)[index]

    def __eq__(self, value):
        if np.isscalar(value):
            result = np.ndarray(self.shape, dtype=bool)
            result.fill(bool(np.equal(self.dtype.type(0),value)))
            result[self.region] = self.crop == value
            return result
        return np.asarray(self) == value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        plain=tuple(np.asarray(v) if isinstance(v,CompactLabels) else v for v in inputs)
        if kwargs.get('out') is not None:
            if any(isinstance(v,CompactLabels) for v in kwargs['out']):
                raise TypeError('Compact component maps are read-only')
        return getattr(ufunc,method)(*plain,**kwargs)

    def __getattr__(self,name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(np.asarray(self),name)


class CompactLabeller:
    def __init__(self, original, counters):
        self.original,self.counters=original,counters

    def label(self, *args, **kwargs):
        result=self.original.label(*args, **kwargs)
        # An explicit output array belongs to the caller; preserve its contract.
        if (len(args)<3 and kwargs.get('output') is None and isinstance(result,tuple)
                and isinstance(result[0],np.ndarray) and result[0].ndim==2):
            return CompactLabels(result[0],self.counters),result[1]
        return result

    def __getattr__(self,name):
        return getattr(self.original,name)


@contextmanager
def install(module):
    counters=dict(maps=0,live_dense_equivalent_bytes=0,live_crop_bytes=0,
                  peak_dense_equivalent_bytes=0,peak_crop_bytes=0)
    old=module.ndi
    module.ndi=CompactLabeller(old,counters)
    try:
        yield counters
    finally:
        module.ndi=old
