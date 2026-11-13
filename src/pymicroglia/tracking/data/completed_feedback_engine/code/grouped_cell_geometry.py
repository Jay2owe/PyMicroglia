"""Group occupied coordinates once while preserving row-major cell geometry."""
from contextlib import contextmanager
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


def groups(frame):
    yy, xx = np.nonzero(frame > 0)
    if not len(yy):
        return
    values = frame[yy, xx]
    order = np.argsort(values, kind='stable')
    ordered = values[order]
    boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1])+1, len(order)]
    for first,last in zip(boundaries[:-1],boundaries[1:]):
        indices = order[first:last]
        yield int(ordered[first]), yy[indices], xx[indices]


# Speed (folded from correction_geometry_storage): a grouped frame is copied
# once into a read-only snapshot and every cell's mask is the equality test of
# its identity against that snapshot, instead of one full-size Boolean image
# per cell; a cyclic-alias descriptor forms its centred pixel set, from the
# same coordinates and centre by the same expression, only when it is read.
class LabelMask(NDArrayOperatorsMixin):
    """A Boolean mask held as the equality test of one identity against a
    read-only labelled frame (the same class, by name, as PyMicroglia's
    frozen_io.LabelMask, so checks by type name treat both alike)."""
    __array_priority__ = 1000

    def __init__(self, labels, identity):
        self.labels = labels
        self.identity = int(identity)

    @property
    def shape(self):
        return self.labels.shape

    @property
    def dtype(self):
        return np.dtype(bool)

    @property
    def ndim(self):
        return self.labels.ndim

    @property
    def size(self):
        return self.labels.size

    def __len__(self):
        return len(self.labels)

    def __array__(self, dtype=None, copy=None):
        value = self.labels == self.identity
        return value if dtype is None else value.astype(dtype, copy=False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        converted = tuple(np.asarray(v) if isinstance(v, LabelMask) else v for v in inputs)
        if kwargs.get("out") is not None:
            kwargs["out"] = tuple(np.asarray(v) if isinstance(v, LabelMask) else v for v in kwargs["out"])
        return getattr(ufunc, method)(*converted, **kwargs)

    def __getitem__(self, key):
        return self.labels[key] == self.identity

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)


def _snapshot(frame):
    result = np.array(frame, copy=True)
    result.flags.writeable = False
    return result


class _LazyCentred(dict):
    """A cyclic-alias descriptor whose 'centred' entry is formed on first read."""
    __slots__ = ('_points',)

    def __init__(self, yy, xx, **values):
        super().__init__(**values)
        self._points = (yy, xx)

    def __missing__(self, key):
        if key != 'centred':
            raise KeyError(key)
        yy, xx = self._points
        cy, cx = self['y'], self['x']
        centred = self['centred'] = frozenset(zip(np.rint(yy-cy).astype(int).tolist(),
                                                  np.rint(xx-cx).astype(int).tolist()))
        self._points = None
        return centred


@contextmanager
def grouped_geometry(density, cyclic):
    old_density, old_cyclic = density._objects, cyclic._objects
    def density_objects(frame):
        if frame.ndim != 2 or frame.dtype.kind not in 'iu':
            return old_density(frame)
        result = {}
        immutable = _snapshot(frame)
        for identity, yy, xx in groups(frame):
            mask = LabelMask(immutable,identity)
            y0,y1,x0,x1 = int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1
            result[identity] = dict(mask=mask,area=int(len(xx)),y=float(yy.mean()),x=float(xx.mean()),
                radius=float(np.sqrt(len(xx)/np.pi)),components=int(density.ndi.label(mask[y0:y1,x0:x1],density.STRUCTURE)[1]),
                first_y=int(yy[0]),first_x=int(xx[0]))
        return result
    def cyclic_objects(labels):
        if labels.ndim != 3 or labels.dtype.kind not in 'iu':
            return old_cyclic(labels)
        result = []
        for frame in labels:
            immutable = _snapshot(frame)
            row = {}
            for identity, yy, xx in groups(frame):
                mask = LabelMask(immutable,identity)
                area=int(len(yy));cy,cx=float(yy.mean()),float(xx.mean())
                row[identity]=_LazyCentred(yy,xx,mask=mask,area=area,y=cy,x=cx,radius=float(np.sqrt(area/np.pi)))
            result.append(row)
        return result
    density._objects, cyclic._objects = density_objects, cyclic_objects
    try:
        yield
    finally:
        density._objects, cyclic._objects = old_density, old_cyclic
