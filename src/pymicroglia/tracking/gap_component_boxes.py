"""Native gap owner transport: each name's components labelled inside its bounding box.

carry_measured_gap_owners labels one name of one frame at a time
(ndi.label(parent[f] == name, 3x3)) over the whole frame. The labeller numbers
components by their first pixel in row-major order and every foreground pixel
lies inside the mask's bounding rectangle, so labelling that rectangle gives
the same values there, zeros elsewhere and the same count (the engine's
merge_partition_speed.BoundedNdimage.label relies on the same fact). For the
audited version whose maps are kept compact, the compact map is built from
that rectangle directly: it is the map's own nonzero rectangle. Other calls,
inputs that are not plain two-dimensional boolean arrays, empty masks and
explicit outputs keep the original labeller.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage

VERSIONS = {
    '81a7217a8ef0b56fc768522f1f797c0fd597e25235138f267526ef50ab212fa5': True,   # r215 a003, compact maps
    '208766a7f376fae2870592f657583c01c3014206e6def06af99afa45a1870bbb': False,  # r213 a010, dense maps
}


def bounds(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    columns = np.flatnonzero(mask[y0:y1].any(axis=0))
    return slice(y0, y1), slice(int(columns[0]), int(columns[-1]) + 1)


class BoxedLabeller:
    def __init__(self, inner, stats, compact):
        self.inner, self.stats, self.compact = inner, stats, compact

    def label(self, input, structure=None, *args, **kwargs):
        if (args or kwargs or type(input) is not np.ndarray or input.ndim != 2
                or input.dtype != np.bool_):
            self.stats['fallback'] += 1
            return self.inner.label(input, structure, *args, **kwargs)
        region = bounds(input)
        if region is None:
            self.stats['fallback'] += 1
            return self.inner.label(input, structure)
        local, count = ndimage.label(input[region], structure)
        self.stats['boxed'] += 1
        if self.compact:
            from .compact_components import CompactLabels
            return CompactLabels.from_crop(input.shape, region, local, self.inner.counters), count
        result = np.ndarray(input.shape, dtype=local.dtype)
        result.fill(0)
        result[region] = local
        return result, count

    def __getattr__(self, name):
        return getattr(self.inner, name)


def version(module):
    try:
        return VERSIONS.get(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    except (OSError, TypeError, AttributeError):
        return None


@contextmanager
def install(module, stats):
    """Box the module's labeller when its file is an audited version; restore afterwards."""
    compact = version(module)
    old = getattr(module, 'ndi', None)
    usable = compact is not None and old is not None and (
        not compact or type(old).__name__ == 'CompactLabeller')
    if usable:
        stats['versions'] += 1
        module.ndi = BoxedLabeller(old, stats, compact)
    try:
        yield stats
    finally:
        if usable:
            module.ndi = old
