"""scipy.ndimage.label of a two-dimensional boolean mask, labelled inside its bounding rectangle.

Corrections label one name's mask of a whole frame hundreds of thousands of
times. A 2-D structure is 3x3, so components join only through neighbouring
foreground pixels, all inside the mask's bounding rectangle, and the labeller
numbers components by their first pixel in row-major order, which the
rectangle keeps. Labelling the rectangle gives the same values there, zeros
elsewhere and the same count (the engine's
merge_partition_speed.BoundedNdimage.label relies on the same fact). The
output is an int32 array of the input shape from np.zeros, which working-array
storage allocates exactly as np.empty (the same pool allocation; file-backed
arrays are already zero, so only resident ones are filled). An empty mask with
no structure or a plain 3x3 one returns zeros and 0 without labelling. Other
inputs, an explicit output, other structures for empty masks, rectangles
covering the whole frame and images large enough for 64-bit labels keep the
original call.
"""
from contextlib import contextmanager
import sys

import numpy as np


def bounded(original, stats):
    def label(input, structure=None, output=None):
        if (output is not None or type(input) is not np.ndarray or input.ndim != 2
                or input.dtype != np.bool_ or input.size >= 2**31 - 2):
            stats['fallback'] += 1
            return original(input, structure, output)
        rows = np.flatnonzero(input.any(axis=1))
        if not len(rows):
            # An empty mask has no components: scipy returns int32 zeros and
            # the Python int 0 once its structure check passes.
            if structure is None or (type(structure) is np.ndarray and structure.shape == (3, 3)
                                     and structure.dtype.kind in 'biuf'):
                stats['empty'] += 1
                return np.zeros(input.shape, np.int32), 0
            stats['fallback'] += 1
            return original(input, structure)
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        columns = np.flatnonzero(input[y0:y1].any(axis=0))
        x0, x1 = int(columns[0]), int(columns[-1]) + 1
        if y0 == 0 and x0 == 0 and y1 == input.shape[0] and x1 == input.shape[1]:
            stats['whole'] += 1
            return original(input, structure)
        local, count = original(input[y0:y1, x0:x1], structure)
        result = np.zeros(input.shape, np.int32)
        result[y0:y1, x0:x1] = local
        stats['boxed'] += 1
        return result, count
    label.__wrapped__ = original
    return label


@contextmanager
def install(stats=None):
    """Replace scipy.ndimage.label and names bound to it; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('boxed', 'whole', 'empty', 'fallback'):
        stats.setdefault(key, 0)
    import scipy.ndimage
    from scipy.ndimage import _measurements
    original = _measurements.label
    if getattr(scipy.ndimage.label, '__wrapped__', None) is not None:
        yield stats
        return
    replacement = bounded(original, stats)
    aliases = []
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        if child is _measurements:
            continue
        names = getattr(child, '__dict__', None)
        if not names:
            continue
        for name, value in list(names.items()):
            if value is original:
                aliases.append((child, name, value))
                setattr(child, name, replacement)
    try:
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
