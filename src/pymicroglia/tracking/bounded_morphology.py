"""scipy.ndimage binary erosion and dilation of a 2-D Boolean mask, inside its padded rectangle.

Corrections erode or dilate one cell's mask of a whole frame once per pair or
per seed. With one iteration, no mask, no output and border value 0, a dilated
pixel lies within the structure's reach r of a True pixel, and an eroded pixel
is a True pixel itself when the structure holds its centre. Every such pixel
lies in the mask's bounding rectangle grown by r, and every neighbour that
decides it lies in the rectangle grown by r or outside the frame, where both
calls see the border value 0. Processing that rectangle (clipped to the frame)
gives the same values there and False elsewhere. The output is allocated as
scipy allocates it (np.empty of the input shape, Boolean). Other inputs or
arguments, empty masks and rectangles covering the whole frame keep the
original call.
"""
from contextlib import contextmanager
import sys

import numpy as np

NAMES = ('binary_erosion', 'binary_dilation')
SMALL = 64 * 64


def bounded(original, stats, erosion):
    def morphology(input, structure=None, iterations=1, mask=None, output=None,
                   border_value=0, origin=0, brute_force=False):
        arguments = (input, structure, iterations, mask, output, border_value, origin, brute_force)
        if type(input) is np.ndarray and input.size <= SMALL:
            # A crop of a small image saves less than finding its rectangle costs.
            stats['small'] += 1
            return original(*arguments)
        if (type(input) is not np.ndarray or input.ndim != 2 or input.dtype != np.bool_
                or type(iterations) is not int or iterations != 1 or mask is not None
                or output is not None or type(border_value) is not int or border_value != 0
                or type(origin) is not int or origin != 0):
            stats['fallback'] += 1
            return original(*arguments)
        if structure is None:
            reach = (1, 1)
        else:
            shape = np.shape(structure)
            if (len(shape) != 2 or not all(size % 2 for size in shape)
                    or erosion and not bool(np.asarray(structure)[shape[0] // 2, shape[1] // 2])):
                stats['fallback'] += 1
                return original(*arguments)
            reach = (shape[0] // 2, shape[1] // 2)
        rows = np.flatnonzero(input.any(axis=1))
        if not len(rows):
            stats['fallback'] += 1
            return original(*arguments)
        y0, y1 = max(int(rows[0]) - reach[0], 0), min(int(rows[-1]) + 1 + reach[0], input.shape[0])
        columns = np.flatnonzero(input[int(rows[0]):int(rows[-1]) + 1].any(axis=0))
        x0, x1 = max(int(columns[0]) - reach[1], 0), min(int(columns[-1]) + 1 + reach[1], input.shape[1])
        if y0 == 0 and x0 == 0 and y1 == input.shape[0] and x1 == input.shape[1]:
            stats['whole'] += 1
            return original(*arguments)
        local = original(input[y0:y1, x0:x1], structure, 1, None, None, 0, 0, brute_force)
        result = np.empty(input.shape, np.bool_)
        result[...] = False
        result[y0:y1, x0:x1] = local
        stats['boxed'] += 1
        return result
    morphology.__wrapped__ = original
    return morphology


@contextmanager
def install(stats=None):
    """Replace scipy.ndimage binary_erosion and binary_dilation and names bound
    to them outside scipy's own morphology module; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('boxed', 'whole', 'fallback', 'small'):
        stats.setdefault(key, 0)
    import scipy.ndimage
    from scipy.ndimage import _morphology
    if any(getattr(getattr(scipy.ndimage, name), '__wrapped__', None) is not None for name in NAMES):
        yield stats
        return
    replacements = {}
    for name in NAMES:
        original = getattr(_morphology, name)
        replacements[id(original)] = (original, bounded(original, stats, name == 'binary_erosion'))
    aliases = []
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        if child is _morphology:
            continue
        names = getattr(child, '__dict__', None)
        if not names:
            continue
        for key, value in list(names.items()):
            entry = replacements.get(id(value))
            if entry is not None and value is entry[0]:
                aliases.append((child, key, value))
                setattr(child, key, entry[1])
    try:
        yield stats
    finally:
        for child, key, value in reversed(aliases):
            setattr(child, key, value)
