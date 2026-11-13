"""Centres of 2-D Boolean masks from integer coordinate sums.

scipy's center_of_mass(mask) (no labels) divides, for each axis, the float sum
of mask x coordinate over the whole frame by the pixel count; the physical
lineage _centre divides the float sum of np.nonzero's coordinates by their
number. Every partial sum is a whole number below 2**53, so each float sum
equals the integer sum of the pixels' row (or column) indices, formed here
from one flatnonzero split into rows and columns; the same division then
gives the same float64. Empty masks, other dtypes, shapes, labels or indices keep the
original call.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np

CENTRE_OF_MASS = '7a9931beee748642eb4a02353848b16fc528cd3a0839c1531729b7db87b994c6'  # scipy 1.14.1
LINEAGE = '94bfef1f2908b888b4397741134d7f389b68b32ecf4d09b7ebe74b58d68bc4b8'
EXACT = 1 << 53


def sums(mask):
    """(pixel count, row-index sum, column-index sum) of a plain 2-D Boolean mask, or None."""
    if type(mask) is not np.ndarray or mask.ndim != 2 or mask.dtype != np.bool_:
        return None
    # One pass over the frame: flat indices of the pixels split into row and column.
    flat = np.flatnonzero(mask)
    total = int(flat.size)
    if total == 0:
        return None
    rows, columns = np.divmod(flat, mask.shape[1])
    row_sum = int(rows.sum(dtype=np.int64))
    column_sum = int(columns.sum(dtype=np.int64))
    if row_sum >= EXACT or column_sum >= EXACT:
        return None
    return total, row_sum, column_sum


def centre_of_mass(original, stats):
    def center_of_mass(input, labels=None, index=None):
        found = sums(input) if labels is None and index is None else None
        if found is None:
            stats['original'] += 1
            return original(input, labels, index)
        total, row_sum, column_sum = found
        stats['counted'] += 1
        return (np.float64(row_sum) / np.int64(total), np.float64(column_sum) / np.int64(total))
    center_of_mass.__wrapped__ = original
    return center_of_mass


@contextmanager
def install(stats=None):
    """Replace scipy.ndimage.center_of_mass and names bound to it outside scipy's measurements module."""
    stats = stats if stats is not None else {}
    for key in ('supported', 'counted', 'original', 'aliases'):
        stats.setdefault(key, 0)
    from scipy.ndimage import _measurements
    original = _measurements.center_of_mass
    if (getattr(original, '__wrapped__', None) is not None
            or hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != CENTRE_OF_MASS):
        yield stats
        return
    replacement = centre_of_mass(original, stats)
    aliases = []
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        names = getattr(child, '__dict__', None)
        if child is _measurements or not names:
            continue
        for key, value in list(names.items()):
            if value is original:
                aliases.append((child, key))
                setattr(child, key, replacement)
    stats.update(supported=True, aliases=len(aliases))
    try:
        yield stats
    finally:
        for child, key in reversed(aliases):
            setattr(child, key, original)


@contextmanager
def install_lineage(module):
    """Replace physical_lineage_aliases._centre with the counted centre."""
    stats = dict(supported=False, counted=0, original=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != LINEAGE:
        yield stats
        return
    original = module._centre

    def _centre(mask):
        found = sums(mask)
        if found is None:
            stats['original'] += 1
            return original(mask)
        total, row_sum, column_sum = found
        stats['counted'] += 1
        return float(np.float64(row_sum) / total), float(np.float64(column_sum) / total)

    _centre.__wrapped__ = original
    module._centre = _centre
    stats['supported'] = True
    try:
        yield stats
    finally:
        module._centre = original
