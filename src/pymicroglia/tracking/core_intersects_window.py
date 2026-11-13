"""A soma-core test returns False at once when the mask is empty inside the core's window.

shared_parent_soma_ownership's _core_intersects (one text, in a002 and a003,
imported by name into the detector, birth and memory families) builds the
point's disc over its clipped square window and asks whether any mask pixel in
that window is inside it. Callers test every point of a frame against one
outline, so most windows hold no mask pixel at all; the answer there is False
whatever the disc is. Windows with a mask pixel run the original expression.
The window arithmetic (and so every error it raises) is unchanged.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
import sys

import numpy as np

SOURCE = '24192921c2b83e62'
_PLAIN = (np.ndarray, np.memmap)


def _core_intersects(mask, p):
    radius = float(p.radius_px)
    x0 = max(0, math.floor(p.x-radius)); x1 = min(mask.shape[1], math.ceil(p.x+radius)+1)
    y0 = max(0, math.floor(p.y-radius)); y1 = min(mask.shape[0], math.ceil(p.y+radius)+1)
    window = mask[y0:y1, x0:x1]
    # A reversed range (negative radius) slices from the end and the original
    # raises on the disc's shape; it keeps the original expression.
    # count_nonzero answers .any() for a plain array about five times faster on
    # a window this small; other array types (masked arrays) keep .any().
    if (y1 >= y0 and x1 >= x0 and isinstance(window, np.ndarray) and window.dtype.kind in 'biu'
            and not (np.count_nonzero(window) if type(window) in _PLAIN else window.any())):
        return False
    yy, xx = np.ogrid[y0:y1, x0:x1]
    return bool(np.any(window & ((xx-p.x)**2+(yy-p.y)**2 <= radius**2)))


def _verified(function):
    try:
        text = inspect.getsource(function).strip()
    except (OSError, TypeError):
        return False
    return hashlib.sha256(text.encode()).hexdigest()[:16] == SOURCE


@contextmanager
def install(stats=None):
    """Replace every loaded verified _core_intersects and the names callers imported it under."""
    stats = stats if stats is not None else {}
    stats.setdefault('versions', 0)
    swap = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, '_core_intersects', None)
        if (original is None or original is _core_intersects or id(original) in swap
                or getattr(original, '__module__', None) != module.__name__ or not _verified(original)):
            continue
        swap[id(original)] = original
    stats['versions'] += len(swap)
    aliases = []
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        names = getattr(child, '__dict__', None)
        if not names:
            continue
        if not swap or swap.keys().isdisjoint(map(id, names.values())):  # alias loop skip
            continue
        for name, value in list(names.items()):
            entry = swap.get(id(value))
            if entry is not None and entry is value:
                aliases.append((child, name, value))
                setattr(child, name, _core_intersects)
    try:
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
