"""Local fragment reunion's adjacent component pairs read from foreground pixels.

_nearby_pairs compared the whole frame with itself at every offset within
maximum_gap_px. A pair needs both pixels inside components, so the same pixel
pairs are read from the first pixel's foreground coordinates. Each offset's
pairs come out in the same sorted order (one int64 key per pair), so the
nearest-gap dictionary is filled in the original order. _components took each
component's frame value at its first pixel; every pixel of a connected
component holds that value, so it is written from all pixels.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

SOURCE = '0e80070f4c64e06804ecc88fc09e21244cb77adf612e4b278dc9b1216b671c88'


def nearby_pairs(original):
    def _nearby_pairs(components, maximum_gap_px):
        if (type(components) is not np.ndarray or components.ndim != 2
                or components.dtype.kind not in 'iu' or not components.size):
            return original(components, maximum_gap_px)
        top = int(components.max())
        if top >= 2**31 or int(components.min()) < 0:
            return original(components, maximum_gap_px)
        height, width = components.shape
        nearest = {}
        flat = components.ravel()
        where = np.flatnonzero(flat != 0)
        yy, xx = np.divmod(where, width)
        ids = flat[where].astype(np.int64)
        base = top + 1
        for dy in range(maximum_gap_px + 1):
            for dx in range(-maximum_gap_px, maximum_gap_px + 1):
                if dy == 0 and dx <= 0:
                    continue
                distance = math.hypot(dy, dx)
                if distance > maximum_gap_px:
                    continue
                y0, y1 = max(0, -dy), min(height, height - dy)
                x0, x1 = max(0, -dx), min(width, width - dx)
                inside = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
                first = ids[inside]
                second = components[yy[inside] + dy, xx[inside] + dx].astype(np.int64)
                valid = (second > 0) & (first != second)
                if not np.any(valid):
                    continue
                first, second = first[valid], second[valid]
                keys = np.unique(np.minimum(first, second) * base + np.maximum(first, second))
                for a, b in zip((keys // base).tolist(), (keys % base).tolist()):
                    pair = a, b
                    nearest[pair] = min(nearest.get(pair, math.inf), distance)
        return nearest
    _nearby_pairs.__wrapped__ = original
    return _nearby_pairs


def components(module, original):
    def _components(frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.dtype.kind not in 'iub':
            return original(frame)
        # Disk-backed working frames are read through a plain view of the same memory.
        frame = frame.view(np.ndarray)
        labelled = module.connected_labels(frame, background=0, connectivity=2)
        count = int(labelled.max())
        areas = np.bincount(labelled.ravel(), minlength=count + 1)
        identities = np.zeros(count + 1, dtype=np.int32)
        # Row-major positions of the labelled pixels split by the row length
        # are np.nonzero's rows and columns; background keeps identity 0.
        flat = labelled.ravel()
        where = np.flatnonzero(flat != 0)
        yy, xx = np.divmod(where, labelled.shape[1]) if labelled.size else np.nonzero(labelled)
        ids = flat[where]
        if count:
            identities[ids] = frame.ravel()[where]
        cy = np.bincount(ids, weights=yy, minlength=count + 1) / np.maximum(areas, 1)
        cx = np.bincount(ids, weights=xx, minlength=count + 1) / np.maximum(areas, 1)
        return labelled, areas, identities, cy, cx
    _components.__wrapped__ = original
    return _components


@contextmanager
def install(module):
    try:
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except (AttributeError, OSError, TypeError):
        digest = None
    if digest != SOURCE or getattr(module._nearby_pairs, '__wrapped__', None) is not None:
        yield False
        return
    original_pairs, original_components = module._nearby_pairs, module._components
    module._nearby_pairs = nearby_pairs(original_pairs)
    module._components = components(module, original_components)
    try:
        yield True
    finally:
        module._nearby_pairs, module._components = original_pairs, original_components


@contextmanager
def install_loaded():
    """install() on the loaded local_fragment_reunion, if any."""
    module = sys.modules.get('local_fragment_reunion')
    if module is None:
        yield False
        return
    with install(module) as installed:
        yield installed
