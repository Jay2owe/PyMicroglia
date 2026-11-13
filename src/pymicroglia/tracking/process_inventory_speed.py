"""Process inventories find outline edges inside the outline's box and sample occupancy in one gather.

process_inventory_bridge.measure and process_inventory_interface_v2.interface
took each outline's edge pixels as np.argwhere(mask & ~binary_erosion(mask))
over the whole frame. The 3x3 erosion of a pixel reads only its neighbours, so
the mask's bounding box grown by one pixel (clipped to the frame, where both
see the border value 0) holds every edge pixel and every neighbour that
decides it. Row-major order inside the box is row-major order in the frame,
so the same coordinates come back in the same order.

process_inventory_motion.sample_occupancy compared the frame with the name at
the shifted pixels and then at each of 12 neighbour offsets in turn, each time
through the disk-backed frame's wrappers. The same comparisons are made on a
plain view of the same memory in one gather; a pixel is near when any offset
hits, which is the same OR.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np
try:
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from module_scan import file_sha256 as _file_sha256

EDGE_VERSIONS = {'b1e51f7e4e7f0b83': 'measure', 'f594f666f6b106e7': 'interface'}
OCCUPANCY_VERSIONS = {'b611c1b1513e7385'}
EDGES = (('np.argwhere(large & ~ndi.binary_erosion(large, structure=STRUCTURE))', '_inventory_edge(large, ndi, STRUCTURE)'),
         ('np.argwhere(small & ~ndi.binary_erosion(small, structure=STRUCTURE))', '_inventory_edge(small, ndi, STRUCTURE)'))


def _digest(module):
    try:
        return _file_sha256(module.__file__)[:16]
    except (AttributeError, OSError, TypeError):
        return None


def edge(stats):
    eight = np.ones((3, 3), bool)

    def edge_pixels(mask, ndi, structure):
        if (type(mask) is not np.ndarray or mask.dtype != bool or mask.ndim != 2
                or type(structure) is not np.ndarray or structure.shape != (3, 3)
                or not np.array_equal(structure, eight)):
            stats['edges_direct'] += 1
            return np.argwhere(mask & ~ndi.binary_erosion(mask, structure=structure))
        rows = np.flatnonzero(mask.any(axis=1))
        if not len(rows):
            stats['edges_empty'] += 1
            return np.argwhere(mask & ~ndi.binary_erosion(mask, structure=structure))
        cols = np.flatnonzero(mask.any(axis=0))
        y0, y1 = max(0, int(rows[0]) - 1), min(mask.shape[0], int(rows[-1]) + 2)
        x0, x1 = max(0, int(cols[0]) - 1), min(mask.shape[1], int(cols[-1]) + 2)
        box = mask[y0:y1, x0:x1]
        found = np.argwhere(box & ~ndi.binary_erosion(box, structure=structure))
        found += (y0, x0)
        stats['edges_boxed'] += 1
        return found
    return edge_pixels


def occupancy(original, stats):
    def sample_occupancy(image, name, yy, xx, dy, dx):
        offsets = original.__globals__.get('OFFSETS')
        plain = image.view(np.ndarray) if isinstance(image, np.ndarray) else image
        if (type(plain) is not np.ndarray or plain.ndim != 2 or type(yy) is not np.ndarray
                or type(xx) is not np.ndarray or yy.ndim != 1 or xx.shape != yy.shape
                or yy.dtype.kind not in 'iu' or xx.dtype.kind not in 'iu' or not isinstance(offsets, list)
                or not isinstance(dy, (int, np.integer)) or not isinstance(dx, (int, np.integer))):
            stats['occupancy_direct'] += 1
            return original(image, name, yy, xx, dy, dx)
        y, x = yy+dy, xx+dx
        height, width = plain.shape
        inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        if not inside.any():
            return 0.0, 0.0
        direct = np.zeros(len(yy), bool)
        direct[inside] = plain[y[inside], x[inside]] == name
        shifts = [(oy, ox) for oy, ox in offsets if oy or ox]
        near = direct.copy()
        if shifts:
            oy = np.array([s[0] for s in shifts])[:, None]
            ox = np.array([s[1] for s in shifts])[:, None]
            sy, sx = y[None, :] + oy, x[None, :] + ox
            valid = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
            hits = np.zeros(sy.shape, bool)
            hits[valid] = plain[sy[valid], sx[valid]] == name
            near |= hits.any(axis=0)
        stats['occupancy_gathered'] += 1
        return float(direct.mean()), float(near.mean())
    sample_occupancy.__wrapped__ = original
    return sample_occupancy


@contextmanager
def install(stats=None):
    """Replace the verified inventory functions (and their imported names) for the duration."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'edges_boxed', 'edges_empty', 'edges_direct', 'occupancy_gathered', 'occupancy_direct'):
        stats.setdefault(key, 0)
    swap = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        if not any(callable(getattr(module, n, None)) for n in ('measure', 'interface', 'sample_occupancy')):
            continue
        digest = _digest(module) if getattr(module, '__file__', None) else None
        name = EDGE_VERSIONS.get(digest)
        if name is not None:
            original = getattr(module, name, None)
            if (original is not None and getattr(original, '__module__', None) == module.__name__
                    and getattr(original, '__wrapped__', None) is None and id(original) not in swap):
                source = inspect.getsource(original)
                if all(source.count(old) == 1 for old, _ in EDGES):
                    for old, new in EDGES:
                        source = source.replace(old, new)
                    globals_ = original.__globals__
                    globals_['_inventory_edge'] = edge(stats)
                    namespace = {}
                    exec(compile(source, '<process-inventory-edges>', 'exec',
                                 flags=__future__.annotations.compiler_flag
                                 if original.__code__.co_flags & __future__.annotations.compiler_flag else 0,
                                 dont_inherit=True), globals_, namespace)
                    replacement = namespace[name]
                    replacement.__wrapped__ = original
                    swap[id(original)] = (original, replacement)
        if digest in OCCUPANCY_VERSIONS:
            original = getattr(module, 'sample_occupancy', None)
            if (original is not None and getattr(original, '__module__', None) == module.__name__
                    and getattr(original, '__wrapped__', None) is None and id(original) not in swap):
                swap[id(original)] = (original, occupancy(original, stats))
    stats['versions'] += len(swap)
    aliases = []
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        names = getattr(child, '__dict__', None)
        if not names:
            continue
        for key, value in list(names.items()):
            entry = swap.get(id(value))
            if entry is not None and value is entry[0]:
                aliases.append((child, key, value))
                setattr(child, key, entry[1])
    try:
        yield stats
    finally:
        for child, key, value in reversed(aliases):
            setattr(child, key, value)
        for original, _ in swap.values():
            original.__globals__.pop('_inventory_edge', None)
