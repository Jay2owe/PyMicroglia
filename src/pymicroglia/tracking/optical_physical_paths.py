"""Optical physical paths collect per-component statistics from one pass over foreground pixels.

optical_basin_soma_paths.physical_paths (one audited file, imported by eight
correction families) visited every connected component of every proposal
frame through find_objects, re-cropped the frame and observed movie per
component, and found reciprocal overlaps through a row-wise np.unique.

skimage's label numbers components 1..n in raster order of their first pixel,
so the first foreground pixel of each label in raster order is the original's
seed (and its identity), bincount gives the same proposed and measured pixel
counts, a box touches the frame edge exactly when the component has a pixel on
that edge, and the four forward neighbour offsets inside the frame are the
original's zero-padded comparisons. Overlap pairs encoded as one integer key
sort in the same lexicographic order as the row-wise unique, and the overlap
helper casts its pairs to int64 itself. Inputs outside 3-D aligned arrays keep
the original.

Several reviewed stages call it on byte-identical movies. The last result is
held (nonzero node positions and values, record and group copies, and every
node added to the blocked set in insertion order) under the SHA-256 of both
movies' buffers, shapes, strides and dtypes; a repeated pair is answered with
fresh copies built the same way, so the blocked set iterates in the same
order. A result that fell back to the original is not held.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

import numpy as np
try:
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from module_scan import file_sha256 as _file_sha256

VERSIONS = {'3ed4d577a4868402'}


def _digest(module):
    try:
        return _file_sha256(module.__file__)[:16]
    except (AttributeError, OSError, TypeError):
        return None


def _plain(value):
    return value.view(np.ndarray) if type(value) is not np.ndarray else value


# The last movie pair's result: (key, shape, positions, values, records, groups, blocked order).
_HELD = {}


def _token(array):
    """SHA-256 of the array's whole contiguous buffer and the view's place in it, or None."""
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    if not root.flags.c_contiguous:
        return None
    start = array.__array_interface__['data'][0] - root.__array_interface__['data'][0]
    if not 0 <= start < max(root.nbytes, 1):
        return None
    try:
        digest = hashlib.sha256(memoryview(root).cast('B')).hexdigest()
    except (TypeError, ValueError, BufferError):
        return None
    return (digest, root.shape, root.dtype.str, start, array.shape, array.strides, array.dtype.str)


def _block(blocked, order, values):
    values = list(values)
    blocked.update(values)
    order.extend(values)


def _hold(key, nodes, records, groups, order):
    # A number is nonzero exactly when it compares unequal to 0; NumPy finds
    # those positions about five times faster in the Boolean array.
    positions = (np.flatnonzero(nodes != 0) if isinstance(nodes, np.ndarray) and nodes.dtype.kind in 'iufc'
                 else np.flatnonzero(nodes))
    if len(positions) > nodes.size // 4:
        return
    if nodes.size <= np.iinfo(np.int32).max:
        positions = positions.astype(np.int32)
    _HELD.clear()
    _HELD['entry'] = (key, nodes.shape, positions, nodes.reshape(-1)[positions],
                      [dict(q) for q in records[1:]], {k: list(v) for k, v in groups.items()}, list(order))


def _rebuild(entry):
    _, shape, positions, values, records, groups, order = entry
    nodes = np.zeros(shape, np.int32)
    nodes.reshape(-1)[positions] = values
    blocked = set()
    blocked.update(order)
    return (nodes, [None] + [dict(q) for q in records], {k: list(v) for k, v in groups.items()}, blocked)


def fast(original, stats):
    label = original.__globals__['label']
    reciprocal_overlap_pairs = original.__globals__['reciprocal_overlap_pairs']

    def physical_paths(proposals, observed):
        if (not isinstance(proposals, np.ndarray) or not isinstance(observed, np.ndarray)
                or proposals.ndim != 3 or observed.shape != proposals.shape
                or proposals.shape[1] == 0 or proposals.shape[2] == 0):
            stats['fallback'] += 1
            return original(proposals, observed)
        proposals, observed = _plain(proposals), _plain(observed)
        key = (_token(proposals), _token(observed))
        if None in key:
            key = None
        entry = _HELD.get('entry')
        if key is not None and entry is not None and entry[0] == key:
            stats['reused'] = stats.get('reused', 0) + 1
            return _rebuild(entry)
        result = measure(proposals, observed)
        if result is None:
            stats['fallback'] += 1
            return original(proposals, observed)
        nodes, records, groups, blocked, order = result
        if key is not None:
            _hold(key, nodes, records, groups, order)
        stats['fast'] += 1
        return nodes, records, groups, blocked

    def measure(proposals, observed):
        nodes = np.zeros(proposals.shape, np.int32); records = [None]; parents = [0]; blocked = set()
        order = []
        height, width = proposals.shape[1:]
        for t, frame in enumerate(proposals):
            components = label(frame, background=0, connectivity=2)
            offset = len(records) - 1
            # A boolean mask is scanned about four times faster than the labels; same positions.
            positive = np.flatnonzero(components != 0)
            py, px = np.divmod(positive, width)
            labels_here = components.ravel()[positive]
            count = int(components.max())
            field = nodes[t]
            field[py, px] = labels_here + offset
            if count:
                _, first = np.unique(labels_here, return_index=True)
                if len(first) != count:
                    return None
                proposed = np.bincount(labels_here, minlength=count + 1)[1:]
                measured = np.bincount(labels_here[observed[t][py, px] != 0], minlength=count + 1)[1:]
                seed_y = py[first]; seed_x = px[first]
                identity = frame[seed_y, seed_x]
                for index in range(count):
                    node = offset + index + 1
                    records.append(dict(node=node, identity=int(identity[index]), frame=t + 1,
                        seed_y=int(seed_y[index]), seed_x=int(seed_x[index]),
                        proposed_pixels=int(proposed[index]), measured_pixels=int(measured[index])))
                    parents.append(node)
                border = np.unique(np.concatenate((components[0], components[-1],
                                                   components[:, 0], components[:, -1])))
                _block(blocked, order, (int(offset + v) for v in border.tolist() if v))
                values = field[py, px]
                for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
                    ny = py + dy; nx = px + dx
                    inside = (ny < height) & (nx >= 0) & (nx < width)
                    other = field[ny[inside], nx[inside]]; mine = values[inside]
                    hit = (other > 0) & (other != mine)
                    _block(blocked, order, map(int, np.unique(mine[hit])))
                    _block(blocked, order, map(int, np.unique(other[hit])))

        def find(a):
            while parents[a] != a:
                parents[a] = parents[parents[a]]; a = parents[a]
            return a
        areas = np.array([0] + [q['proposed_pixels'] for q in records[1:]])
        base = len(records)
        for t in range(1, len(nodes)):
            shared = (nodes[t - 1] > 0) & (nodes[t] > 0)
            keys = nodes[t - 1][shared].astype(np.int64) * base + nodes[t][shared].astype(np.int64)
            unique, counts = np.unique(keys, return_counts=True)
            pairs = np.column_stack((unique // base, unique % base))
            for a, b in reciprocal_overlap_pairs(pairs, counts, areas):
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parents[rb] = ra
        groups = {}
        for node in range(1, len(records)):
            groups.setdefault(find(node), []).append(node)
        return nodes, records, groups, blocked, order
    physical_paths.__wrapped__ = original
    return physical_paths


@contextmanager
def install(stats=None):
    """Replace every loaded verified physical_paths and the names callers imported it under."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'fast', 'fallback'):
        stats.setdefault(key, 0)
    swap = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, 'physical_paths', None)
        if (original is None or getattr(original, '__module__', None) != module.__name__
                or getattr(original, '__wrapped__', None) is not None or id(original) in swap
                or _digest(module) not in VERSIONS):
            continue
        swap[id(original)] = (original, fast(original, stats))
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
            if entry is not None and value is entry[0]:
                aliases.append((child, name, value))
                setattr(child, name, entry[1])
    try:
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
