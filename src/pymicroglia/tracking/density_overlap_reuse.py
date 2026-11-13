"""Density reciprocal overlap measures each frame's objects once and counts neighbours as arrays.

density_reciprocal_overlap.correct (one audited file) described every frame
twice: as `current` of one step and as `previous` of the next. The second
description is of the same frame unless that step's swaps changed it, so the
previous step's objects are taken when the frame's bytes (dtype, shape and
SHA-256) are unchanged and the same object function is installed. The object
tables are only read.

Each object's neighbour count compared every pair of objects in Python. The
same subtractions, np.hypot, sums, max(..., 1.0) and products run elementwise
on float64 arrays of the same Python floats, so each comparison is the same;
the counts keep the original's types (NumPy integers once a frame has two
objects, Python 0 for a lone object) and the objects' order.
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

VERSIONS = {'5b2649db96a293aa'}
EDITS = (
    ('        old, new = _objects(previous), _objects(current)\n',
     '        old, new = _density_objects(_objects, previous, current, _density_held)\n'),
    ('        neighbours = {}\n'
     '        for identity, item in old.items():\n'
     '            neighbours[identity] = sum(\n'
     '                other != identity and np.hypot(\n'
     '                    item["y"] - candidate["y"], item["x"] - candidate["x"]\n'
     '                ) <= maximum_neighbour_sum_radii * max(\n'
     '                    item["radius"] + candidate["radius"], 1.0)\n'
     '                for other, candidate in old.items())\n',
     '        neighbours = _density_neighbours(old, maximum_neighbour_sum_radii)\n'),
    ('    for frame in range(1, len(output)):\n',
     '    _density_held = [None]\n    for frame in range(1, len(output)):\n'),
)


def _digest(module):
    try:
        return _file_sha256(module.__file__)[:16]
    except (AttributeError, OSError, TypeError):
        return None


def _key(frame):
    if not isinstance(frame, np.ndarray):
        return None
    plain = np.ascontiguousarray(frame.view(np.ndarray))
    return frame.dtype.str, frame.shape, hashlib.sha256(memoryview(plain).cast('B')).digest()


def objects_pair(stats):
    def pair(objects, previous, current, held):
        entry = held[0]
        key = _key(previous) if entry is not None else None
        if entry is not None and key is not None and entry[0] is objects and entry[1] == key:
            old = entry[2]
            stats['objects_reused'] += 1
        else:
            old = objects(previous)
            stats['objects_measured'] += 1
        new = objects(current)
        stats['objects_measured'] += 1
        current_key = _key(current)
        held[0] = None if current_key is None else (objects, current_key, new)
        return old, new
    return pair


def neighbour_counts(stats):
    def counts(old, maximum):
        items = list(old.items())
        if len(items) < 2 or not all(type(item[k]) is float for _, item in items for k in ('y', 'x', 'radius')) \
                or type(maximum) is not float:
            stats['neighbours_direct'] += 1
            result = {}
            for identity, item in items:
                result[identity] = sum(
                    other != identity and np.hypot(
                        item["y"] - candidate["y"], item["x"] - candidate["x"]
                    ) <= maximum * max(item["radius"] + candidate["radius"], 1.0)
                    for other, candidate in items)
            return result
        stats['neighbours_arrays'] += 1
        y = np.array([item['y'] for _, item in items], np.float64)
        x = np.array([item['x'] for _, item in items], np.float64)
        r = np.array([item['radius'] for _, item in items], np.float64)
        near = np.hypot(y[:, None] - y[None, :], x[:, None] - x[None, :]) <= maximum * np.maximum(r[:, None] + r[None, :], 1.0)
        np.fill_diagonal(near, False)
        found = near.sum(axis=1)
        return {identity: np.int64(found[index]) for index, (identity, _) in enumerate(items)}
    return counts


def rewrite(source):
    for old, _ in EDITS:
        if source.count(old) != 1:
            return None
    for old, new in EDITS:
        source = source.replace(old, new)
    return source


@contextmanager
def install(stats=None):
    """Replace every loaded verified correct (and its aliases) for the duration; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'objects_reused', 'objects_measured', 'neighbours_arrays', 'neighbours_direct'):
        stats.setdefault(key, 0)
    swap = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, 'correct', None)
        if (original is None or getattr(original, '__module__', None) != module.__name__
                or getattr(original, '__wrapped__', None) is not None or id(original) in swap
                or _digest(module) not in VERSIONS):
            continue
        source = rewrite(inspect.getsource(original))
        if source is None:
            continue
        globals_ = original.__globals__
        globals_['_density_objects'] = objects_pair(stats)
        globals_['_density_neighbours'] = neighbour_counts(stats)
        namespace = {}
        exec(compile(source, '<density-overlap-reuse>', 'exec',
                     flags=__future__.annotations.compiler_flag
                     if original.__code__.co_flags & __future__.annotations.compiler_flag else 0,
                     dont_inherit=True), globals_, namespace)
        replacement = namespace['correct']
        replacement.__wrapped__ = original
        swap[id(original)] = (original, replacement)
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
        for original, _ in swap.values():
            original.__globals__.pop('_density_objects', None)
            original.__globals__.pop('_density_neighbours', None)
