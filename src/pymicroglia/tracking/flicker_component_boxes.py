"""Bookended owner flickers read each candidate outline inside its own bounding box.

recover_large_bookended_owner_flickers.correct (also run with smaller
thresholds by recover_smaller_bookended_owner_flickers) built a full-frame
mask `components == cid` for every candidate outline and indexed the frame,
both neighbouring label frames, the protected pixels and the proposal through
it. Every True pixel of that mask lies inside the outline's bounding box
(scipy.ndimage.find_objects of the same component image), and boolean
indexing visits pixels in row-major order, which the box keeps. So the first
name, the two neighbour counts, the protected test, the prior-proposal test
and the write see the same pixels in the same order. The opening whole-movie
name count is summed frame by frame (movie_label_counts).
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys
try:
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from module_scan import file_sha256 as _file_sha256

VERSIONS = {'684215c725ae3b8e'}
EDITS = (
    ('    remaining = np.bincount(labels.ravel().astype(np.int64))\n',
     '    remaining = _flicker_movie_counts(labels)\n'),
    ('        areas = np.bincount(components.ravel())\n',
     '        areas = np.bincount(components.ravel())\n'
     '        _boxes = _flicker_find_objects(components)\n'),
    ('            mask = components == cid\n'
     '            old_name = int(frame[mask][0])\n'
     '            before_counts = np.bincount(labels[t-1][mask].astype(np.int64))\n'
     '            after_counts = np.bincount(labels[t+1][mask].astype(np.int64))\n',
     '            _box = _boxes[cid-1]\n'
     '            _sub = components[_box] == cid\n'
     '            _flicker_stats["boxed"] += 1\n'
     '            old_name = int(frame[_box][_sub][0])\n'
     '            before_counts = np.bincount(labels[t-1][_box][_sub].astype(np.int64))\n'
     '            after_counts = np.bincount(labels[t+1][_box][_sub].astype(np.int64))\n'),
    ('            if np.any(protected[t] & mask):\n',
     '            if np.any(protected[t][_box] & _sub):\n'),
    ('            if not reason and not np.all(candidate[t][mask] == old_name):\n',
     '            if not reason and not np.all(candidate[t][_box][_sub] == old_name):\n'),
    ('            candidate[t][mask] = new_name\n',
     '            candidate[t][_box][_sub] = new_name\n'),
)


def _digest(module):
    try:
        return _file_sha256(module.__file__)[:16]
    except (AttributeError, OSError, TypeError):
        return None


def rewrite(source):
    for old, _ in EDITS:
        if source.count(old) != 1:
            return None
    for old, new in EDITS:
        source = source.replace(old, new)
    # `mask` must have no other reader.
    return None if 'mask' in source.replace('_flicker_', '') else source


@contextmanager
def install(stats=None):
    """Replace every loaded verified correct (and its aliases) for the duration; restore afterwards."""
    from scipy import ndimage
    try:
        from .movie_label_counts import counts
    except ImportError:
        from movie_label_counts import counts
    stats = stats if stats is not None else {}
    for key in ('versions', 'boxed', 'framed', 'direct'):
        stats.setdefault(key, 0)
    movie_counts = counts(stats)
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
        globals_['_flicker_stats'] = stats
        globals_['_flicker_find_objects'] = ndimage.find_objects
        globals_['_flicker_movie_counts'] = movie_counts
        namespace = {}
        exec(compile(source, '<flicker-component-boxes>', 'exec',
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
            original.__globals__.pop('_flicker_stats', None)
            original.__globals__.pop('_flicker_find_objects', None)
            original.__globals__.pop('_flicker_movie_counts', None)
