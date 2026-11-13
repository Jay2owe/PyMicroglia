"""Bracketed consensus memory takes each track's named rows from one sort.

restore_bracketed_consensus_memory walked every track of the observed strong
points and took its rows with a positive proposal name,
g[g.native.gt(0)].sort_values('frame'): a Boolean selection and a sort per
track. Tracks without such rows were skipped. With integer track, frame and
native columns (unique column names), the positive rows are taken once; when
no track repeats a frame among them, each track's frames sort to one order
whatever the sort, so one stable sort of those rows by (track, frame) followed
by the same groupby (which keeps row order within a track) gives every track
that has positive rows the same rows in the same order, index, columns and
dtypes. Repeated frames or other dtypes keep the per-track steps.
"""
import __future__
from contextlib import contextmanager
import inspect
import sys

import numpy as np
try:
    from .module_scan import file_sha256 as _file_sha256, loaded as _loaded
except ImportError:
    from module_scan import file_sha256 as _file_sha256, loaded as _loaded

NAME = 'bracketed_consensus_memory'
FUNCTION = 'restore_bracketed_consensus_memory'
VERSIONS = {'5916ad335234b93b405a315ce1c8690ef9a3371ded633ff85e4bac63ab4bc5ea'}
OLD = ("    for track,g in pts.groupby('track_id'):\n"
       "        g=g[g.native.gt(0)].sort_values('frame')\n")
NEW = "    for track,g in _bracketed_tracks(pts):\n"


def tracks(stats):
    def _bracketed_tracks(pts):
        columns = pts.columns
        if columns.is_unique and all(name in columns for name in ('track_id', 'frame', 'native')):
            track, frame, native = pts['track_id'], pts['frame'], pts['native']
            if all(isinstance(c.dtype, np.dtype) and c.dtype.kind in 'iu' for c in (track, frame, native)):
                positive = np.flatnonzero(native.to_numpy() > 0)
                tracks_, frames = track.to_numpy()[positive], frame.to_numpy()[positive]
                order = np.lexsort((frames, tracks_))
                tracks_, frames = tracks_[order], frames[order]
                if not np.any((tracks_[1:] == tracks_[:-1]) & (frames[1:] == frames[:-1])):
                    stats['sorted_once'] += 1
                    return pts.take(positive[order]).groupby('track_id')
        stats['per_track'] += 1
        return ((tr, g[g.native.gt(0)].sort_values('frame')) for tr, g in pts.groupby('track_id'))
    return _bracketed_tracks


def _digest(module):
    try:
        return _file_sha256(module.__file__)
    except (AttributeError, OSError, TypeError):
        return None


@contextmanager
def install(stats=None):
    """Replace every loaded verified version (and its aliases) for the duration; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'sorted_once', 'per_track'):
        stats.setdefault(key, 0)
    helper = tracks(stats)
    swap, added = {}, []
    for module in _loaded(NAME):
        original = getattr(module, FUNCTION, None)
        if (original is None or getattr(original, '__wrapped__', None) is not None
                or id(original) in swap or _digest(module) not in VERSIONS):
            continue
        try:
            source = inspect.getsource(original)
        except (OSError, TypeError):
            continue
        if source.count(OLD) != 1:
            continue
        vars(module)['_bracketed_tracks'] = helper
        added.append(module)
        namespace = {}
        flags = original.__code__.co_flags & __future__.annotations.compiler_flag
        exec(compile(source.replace(OLD, NEW), '<bracketed-consensus-tracks>', 'exec',
                     flags=flags, dont_inherit=True), vars(module), namespace)
        replacement = namespace[FUNCTION]
        replacement.__wrapped__ = original
        swap[id(original)] = (original, replacement)
    stats['versions'] += len(swap)
    aliases = []
    if swap:
        children = [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']
        children += added + _loaded('complete_body_consensus_memory')
        for child in children:
            names = getattr(child, '__dict__', None)
            if not names or swap.keys().isdisjoint(map(id, names.values())):
                continue
            for key, value in list(names.items()):
                entry = swap.get(id(value))
                # A module listed twice already holds the replacement the second time.
                if entry is not None and value is entry[0]:
                    aliases.append((child, key, value))
                    setattr(child, key, entry[1])
    try:
        yield stats
    finally:
        for child, key, value in reversed(aliases):
            setattr(child, key, value)
        for module in added:
            vars(module).pop('_bracketed_tracks', None)
