"""Whole-movie name counts are summed frame by frame instead of through an int64 copy of the movie.

Four correction families (complete short soma-owner excursions, recurrent soma
name conflicts a002, separable bookended soma basins, soma-free temporal
satellites; flicker_component_boxes applies the same edit to the bookended
flickers) begin `correct` with np.bincount(labels.ravel().astype(np.int64)):
a 0.85 GB int64 copy of a 476-frame movie just to count each name's pixels.
For a non-negative integer movie the same counts are the sum of each frame's
np.bincount, padded to the longest frame histogram; the longest is max name + 1,
the whole movie's length. The result is a fresh int64 array, as before, and
callers may write to it. Other dtypes, signed movies and empty movies evaluate
the original expression.

Recurrent soma name conflicts a002 also filtered the whole strong table once per
track (strong[strong.track_id == track]). Its strong rows are grouped by track
once instead; a group holds exactly the rows whose track equals the key, in
table order with their index, so the frame sort that follows sees the same
table. Tracks without a group, and non-numeric track columns, use the original
filter. Both of its per-track frame sorts are skipped when the track's integer
frames already increase strictly, which a sort leaves as they are.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
try:
    from .module_scan import file_sha256 as _file_sha256
    from .track_sort_skips import frame_ordered as _frame_ordered_helper
except ImportError:
    from module_scan import file_sha256 as _file_sha256
    from track_sort_skips import frame_ordered as _frame_ordered_helper

VERSIONS = {'3b7520ef6810cde0', 'cc64ec62cafcc1c6', 'a26d0af1c760f6de', '8953290e2ef5ccdb'}
TEXT = 'np.bincount(labels.ravel().astype(np.int64))'
# Per-version statement edits applied with the count edit (each must occur once).
EXTRA = {'cc64ec62cafcc1c6': (
    ('    strong = scored[(scored.state.astype(str) == "observed") & scored.strong]\n',
     '    strong = scored[(scored.state.astype(str) == "observed") & scored.strong]\n'
     '    _strong_tracks = _track_groups(strong)\n'),
    ('        sole_strong = strong[strong.track_id == track].sort_values("frame")\n',
     '        sole_strong = _frame_ordered(_sole_rows(_strong_tracks, strong, track))\n'),
    # Each track's rows skip the frame sort when their integer frames already
    # increase strictly (track_sort_skips.frame_ordered); a sort leaves them so.
    ('        group = group.sort_values("frame")\n',
     '        group = _frame_ordered(group)\n'),
)}


def _digest(module):
    try:
        return _file_sha256(module.__file__)[:16]
    except (AttributeError, OSError, TypeError):
        return None


def counts(stats):
    def movie_counts(labels):
        if (not isinstance(labels, np.ndarray) or labels.ndim != 3 or labels.dtype.kind != 'u'
                or labels.dtype.itemsize > 4 or not labels.size):
            stats['direct'] += 1
            return np.bincount(labels.ravel().astype(np.int64))
        total = np.zeros(0, np.int64)
        plain = labels.view(np.ndarray)
        for frame in plain:
            frame_counts = np.bincount(frame.ravel())
            if len(frame_counts) > len(total):
                grown = np.zeros(len(frame_counts), np.int64)
                grown[:len(total)] = total
                total = grown
            total[:len(frame_counts)] += frame_counts
        stats['framed'] += 1
        return total
    return movie_counts


def track_groups(stats):
    def _track_groups(strong):
        if (not isinstance(strong, pd.DataFrame) or not strong.columns.is_unique
                or 'track_id' not in strong.columns or strong['track_id'].dtype.kind not in 'iuf'):
            return None
        return {key: group for key, group in strong.groupby('track_id', sort=False)}

    def _sole_rows(groups, strong, track):
        found = groups.get(track) if groups is not None else None
        if found is None:
            stats['filtered'] += 1
            return strong[strong.track_id == track]
        stats['grouped'] += 1
        return found
    return _track_groups, _sole_rows


@contextmanager
def install(stats=None):
    """Replace every loaded verified correct (and its aliases) for the duration; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'framed', 'direct', 'grouped', 'filtered', 'in_order', 'sorted'):
        stats.setdefault(key, 0)
    swap = {}
    rewritten = []
    helper = counts(stats)
    grouping, sole = track_groups(stats)
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, 'correct', None)
        if (original is None or getattr(original, '__module__', None) != getattr(module, '__name__', None)
                or getattr(original, '__wrapped__', None) is not None or id(original) in swap
                or _digest(module) not in VERSIONS):
            continue
        source = inspect.getsource(original)
        extra = EXTRA.get(_digest(module), ())
        if source.count(TEXT) != 1 or any(source.count(old) != 1 for old, _ in extra):
            continue
        source = source.replace(TEXT, '_movie_counts(labels)')
        for old, new in extra:
            source = source.replace(old, new)
        globals_ = original.__globals__
        globals_['_movie_counts'] = helper
        if extra:
            globals_['_track_groups'], globals_['_sole_rows'] = grouping, sole
            globals_['_frame_ordered'] = _frame_ordered_helper(stats)
        rewritten.append(globals_)
        namespace = {}
        exec(compile(source, '<movie-label-counts>', 'exec',
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
        for globals_ in rewritten:
            globals_.pop('_movie_counts', None)
            globals_.pop('_track_groups', None)
            globals_.pop('_sole_rows', None)
            globals_.pop('_frame_ordered', None)
