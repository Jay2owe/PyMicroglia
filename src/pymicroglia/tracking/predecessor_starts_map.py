"""Predecessor name memory looks up track starts through one Series per call.

predecessor_name_memory_a002.preserve_predecessors filters each candidate
successor frame with new.track_id.map(starts), starts being a dict of every
track's first frame. Series.map turns a dict into pd.Series(dict) (float64
when empty) on every call and then looks values up through that Series' index,
so the map rebuilt a Series and its index table once per track. The same
Series is made once, before the loop, and passed to map, which takes the same
lookup path. Only that statement changes; the module is pinned by content
hash and callers that imported the function by name are re-pointed.

Every track's rows are only read after their frame sort, so when no track
repeats or reverses a frame in table order they skip it
(track_sort_skips.tracks_in_order; same keys in the same order).
"""
from contextlib import contextmanager
import __future__
import hashlib
import inspect
from pathlib import Path
import sys

import pandas as pd
try:
    from .track_sort_skips import tracks_in_order as _tracks_in_order_helper
except ImportError:
    from track_sort_skips import tracks_in_order as _tracks_in_order_helper

SOURCE = 'f87be3abc8afb43c403407ab0fd947aae5a637406e05d21abbcdefb83a70173f'
MODULE = 'predecessor_name_memory_a002'
EDITS = (
    ("    full = {int(tr):g.sort_values('frame') for tr,g in scored.groupby('track_id')}\n",
     "    full = _predecessor_tracks(scored,'track_id')\n"),
    ("    ends = {int(tr):int(g.frame.max()) for tr,g in full.items()}\n",
     "    ends = {int(tr):int(g.frame.max()) for tr,g in full.items()}\n"
     "    _starts = _starts_series(starts)\n"),
    ("        new = new[new.track_id.map(starts).eq(t+1)]\n",
     "        new = new[new.track_id.map(_starts).eq(t+1)]\n"),
)


def starts_series(starts):
    """The Series pandas' map makes from this dict."""
    return pd.Series(starts, dtype='float64') if len(starts) == 0 else pd.Series(starts)


@contextmanager
def install(stats=None):
    stats = stats if stats is not None else {}
    for key in ('versions', 'groups_in_order', 'groups_sorted'):
        stats.setdefault(key, 0)
    module = sys.modules.get(MODULE)
    aliases = []
    added = []
    try:
        original = getattr(module, 'preserve_predecessors', None) if module is not None else None
        digest = None
        if original is not None and getattr(original, '__wrapped__', None) is None:
            try:
                digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            except (OSError, TypeError):
                digest = None
        if digest == SOURCE:
            source = inspect.getsource(original)
            if all(source.count(old) == 1 for old, _ in EDITS):
                for old, new in EDITS:
                    source = source.replace(old, new)
                # The module's own globals, so later rebinding there is still seen.
                vars(module)['_starts_series'] = starts_series
                vars(module)['_predecessor_tracks'] = _tracks_in_order_helper(stats)
                added.append(module)
                namespace = {}
                exec(compile(source, '<predecessor-starts-map>', 'exec',
                             flags=__future__.annotations.compiler_flag, dont_inherit=True),
                     vars(module), namespace)
                replacement = namespace['preserve_predecessors']
                replacement.__wrapped__ = original
                for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
                    names = getattr(child, '__dict__', None)
                    if not names or id(original) not in map(id, names.values()):
                        continue
                    for name, value in list(names.items()):
                        if value is original:
                            aliases.append((child, name, value))
                            setattr(child, name, replacement)
                stats['versions'] += 1
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
        for module in added:
            vars(module).pop('_starts_series', None)
            vars(module).pop('_predecessor_tracks', None)
