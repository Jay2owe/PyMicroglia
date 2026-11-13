"""Owner runs sliced from the sorted rows instead of grouped by their run number.

coobserved_bookended_soma_memory_a001._runs (also imported by
repeated_retired_owner_memory_a002) and recurrent_core_memory_a002._anchors
split a track's frame-sorted rows into runs: a run number that is the
cumulative sum of a Boolean "owner or frame step changed" column, then one
groupby over that number. A cumulative sum of a plain Boolean column is an
int64 count that never decreases, so every run is one contiguous block of the
sorted rows, the blocks come in ascending run order (groupby's order whether
sorted or not), and groupby hands back each block's rows with the same index,
columns, dtypes and order as rows.iloc over it. The callers only read the
runs. A run number of any other dtype (a nullable owner column makes missing
values, which groupby drops) keeps the original groupby.

_anchors' run number itself is taken from NumPy for integer owner and frame
columns below 2**53: the first row starts a run (pandas compares it with the
missing shifted value), and a later row starts one where its owner differs
from the previous row's or its frame is not the previous frame plus one,
which is what the float comparisons pandas makes give for such integers. The
cumulative sum is the same int64 count on the same index.

Both take a track's rows as they are when its integer frames already strictly
increase (track_sort_skips.frame_ordered): the sort would keep every row in
place, and the runs cut from the rows are only read by their callers.
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
    from .module_scan import loaded as _loaded
    from .track_sort_skips import frame_ordered as _frame_ordered_helper
except ImportError:
    from module_scan import loaded as _loaded
    from track_sort_skips import frame_ordered as _frame_ordered_helper

VERSIONS = {
    'coobserved_bookended_soma_memory_a001': (
        'dafc2cfecd65ef3b2462716bc77e79f9b7973cf367ff97b0c0a2f4edc3174b73', '_runs',
        (('    return [run for _, run in rows.groupby(breaks, sort=False)]\n',
          '    return _run_blocks(rows, breaks, False)\n'),)),
    'recurrent_core_memory_a002': (
        '043b4af314083c0be0acdd734110b5708b40d81def9bf2d4a946add0e48fb052', '_anchors',
        (('        for _, run in group.groupby(runs):\n',
          '        for run in _run_blocks(group, runs, True):\n'),)),
}
# Applied when present; the version keeps its required edits either way.
OPTIONAL = {
    'coobserved_bookended_soma_memory_a001': (
        ('    rows = group.sort_values("frame")\n',
         '    rows = _run_frames(group)\n'),),
    'recurrent_core_memory_a002': (
        ("        group=group.sort_values('frame')\n",
         "        group=_run_frames(group)\n"),
        ('        runs=(group.candidate_owner.ne(group.candidate_owner.shift()) | group.frame.diff().ne(1)).cumsum()\n',
         '        runs=_run_numbers(group)\n'),),
}
LIMIT = 1 << 53
# Modules that import a rewritten function by name.
IMPORTERS = ('repeated_retired_owner_memory_a002',)


def run_blocks(stats):
    def _run_blocks(rows, numbers, sort):
        keys = numbers.to_numpy() if hasattr(numbers, 'to_numpy') else None
        if (keys is None or keys.dtype != np.int64 or keys.ndim != 1 or len(keys) != len(rows)
                or not numbers.index.equals(rows.index)):
            stats['grouped'] += 1
            return [run for _, run in rows.groupby(numbers, sort=sort)]
        stats['sliced'] += 1
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else keys[:0]
        ends = np.r_[starts[1:], len(keys)]
        return [rows.iloc[start:end] for start, end in zip(starts.tolist(), ends.tolist())]
    return _run_blocks


def run_numbers(stats):
    def _run_numbers(group):
        owner = group['candidate_owner'] if group.columns.is_unique and 'candidate_owner' in group.columns else None
        frame = group['frame'] if owner is not None and 'frame' in group.columns else None
        # Plain NumPy integer columns only: nullable Int64 also reports kind 'i'.
        if (frame is not None and len(group) and isinstance(owner.dtype, np.dtype) and isinstance(frame.dtype, np.dtype)
                and owner.dtype.kind in 'iu' and frame.dtype.kind in 'iu'):
            owners, frames = owner.to_numpy(), frame.to_numpy()
            if (int(owners.max()) < LIMIT and int(owners.min()) > -LIMIT
                    and int(frames.max()) < LIMIT and int(frames.min()) > -LIMIT):
                owners, frames = owners.astype(np.int64), frames.astype(np.int64)
                breaks = np.ones(len(group), bool)
                breaks[1:] = (owners[1:] != owners[:-1]) | (frames[1:] - frames[:-1] != 1)
                stats['numbered'] += 1
                return pd.Series(np.cumsum(breaks, dtype=np.int64), index=group.index)
        stats['numbers_original'] += 1
        return (group.candidate_owner.ne(group.candidate_owner.shift()) | group.frame.diff().ne(1)).cumsum()
    return _run_numbers


@contextmanager
def install(stats=None):
    """Rewrite every loaded verified version (also those only a composition holds); restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'sliced', 'grouped', 'numbered', 'numbers_original', 'in_order', 'sorted'):
        stats.setdefault(key, 0)
    helper = run_blocks(stats)
    numbers = run_numbers(stats)
    frames = _frame_ordered_helper(stats)
    swap = {}
    added = []
    for name, (digest, function, edits) in VERSIONS.items():
        for module in _loaded(name):
            original = getattr(module, function, None)
            if (original is None or getattr(original, '__wrapped__', None) is not None
                    or id(original) in swap):
                continue
            try:
                if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
                    continue
                source = inspect.getsource(original)
            except (OSError, TypeError, AttributeError):
                continue
            if not all(source.count(old) == 1 for old, _ in edits):
                continue
            for old, new in edits:
                source = source.replace(old, new)
            for old, new in OPTIONAL.get(name, ()):
                if source.count(old) == 1:
                    source = source.replace(old, new)
            vars(module)['_run_blocks'] = helper
            vars(module)['_run_numbers'] = numbers
            vars(module)['_run_frames'] = frames
            added.append(module)
            namespace = {}
            flags = original.__code__.co_flags & __future__.annotations.compiler_flag
            exec(compile(source, '<run-blocks>', 'exec', flags=flags, dont_inherit=True), vars(module), namespace)
            replacement = namespace[function]
            replacement.__wrapped__ = original
            swap[id(original)] = (original, replacement)
            stats['versions'] += 1
    aliases = []
    if swap:
        children = [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']
        children += [m for name in (*VERSIONS, *IMPORTERS) for m in _loaded(name)]
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
            vars(module).pop('_run_blocks', None)
            vars(module).pop('_run_frames', None)
            vars(module).pop('_run_numbers', None)
