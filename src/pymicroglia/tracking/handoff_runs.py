"""Owner runs of the handoff-episode modules from frame and owner arrays.

complete_owner_handoff_episodes_*._runs sorted a track's rows by frame and
walked them with itertuples, starting a new run whenever a row's frame is not
the previous row's frame plus one or its owner differs from the previous
row's. Those are adjacent-row comparisons, so after the same sort_values call
the run starts come from two array comparisons. int() and astype(int64) both
truncate toward zero; columns whose values int() would reject or that int64
cannot hold (NaN, infinities, nullable or object columns, uint64 beyond the
int64 range) keep the original loop. Run dicts hold Python ints, as before.

Rows whose integer frame column already strictly increases are read as they
are: the sort keeps every row where it is, and only the two columns' values
are read from the result.
"""
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path

import numpy as np

SOURCES = {
    'complete_owner_handoff_episodes_r10_a002': '5305c91dc515bcfcf7612825fbdce55d4d6773edae27d71bfd86bfca8f8938f1',
    'complete_owner_handoff_episodes_a003': '4cb7f54fda6732a2308b6ceee9109794c0bf9de77ef60893848e667f7ee61c09',
}


def _integers(column):
    dtype = getattr(column, 'dtype', None)
    if not isinstance(dtype, np.dtype) or dtype.kind not in 'biuf':
        return None
    values = column.to_numpy()
    if dtype.kind == 'f':
        if not np.isfinite(values).all() or (len(values) and np.abs(values).max() >= 2.0**62):
            return None
    elif dtype.kind == 'u' and dtype.itemsize == 8 and len(values) and values.max() >= 2**62:
        return None
    return values.astype(np.int64)


def _in_order(rows, stats):
    """rows itself when its integer frames already strictly increase, which the sort keeps."""
    columns = rows.columns
    if columns.is_unique and "frame" in columns:
        dtype = rows["frame"].dtype
        if isinstance(dtype, np.dtype) and dtype.kind in 'iu':
            frames = rows["frame"].to_numpy()
            if len(frames) < 2 or bool(np.all(frames[1:] > frames[:-1])):
                stats['in_order'] += 1
                return rows
    return rows.sort_values("frame")


def runs(original, stats):
    def _runs(rows):
        ordered = _in_order(rows, stats)
        frames = _integers(ordered["frame"]) if "frame" in ordered else None
        owners = _integers(ordered["accepted_owner"]) if "accepted_owner" in ordered else None
        if frames is None or owners is None or frames.ndim != 1 or owners.ndim != 1:
            stats['original'] += 1
            return original(rows)
        stats['arrays'] += 1
        if not len(frames):
            return []
        new = np.ones(len(frames), bool)
        new[1:] = (frames[1:] != frames[:-1] + 1) | (owners[1:] != owners[:-1])
        starts = np.flatnonzero(new).tolist()
        ends = starts[1:] + [len(frames)]
        t, o = frames.tolist(), owners.tolist()
        return [{"first": t[s], "last": t[e - 1], "owner": o[s], "length": e - s}
                for s, e in zip(starts, ends)]
    _runs.__wrapped__ = original
    return _runs


@contextmanager
def install(stats=None):
    stats = stats if stats is not None else {}
    for key in ('supported', 'arrays', 'original', 'in_order'):
        stats.setdefault(key, 0)
    replaced = []
    for name, digest in SOURCES.items():
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        original = getattr(module, '_runs', None)
        if (original is None or getattr(original, '__wrapped__', None) is not None
                or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest):
            continue
        module._runs = runs(original, stats)
        replaced.append((module, original))
        stats['supported'] += 1
    try:
        yield stats
    finally:
        for module, original in replaced:
            module._runs = original
