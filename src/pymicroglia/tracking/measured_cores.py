"""Measured soma cores per frame from column arrays instead of per-frame tables.

observed_soma_reservation.measured_cores_by_frame grouped the strong observed
points by frame (first-appearance order), dropped each frame's duplicate
(track_id, x, y) rows keeping the first, and listed (track, round(y),
round(x)) of the finite rows from itertuples. Integer frames group exactly,
so one duplicated() over (frame, track_id, x, y) marks the same rows (the
same pandas hashing, which also equates 0.0 and -0.0 and NaN with NaN);
factorize gives the frames in first-appearance order and a stable sort keeps
row order within each. itertuples gives Python ints and floats for integer
and float64 columns; np.rint rounds halves to even as round does and is exact
below 2**52, so the same integers result. Frames whose rows are all dropped
keep their empty list. Other column types, repeated column names and larger
coordinates keep the original.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

VERSIONS = {'8079ea45265a8f1a'}
LIMIT = 2.0 ** 52


def _digest(module):
    try:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()[:16]
    except (AttributeError, OSError, TypeError):
        return None


def adapt(original, stats):
    def measured_cores_by_frame(points):
        required = {'frame', 'track_id', 'x', 'y', 'strong', 'state'}
        if required - set(points.columns) or not points.columns.is_unique:
            stats['original'] += 1
            return original(points)
        selected = points[points.strong.astype(bool) & points.state.eq('observed')]
        if 'physically_visible' in selected:
            selected = selected[selected.physically_visible.astype(bool)]
        elif 'visibility' in selected:
            selected = selected[selected.visibility.gt(0)]
        frame, track = selected['frame'].to_numpy(), selected['track_id'].to_numpy()
        x, y = selected['x'].to_numpy(), selected['y'].to_numpy()
        if not (frame.dtype.kind in 'iu' and track.dtype.kind in 'iu'
                and x.dtype == np.float64 and y.dtype == np.float64):
            stats['original'] += 1
            return original(points)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(np.abs(x[finite]) >= LIMIT) or np.any(np.abs(y[finite]) >= LIMIT):
            stats['original'] += 1
            return original(points)
        keep = finite & ~selected.duplicated(['frame', 'track_id', 'x', 'y']).to_numpy()
        codes, uniques = pd.factorize(frame, sort=False)
        order = np.argsort(codes, kind='stable')
        order = order[keep[order]]
        counts = np.bincount(codes[order], minlength=len(uniques))
        rows = list(zip(track[order].tolist(), np.rint(y[order]).astype(np.int64).tolist(),
                        np.rint(x[order]).astype(np.int64).tolist()))
        frames = {}
        start = 0
        for value, count in zip(uniques.tolist(), counts.tolist()):
            frames[int(value)] = rows[start:start + count]
            start += count
        stats['vectorised'] += 1
        return frames
    measured_cores_by_frame.__wrapped__ = original
    return measured_cores_by_frame


@contextmanager
def install(stats=None):
    """Replace every loaded verified copy and its imported aliases; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'vectorised', 'original'):
        stats.setdefault(key, 0)
    modules = [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']
    swap = {}
    for module in modules:
        function = getattr(module, '__dict__', {}).get('measured_cores_by_frame')
        if (function is None or getattr(function, '__module__', None) != module.__name__
                or getattr(function, '__wrapped__', None) is not None or id(function) in swap
                or _digest(module) not in VERSIONS):
            continue
        swap[id(function)] = (function, adapt(function, stats))
    aliases = []
    if swap:
        stats['versions'] += len(swap)
        for child in modules:
            names = getattr(child, '__dict__', None)
            if not names or swap.keys().isdisjoint(map(id, names.values())):
                continue
            for name, value in list(names.items()):
                found = swap.get(id(value))
                if found is not None and value is found[0]:
                    aliases.append((child, name, value))
                    setattr(child, name, found[1])
    try:
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
