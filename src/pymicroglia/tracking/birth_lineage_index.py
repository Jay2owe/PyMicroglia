"""Measured birth lineage: first native frames and per-frame nodes in one pass.

restore_measured_birth_lineage found each native name's first frame with a
whole-movie comparison per name, and listed each frame's fully measured nodes
by scanning every node record once per frame. One count table per frame gives
the same first frames (frames are visited in order), and one pass over the
records gives the same per-frame lists in record order. Movies that are not
non-negative integer images keep the original statements.

The disjoint-shape linking compared every measured node of one frame with
every node of the next in Python. Its box-gap distance test (np.hypot of the
box gaps against the smaller equal-area radius) is evaluated for all pairs at
once with the same arithmetic and a small upward slack, and only the pairs
that can pass reach the original statements, in the original order, which
repeat the exact test. The skipped pairs only failed that test; their find()
calls only compressed union-find paths, which leaves every root unchanged.
Same-name fragment joins read each frame's records from one grouping pass.
"""
from contextlib import contextmanager
import hashlib
import inspect
import itertools
from pathlib import Path
import sys

import numpy as np

SOURCE = 'b71552c75ee1f2da2e6672f92330a4b4c50ce1f5fdcfca9e4619ee2de24c990c'
MAXIMUM_LABEL = 1 << 20
REPLACEMENTS = [
    ("""    for name in np.unique(proposals):
        if name:first_native[int(name)]=int(np.flatnonzero(np.any(proposals==name,axis=(1,2)))[0])
""", """    _first=_first_frames(proposals)
    for name in np.unique(proposals):
        if name:first_native[int(name)]=(_first[int(name)] if _first is not None
            else int(np.flatnonzero(np.any(proposals==name,axis=(1,2)))[0]))
"""),
    ("""    per_frame={t:[q['node'] for q in records[1:] if q['frame']==t+1 and q['measured_pixels']==q['proposed_pixels']] for t in range(len(labels))}
""", """    _measured={}
    for q in records[1:]:
        if q['measured_pixels']==q['proposed_pixels']:_measured.setdefault(q['frame'],[]).append(q['node'])
    per_frame={t:list(_measured.get(t+1,())) for t in range(len(labels))}
"""),
    ("""        for a in per_frame[t-1]:
            ba=boxes[a];ra=np.sqrt(areas[a]/np.pi)
            for b in per_frame[t]:
                if find(a)==find(b):continue
""", """        for a,b,ba,ra in _birth_pairs(per_frame[t-1],per_frame[t],boxes,areas):
                if find(a)==find(b):continue
"""),
    ("""    for t in range(len(labels)):
        by_owner={}
        for q in records[1:]:
            if q['frame']==t+1:by_owner.setdefault(q['identity'],[]).append(q['node'])
""", """    _by_frame={}
    for q in records[1:]:_by_frame.setdefault(q['frame'],[]).append(q)
    for t in range(len(labels)):
        by_owner={}
        for q in _by_frame.get(t+1,()):
            by_owner.setdefault(q['identity'],[]).append(q['node'])
"""),
]
# Relative slack of the vectorised distance screen; the original test decides.
SLACK = 1e-9


def pairs(stats):
    def _birth_pairs(before, after, boxes, areas):
        known = (type(areas) is np.ndarray and areas.ndim == 1 and areas.dtype.kind in 'iuf'
                 and all(type(n) in (int, np.int32, np.int64) and 0 <= n < len(areas) and n in boxes
                         for n in itertools.chain(before, after)))
        if not known or not before or not after:
            # The original loops, statement for statement.
            stats['pairs_direct'] += 1
            for a in before:
                ba = boxes[a]; ra = np.sqrt(areas[a] / np.pi)
                for b in after:
                    yield a, b, ba, ra
            return
        stats['pairs_screened'] += 1
        def edges(nodes):
            return np.array([(boxes[n][0].start, boxes[n][0].stop, boxes[n][1].start, boxes[n][1].stop)
                             for n in nodes], np.int64).reshape(-1, 4)
        left, right = edges(before), edges(after)
        radius = np.minimum(np.sqrt(areas[np.asarray(before)] / np.pi)[:, None],
                            np.sqrt(areas[np.asarray(after)] / np.pi)[None, :])
        dy = np.maximum(0, np.maximum(left[:, None, 0] - right[None, :, 1] + 1, right[None, :, 0] - left[:, None, 1] + 1))
        dx = np.maximum(0, np.maximum(left[:, None, 2] - right[None, :, 3] + 1, right[None, :, 2] - left[:, None, 3] + 1))
        near = np.hypot(dx, dy) <= radius * (1 + SLACK) + SLACK
        rows, cols = np.nonzero(near)
        stats['pairs_kept'] += len(rows)
        stats['pairs_skipped'] += near.size - len(rows)
        held = {}
        for i, j in zip(rows.tolist(), cols.tolist()):
            a = before[i]
            if i not in held:
                held[i] = (boxes[a], np.sqrt(areas[a] / np.pi))
            ba, ra = held[i]
            yield a, after[j], ba, ra
    return _birth_pairs


def first_frames(stats):
    def _first_frames(movie):
        first = {}
        for t in range(len(movie)):
            plain = np.asarray(movie[t])
            if plain.dtype.kind not in 'iu':
                stats['fallback'] += 1
                return None
            flat = plain.ravel()
            if not flat.size:
                continue
            if int(flat.min()) < 0 or int(flat.max()) >= MAXIMUM_LABEL:
                stats['fallback'] += 1
                return None
            for name in np.flatnonzero(np.bincount(flat)):
                first.setdefault(int(name), t)
        stats['tables'] += 1
        return first
    return _first_frames


def adapt(module, stats):
    original = getattr(module, 'restore_measured_birth_lineage', None)
    path = getattr(module, '__file__', None)
    if (original is None or not path or getattr(original, '__module__', None) != module.__name__
            or getattr(original, '__wrapped__', None) is not None):
        return None
    try:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE:
            return None
        source = inspect.getsource(original)
    except (OSError, TypeError):
        return None
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            return None
        source = source.replace(old, new)
    namespace = dict(vars(module))
    namespace['_first_frames'] = first_frames(stats)
    for key in ('pairs_direct', 'pairs_screened', 'pairs_kept', 'pairs_skipped'):
        stats.setdefault(key, 0)
    namespace['_birth_pairs'] = pairs(stats)
    exec(compile(source, '<birth-lineage-index>', 'exec'), namespace)
    replacement = namespace['restore_measured_birth_lineage']
    replacement.__wrapped__ = original
    return original, replacement


@contextmanager
def install(stats=None):
    """Replace the loaded verified version and its imported aliases; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'tables', 'fallback'):
        stats.setdefault(key, 0)
    swap = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        found = adapt(module, stats) if hasattr(module, 'restore_measured_birth_lineage') else None
        if found is not None:
            swap[id(found[0])] = found
    aliases = []
    if swap:
        stats['versions'] += len(swap)
        for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
            names = getattr(child, '__dict__', None)
            if not names:
                continue
            if not swap or swap.keys().isdisjoint(map(id, names.values())):  # alias loop skip
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
