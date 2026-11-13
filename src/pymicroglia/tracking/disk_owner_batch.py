"""Merge residency and persistent body identity: every point's disk owner voted in one batch.

Both modules attach owners point by point with the same _disk_owner (radius
0.75 r clipped to [2, 6], window floor(c - r) .. ceil(c + r + 1) clipped to
the image, owner the most frequent positive label, smallest on ties). The
attach function is rewritten with only its per-row list replaced: the list is
built from point_disk_votes.votes, with coverage and purity divided as the
original divides them. A frame outside 0 .. len(labels) - 1, a non-finite or
huge coordinate or radius, a coordinate below -2 (where the original window's
end turns negative and the slice counts from the far edge), a non-integer
stack or a conversion the batch cannot repeat keeps the original per-point
list, so errors stay the original's. Persistent body identity's unclaimed
owner under each rounded, clipped point is read in one gather the same way.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

import numpy as np
try:
    from .point_disk_votes import votes
except ImportError:
    from point_disk_votes import votes

MERGE_RESIDENCY = ('a20e87d0ecf079a4c77e2aa279e1c91a9ecbf2db3f7a3ce53ae4151f1494b7f3', '_attach_owners', (
    ('    evidence = [_disk_owner(labels[int(row.frame)], float(row.x), float(row.y),\n'
     '                            float(row.radius_px))\n'
     '                for row in rows.itertuples(index=False)]\n',
     '    evidence = _disk_owner_batch(rows, labels)\n'),))
UNCLAIMED = ('    unclaimed_owner = []\n'
             '    for row in rows.itertuples(index=False):\n'
             '        frame = int(row.frame)\n'
             '        y = int(np.clip(round(float(row.y)), 0, labels.shape[1] - 1))\n'
             '        x = int(np.clip(round(float(row.x)), 0, labels.shape[2] - 1))\n'
             '        unclaimed_owner.append(int(unclaimed[frame, y, x]))\n')
PERSISTENT_BODY = ('11c90fde1bd6f831e1ec5ea19dfb79faf9f3aad473aef2cf6780e04ebc8f6eb7', '_attach_evidence', (
    ('    owner_evidence = [\n'
     '        _disk_owner(labels[int(row.frame)], float(row.x), float(row.y),\n'
     '                    float(row.radius_px))\n'
     '        for row in rows.itertuples(index=False)]\n',
     '    owner_evidence = _disk_owner_batch(rows, labels)\n'),
    (UNCLAIMED,
     '    unclaimed_owner = _unclaimed_batch(rows, labels, unclaimed)\n'),))


def unclaimed_owners(rows, labels, unclaimed, stats):
    """The original list of unclaimed labels under each rounded, clipped point, read in one gather where audited."""
    def each():
        stats['unclaimed_fallback_calls'] += 1
        namespace = dict(np=np, rows=rows, labels=labels, unclaimed=unclaimed)
        exec(textwrap.dedent(UNCLAIMED), namespace)
        return namespace['unclaimed_owner']
    if not (isinstance(unclaimed, np.ndarray) and isinstance(labels, np.ndarray) and unclaimed.ndim == 3
            and labels.ndim == 3 and unclaimed.shape[1:] == labels.shape[1:] and len(rows)):
        return each()
    try:
        frames = np.array([int(value) for value in rows['frame'].tolist()], dtype=np.int64)
        x = np.array([float(value) for value in rows['x'].tolist()], dtype=float)
        y = np.array([float(value) for value in rows['y'].tolist()], dtype=float)
    except Exception:
        return each()
    if not (len(frames) == len(x) == len(y) == len(rows) and np.isfinite(x).all() and np.isfinite(y).all()
            and not np.any((np.abs(x) > 1e9) | (np.abs(y) > 1e9) | (frames < 0) | (frames >= len(unclaimed)))):
        return each()
    # round() and rint both round halves to even, exactly, at these magnitudes.
    rows_at = np.clip(np.rint(y).astype(np.int64), 0, labels.shape[1] - 1)
    columns_at = np.clip(np.rint(x).astype(np.int64), 0, labels.shape[2] - 1)
    stats['unclaimed_points'] += len(frames)
    return [int(value) for value in np.asarray(unclaimed[frames, rows_at, columns_at]).tolist()]


def evidence(module, rows, labels, stats):
    """The original list of (owner, coverage, purity), voted in one batch where audited."""
    def each():
        stats['fallback_calls'] += 1
        return [module._disk_owner(labels[int(row.frame)], float(row.x), float(row.y), float(row.radius_px))
                for row in rows.itertuples(index=False)]
    stats['calls'] += 1
    if not (isinstance(labels, np.ndarray) and labels.ndim == 3 and labels.dtype.kind in 'iu'):
        return each()
    if not len(rows):
        return []
    try:
        frames = [int(value) for value in rows['frame'].tolist()]
        x = np.array([float(value) for value in rows['x'].tolist()], dtype=float)
        y = np.array([float(value) for value in rows['y'].tolist()], dtype=float)
        radius = np.array([float(value) for value in rows['radius_px'].tolist()], dtype=float)
    except Exception:
        return each()
    if not (len(frames) == len(x) == len(y) == len(radius) == len(rows)):
        return each()
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(radius).all()):
        return each()
    if (np.any((x < -2) | (y < -2) | (np.abs(x) > 1e9) | (np.abs(y) > 1e9) | (np.abs(radius) > 1e9))
            or any(frame < 0 or frame >= len(labels) for frame in frames)):
        return each()
    result = votes(labels, frames, x, y, radius)
    if result is None:
        return each()
    owner, positive, inside, winning = result
    stats['points'] += len(frames)
    listed = []
    for name, found, total, best in zip(owner.tolist(), positive.tolist(), inside.tolist(), winning.tolist()):
        coverage = float(found / max(total, 1))
        listed.append((int(name), coverage, float(best / found)) if found else (0, coverage, 0.0))
    return listed


@contextmanager
def _rewrite(module, spec, stats):
    digest, name, edits = spec
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
        yield False
        return
    original = getattr(module, name)
    source = inspect.getsource(original)
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError('Unexpected %s source' % name)
        source = source.replace(old, new)
    scratch = {}
    exec(compile(source, '<disk-owner-batch:%s>' % module.__name__, 'exec'),
         dict(module.__dict__), scratch)
    # The module's own globals, so _disk_owner_batch and later installs resolve there.
    rewritten = types.FunctionType(scratch[name].__code__, module.__dict__, name,
                                   original.__defaults__, original.__closure__)
    rewritten.__wrapped__ = original
    helpers = dict(_disk_owner_batch=lambda rows, labels: evidence(module, rows, labels, stats),
                   _unclaimed_batch=lambda rows, labels, unclaimed: unclaimed_owners(rows, labels, unclaimed, stats))
    for helper, value in helpers.items():
        setattr(module, helper, value)
    setattr(module, name, rewritten)
    try:
        yield True
    finally:
        setattr(module, name, original)
        for helper in helpers:
            if helper in module.__dict__:
                delattr(module, helper)


@contextmanager
def install(merge_residency=None, persistent_body=None):
    stats = dict(supported=[], calls=0, points=0, fallback_calls=0, unclaimed_points=0, unclaimed_fallback_calls=0)
    with ExitStack() as stack:
        for module, spec in ((merge_residency, MERGE_RESIDENCY), (persistent_body, PERSISTENT_BODY)):
            if module is not None and stack.enter_context(_rewrite(module, spec, stats)):
                stats['supported'].append(module.__name__)
        yield stats
