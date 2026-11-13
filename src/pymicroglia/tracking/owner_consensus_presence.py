"""Owner-consensus application with one per-frame label census.

For each eligible proposal the application copied every frame outside the run
to ask whether the source still appears there, and counted the proposed
owner's pixels on every frame to take its median area. Both answers come from
the unchanged input movie, so one pass of per-frame values and pixel counts is
made the first time either is needed: the source appears outside the run when
an outside frame holds it, and the owner's positive areas are its counts in
frame order.
"""
import __future__
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SOURCE = '0080d7c3bf52221d3ae9407eef762dbe01920a2f848732656d4d773f5d625c71'
REPLACEMENTS = (
    ("    medians: dict[int, float] = {}\n",
     "    medians: dict[int, float] = {}\n    _label_memo = {}\n"),
    ("        if not np.any(labels[outside] == source):\n",
     "        if not _present_outside(labels, source, outside, _label_memo):\n"),
    ("        medians.setdefault(owner, _identity_median_area(labels, owner))\n",
     "        medians.setdefault(owner, _median_area(labels, owner, _label_memo))\n"),
)


def _census(labels, memo):
    if 'counts' not in memo:
        counts = []
        for t in range(len(labels)):
            frame = labels[t]
            frame = frame.view(np.ndarray) if isinstance(frame, np.ndarray) else np.asarray(frame)
            values, number = np.unique(frame, return_counts=True)
            counts.append(dict(zip(values.tolist(), number.tolist())))
        memo['counts'] = counts
    return memo['counts']


def helpers(stats):
    def present_outside(labels, source, outside, memo):
        counts = _census(labels, memo)
        stats['presence'] += 1
        return any(int(source) in counts[t] for t in np.flatnonzero(outside).tolist())

    def median_area(labels, owner, memo):
        counts = _census(labels, memo)
        areas = [counts[t].get(int(owner), 0) for t in range(len(counts))]
        positive = [area for area in areas if area > 0]
        stats['medians'] += 1
        return float(np.median(positive)) if positive else 0.0

    return dict(_present_outside=present_outside, _median_area=median_area)


@contextmanager
def install(module):
    stats = dict(supported=False, presence=0, medians=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    original = module.apply_dependency_ordered_proposals
    source = inspect.getsource(original)
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            yield stats
            return
        source = source.replace(old, new)
    globals_ = vars(module)
    added = helpers(stats)
    globals_.update(added)
    namespace = {}
    exec(compile(source, '<owner-consensus-presence>', 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), globals_, namespace)
    module.apply_dependency_ordered_proposals = namespace['apply_dependency_ordered_proposals']
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.apply_dependency_ordered_proposals = original
        for name in added:
            globals_.pop(name, None)
