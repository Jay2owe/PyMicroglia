"""Projection-tolerant seat preservation: expected areas from owner components labelled once, in the owner's box.

_expected_area labelled the whole frame's owner mask (_component_area) for
every earlier point of a track, once per proposal and again for each collapse
frame's resident. Discovery does not change labels (the candidate is a copy
made after it), so within one discover_and_apply call each (frame, owner) mask
is labelled once, inside the owner's box, by censored_area_boxes.OwnerParts.
A point's area is that of separable_merge_recovery._component_at's component
at the default search radius of 6, which OwnerParts.area_at reproduces. The
rewritten _expected_area is the original source with only the area call
replaced; calls on any other labels, or outside discovery, keep
_component_area.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import threading
import types

import numpy as np
try:
    from .censored_area_boxes import OwnerParts
except ImportError:
    from censored_area_boxes import OwnerParts

SOURCE = 'beff005d513a7bd62fe3acd7babda72448ad837f31153c40f44827902d795f9e'
EDIT = ('        area = _component_area(labels[int(point.frame)], owner, point)\n',
        '        area = _seat_area(labels, int(point.frame), owner, point)\n')
# The current discovery's OwnerParts, per thread (discoveries in other threads keep their own).
_local = threading.local()


@contextmanager
def install(module):
    stats = dict(supported=False, calls=0, labelled=0, reused=0, boxed=0, fallback=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    original = module.discover_and_apply
    expected = module._expected_area
    source = inspect.getsource(expected)
    if source.count(EDIT[0]) != 1:
        raise ValueError('Unexpected _expected_area source')
    scratch = {}
    exec(compile(source.replace(*EDIT), '<seat-projection-boxes>', 'exec'), dict(module.__dict__), scratch)
    # The module's own globals, so _seat_area and later installs resolve there.
    rewritten = types.FunctionType(scratch['_expected_area'].__code__, module.__dict__, '_expected_area',
                                   expected.__defaults__, expected.__closure__)
    rewritten.__wrapped__ = expected

    def _seat_area(labels, frame, owner, point):
        plane = labels[frame]
        memo = getattr(_local, 'memo', None)
        if memo is None or labels is not memo.labels:
            stats['fallback'] += 1
            return module._component_area(plane, owner, point)
        return memo.area_at(frame, int(owner), float(point.x), float(point.y), 6)

    def discover_and_apply(labels, raw, points, params):
        if not (isinstance(labels, np.ndarray) and labels.ndim == 3 and np.issubdtype(labels.dtype, np.integer)):
            return original(labels, raw, points, params)
        outer = getattr(_local, 'memo', None)
        _local.memo = OwnerParts(labels, module.STRUCTURE, stats)
        stats['calls'] += 1
        try:
            return original(labels, raw, points, params)
        finally:
            _local.memo = outer
    discover_and_apply.__wrapped__ = original
    module._seat_area = _seat_area
    module._expected_area = rewritten
    module.discover_and_apply = discover_and_apply
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.discover_and_apply = original
        module._expected_area = expected
        if '_seat_area' in module.__dict__:
            del module._seat_area
