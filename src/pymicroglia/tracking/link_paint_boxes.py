"""Paint each linked or new observation inside its mask's known box.

A packed prepared observation is rebuilt as a zero frame with its packed box
filled, so its pixels are painted from that box.

track_direction wrote `output[t][observation["mask"]] = identity` for every
observation it keeps. A prepared observation's mask is a LabelMask, so each
write compared the whole frame with the identity and scanned the whole frame
again. Its pixels all lie in that label's find_objects box of the read-only
snapshot, found once per snapshot; the same pixels receive the same value
through the target's own fancy assignment. The new state's own copy of the
mask (`observation["mask"].copy()`, a whole-frame comparison and copy) is the
same Boolean frame, filled inside that box. Any other mask or target uses the
original statement.
"""
from contextlib import contextmanager
import hashlib
import inspect
import weakref

import numpy as np
from scipy import ndimage

SOURCE = 'e3d4e3ad85b46f9498b86c69ffed443a6f12ce20577e5a37485ca875bedecc57'
STATEMENTS = (
    ('            output[t][observation["mask"]] = state.identity\n',
     '            _paint_label_mask(output[t], observation["mask"], state.identity)\n'),
    ('            output[t][observation["mask"]] = identity\n',
     '            _paint_label_mask(output[t], observation["mask"], identity)\n'),
)
COPY = ('observation["mask"].copy()', '_copy_label_mask(observation["mask"])', 2)
# A packed observation's full mask is rebuilt as zeros with its packed box
# filled; the frame's rebuilt masks are remembered with that box.
MATERIALIZE = ('_materialize_observations(prepared_observations[t])',
               '_materialize_boxed(prepared_observations[t])')


def painter(label_mask, stats, module=None):
    """(paint, copy, materialize) sharing one find_objects table per read-only
    snapshot and the current frame's rebuilt packed masks."""
    held = {}
    rebuilt = {}

    def materialize(frame):
        result = module._materialize_observations(frame)
        rebuilt.clear()
        packed = module._PackedObservationMask
        for local, observation in frame.items():
            stored = observation.get("mask")
            if isinstance(stored, packed):
                mask = result[local]["mask"]
                if type(mask) is np.ndarray and mask.dtype == bool and mask.ndim == 2:
                    # Weak: the frame's own dictionary keeps its masks alive.
                    rebuilt[id(mask)] = (weakref.ref(mask), tuple(stored.bounds))
        return result

    def boxed(mask):
        labels = getattr(mask, 'labels', None) if type(mask) is label_mask else None
        if (labels is None or type(labels) is not np.ndarray or labels.flags.writeable
                or labels.ndim != 2 or labels.dtype.kind not in 'iu' or mask.identity <= 0):
            return None
        entry = held.get(id(labels))
        if entry is None or entry[0] is not labels:
            if len(held) >= 8:
                held.clear()
            entry = held[id(labels)] = (labels, ndimage.find_objects(labels))
            stats['snapshots'] += 1
        boxes = entry[1]
        identity = mask.identity
        return labels, (boxes[identity - 1] if identity <= len(boxes) else None)

    def copy(mask):
        found = boxed(mask)
        if found is None:
            stats['copy_fallback'] += 1
            return mask.copy()
        labels, box = found
        stats['copied'] += 1
        # np.ndarray itself: the array pool's np.zeros may choose other storage.
        result = np.ndarray(labels.shape, dtype=bool)
        result.fill(False)
        if box is not None:
            result[box] = labels[box] == mask.identity
        return result

    def paint(target, mask, value):
        entry = rebuilt.get(id(mask))
        if entry is not None and entry[0]() is mask and isinstance(target, np.ndarray) and target.shape == mask.shape:
            y0, y1, x0, x1 = entry[1]
            stats['packed'] += 1
            # The box's Boolean selection writes the same pixels, in the same
            # row-major order, with the same casting as their coordinates did.
            target[y0:y1, x0:x1][mask[y0:y1, x0:x1]] = value
            return
        found = boxed(mask)
        if found is None or not isinstance(target, np.ndarray) or target.shape != found[0].shape:
            stats['fallback'] += 1
            target[mask] = value
            return
        labels, box = found
        identity = mask.identity
        stats['boxed'] += 1
        if box is None:
            # No pixel holds this identity: the original assignment selected none.
            target[np.zeros(target.shape, bool)] = value
            return
        target[box][labels[box] == identity] = value
    return paint, copy, materialize


@contextmanager
def install(module, label_mask):
    stats = dict(supported=False, boxed=0, packed=0, fallback=0, snapshots=0, copied=0, copy_fallback=0)
    original = module.track_direction
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE:
        yield stats
        return
    for old, new in STATEMENTS:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    assert source.count(COPY[0]) == COPY[2] and source.count(MATERIALIZE[0]) == 1
    source = source.replace(COPY[0], COPY[1]).replace(MATERIALIZE[0], MATERIALIZE[1])
    globals_ = original.__globals__
    sentinel = object()
    names = ('_paint_label_mask', '_copy_label_mask', '_materialize_boxed')
    previous = {name: globals_.get(name, sentinel) for name in names}
    globals_.update(zip(names, painter(label_mask, stats, module)))
    namespace = {}
    exec(compile(source, '<link-paint-boxes>', 'exec'), globals_, namespace)
    replacement = namespace[original.__name__]
    replacement.__wrapped__ = original
    module.track_direction = replacement
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.track_direction = original
        for name, value in previous.items():
            if value is sentinel:
                globals_.pop(name, None)
            else:
                globals_[name] = value
