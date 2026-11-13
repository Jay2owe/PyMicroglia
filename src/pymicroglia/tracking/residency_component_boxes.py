"""Merge residency: owner components labelled once per (frame, owner), in the owner's box.

_component_at labelled the whole frame's owner mask and tested a disk over the
whole frame for every resident and host point. recover_merge_residents never
writes its labels (it edits a copy), so within one call each (frame, owner)
mask is labelled once inside the owner's box by censored_area_boxes.OwnerParts.
Every component lies in that box, and labelling numbers components by their
first pixel in row-major order, which the box keeps, so the component numbers
are the whole-frame ones. The disk is tested only inside the box, with the
same arithmetic on the same integer coordinates; outside it no pixel is part
of a component. The chosen component is returned as a whole-frame mask, as
before. A plane is served from the memo only when it is exactly the memory of
one frame of the call's labels (same address, shape, strides and dtype);
anything else takes the original route.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading

import numpy as np
try:
    from .censored_area_boxes import OwnerParts
except ImportError:
    from censored_area_boxes import OwnerParts

SOURCE = 'a20e87d0ecf079a4c77e2aa279e1c91a9ecbf2db3f7a3ce53ae4151f1494b7f3'
# The current recovery's OwnerParts, per thread.
_local = threading.local()


def frame_index(labels, frame):
    """t when frame is exactly the memory of labels[t], else None."""
    if not (isinstance(frame, np.ndarray) and frame.ndim == 2 and frame.dtype == labels.dtype
            and frame.shape == labels.shape[1:] and frame.strides == labels.strides[1:]):
        return None
    step = labels.strides[0]
    if step <= 0 or not len(labels):
        return None
    offset = frame.__array_interface__['data'][0] - labels.__array_interface__['data'][0]
    if offset < 0 or offset % step:
        return None
    t = offset // step
    return t if t < len(labels) else None


def component_at(original, stats):
    def _component_at(frame, owner, x, y, radius):
        memo = getattr(_local, 'memo', None)
        t = None if memo is None else frame_index(memo.labels, frame)
        if t is None:
            stats['fallback'] += 1
            return original(frame, owner, x, y, radius)
        origin, parts, _ = memo.get(t, owner)
        if parts is None:
            return None
        y0, x0 = origin
        height, width = parts.shape
        yy = np.arange(y0, y0 + height)[:, None]
        xx = np.arange(x0, x0 + width)[None, :]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= max(2.0, radius) ** 2
        values = parts[disk & (parts > 0)]
        if not len(values):
            return None
        ids, counts = np.unique(values, return_counts=True)
        mask = np.zeros(frame.shape, bool)
        mask[y0:y0 + height, x0:x0 + width] = parts == int(ids[np.argmax(counts)])
        stats['served'] += 1
        return mask
    _component_at.__wrapped__ = original
    return _component_at


@contextmanager
def install(module):
    stats = dict(supported=False, calls=0, labelled=0, reused=0, boxed=0, served=0, fallback=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    original = module.recover_merge_residents
    found = module._component_at

    def recover_merge_residents(labels, unclaimed, raw, physical_points, params):
        if not (isinstance(labels, np.ndarray) and labels.ndim == 3 and np.issubdtype(labels.dtype, np.integer)):
            return original(labels, unclaimed, raw, physical_points, params)
        outer = getattr(_local, 'memo', None)
        _local.memo = OwnerParts(labels, module.STRUCTURE, stats)
        stats['calls'] += 1
        try:
            return original(labels, unclaimed, raw, physical_points, params)
        finally:
            _local.memo = outer
    recover_merge_residents.__wrapped__ = original
    module._component_at = component_at(found, stats)
    module.recover_merge_residents = recover_merge_residents
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.recover_merge_residents = original
        module._component_at = found
