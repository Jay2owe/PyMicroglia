"""Recording-start duplicate summary: component excess over changed frames, inside identity boxes.

excess_components sums an independent term per frame, so the difference between
the candidate and the input equals the difference of their sums over the frames
where they differ. Each term labels every identity in the whole frame; an
8-connected component never leaves its identity's bounding box, so labelling
inside that box gives the same component count.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import numpy as np
from scipy import ndimage

SOURCE = 'bf05ad73e239d54e50d48213e9ff237d938a5744fd6550fb51411e21a0b0fa31'
OLD = ("            0, excess_components(candidate) - excess_components(labels)),\n")
NEW = ("            0, _excess_frames(candidate, changed) - _excess_frames(labels, changed)),\n")


def frame_excess(frame, structure):
    """excess_components term of one 2-D frame."""
    frame = np.asarray(frame)
    if frame.ndim != 2 or not np.issubdtype(frame.dtype, np.integer) or not frame.size:
        return None
    values = np.unique(frame)
    if values[0] < 0:
        return None
    if values[-1] <= 1 << 24:
        found = ndimage.find_objects(frame)
        boxes = {int(v): found[int(v) - 1] for v in values if v}
    else:
        # Compact numbering keeps find_objects small for very large identities.
        _, inverse = np.unique(frame, return_inverse=True)
        found = ndimage.find_objects(inverse.reshape(frame.shape) + 1)
        boxes = {int(v): found[index] for index, v in enumerate(values) if v}
    total = 0
    for identity in set(map(int, values)) - {0}:
        total += max(0, ndimage.label(frame[boxes[identity]] == identity, structure)[1] - 1)
    return total


def helpers(module, stats):
    def excess_frames(stack, changed):
        frames = np.flatnonzero(np.any(changed, axis=(1, 2)))
        total = 0
        for t in frames:
            term = frame_excess(stack[t], module.STRUCTURE)
            if term is None:
                stats['fallback_frames'] += 1
                term = module.excess_components(stack[t:t + 1])
            total += term
        stats['frames'] += len(frames)
        stats['calls'] += 1
        return int(total)
    return dict(_excess_frames=excess_frames)


@contextmanager
def install(module):
    stats = dict(supported=False, calls=0, frames=0, fallback_frames=0)
    original = module.summarize
    source = inspect.getsource(original)
    if (hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE
            or source.count(OLD) != 1):
        yield stats
        return
    namespace = dict(module.__dict__)
    namespace.update(helpers(module, stats))
    flags = __future__.annotations.compiler_flag
    exec(compile(source.replace(OLD, NEW), '<start-duplicate-summary>', 'exec', flags, dont_inherit=True), namespace)
    module.summarize = namespace['summarize']
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.summarize = original
