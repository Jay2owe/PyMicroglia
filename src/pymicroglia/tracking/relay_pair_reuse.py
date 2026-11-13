"""Skip relay frame-set membership once the solo test has failed; reuse transition points.

_relay_orientation computed np.isin for both orientations of every identity
pair before testing the cheaper before/after counts. The three conditions are
joined by `and` and have no side effects, so the membership count is now taken
only when the counts pass. Consecutive relay transitions share a path mask
(the later mask of one is the earlier mask of the next), so its occupied
coordinates are kept from the previous call and passed to the same evidence
statements as relay_transition_geometry. Coordinates are listed inside the
mask's occupied rows and columns: the same row-major arrays.
"""
import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SOURCE = '34603089a862624829d01040ba36c00af126b76b787d9fa3abc0217af1b90658'


def orientation(first, second, identity_frames, minimum_solo_frames):
    a = identity_frames[first]
    b = identity_frames[second]
    if not len(a) or not len(b):
        return None
    for persistent, relay, outer, inner in ((first, second, a, b),
                                             (second, first, b, a)):
        before = int(np.count_nonzero(outer < int(inner.min())))
        if not before >= minimum_solo_frames:
            continue
        after = int(np.count_nonzero(outer > int(inner.max())))
        if not after >= minimum_solo_frames:
            continue
        relay_solo = int(np.count_nonzero(~np.isin(inner, outer)))
        if relay_solo >= minimum_solo_frames:
            return persistent, relay
    return None


def bounds(points):
    if not len(points):
        return 0, 0, 0, 0
    low = points.min(axis=0); high = points.max(axis=0) + 1
    return int(low[0]), int(high[0]), int(low[1]), int(high[1])


def evidence(old, new, lag_frame, module, before, after):
    """relay_transition_geometry.evidence with the coordinates supplied."""
    support = module.motion_support(old, new, lag_frame, bounds(before), bounds(after))
    blue = before[lag_frame[before[:, 0], before[:, 1]] <= -1.5]
    red = after[lag_frame[after[:, 0], after[:, 1]] >= 1.5]
    displacement = np.mean(after, axis=0) - np.mean(before, axis=0)
    cosine = float('nan')
    if len(blue) >= 8 and len(red) >= 8:
        motion_vector = red.mean(axis=0) - blue.mean(axis=0)
        denominator = float(np.linalg.norm(motion_vector) * np.linalg.norm(displacement))
        if denominator > 1e-12:
            cosine = float(np.dot(motion_vector, displacement) / denominator)
    return {**support, 'direction_cosine': cosine}


def _coordinates(mask):
    """np.column_stack(np.nonzero(mask)), listed inside the mask's occupied rows and columns."""
    mask = np.asarray(mask)
    if type(mask) is not np.ndarray or mask.ndim != 2 or mask.dtype != np.bool_:
        return np.column_stack(np.nonzero(mask))
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return np.column_stack(np.nonzero(mask))
    columns = np.flatnonzero(mask[rows[0]:rows[-1] + 1].any(axis=0))
    y0, x0 = int(rows[0]), int(columns[0])
    yy, xx = np.nonzero(mask[y0:int(rows[-1]) + 1, x0:int(columns[-1]) + 1])
    return np.column_stack((yy + y0, xx + x0))


@contextmanager
def install(module, transition_geometry):
    stats = dict(supported=False, reused_points=0, measured_points=0)
    current = module._transition_evidence
    if (hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE
            or getattr(current, '__module__', None) != transition_geometry.__name__):
        yield stats
        return
    original_orientation = module._relay_orientation
    last = [None, None]

    def points(mask):
        if last[0] is mask:
            stats['reused_points'] += 1
            return last[1]
        stats['measured_points'] += 1
        return _coordinates(mask)

    def transition(old, new, lag_frame):
        if (old.ndim != 2 or new.shape != old.shape or lag_frame.shape != old.shape
                or old.dtype != np.bool_ or new.dtype != np.bool_):
            last[0] = last[1] = None
            return current(old, new, lag_frame)
        before = points(old)
        after = _coordinates(new)
        stats['measured_points'] += 1
        last[0], last[1] = new, after
        return evidence(np.asarray(old), np.asarray(new), np.asarray(lag_frame), module, before, after)

    module._relay_orientation = orientation
    module._transition_evidence = transition
    stats['supported'] = True
    try:
        yield stats
    finally:
        module._relay_orientation = original_orientation
        module._transition_evidence = current
        last[0] = last[1] = None
