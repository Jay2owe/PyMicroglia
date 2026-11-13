"""Complete-flip boundary steps measured inside the two masks' joint box.

The boundary step labelled both whole frames and, for every pair of parts,
compared each whole labelled frame with the part number again. Every part
lies inside the bounding box of the two masks, and labelling that box numbers
the parts in the same raster order. Each part is read from its own box, and
its coordinates are shifted back to frame positions before averaging, so the
same integers are averaged in the same order; areas and the pair loop are
unchanged. Emptiness and overlap are decided from row and column occupancy:
two masks can share a pixel only on rows both occupy.
"""
import hashlib
import inspect
import math
from contextlib import contextmanager

import numpy as np
from scipy import ndimage as scipy_ndimage

SOURCE = '4a8158c25fc290fa54a5706f479585cf97ae18f116482b6b37d4173df748399b'
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def _parts(labels, count, y0, x0):
    parts = []
    for number, box in enumerate(scipy_ndimage.find_objects(labels, max_label=count), 1):
        mask = labels[box] == number
        rows, columns = np.nonzero(mask)
        points = np.column_stack((rows + (y0 + box[0].start), columns + (x0 + box[1].start)))
        parts.append((np.mean(points, axis=0), mask.sum()))
    return parts


def boundary_step(module, stats):
    original = module._mask_boundary_step

    def step(left, right):
        plain_left, plain_right = np.asarray(left), np.asarray(right)
        if (plain_left.ndim != 2 or plain_left.shape != plain_right.shape
                or plain_left.dtype != np.bool_ or plain_right.dtype != np.bool_):
            stats['fallback'] += 1
            return original(left, right)
        left_rows = plain_left.any(axis=1)
        if not left_rows.any():
            return float("inf")
        right_rows = plain_right.any(axis=1)
        if not right_rows.any():
            return float("inf")
        shared = np.flatnonzero(left_rows & right_rows)
        if len(shared):
            band = slice(int(shared[0]), int(shared[-1]) + 1)
            if np.any(plain_left[band] & plain_right[band]):
                return 0.0
        rows = np.flatnonzero(left_rows | right_rows)
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        columns = np.flatnonzero(plain_left[y0:y1].any(axis=0) | plain_right[y0:y1].any(axis=0))
        x0, x1 = int(columns[0]), int(columns[-1]) + 1
        left_labels, left_count = scipy_ndimage.label(plain_left[y0:y1, x0:x1], STRUCTURE)
        right_labels, right_count = scipy_ndimage.label(plain_right[y0:y1, x0:x1], STRUCTURE)
        right_parts = _parts(right_labels, right_count, y0, x0)
        best = float("inf")
        for left_yx, left_area in _parts(left_labels, left_count, y0, x0):
            for right_yx, right_area in right_parts:
                radius = math.sqrt(max(
                    ((left_area / math.pi)
                     + (right_area / math.pi)) / 2.0,
                    np.finfo(float).eps))
                best = min(best, float(np.linalg.norm(left_yx - right_yx))
                           / radius)
        stats['cropped'] += 1
        return best
    return step


def source_hash(module):
    return hashlib.sha256(inspect.getsource(module._mask_boundary_step).encode()).hexdigest()


@contextmanager
def install(module):
    stats = dict(supported=False, cropped=0, fallback=0)
    if source_hash(module) != SOURCE:
        yield stats
        return
    original = module._mask_boundary_step
    module._mask_boundary_step = boundary_step(module, stats)
    stats['supported'] = True
    try:
        yield stats
    finally:
        module._mask_boundary_step = original
