"""Complete-flip nested-alias search with per-source support and plain frames.

For every predecessor and source pair the search filtered the visible point
table by the source and took its strong fraction; that value depends only on
the source, so it is formed by the same statements once per source. Its frame
comparisons read disk-backed frames through the storage adapter; a plain view
of the same memory gives the same Boolean masks. Boundary steps read each
frame's identity mask parts once per search (labelled in the mask's own box,
so numbered in the same raster order) instead of once per pair and run.
"""
import __future__
import hashlib
import inspect
import math
from contextlib import contextmanager

import numpy as np
from scipy import ndimage

SOURCE = '43d6a529a44aa835c5b7d52623c298914bcbd8a759ca627ede23e152935bb69a'
STRONG = (
    "            source_points = visible[\n"
    "                visible.candidate_owner.astype(int).eq(source)]\n"
    "            strong_fraction = float(source_points.strong.astype(bool).mean()) \\\n"
    "                if len(source_points) else 0.0\n")
REPLACEMENTS = (
    ("    proposals: list[dict] = []\n",
     "    proposals: list[dict] = []\n"
     "    _strong_memo = {}\n"
     "    _plain_labels = labels.view(np.ndarray) if isinstance(labels, np.ndarray) else labels\n"),
    (STRONG, "            strong_fraction = _strong_fraction(visible, source, _strong_memo)\n"),
    ("labels[run[0] - 1] == predecessor", "_plain_labels[run[0] - 1] == predecessor"),
    ("labels[run[0]] == source", "_plain_labels[run[0]] == source"),
    ("labels[run[-1]] == source", "_plain_labels[run[-1]] == source"),
    ("labels[run[-1] + 1] == predecessor", "_plain_labels[run[-1] + 1] == predecessor"),
    ("    _strong_memo = {}\n", "    _strong_memo = {}\n    _parts_memo = {}\n"),
    ("                boundary_steps.append(_mask_boundary_step(\n"
     "                    _plain_labels[run[0] - 1] == predecessor,\n"
     "                    _plain_labels[run[0]] == source))\n",
     "                boundary_steps.append(_boundary_step(\n"
     "                    _plain_labels, run[0] - 1, predecessor, run[0], source,\n"
     "                    _parts_memo, _mask_boundary_step))\n"),
    ("                boundary_steps.append(_mask_boundary_step(\n"
     "                    _plain_labels[run[-1]] == source,\n"
     "                    _plain_labels[run[-1] + 1] == predecessor))\n",
     "                boundary_steps.append(_boundary_step(\n"
     "                    _plain_labels, run[-1], source, run[-1] + 1, predecessor,\n"
     "                    _parts_memo, _mask_boundary_step))\n"),
)
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def _mask_parts(labels, frame, identity, memo):
    # One frame's identity mask, its box and its parts (centroid of the
    # raster-ordered points, area) in the frame's raster numbering; the same
    # mask is met for many predecessor and source pairs.
    key = (frame, identity)
    if key not in memo:
        if isinstance(identity, (int, np.integer)) and identity > 0 and labels.dtype.kind == "u":
            # Every identity's box in this frame from one pass: find_objects
            # gives the first and last occupied row and column of each label,
            # the same box as the scans of the identity's mask.
            objects = memo.get(('objects', frame))
            if objects is None:
                objects = memo[('objects', frame)] = ndimage.find_objects(labels[frame])
            box = objects[identity - 1] if identity <= len(objects) else None
            if box is None:
                memo[key] = None
                return None
            box = (slice(box[0].start, box[0].stop), slice(box[1].start, box[1].stop))
            crop = labels[frame][box] == identity
            rows = None
        else:
            mask = labels[frame] == identity
            rows = mask.any(axis=1)
        if rows is not None and not rows.any():
            memo[key] = None
        else:
            if rows is not None:
                y = np.flatnonzero(rows)
                x = np.flatnonzero(mask[y[0]:y[-1] + 1].any(axis=0))
                box = (slice(int(y[0]), int(y[-1]) + 1), slice(int(x[0]), int(x[-1]) + 1))
                crop = mask[box]
            numbered, count = ndimage.label(crop, STRUCTURE)
            parts = []
            for number, part in enumerate(ndimage.find_objects(numbered, max_label=count), 1):
                inside = numbered[part] == number
                ys, xs = np.nonzero(inside)
                points = np.column_stack((ys + (box[0].start + part[0].start),
                                          xs + (box[1].start + part[1].start)))
                parts.append((np.mean(points, axis=0), inside.sum()))
            memo[key] = (box, crop, parts)
    return memo[key]


def boundary_step(labels, left_frame, left_identity, right_frame, right_identity, memo, step):
    """_mask_boundary_step(labels[left_frame] == left_identity, labels[right_frame] == right_identity)."""
    if not (type(labels) is np.ndarray and labels.ndim == 3 and labels.dtype.kind in 'iu'):
        return step(labels[left_frame] == left_identity, labels[right_frame] == right_identity)
    left = _mask_parts(labels, left_frame, left_identity, memo)
    if left is None:
        return float("inf")
    right = _mask_parts(labels, right_frame, right_identity, memo)
    if right is None:
        return float("inf")
    (left_box, left_crop, left_parts), (right_box, right_crop, right_parts) = left, right
    shared = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(left_box, right_box))
    if all(part.start < part.stop for part in shared):
        def within(box, crop):
            return crop[tuple(slice(s.start - b.start, s.stop - b.start) for s, b in zip(shared, box))]
        if np.any(within(left_box, left_crop) & within(right_box, right_crop)):
            return 0.0
    best = float("inf")
    for left_yx, left_area in left_parts:
        for right_yx, right_area in right_parts:
            radius = math.sqrt(max(
                ((left_area / math.pi)
                 + (right_area / math.pi)) / 2.0,
                np.finfo(float).eps))
            best = min(best, float(np.linalg.norm(left_yx - right_yx))
                       / radius)
    return best


def strong_fraction(visible, source, memo):
    if source not in memo:
        source_points = visible[
            visible.candidate_owner.astype(int).eq(source)]
        memo[source] = float(source_points.strong.astype(bool).mean()) \
            if len(source_points) else 0.0
    return memo[source]


@contextmanager
def install(module):
    stats = dict(supported=False)
    original = module._nested_alias_proposals
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE:
        yield stats
        return
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            yield stats
            return
        source = source.replace(old, new)
    globals_ = vars(module)
    globals_['_strong_fraction'] = strong_fraction
    globals_['_boundary_step'] = boundary_step
    namespace = {}
    exec(compile(source, '<nested-alias-reuse>', 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), globals_, namespace)
    module._nested_alias_proposals = namespace['_nested_alias_proposals']
    stats['supported'] = True
    try:
        yield stats
    finally:
        module._nested_alias_proposals = original
        globals_.pop('_strong_fraction', None)
        globals_.pop('_boundary_step', None)
