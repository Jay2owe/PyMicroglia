"""Offer suppressed continuation peaks only outside existing measured cores.

The original detector's observations remain intact. A supplementary centre
inside either peak's estimated soma radius is another observation of the same
core, not independent continuation evidence. This is the same separation
geometry used by the established strong-other-soma check. All radii come from
the current recording; there is no identity, location or feedback selector.
"""
import numpy as np

import continuation_only_soma_peaks as base


def select_distinct_core_continuations(rows):
    original = [row for row in rows if not row['continuation_only']]
    extra = [row for row in rows if row['continuation_only']]
    if not original or not extra:
        return rows
    original_xy = np.asarray([(q['x'], q['y']) for q in original], dtype=np.float64)
    extra_xy = np.asarray([(q['x'], q['y']) for q in extra], dtype=np.float64)
    original_r = np.asarray([q['radius_px'] for q in original], dtype=np.float64)
    extra_r = np.asarray([q['radius_px'] for q in extra], dtype=np.float64)
    distance_squared = np.sum((extra_xy[:, None, :]-original_xy[None, :, :])**2, axis=2)
    core_squared = np.maximum(extra_r[:, None], original_r[None, :])**2
    distinct = ~np.any(distance_squared <= core_squared, axis=1)
    # Preserve row order and every original field, including original peak IDs.
    decisions = iter(distinct)
    return [row for row in rows if not row['continuation_only'] or next(decisions)]


def frame_hypotheses(frame, frame_index, params):
    rows, thresholds = base.frame_hypotheses(frame, frame_index, params)
    return select_distinct_core_continuations(rows), thresholds


link_hypotheses = base.link_hypotheses
