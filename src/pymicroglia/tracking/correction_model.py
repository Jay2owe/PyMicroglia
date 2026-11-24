"""Model grow for the correction reviewer: a cell takes the LV200 outline model's piece of it (Jamie 2026-11-19:
"make something thats better at seeding/growing projections from a cell").

Chosen in round R04_projection_grow (development/lv200_soma_branch_tuning): on the 84 hand-outlined LV200 frames,
starting from the tracking bodies, the outline model's piece filled 62% of each cell's missed pixels with 77% of what
it added inside Jamie's outlines (held-out rows; Dice 0.69), against 22% / 89% (Dice 0.35) for Grow at level 0.2 and
Dice 0.40 for a click-on-the-tip brightest-route tool.

The model is learned_mask.lv200 (three-frame network, its own background removal and 0.70 cut). Its pieces join cells
by the package's overlap rule (tracking.mask_proposals.overlap_frame): a piece overlapping cells is divided between
them by nearest cell pixel; a piece overlapping no cell is ignored. Only unassigned pixels are taken, so no other
cell's outline changes. No size or distance setting.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def predict_labels(raw_stack, threads=8) -> np.ndarray:
    """The outline model's pieces for a whole recording (time, height, width) as a label stack."""
    from ..learned_mask import lv200
    from .mask_proposals import normalise
    raw_stack = np.asarray(raw_stack, np.float32)
    labels, _ = normalise(lv200.predict(raw_stack, threads=threads), raw_stack.shape)
    labels = np.asarray(labels)
    return labels.astype(np.uint16 if labels.max() < 65535 else np.uint32)


def grow_from_model(lab, model, n) -> np.ndarray:
    """Free pixels cell n gets from the model's pieces in this frame (empty when n is absent)."""
    from .mask_proposals import overlap_frame
    cell = lab == n
    if not cell.any():
        return np.zeros(lab.shape, bool)
    pieces = np.unique(model[cell])
    pieces = pieces[pieces > 0]
    if not len(pieces):
        return np.zeros(lab.shape, bool)
    mine = np.isin(model, pieces)
    ys, xs = np.nonzero(mine | cell)
    box = slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)
    proposals, _ = ndi.label(mine[box], np.ones((3, 3), bool))   # only the pieces cell n overlaps
    out, _ = overlap_frame(lab[box], proposals)
    got = np.zeros(lab.shape, bool)
    got[box] = (out == n) & (lab[box] == 0)
    return got


def model_grow_track(frame, model_frame, n, f0, last) -> dict:
    """{frame: added pixels} when cell n takes its model piece, frames f0..last."""
    out = {}
    for f in range(f0, last + 1):
        add = grow_from_model(frame(f), model_frame(f), n)
        if add.any():
            out[f] = add
    return out
