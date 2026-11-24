"""Seed and grow outlines for the correction reviewer (not in Motion; Jamie 2026-11-17).

Signal: the frame with its slowly varying background removed (the approved CD68 display filter,
figure_tables.tracking_tower.DISPLAY_PRESETS['cd68']), lightly denoised, then pixels brighter than the frame's own
Otsu threshold times a level (the reviewer's 'grow level' box, DEFAULT_LEVEL). Without the background step a
seed in position 09 flooded diffuse glow up to ~24,000 px; with it, free regions stay cell-sized (checked
2026-11-17 on frames 20/120/220: largest free region <= ~320 px at level 0.4, two thirds of outlined pixels
counted as signal). Growth is by connectivity only: a seed takes the connected signal it sits on, a cell takes the
connected signal touching it, and a seed is followed into the next frame through the signal that overlaps it.
Only unassigned pixels are taken, so another cell's outline never changes. Settings: the level, DENOISE_SIGMA
(1-pixel smoothing) and the display filter's two blur widths, which belong to the CD68 display recipe.

frame(f) returns the full current labels of frame f; raw(f) the matching raw image (both 0-based label frames).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)
DENOISE_SIGMA = 1.0   # light Gaussian denoising before the threshold (display-style smoothing, not a size rule)
DEFAULT_LEVEL = 0.2   # Jamie's most used level on recording 3; no per-cell rule did better (R03, 2026-11-19)


def signal(raw, level=DEFAULT_LEVEL):
    """Pixels above level x the frame's Otsu threshold, after the CD68 display background removal."""
    from skimage.filters import threshold_otsu
    from ..figure_tables.tracking_tower import DISPLAY_PRESETS
    recipe = DISPLAY_PRESETS["cd68"]
    smooth = ndi.gaussian_filter(np.nan_to_num(np.asarray(raw, np.float32)), recipe["gaussian_sigma_px"])
    lifted = np.maximum(smooth - ndi.gaussian_filter(smooth, recipe["background_sigma_px"]), 0)
    lifted = ndi.gaussian_filter(lifted, DENOISE_SIGMA)
    return lifted > threshold_otsu(lifted) * float(level)


def _free_parts(lab, fg):
    parts, _ = ndi.label(fg & (lab == 0), structure=EIGHT)
    return parts


def seed_region(lab, fg, y, x):
    """The unassigned signal connected to (y, x); empty when the click is not on free signal."""
    parts = _free_parts(lab, fg)
    if not parts[y, x]:
        return np.zeros(lab.shape, bool)
    return parts == parts[y, x]


def follow_region(previous, lab, fg):
    """The unassigned signal in this frame that overlaps last frame's region."""
    parts = _free_parts(lab, fg)
    hit = np.unique(parts[previous & (parts > 0)])
    return np.isin(parts, hit[hit > 0])


def grow_cell(lab, fg, n):
    """Unassigned signal connected to cell n (pixels to add; empty if n is absent)."""
    cell = lab == n
    if not cell.any():
        return np.zeros(lab.shape, bool)
    parts = _free_parts(lab, fg)
    hit = np.unique(parts[ndi.binary_dilation(cell, EIGHT) & (parts > 0)])
    return np.isin(parts, hit[hit > 0])


def keeps_going(ref, a, b):
    """Whether a followed region of b pixels (a in the frame before) is still the same cell, judged against the
    cell's own size ref: not below ref / AREA_RATIO and no AREA_RATIO-fold change in one frame. Relative only
    (Jamie 2026-11-18: a fixed 60-px floor stopped every seed of a dim 30-56 px cell after one frame)."""
    from .correction_rules import AREA_RATIO
    return b > 0 and b * AREA_RATIO > ref and max(a, b) < AREA_RATIO * min(a, b)


def seed_track(frame, raw, f0, last, y, x, level=DEFAULT_LEVEL, first=None, sig=None):
    """{frame: mask} of a new cell seeded at (y, x) in frame f0, followed by overlap forward to frame last and
    back to frame first (default f0), each way until no free signal overlaps (e.g. the cell is outlined again)
    or keeps_going says it is no longer the same cell. Checked on position 09 (2026-11-17): without a size check
    a seed drifted over 170 frames through 2-px fragments; a fixed 60-px floor (until 2026-11-18) cut dim cells
    to one frame. sig(f), when given, returns the signal of frame f (a cached signal(raw(f), level)).
    """
    sig = sig or (lambda f: signal(raw(f), level))
    seed = seed_region(frame(f0), sig(f0), y, x)
    if not seed.any():
        raise ValueError("No signal at the click above the grow level: click on the cell, or lower the level")
    out = {f0: seed}
    for frames in (range(f0 + 1, last + 1), range(f0 - 1, (f0 if first is None else first) - 1, -1)):
        region = seed
        for f in frames:
            nxt = follow_region(region, frame(f), sig(f))
            if not keeps_going(int(seed.sum()), int(region.sum()), int(nxt.sum())):
                break
            out[f] = region = nxt
    return out


def _share(lab, fg, raw, prev, cells=()):
    """Free signal divided between several seeds in one frame. prev: one mask per seed (its click pixel in the
    seed frame, else its region a frame earlier; None for a seed that has stopped). Each seed takes the free
    signal it touches; where two reach the same connected patch, it is split by a watershed from their pixels
    on the lightly smoothed image, so the boundary runs along the dimmest path between them (as the split
    engine places a boundary). cells: numbers of existing cells growing at the same time (None for one not
    growing in this frame); each grows from its own outline into the free signal touching it, sharing it with
    the seeds and each other the same way. Returns (one mask per seed, one added-pixels mask per cell); empty
    for a stopped seed or a cell not growing."""
    from skimage.segmentation import watershed
    free = fg & (lab == 0)
    parts, _ = ndi.label(free, structure=EIGHT)
    markers = np.zeros(lab.shape, np.int32)
    for i, m in enumerate(prev):
        if m is not None:
            markers[m & free & (markers == 0)] = i + 1
    hit = set(np.unique(parts[markers > 0]).tolist())
    own = np.zeros(lab.shape, bool)
    for j, n in enumerate(cells):
        if n is not None:
            cell = n[1] if isinstance(n, tuple) else lab == n
            markers[cell] = len(prev) + j + 1
            own |= cell
            hit |= set(np.unique(parts[ndi.binary_dilation(cell, EIGHT)]).tolist())
    hit.discard(0)
    area = np.isin(parts, sorted(hit))
    out = [np.zeros(lab.shape, bool) for _ in prev]
    grown = [np.zeros(lab.shape, bool) for _ in cells]
    if not area.any():
        return out, grown
    ys, xs = np.nonzero(area | own)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    image = ndi.gaussian_filter(np.nan_to_num(np.asarray(raw[y0:y1, x0:x1], np.float32)), DENOISE_SIGMA)
    ws = watershed(-image, markers[y0:y1, x0:x1], mask=(area | own)[y0:y1, x0:x1], connectivity=2)
    for i in range(len(prev)):
        out[i][y0:y1, x0:x1] = ws == i + 1
    for j in range(len(cells)):
        grown[j][y0:y1, x0:x1] = (ws == len(prev) + j + 1) & area[y0:y1, x0:x1]
    return out, grown


def _uphill(image, free, y, x):
    """From (y, x), step to the brightest of the 8 neighbours (free signal only) until none is brighter: the top
    of the bright body the click is on. No distance or size setting."""
    H, W = image.shape
    while True:
        best = (image[y, x], y, x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v, u = y + dy, x + dx
                if 0 <= v < H and 0 <= u < W and free[v, u] and image[v, u] > best[0]:
                    best = (image[v, u], v, u)
        if best[1:] == (y, x):
            return y, x
        y, x = best[1], best[2]


def seed_many(frame, raw, f0, last, points, level=DEFAULT_LEVEL, first=None, sig=None, grows=None):
    """seed_track for several seeds at once (Jamie 2026-11-18: "if i do them both at the same time the right
    cell gets the right territory"): every frame the seeds share the free signal (_share) instead of the first
    one taking it all; each seed stops on its own (keeps_going against its own first size). Returns one
    {frame: mask} per point (empty for a point that is not on free signal). grows: [(cell number, last frame)]
    of existing cells growing at the same time from f0 (Jamie 2026-11-18: "by seed i meant seed and growing");
    then returns (seed masks, [{frame: added pixels} per cell])."""
    sig = sig or (lambda f: signal(raw(f), level))
    lab, fg = frame(f0), sig(f0)
    image = ndi.gaussian_filter(np.nan_to_num(np.asarray(raw(f0), np.float32)), DENOISE_SIGMA)
    free = fg & (lab == 0)
    dots, taken = [], set()
    for y, x in points:
        y, x = int(y), int(x)
        if not free[y, x]:
            dots.append(None)
            continue
        top = _uphill(image, free, y, x)   # a click on a cell's dim edge starts from its bright body
        if top in taken:
            top = (y, x)
        taken.add(top)
        d = np.zeros(lab.shape, bool)
        d[top] = True
        dots.append(d)
    g = list(grows or ())
    kept = [None] * len(g)   # grows from one piece: the piece followed so far

    def active(f, here):
        out = []
        for j, item in enumerate(g):
            n, end, click = item[0], item[1], (item[2] if len(item) > 2 else None)
            if f > end:
                out.append(None)
            elif click is None:
                out.append(n)
            else:
                keep, _ = _keep_piece(here, n, kept[j], tuple(click) if f == f0 else None)
                if keep is not None:
                    kept[j] = keep
                out.append(None if keep is None else (n, keep))
        return out
    start, grown = _share(lab, fg, raw(f0), dots, active(f0, lab))
    out = [({f0: m} if m.any() else {}) for m in start]
    gout = [({f0: m} if m.any() else {}) for m in grown]
    refs = [int(m.sum()) for m in start]
    end = max([last] + [item[1] for item in g])
    for frames, with_cells in ((range(f0 + 1, end + 1), True), (range(f0 - 1, (f0 if first is None else first) - 1, -1), False)):
        prev = [m if m.any() else None for m in start]
        for f in frames:
            here = frame(f)
            cells = active(f, here) if with_cells else []
            if all(p is None for p in prev) and not any(c is not None for c in cells):
                break
            nxt, more = _share(here, sig(f), raw(f), [p if f <= last or not with_cells else None for p in prev]
                               if with_cells else prev, cells)
            for i, (p, m) in enumerate(zip(prev, nxt)):
                if p is None or (with_cells and f > last):
                    continue
                if keeps_going(refs[i], int(p.sum()), int(m.sum())):
                    out[i][f] = prev[i] = m
                else:
                    prev[i] = None
            for j, m in enumerate(more):
                if m.any():
                    gout[j][f] = m
    return (out, gout) if grows is not None else out


def continued_number(frame, masks, T):
    """The number a seeded or drawn cell continues: a cell overlapping it in the frame just before (or after) its
    frames that is absent from its first (or last) frame - a track that vanished where this cell appears.
    None when there is no single such number."""
    for edge, beyond in ((min(masks), min(masks) - 1), (max(masks), max(masks) + 1)):
        if not 0 <= beyond < T:
            continue
        there, here = frame(beyond), frame(edge)
        hit = set(np.unique(there[masks[edge]]).tolist()) - {0} - set(np.unique(here).tolist())
        if len(hit) == 1:
            return int(hit.pop())
    return None


def grow_track(frame, raw, n, f0, last, level=DEFAULT_LEVEL, sig=None, piece=None):
    """{frame: added pixels} when cell n takes the free signal touching it, frames f0..last (sig as seed_track).
    piece = (frame, y, x): grow only from the piece of n clicked there, followed through the frames as One island
    follows it (Jamie 2026-11-19: a cell broken into several pieces grew from all of them); the other pieces
    stay as they are and do not grow."""
    sig = sig or (lambda f: signal(raw(f), level))
    if piece is not None:
        fc, y, x = int(piece[0]), int(piece[1]), int(piece[2])
        fc = min(max(fc, f0), last)
        out = {}
        for frames in (range(fc, last + 1), range(fc - 1, f0 - 1, -1)):
            prev = None
            for f in frames:
                lab = frame(f)
                keep, _ = _keep_piece(lab, n, prev, (y, x) if f == fc else None)
                if keep is None:
                    continue
                added = grow_cell(_only_piece(lab, n, keep), sig(f), n)
                if added.any():
                    out[f] = added
                prev = keep
        return out
    out = {}
    for f in range(f0, last + 1):
        added = grow_cell(frame(f), sig(f), n)
        if added.any():
            out[f] = added
    return out


def _islands(lab, n):
    """Each 8-connected piece of cell n, as (pieces label image, count)."""
    return ndi.label(lab == n, structure=EIGHT)


def _keep_piece(lab, n, prev, click=None):
    """The piece of cell n to keep in this frame: the piece at click, else (with no earlier piece) the largest,
    else the piece overlapping the earlier kept piece most, else the piece nearest to it. (None, pieces) when n
    is absent. Connectivity only."""
    parts, k = _islands(lab, n)
    if k == 0:
        return None, parts
    if click is not None and parts[click]:
        return parts == parts[click], parts
    if prev is None:
        sizes = np.bincount(parts.ravel())[1:]
        return parts == int(np.argmax(sizes)) + 1, parts
    hit = np.bincount(parts[prev], minlength=k + 1)[1:]
    if hit.any():
        return parts == int(np.argmax(hit)) + 1, parts
    d = ndi.distance_transform_edt(~prev)
    near = ndi.minimum(d, parts, range(1, k + 1))
    return parts == int(np.argmin(near)) + 1, parts


def _only_piece(lab, n, keep):
    """Labels where cell n's other pieces carry a stand-in number, so they block but do not grow."""
    out = lab.astype(np.int64, copy=True)
    out[(lab == n) & ~keep] = int(lab.max()) + 1
    return out


def one_island_track(frame, n, f0, last, y, x, first=None) -> dict:
    """{frame: pixels to free} so cell n is one island in every frame first..last (Jamie 2026-11-19: a grow
    good for one frame spreads across the video into several pieces that each grow). In frame f0 the piece at
    (y, x) is kept (the largest piece if the click is not on the cell); in each following (and, with first,
    preceding) frame the piece overlapping the last kept piece most is kept, else the piece nearest to it; every
    other piece of n is freed. Connectivity only: no size or distance setting. A frame without n is skipped and
    the last kept piece carries on to the next frame."""
    def keep_in(lab, prev, click=None):
        return _keep_piece(lab, n, prev, click)

    out = {}
    lab0 = frame(f0)
    kept, parts = keep_in(lab0, None, (int(y), int(x)))
    if kept is None:
        return out
    if (parts > 0).sum() > kept.sum():
        out[f0] = (parts > 0) & ~kept
    for frames in (range(f0 + 1, last + 1), range(f0 - 1, (f0 if first is None else first) - 1, -1)):
        prev = kept
        for f in frames:
            got, parts = keep_in(frame(f), prev)
            if got is None:
                continue
            if (parts > 0).sum() > got.sum():
                out[f] = (parts > 0) & ~got
            prev = got
    return out


def polygon_mask(points, shape):
    """Pixels inside a drawn closed outline of (y, x) points (full-frame coordinates)."""
    from skimage.draw import polygon
    pts = np.asarray(points, float)
    rr, cc = polygon(pts[:, 0], pts[:, 1], shape)
    mask = np.zeros(shape, bool)
    mask[rr, cc] = True
    return mask
