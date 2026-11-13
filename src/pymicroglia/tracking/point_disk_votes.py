"""Exact batched label votes inside clipped disks around physical-track points.

Several frozen samplers visit points one at a time with

    r = float(np.clip(scale * radius, lo, hi))
    window = [floor(c - r), ceil(c + r + 1)) clipped to the image
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2

then vote with np.unique over the positive labels. Every disk pixel lies in
floor(c) +/- (ceil(max(lo, hi)) + 1), and pixels in that range but outside the
original window are more than r + 1 away, so they fail the same test. Votes
for all points of one frame are therefore gathered together here.
"""
import math
import numpy as np


def votes(stack, frames, x, y, radius, *, scale=0.75, lo=2.0, hi=6.0, batch_rows=512):
    """Return owner, positive, inside and winning counts per point, or None.

    owner is the most frequent positive label (smallest on ties), 0 if none.
    None means the inputs fall outside the audited case; callers then use the
    frozen per-point sampler.
    """
    if stack.ndim != 3 or stack.dtype.kind not in 'iu' or not stack.size:
        return None
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    radius = np.asarray(radius, dtype=float)
    frames = np.asarray(frames, dtype=np.int64)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(radius).all()):
        return None
    if np.any((frames < 0) | (frames >= len(stack))):
        return None
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    count = len(frames)
    owner = np.zeros(count, dtype=np.int64)
    positive = np.zeros(count, dtype=np.int64)
    inside = np.zeros(count, dtype=np.int64)
    winning = np.zeros(count, dtype=np.int64)
    if not count:
        return owner, positive, inside, winning
    height, width = stack.shape[1:]
    reach = int(math.ceil(max(lo, hi))) + 1
    offsets = np.arange(-reach, reach + 1, dtype=np.int64)
    largest = int(stack.max())
    base = largest + 1
    if largest < 0 or base >= np.iinfo(np.int64).max // max(batch_rows, 1):
        return None
    order = np.argsort(frames, kind='stable')
    cuts = np.flatnonzero(np.diff(frames[order])) + 1
    for selected in np.split(order, cuts):
        plane = np.asarray(stack[int(frames[selected[0]])])
        for start in range(0, len(selected), batch_rows):
            index = selected[start:start + batch_rows]
            px, py = x[index], y[index]
            # Multiplication and clipping are exact elementwise; r ** 2 uses
            # Python's float power as the original does.
            clipped = np.clip(scale * radius[index], lo, hi)
            squared = np.array([float(value) ** 2 for value in clipped])
            xx = np.floor(px).astype(np.int64)[:, None] + offsets
            yy = np.floor(py).astype(np.int64)[:, None] + offsets
            dx = (xx - px[:, None]) ** 2
            dy = (yy - py[:, None]) ** 2
            disk = dx[:, None, :] + dy[:, :, None] <= squared[:, None, None]
            disk &= ((xx >= 0) & (xx < width))[:, None, :]
            disk &= ((yy >= 0) & (yy < height))[:, :, None]
            values = plane[np.clip(yy, 0, height - 1)[:, :, None],
                           np.clip(xx, 0, width - 1)[:, None, :]]
            hit = disk & (values > 0)
            inside[index] = np.count_nonzero(disk, axis=(1, 2))
            positive[index] = np.count_nonzero(hit, axis=(1, 2))
            rows, rr, cc = np.nonzero(hit)
            if not len(rows):
                continue
            encoded = rows.astype(np.int64) * base + values[rows, rr, cc].astype(np.int64)
            keys, counts = np.unique(encoded, return_counts=True)
            body = keys // base
            identity = keys % base
            best = np.zeros(len(index), dtype=np.int64)
            np.maximum.at(best, body, counts)
            winners = np.flatnonzero(counts == best[body])
            # Sorted by body then identity: first winner is the smallest label.
            _, first = np.unique(body[winners], return_index=True)
            winners = winners[first]
            owner[index[body[winners]]] = identity[winners]
            winning[index[body[winners]]] = counts[winners]
    return owner, positive, inside, winning
