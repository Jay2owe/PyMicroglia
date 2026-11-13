"""Measured soma footprints: every point's disk of a frame tested in one pass.

certify_raw_soma_footprints and constrain_destination_soma built a frame's
footprint point by point: (yy-y)**2 + (xx-x)**2 <= radius**2 over the point's
clipped box (np.ogrid's int64 rows and columns), OR-ed into the frame, and
(destination) the rounded centre set True when it lies in the frame. Here the
same per-point scalars, bounds and skips are taken row by row (math floor,
ceil and isfinite give the same integers and verdicts for finite floats), the
frame gets its array when its first kept point arrives (so the same keys in
the same order), and each frame's box pixels are then listed together: each
pixel's int64 row and column minus that point's float y and x, squared and
added in the same order, compared with that point's Python radius**2, give
the same verdicts. Setting True is idempotent and order-free, so every frame
holds the same Boolean values. Pixels are tested in chunks of at most 2**20.
"""
import math

import numpy as np

CHUNK = 1 << 20


def _fill(coverage, boxes):
    y0, y1, x0, x1, ys, xs, r2 = (np.array(column) for column in zip(*boxes))
    y0, x0 = y0.astype(np.int64), x0.astype(np.int64)
    width = (x1 - x0).astype(np.int64)
    counts = (y1 - y0).astype(np.int64) * width
    ys, xs, r2 = ys.astype(np.float64), xs.astype(np.float64), r2.astype(np.float64)
    start = 0
    while start < len(counts):
        stop = start + 1
        total = int(counts[start])
        while stop < len(counts) and total + int(counts[stop]) <= CHUNK:
            total += int(counts[stop])
            stop += 1
        n = counts[start:stop]
        point = np.repeat(np.arange(start, stop), n)
        offset = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
        rows = y0[point] + offset // width[point]
        columns = x0[point] + offset % width[point]
        inside = (rows - ys[point])**2 + (columns - xs[point])**2 <= r2[point]
        coverage[rows[inside], columns[inside]] = True
        start = stop


def footprints(selected, raw, centre):
    """The footprint loop's {frame: Boolean frame} for these selected points."""
    result = {}
    boxes = {}
    centres = {}
    height, width = raw.shape[1], raw.shape[2]
    for q in selected.itertuples(index=False):
        t = int(q.frame); x = float(q.x); y = float(q.y)
        radius = float(getattr(q, 'radius_px', 0.))
        if (not 0 <= t < len(raw) or not (math.isfinite(x) and math.isfinite(y) and math.isfinite(radius))
                or radius < 0):
            continue
        y0 = max(0, math.floor(y-radius)); y1 = min(height, math.ceil(y+radius)+1)
        x0 = max(0, math.floor(x-radius)); x1 = min(width, math.ceil(x+radius)+1)
        if y0 >= y1 or x0 >= x1:
            continue
        if t not in result:
            result[t] = np.zeros(raw.shape[1:], bool)
        boxes.setdefault(t, []).append((y0, y1, x0, x1, y, x, radius**2))
        if centre:
            iy, ix = math.floor(y+.5), math.floor(x+.5)
            if 0 <= iy < height and 0 <= ix < width:
                centres.setdefault(t, []).append((iy, ix))
    for t, items in boxes.items():
        _fill(result[t], items)
    for t, items in centres.items():
        rows, columns = zip(*items)
        result[t][list(rows), list(columns)] = True
    return result
