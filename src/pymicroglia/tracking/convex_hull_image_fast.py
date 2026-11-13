"""skimage convex_hull_image without a qhull call per object.

On Windows each scipy ConvexHull opens a temporary message file (~1.8 ms),
dwarfing the hull itself for cell-sized masks; regionprops solidity calls it
for every component. In 2-D with pixel-edge offsets the points are multiples of
0.5, so a monotone-chain hull is exact. The polygon then goes to the same
grid_points_in_poly classification. Its vertex and edge tests are exact cross
products, so the start vertex does not change the mask. Every other case uses
the original.
"""
from contextlib import contextmanager
import hashlib
import inspect

import numpy as np

SOURCE = '1c7a7d72d1016d7c6fe5ef0bfae78a844f39bf86e04653ab063d6afc4b16dac7'  # skimage 0.26.0 convex_hull_image


def _hull(points):
    """Counter-clockwise hull vertices of sorted unique points, collinear points dropped."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    xs = points[order, 0].tolist()
    ys = points[order, 1].tolist()
    if len(xs) < 3:
        return None

    def chain(indices):
        out = []
        for i in indices:
            x, y = xs[i], ys[i]
            while len(out) >= 2:
                ax, ay = out[-2]
                bx, by = out[-1]
                if (bx - ax) * (y - ay) - (by - ay) * (x - ax) <= 0:
                    out.pop()
                else:
                    break
            out.append((x, y))
        return out
    lower = chain(range(len(xs)))
    upper = chain(range(len(xs) - 1, -1, -1))
    vertices = lower[:-1] + upper[:-1]
    if len(vertices) < 3:
        return None
    return np.array(vertices, dtype=float)


def _extremes(doubled):
    """Each first-coordinate value's smallest and largest second coordinate.

    Points between them on the same line are inside the hull or on one of its
    edges, and _hull drops both kinds, so it returns the same vertices from
    these points alone. Its own sort fixes the order; duplicates are removed.
    """
    order = np.lexsort((doubled[:, 1], doubled[:, 0]))
    first, second = doubled[order, 0], doubled[order, 1]
    starts = np.flatnonzero(np.r_[True, first[1:] != first[:-1]])
    ends = np.r_[starts[1:] - 1, first.size - 1]
    lone = second[starts] == second[ends]
    picked = np.sort(np.concatenate((starts, ends[~lone])))
    return np.column_stack((first[picked], second[picked])) / 2.0


@contextmanager
def install(expected_sha=SOURCE):
    import skimage
    from skimage.morphology import convex_hull as module
    stats = dict(supported=False, calls=0, fallback_calls=0)
    original = module.convex_hull_image
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha:
        yield stats
        return
    possible_hull = module.possible_hull
    unique_rows = module.unique_rows
    grid_points_in_poly = module.grid_points_in_poly
    offsets = module._offsets_diamond(2)
    # The half-pixel offsets doubled are whole numbers, so candidate points are
    # handled as exact doubled integers until _extremes halves them back.
    doubled_offsets = (offsets * 2).astype(np.int64)
    exact_offsets = np.array_equal(doubled_offsets / 2.0, offsets)

    def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10, include_borders=True):
        stats['calls'] += 1
        if (not isinstance(image, np.ndarray) or image.ndim != 2 or not offset_coordinates
                or np.count_nonzero(image) == 0):
            stats['fallback_calls'] += 1
            return original(image, offset_coordinates, tolerance, include_borders)
        coords = possible_hull(np.ascontiguousarray(image, dtype=np.uint8))
        if exact_offsets:
            doubled = (coords[:, np.newaxis, :].astype(np.int64) * 2 + doubled_offsets).reshape(-1, 2)
            coords = _extremes(doubled)
        else:
            coords = unique_rows((coords[:, np.newaxis, :] + offsets).reshape(-1, 2))
        vertices = _hull(coords)
        if vertices is None:
            stats['fallback_calls'] += 1
            return original(image, offset_coordinates, tolerance, include_borders)
        labels = grid_points_in_poly(image.shape, vertices, binarize=False)
        return labels >= 1 if include_borders else labels == 1

    convex_hull_image.__wrapped__ = original
    module.convex_hull_image = convex_hull_image
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.convex_hull_image = original


@contextmanager
def aliases():
    """Point names bound to the original at import time to the installed fast hull.

    Correction steps import convex_hull_image into their own modules; install()
    replaces only skimage's module attribute. Restored on exit.
    """
    import sys
    from skimage.morphology import convex_hull as module
    fast = module.convex_hull_image
    original = getattr(fast, '__wrapped__', None)
    swapped = []
    if original is not None:
        for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
            names = getattr(child, '__dict__', None)
            if not names or child is module:
                continue
            # Modules holding no name bound to the original are left as they are.
            if id(original) not in map(id, names.values()):
                continue
            for name, value in list(names.items()):
                if value is original:
                    swapped.append((child, name))
                    setattr(child, name, fast)
    try:
        yield len(swapped)
    finally:
        for child, name in reversed(swapped):
            setattr(child, name, original)
