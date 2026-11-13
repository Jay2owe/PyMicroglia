"""Exact occupied bounds without allocating every occupied coordinate."""
import numpy as np

def occupied_bounds(mask):
    """Return y0,y1,x0,x1, or None for an empty mask; bounds are exclusive."""
    if type(mask) is np.ndarray and mask.ndim == 2 and mask.dtype == np.bool_:
        rows = np.flatnonzero(np.any(mask, axis=1))
        if not len(rows):
            return None
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        # Speed (folded from bounds_band): rows outside the occupied band hold
        # no set pixel, so the band's columns are the whole mask's.
        columns = np.flatnonzero(np.any(mask[y0:y1], axis=0))
        return y0, y1, int(columns[0]), int(columns[-1]) + 1
    # Preserve the preceding coercion and dimensionality behavior elsewhere.
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return None
    return int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1


def occupied_coordinates(mask):
    """Return the same row-major integer coordinates as numpy.nonzero."""
    if not (type(mask) is np.ndarray and mask.ndim == 2 and mask.dtype == np.bool_):
        return np.nonzero(mask)
    bounds = occupied_bounds(mask)
    if bounds is None:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    y0, y1, x0, x1 = bounds
    yy, xx = np.nonzero(mask[y0:y1, x0:x1])
    return yy + y0, xx + x0
