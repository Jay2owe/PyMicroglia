"""Exact positive-intensity display percentile without copying an integer movie."""
import numpy as np


def positive_display_high(raw):
    """Match percentile(raw[raw > 0], 99.5), including NumPy interpolation.

For 8/16-bit integer images, a bounded histogram identifies the same two
order statistics. NumPy itself interpolates those endpoints. Other dtypes
retain the original percentile operation on its private selected values.
"""
    if raw.dtype.kind in 'ui' and raw.dtype.itemsize <= 2:
        bins = 1 << (raw.dtype.itemsize * 8)
        counts = np.zeros(bins, dtype=np.int64)
        for frame in raw:
            values = frame.reshape(-1)
            # bincount converts integers to platform-width indices. Bound that
            # temporary too, including for very wide future camera frames.
            for start in range(0, values.size, 1024 * 1024):
                block = values[start:start + 1024 * 1024]
                if raw.dtype.kind == 'i': block = block[block > 0]
                counts += np.bincount(block, minlength=bins)
        counts[0] = 0
        n = int(counts.sum())
        if not n: return 1.0
        cumulative = np.cumsum(counts)
        position = (n - 1) * np.float64(99.5 / 100)
        lower = int(np.floor(position))
        upper = min(lower + 1, n - 1)
        endpoints = np.searchsorted(cumulative, [lower, upper], side='right').astype(raw.dtype)
        return float(np.quantile(endpoints, position - lower, method='linear'))
    positive = raw[raw > 0]
    return float(np.percentile(positive, 99.5, overwrite_input=True)) if positive.size else 1.0
