"""Default microglial reporter trace: original photons summed inside the mask."""

from __future__ import annotations

import numpy as np


DEFAULT_REPORTER_METRIC = "integrated_density_mean3"
DEFAULT_REPORTER_MEAN_FRAMES = 3


def centered_frame_mean(values: np.ndarray, frames: np.ndarray, *,
                        window: int = DEFAULT_REPORTER_MEAN_FRAMES
                        ) -> np.ndarray:
    """Mean full consecutive native-frame windows without bridging missing masks."""
    source = np.asarray(values, dtype=float)
    indices = np.asarray(frames, dtype=int)
    if source.shape != indices.shape:
        raise ValueError("values and native frame indices must align")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd frame count")
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError("native frame indices must increase")
    result = np.full(source.shape, np.nan, dtype=float)
    radius = window // 2
    for center in range(radius, len(source) - radius):
        start = center - radius
        stop = center + radius + 1
        segment = source[start:stop]
        if np.isfinite(segment).all() and np.all(np.diff(indices[start:stop]) == 1):
            result[center] = float(np.mean(segment))
    return result
