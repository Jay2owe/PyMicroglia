"""Identity boundary geometry used by tracked-cell displays."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def outer_boundaries(labels: np.ndarray, width_px: int = 1) -> np.ndarray:
    """Identity-valued pixels immediately outside each labelled cell."""
    values = np.asarray(labels)
    if values.ndim not in (2, 3):
        raise ValueError(f"tracked outlines need 2D or 3D labels; got {values.shape}")
    width = int(width_px)
    if (isinstance(width_px, bool) or width < 1
            or float(width_px) != float(width)):
        raise ValueError("outline_width_px must be a whole number of at least 1")
    planes = values[None] if values.ndim == 2 else values
    boundaries = np.zeros(planes.shape, values.dtype)
    for frame_index, frame in enumerate(planes):
        for identity in sorted(int(value) for value in np.unique(frame)
                               if int(value) > 0):
            mask = frame == identity
            edge = ndi.binary_dilation(mask, iterations=width) & ~mask
            # The accepted renderer loops identities in order, so a later
            # identity wins the rare pixel where two outside boundaries meet.
            boundaries[frame_index][edge] = identity
    return boundaries[0] if values.ndim == 2 else boundaries
