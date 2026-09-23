"""Frame-specific tracked-cell boundaries for display-only exports."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
from scipy import ndimage as ndi

from auto_organotypic.render.outlines import OutlineOverlay

from ..tracking.contract import sha256_of

# The accepted review-video cycle from the mask tuning round.  These are
# positional identity colours, never biological categories.
ACCEPTED_COLOURS: tuple[tuple[int, int, int], ...] = (
    (255, 70, 70), (80, 210, 255), (110, 255, 100), (255, 210, 70),
    (220, 100, 255), (255, 145, 60), (80, 255, 210), (160, 160, 255),
    (255, 100, 190), (180, 255, 70), (90, 170, 255), (255, 180, 190),
)


def _palette(values: Sequence[Sequence[int]] | str) -> tuple[tuple[int, int, int], ...]:
    if isinstance(values, str):
        if values.strip().lower() != "accepted":
            raise ValueError("outline_colours must be 'accepted' or a list of RGB triples")
        return ACCEPTED_COLOURS
    colours = tuple(tuple(int(channel) for channel in colour) for colour in values)
    if not colours or any(len(colour) != 3 or any(
            channel < 0 or channel > 255 for channel in colour)
                          for colour in colours):
        raise ValueError("outline_colours must contain RGB triples from 0 to 255")
    return colours


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


def overlay(labels, *, width_px: int = 1, opacity: float = 1.0,
            colours: Sequence[Sequence[int]] | str = "accepted",
            frame_index: int | None = None) -> OutlineOverlay:
    """Resolve tracked labels into Auto-Organotypic's shared outline painter."""
    path = Path(labels)
    values = np.asarray(tifffile.imread(path))
    if frame_index is not None:
        if values.ndim != 3 or not 0 <= int(frame_index) < values.shape[0]:
            raise ValueError("frame_index is outside the tracked label stack")
        values = values[int(frame_index)]
    alpha = float(opacity)
    if not 0 <= alpha <= 1:
        raise ValueError("outline_opacity must be between 0 and 1")
    palette = _palette(colours)
    return OutlineOverlay(
        labels=values,
        boundary=outer_boundaries(values, width_px=width_px),
        source=str(path),
        sha256=sha256_of(path),
        discovery="tracked identity labels supplied by PyMicroglia",
        colour=(0, 0, 0),
        width_px=int(width_px),
        opacity=alpha,
        colour_by_label=True,
        palette=palette,
    )
