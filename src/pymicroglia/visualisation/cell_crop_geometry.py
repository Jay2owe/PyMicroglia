"""Existing sharp-pixel crop geometry shared by tracked-cell displays."""
from __future__ import annotations
import math
from auto_organotypic.outline.crop import CROP_SCALES
import numpy as np
from PIL import Image

def _resize_photons(crop: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fill a grid slot without blending source pixels or missing padding.

    Use the same nearest-neighbour geometry as the mask overlay so display
    enlargement cannot soften a boundary or invent intermediate intensities.
    """
    return np.asarray(Image.fromarray(crop).resize(
        size, Image.Resampling.NEAREST), np.float32)


def _window(image, centre, size, *, fill, dtype, clamp=False):
    width, height = size
    top, left = _window_origin(centre, size, image.shape, clamp)
    out = np.full((height, width), fill, dtype)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + height, image.shape[0]), min(left + width, image.shape[1])
    if y1 > y0 and x1 > x0:
        out[y0 - top:y1 - top, x0 - left:x1 - left] = image[y0:y1, x0:x1]
    return out


def _window_origin(centre, size, shape, clamp):
    width, height = size
    cy, cx = (int(round(value)) for value in centre)
    top, left = cy - height // 2, cx - width // 2
    if clamp:
        top = min(max(top, 0), max(0, shape[0] - height))
        left = min(max(left, 0), max(0, shape[1] - width))
    return top, left



def _size_for(reach: tuple[int, int], crop: str) -> tuple[int, int]:
    factor = float(CROP_SCALES[crop])
    side = max(2 * reach[0] + 1, 2 * reach[1] + 1)
    scaled = math.ceil(side * factor)
    if scaled % 2 == 0:
        scaled += 1
    return scaled, scaled

