"""Solid cell outlines: enlarge the picture first, then draw a one-display-pixel outline.

Jamie's rule (2026-11-10): outlines must be continuous lines. An outline drawn at the image's own
resolution and then enlarged (by the browser, a video encoder or a resize) breaks into dots and
gaps along thin processes and diagonals. Enlarging the label image by whole pixels first and then
marking every pixel whose neighbour has another label gives a closed line one display pixel wide
at any zoom - the drawing used by Motion's correction reviewer
(Motion/manual_editing/trajectory_review/correction_reviewer.py: up, edges, thick).

    rgb = outline_image(grey_uint8, labels, zoom=4, colours={7: (0, 220, 255)})

Display only: labels and pixel values used for measurement are never changed.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

GREY = (120, 120, 120)


def up(a: np.ndarray, k: int) -> np.ndarray:
    """Enlarge by a whole number k (nearest neighbour), so each source pixel becomes a k x k block."""
    return np.repeat(np.repeat(a, k, 0), k, 1) if k > 1 else a


def edges(lab: np.ndarray) -> np.ndarray:
    """Pixels inside a cell whose 4-neighbour belongs to another label (or background)."""
    e = np.zeros(lab.shape, bool)
    e[1:] |= lab[1:] != lab[:-1]
    e[:-1] |= lab[:-1] != lab[1:]
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return e & (lab > 0)


def thick(mask: np.ndarray) -> np.ndarray:
    """Grow a line by one pixel in each direction (used for a highlighted cell)."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def zoom_for(shape: tuple[int, int], target_px: int) -> int:
    """Smallest whole zoom that makes the longer side at least target_px (at least 1)."""
    return max(1, -(-int(target_px) // max(int(shape[0]), int(shape[1]))))


def outline_image(grey: np.ndarray, labels: np.ndarray, zoom: int, *,
                  colours: Mapping[int, tuple[int, int, int]] | None = None,
                  default: tuple[int, int, int] | None = GREY,
                  highlight: set[int] | frozenset[int] = frozenset()) -> np.ndarray:
    """RGB uint8 image at zoom x size with solid outlines.

    grey: 2D uint8 display image (already contrast-mapped); labels: 2D integer labels, same shape.
    colours: label -> RGB; labels not listed get `default` (None leaves them unoutlined).
    highlight: labels drawn two display pixels wide.
    """
    if grey.shape != labels.shape:
        raise ValueError("grey and labels must have the same shape")
    lab = up(labels, zoom)
    image = np.repeat(up(grey, zoom)[..., None], 3, 2).astype(np.uint8)
    line = edges(lab)
    if default is not None:
        listed = np.isin(lab, list(colours)) if colours else np.zeros(lab.shape, bool)
        image[line & ~listed] = default
    for label, colour in (colours or {}).items():
        mask = edges(np.where(lab == label, lab, 0))
        image[thick(mask) if label in highlight else mask] = colour
    return image


def open_outlines(rgb: np.ndarray, colour: tuple[int, int, int], *, tolerance: int = 60,
                  min_pixels: int = 12) -> dict:
    """Check a rendered picture: every outline drawn in `colour` should be a closed loop.

    An outline drawn at native size and then enlarged with smoothing (a viewer, resize or video
    encoder) breaks into open dashes; one drawn after enlarging encloses its cell. Counts the
    connected outline pieces of at least `min_pixels` (8-connected, colour within `tolerance` summed
    over RGB) and how many enclose nothing. Returns {"closed", "open", "open_fraction"}. A good
    picture has open == 0 and closed equal to the number of cells drawn in that colour (smoothing
    can also wash a line out of the colour range, which shows as fewer closed outlines).
    """
    from scipy import ndimage as ndi
    line = np.abs(rgb[..., :3].astype(int) - np.asarray(colour)).sum(-1) < tolerance
    pieces, _ = ndi.label(line, structure=np.ones((3, 3)))
    closed = opened = 0
    for index, box in enumerate(ndi.find_objects(pieces), 1):
        piece = pieces[box] == index
        if piece.sum() < min_pixels:
            continue
        if ndi.binary_fill_holes(piece).sum() > piece.sum():
            closed += 1
        else:
            opened += 1
    total = closed + opened
    return {"closed": closed, "open": opened, "open_fraction": opened / total if total else 0.0}
