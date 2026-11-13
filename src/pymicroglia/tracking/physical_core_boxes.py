"""Physical-lineage core measurements inside the mask's box.

largest_core and substantial_core_count eroded and labelled a whole frame for
one cell-sized mask. Every mask pixel lies in the mask's bounding box; widened
by the 3 x 3 structure's reach (clipped to the frame) the box is bordered by
background or by the frame edge, exactly as in the whole frame, so the eroded
pixels, their components and the components' raster numbering are the same.
The returned core is the same Boolean frame. Masks that are not 2-D Boolean
arrays keep the original functions.
"""
import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SOURCE = '94bfef1f2908b888b4397741134d7f389b68b32ecf4d09b7ebe74b58d68bc4b8'


def _window(mask, reach):
    """The mask's bounding box widened by reach, or None for an empty mask."""
    plain = mask.view(np.ndarray)
    rows = np.flatnonzero(plain.any(axis=1))
    if not len(rows):
        return plain, None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    columns = np.flatnonzero(plain[y0:y1].any(axis=0))
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    height, width = plain.shape
    return plain, (slice(max(0, y0 - reach), min(height, y1 + reach)),
                   slice(max(0, x0 - reach), min(width, x1 + reach)))


def _supported(mask, structure):
    return (isinstance(mask, np.ndarray) and mask.ndim == 2 and mask.dtype == np.bool_
            and structure.shape == (3, 3))


def boxed(module, stats):
    original_largest, original_count = module.largest_core, module.substantial_core_count

    def largest_core(mask):
        structure = module.CONNECTIVITY
        if not _supported(mask, structure):
            stats['whole'] += 1
            return original_largest(mask)
        plain, box = _window(mask, 1)
        if box is None:
            # Nothing to erode: no component.
            stats['boxed'] += 1
            return 0, None
        eroded = ndi.binary_erosion(plain[box], structure=structure)
        components, count = ndi.label(eroded, structure=structure)
        stats['boxed'] += 1
        if not count:
            return 0, None
        sizes = np.bincount(components.ravel())[1:]
        number = int(np.argmax(sizes)) + 1
        core = np.zeros(plain.shape, bool)
        core[box] = components == number
        return int(sizes[number - 1]), core

    def substantial_core_count(mask, minimum_px):
        structure = module.CONNECTIVITY
        if not _supported(mask, structure):
            stats['whole'] += 1
            return original_count(mask, minimum_px)
        plain, box = _window(mask, 1)
        stats['boxed'] += 1
        if box is None:
            return 0
        eroded = ndi.binary_erosion(plain[box], structure=structure)
        components, count = ndi.label(eroded, structure=structure)
        if not count:
            return 0
        sizes = np.bincount(components.ravel())[1:]
        return int(np.count_nonzero(sizes >= int(minimum_px)))

    return largest_core, substantial_core_count


@contextmanager
def install(module):
    stats = dict(supported=False, boxed=0, whole=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    originals = (module.largest_core, module.substantial_core_count)
    module.largest_core, module.substantial_core_count = boxed(module, stats)
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.largest_core, module.substantial_core_count = originals
