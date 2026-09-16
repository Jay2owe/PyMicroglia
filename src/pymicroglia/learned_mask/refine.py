"""Turn a probability map into cells: split what is too big, grow what is too tight.

A plain threshold gets two things wrong, and they pull in opposite directions.

**Too big.** Where cells sit on top of each other the network masks the whole
sheet as one region. On one recording the largest single blob was 18,103 px --
fifteen times the largest cell anybody has ever drawn by hand (1,177 px) and 7%
of the field. Raising the cut does not separate them; it shrinks the sheet.

**Too tight.** The cut that keeps false blobs down is too strict for processes,
which are dim by nature. The soma clears the cut and the arms do not, so the mask
comes out as a blob where the cell is a spider.

Both are fixed by treating the cut as a *seed* rather than an answer:

    1  seed     cut at ``high``, drop specks under ``min_area``
    2  split    a seed bigger than ``max_area`` is cut apart at the band-passed
                peaks inside it, using the same difference-of-Gaussians the
                accepted detector uses to find somata
    3  reject   a piece still bigger than ``max_area`` after splitting is a
                sheet, not a cell, and is dropped
    4  grow     each seed claims the pixels around it down to a lower cut
                ``low``, out to ``grow_px``, by watershed -- so a process is
                picked up but two neighbours never merge, and dim noise far from
                any seed is never picked up at all. A pixel joins only if the
                picture agrees it is brighter than its surroundings by
                ``grow_sigma`` times the background noise.

Step 4 is what "locally more sensitive" means here: the low cut applies only
next to something the high cut already believed in, and only where the picture
agrees. It is not a lower threshold over the field.

Splitting is **off** by default (``max_area=0``). Whether two touching cells are
one region or two is a counting question, and this module makes a mask: it does
not change which pixels are called cell. Turn it on when the answer is a count.
"""

from __future__ import annotations

import numpy as np

from . import filters, scale as scaling
from .scale import NATIVE

# Sizes come from scale.py as micrometres, so a recording at a different
# magnification asks `settings(its_own_scale)` rather than these.
HIGH = 0.90          # a default seed cut; the accepted operating point is lower
LOW = 0.50           # how far down to follow a process next to a seed
MIN_AREA = NATIVE.area_px(scaling.MIN_AREA_UM2)            # 32 px2
MAX_AREA = NATIVE.area_px(scaling.MAX_AREA_UM2)            # 1177 px2
GROW_PX = NATIVE.px_int(scaling.GROW_UM)                   # 8 px
GROW_SIGMA = 1.5     # how firmly the picture must agree, in background noise.
                     # 2.5 is the accepted detector's cut for a soma; a process
                     # is dimmer than a soma, so the gate sits below it. Already
                     # relative, so it needs no conversion.
SPLIT_DISTANCE = NATIVE.px_int(scaling.SPLIT_DISTANCE_UM)  # 6 px


def _structure():
    from scipy import ndimage  # noqa: PLC0415 - deliberately local

    return ndimage.generate_binary_structure(2, 2)


def settings(recording: scaling.Scale = NATIVE) -> dict:
    """The same sizes, in the pixels a recording at this scale needs.

    Hand the result to :func:`refine` for a recording that is not at 2.0 um/px.
    The numbers a cell is actually made of do not change; only their expression
    does. This is necessary and not sufficient -- see the package docstring.
    """
    return {"min_area": recording.area_px(scaling.MIN_AREA_UM2),
            "max_area": recording.area_px(scaling.MAX_AREA_UM2),
            "grow_px": recording.px_int(scaling.GROW_UM),
            "split_distance": recording.px_int(scaling.SPLIT_DISTANCE_UM)}


def band_pass(picture: np.ndarray,
              recording: scaling.Scale = NATIVE) -> np.ndarray:
    """The accepted difference-of-Gaussians: soma scale in, background out."""
    return (filters.blur(picture, recording.px(scaling.DOG_INNER_UM))
            - filters.blur(picture, recording.px(scaling.DOG_OUTER_UM)))


def seeds(probability: np.ndarray, high: float, min_area: int) -> np.ndarray:
    """What the cut believes in, once the specks are dropped."""
    from scipy import ndimage  # noqa: PLC0415 - deliberately local

    found, count = ndimage.label(probability >= high, _structure())
    if not count:
        return found.astype(np.int32)
    sizes = np.bincount(found.ravel(), minlength=count + 1)
    sizes[0] = 0
    keep = np.flatnonzero(sizes >= min_area)
    lookup = np.zeros(count + 1, np.int32)
    lookup[keep] = np.arange(1, len(keep) + 1)
    return lookup[found]


def split_big(labels: np.ndarray, scored: np.ndarray, max_area: int,
              min_area: int, split_distance: int) -> np.ndarray:
    """Cut apart anything larger than a cell, at the somata inside it."""
    from skimage.feature import peak_local_max  # noqa: PLC0415
    from skimage.segmentation import watershed  # noqa: PLC0415

    sizes = np.bincount(labels.ravel())
    oversized = [i for i in range(1, len(sizes)) if sizes[i] > max_area]
    if not oversized:
        return labels
    out = labels.copy()
    nxt = int(labels.max())
    for blob in oversized:
        where = labels == blob
        peaks = peak_local_max(scored, min_distance=split_distance, labels=where,
                               exclude_border=False)
        if len(peaks) < 2:
            continue
        markers = np.zeros(labels.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        pieces = watershed(-scored, markers, mask=where)
        out[where] = 0
        for piece in range(1, len(peaks) + 1):
            part = pieces == piece
            if part.sum() < min_area:
                continue
            nxt += 1
            out[part] = nxt
    return out


def drop_sheets(labels: np.ndarray, max_area: int, min_area: int) -> np.ndarray:
    """Whatever is still bigger than any cell ever drawn is not a cell."""
    sizes = np.bincount(labels.ravel())
    if len(sizes) < 2:
        return labels
    sizes[0] = 0
    keep = np.flatnonzero((sizes >= min_area) & (sizes <= max_area))
    lookup = np.zeros(len(sizes), np.int32)
    lookup[keep] = np.arange(1, len(keep) + 1)
    return lookup[labels]


def grow(labels: np.ndarray, probability: np.ndarray, scored: np.ndarray,
         low: float, grow_px: int, grow_sigma: float) -> np.ndarray:
    """Follow each cell out into its processes, without letting two meet."""
    from skimage.segmentation import watershed  # noqa: PLC0415

    if not labels.any():
        return labels
    noise = scored[scored < 0].std() or 1.0     # the negative half is background
    room = ((probability >= low)
            & filters.reach(labels, grow_px)
            & (scored > grow_sigma * noise))
    return watershed(-probability, labels, mask=room | (labels > 0))


def refine(picture: np.ndarray, probability: np.ndarray, *, high: float = HIGH,
           low: float = LOW, min_area: int = MIN_AREA, max_area: int = MAX_AREA,
           grow_px: int = GROW_PX, grow_sigma: float = GROW_SIGMA,
           split_distance: int = SPLIT_DISTANCE) -> np.ndarray:
    """One labelled cell per integer, for one picture and its probability map."""
    marked = seeds(probability, high, min_area)
    if not marked.any():
        return marked.astype(np.uint16)
    scored = band_pass(picture.astype(np.float32))
    if max_area > 0:      # 0 leaves a merged lobe merged, which is what you want
        marked = split_big(marked, scored, max_area, min_area, split_distance)
        marked = drop_sheets(marked, max_area, min_area)
    if marked.any() and low < high:
        marked = grow(marked, probability, scored, low, grow_px, grow_sigma)
    return marked.astype(np.uint16)


def refine_stack(pictures: np.ndarray, probability: np.ndarray,
                 **options) -> np.ndarray:
    """The same, frame by frame, with labels unique across the whole stack.

    Unique across the stack, not across time: nothing here says the cell
    labelled 4 in one frame is the cell labelled 4 in the next. Linking identity
    through frames is the Motion project's job, and this deliberately does not
    pretend to have done it.
    """
    out = np.zeros(probability.shape, np.uint16)
    nxt = 0
    for index in range(probability.shape[0]):
        one = refine(pictures[index], probability[index], **options)
        found = int(one.max())
        if found:
            out[index] = np.where(one > 0, one + nxt, 0)
            nxt += found
    return out
