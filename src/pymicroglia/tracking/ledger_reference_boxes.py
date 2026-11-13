"""Substantial identity references of the soma ledger, measured inside each name's box.

global_soma_ledger._substantial_identity_references visits every name of each
bookend frame: model_review.identity_components compares the name with the
whole frame, finds its bounds and writes each 8-connected component into a new
whole-frame mask, and eroded_cores finds that component's bounds, erodes and
labels it there and writes every core into another whole-frame mask, then the
largest core's whole-frame coordinates give the centre. Here the names and
their boxes come from one find_objects pass; components are labelled in the
name's box, each is eroded in that box (pixels outside are background either
way, as erosion treats the border), cores are labelled and counted there, and
the largest core (the first of equal sizes, as the stable descending sort
keeps them) supplies its pixels in row-major order shifted to whole-frame
coordinates, so the mean is taken over the same array. Labelling numbers
components by first pixel in row-major order, which any enclosing box keeps.
Frames that are not two-dimensional non-negative integers with names below
2**24 send the whole call to the original.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as _connected

LEDGER = 'b56dcf690a131bb3526ee6e6d2f1e54cfbd0ebfadab7ec45f6b5fb1f1734531d'
REVIEW = 'f2f0abcfe97024813a3c12aeb90e61b650c92bc836c2590e953c1e85c456b467'
LARGEST_NAME = 1 << 24


def _frame(labels, t):
    frame = np.asarray(labels[t])
    if type(frame) is not np.ndarray:
        frame = frame.view(np.ndarray)
    if frame.ndim != 2 or frame.dtype.kind not in 'iu' or not frame.size:
        return None
    if (frame.dtype.kind == 'i' and int(frame.min()) < 0) or int(frame.max()) >= LARGEST_NAME:
        return None
    return frame


def _frame_rows(t, frame, minimum_core_px, rows):
    """Append one frame's reference rows, in the original's order and values.

    Components of every name come from one labelling of equal neighbouring
    names (8-connected, as CONNECTIVITY), numbered by first pixel in row-major
    order; sorting them by (name, number) gives each name's components in the
    order labelling its own mask numbers them. Eroding a component with that
    connectivity keeps a pixel exactly when its eight neighbours (outside the
    frame counting as background) hold its name: such neighbours are all in its
    component. A core pixel's neighbours are all in its own component, so cores
    of different components never touch and one labelling of all core pixels
    numbers each component's cores in their row-major order. The largest core
    (the first of equal sizes) and the integer sums of its rows and columns, divided
    by its size in float64, give the same centre as np.mean of its coordinates."""
    components = _connected(frame, background=0, connectivity=2)
    count = int(components.max())
    if not count:
        return
    height, width = frame.shape
    padded = np.pad(frame, 1)
    core = frame != 0
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                core &= padded[dy:dy + height, dx:dx + width] == frame
    cores, found = ndi.label(core, structure=np.ones((3, 3), bool))
    if not found:
        return
    flat = components.ravel()
    # Positions found in a Boolean array: the same, five times faster than in labels.
    where = np.flatnonzero(flat != 0)
    names = np.zeros(count + 1, np.int64)
    names[flat[where]] = frame.ravel()[where]
    numbered = cores.ravel()
    where = np.flatnonzero(numbered != 0)
    ids = numbered[where]
    ys, xs = np.divmod(where, width)
    sizes = np.bincount(ids, minlength=found + 1)
    rows_sum = np.bincount(ids, weights=ys, minlength=found + 1)
    columns_sum = np.bincount(ids, weights=xs, minlength=found + 1)
    owner = np.zeros(found + 1, np.int64)
    owner[ids] = flat[where]
    labels = np.arange(1, found + 1)
    # Per component: largest core first, the lowest number among equal sizes.
    order = np.lexsort((labels, -sizes[1:], owner[1:]))
    leading = np.r_[True, owner[1:][order][1:] != owner[1:][order][:-1]]
    best = np.zeros(count + 1, np.int64)
    best[owner[1:][order][leading]] = labels[order][leading]
    for component in np.lexsort((np.arange(1, count + 1), names[1:])) + 1:
        chosen = int(best[component])
        if not chosen or int(sizes[chosen]) < minimum_core_px:
            continue
        centre = np.array([rows_sum[chosen], columns_sum[chosen]]) / sizes[chosen]
        rows.append({
            "t": t, "identity": int(names[component]),
            "y": float(centre[0]), "x": float(centre[1]),
        })


def boxed(original, connectivity, stats):
    def _substantial_identity_references(labels, first, last, minimum_core_px):
        try:
            frames = range(max(0, first), min(len(labels), last + 1))
            planes = [_frame(labels, t) for t in frames] if isinstance(labels, np.ndarray) else [None]
        except Exception:
            planes = [None]
        if any(plane is None for plane in planes) or not isinstance(minimum_core_px, (int, np.integer)):
            stats['fallback'] += 1
            return original(labels, first, last, minimum_core_px)
        stats['boxed'] += 1
        rows = []
        for t, frame in zip(frames, planes):
            _frame_rows(t, frame, minimum_core_px, rows)
        return rows
    _substantial_identity_references.__wrapped__ = original
    return _substantial_identity_references


def _digest(module):
    try:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except (OSError, TypeError, AttributeError):
        return None


@contextmanager
def install(ledger, stats=None):
    """Replace the pinned ledger's reference finder while its review helpers are the pinned ones."""
    stats = stats if stats is not None else {}
    for key in ('supported', 'boxed', 'fallback'):
        stats.setdefault(key, 0)
    original = getattr(ledger, '_substantial_identity_references', None)
    review = __import__('sys').modules.get('model_review')
    supported = (original is not None and getattr(original, '__wrapped__', None) is None
                 and _digest(ledger) == LEDGER and review is not None and _digest(review) == REVIEW
                 and getattr(ledger, 'identity_components', None) is review.identity_components
                 and getattr(ledger, 'eroded_cores', None) is review.eroded_cores
                 and np.array_equal(review.CONNECTIVITY, np.ones((3, 3), np.uint8)))
    if supported:
        ledger._substantial_identity_references = boxed(original, review.CONNECTIVITY, stats)
        stats['supported'] = 1
    try:
        yield stats
    finally:
        if supported:
            ledger._substantial_identity_references = original
