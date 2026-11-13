"""One-core interaction discovery tests each pair's union inside the masks' boxes.

discover_one_core_interactions labelled the whole frame for every incoming and
outgoing fragment pair to ask whether their union is one component. Every
pixel of the union lies in the outer box of the two masks' boxes, so labelling
that box gives the same component count. When both masks are occupied and their
boxes are two or more rows or columns apart, no 3 x 3 neighbourhood joins them
and the union has at least two components, as the whole-frame label would find.
Each mask's box is found once per frame. A frame's track identities are read
once: the outgoing set of one source frame is the incoming set of the next,
and the track movie is not written during discovery. Masks that are not 2-D
Boolean arrays of one shape keep the whole-frame label.
"""
import __future__
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SOURCE = '94bfef1f2908b888b4397741134d7f389b68b32ecf4d09b7ebe74b58d68bc4b8'
EDITS = (
    ("        incoming_ids = set(map(int, np.unique(\n"
     "            neutral_tracks[source_index - 1]))) - {0}\n"
     "        outgoing_ids = set(map(int, np.unique(\n"
     "            neutral_tracks[source_index]))) - {0}\n",
     "        incoming_ids = _one_core_track_ids(neutral_tracks, source_index - 1, _track_ids)\n"
     "        outgoing_ids = _one_core_track_ids(neutral_tracks, source_index, _track_ids)\n"),
    ("    for source_index in range(first_source_index, last_source_index + 1):\n",
     "    _track_ids = {}\n"
     "    for source_index in range(first_source_index, last_source_index + 1):\n"),
    ("        candidates: list[tuple[object, np.ndarray, str, set[int], int]] = []\n",
     "        candidates: list[tuple[object, np.ndarray, str, set[int], int]] = []\n"
     "        _boxes = {}\n"),
    ("                union = mask_a | mask_b\n"
     "                if ndi.label(union, structure=CONNECTIVITY)[1] != 1:\n"
     "                    continue\n",
     "                if not _one_core_piece(mask_a, mask_b, CONNECTIVITY, _boxes):\n"
     "                    continue\n"
     "                union = mask_a | mask_b\n"),
)


def _box(mask, boxes):
    # Keyed by the mask object, which the caller's candidate list keeps alive
    # for the whole frame.
    key = id(mask)
    found = boxes.get(key)
    if found is None:
        plain = mask.view(np.ndarray)
        rows = np.flatnonzero(plain.any(axis=1))
        if not len(rows):
            found = (plain, None)
        else:
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            columns = np.flatnonzero(plain[y0:y1].any(axis=0))
            found = (plain, (y0, y1, int(columns[0]), int(columns[-1]) + 1))
        boxes[key] = (mask, *found)
        return found
    return found[1:]


def helpers(stats):
    def one_piece(mask_a, mask_b, structure, boxes):
        if not (isinstance(mask_a, np.ndarray) and isinstance(mask_b, np.ndarray)
                and mask_a.ndim == mask_b.ndim == 2 and mask_a.shape == mask_b.shape
                and mask_a.dtype == mask_b.dtype == np.bool_
                and np.shape(structure) == (3, 3)):
            stats['whole'] += 1
            return ndi.label(mask_a | mask_b, structure=structure)[1] == 1
        plain_a, box_a = _box(mask_a, boxes)
        plain_b, box_b = _box(mask_b, boxes)
        if box_a is None and box_b is None:
            stats['empty'] += 1
            return False
        if box_a is not None and box_b is not None and (
                box_a[0] > box_b[1] or box_b[0] > box_a[1]
                or box_a[2] > box_b[3] or box_b[2] > box_a[3]):
            stats['apart'] += 1
            return False
        boxes_present = [box for box in (box_a, box_b) if box is not None]
        region = (slice(min(b[0] for b in boxes_present), max(b[1] for b in boxes_present)),
                  slice(min(b[2] for b in boxes_present), max(b[3] for b in boxes_present)))
        stats['boxed'] += 1
        return ndi.label(plain_a[region] | plain_b[region], structure=structure)[1] == 1

    def track_ids(tracks, index, cache):
        if index not in cache:
            cache[index] = set(map(int, np.unique(tracks[index]))) - {0}
        else:
            stats['track_reuse'] += 1
        return set(cache[index])

    return dict(_one_core_piece=one_piece, _one_core_track_ids=track_ids)


@contextmanager
def install(module):
    stats = dict(supported=False, whole=0, empty=0, apart=0, boxed=0, track_reuse=0)
    original = module.discover_one_core_interactions
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    source = inspect.getsource(original)
    if any(source.count(old) != 1 for old, _ in EDITS):
        yield stats
        return
    for old, new in EDITS:
        source = source.replace(old, new)
    names = vars(module)
    added = helpers(stats)
    names.update(added)
    namespace = {}
    exec(compile(source, '<one-core-union-boxes>', 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), names, namespace)
    replacement = namespace['discover_one_core_interactions']
    replacement.__wrapped__ = original
    module.discover_one_core_interactions = replacement
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.discover_one_core_interactions = original
        for name in added:
            names.pop(name, None)
