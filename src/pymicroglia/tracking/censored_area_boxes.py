"""Right-censored seat partition: expected areas from owner components labelled once, in the owner's box.

_expected_area labelled the whole frame's owner mask for every earlier point of
a track, and discovery asks again for each transition and resident. Discovery
does not change labels, so within one discover_and_apply call each (frame,
owner) mask is labelled once. An 8-connected component never leaves the mask's
bounding box, so labelling inside the box gives the same components. The point's
component follows physical._component_at: the pixel under the rounded point, or
else the first nearest mask pixel of the search window in raster order. Mask
pixels lie inside the box, so the window's mask pixels and their order are the
same. Only the chosen component's pixel count is read.

Each owner's box comes from one find_objects pass over its frame (the first to
last row and column holding the owner, the box the mask scan finds), and only
that box is compared with the owner, so frames are not compared whole for every
owner. Negative, very large or non-integer label values keep the mask scan.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

SOURCE = '0b7146ac0a432fe2b8c1b3472f2e40dfa2e5259ded2580c5e7b4399da469aec9'


class OwnerParts:
    def __init__(self, labels, structure, stats):
        self.labels, self.structure, self.stats = labels, structure, stats
        self.parts = {}
        self.boxes = {}

    def _boxes(self, frame):
        # Every owner's bounding box in one pass: find_objects gives, for value v,
        # the first to last row and column holding v, the box get() derives from
        # the owner's mask. Only for non-negative integer frames of modest range.
        held = self.boxes.get(frame, False)
        if held is False:
            held = None
            plane = self.labels[frame]
            if (isinstance(plane, np.ndarray) and plane.ndim == 2 and plane.size
                    and plane.dtype.kind in 'iu'):
                if int(plane.min()) >= 0 and int(plane.max()) < 1 << 20:
                    held = ndimage.find_objects(plane)
            self.boxes[frame] = held
        return held

    def get(self, frame, owner):
        key = (frame, owner)
        held = self.parts.get(key)
        boxes = None
        if (held is None and isinstance(owner, (int, np.integer)) and not isinstance(owner, (bool, np.bool_))
                and owner > 0 and isinstance(frame, (int, np.integer))):
            boxes = self._boxes(frame)
        if held is None and boxes is not None:
            box = boxes[int(owner) - 1] if int(owner) <= len(boxes) else None
            if box is None:
                held = None, None, None
            else:
                crop = self.labels[frame][box] == owner
                parts, _ = ndimage.label(crop, self.structure)
                held = (int(box[0].start), int(box[1].start)), parts, np.bincount(parts.ravel())
            self.parts[key] = held
            self.stats['labelled'] += 1
            self.stats['boxed'] = self.stats.get('boxed', 0) + 1
        elif held is None:
            mask = self.labels[frame] == owner
            rows = np.flatnonzero(mask.any(axis=1))
            if not len(rows):
                held = None, None, None
            else:
                columns = np.flatnonzero(mask[rows[0]:rows[-1] + 1].any(axis=0))
                y0, x0 = int(rows[0]), int(columns[0])
                crop = mask[y0:int(rows[-1]) + 1, x0:int(columns[-1]) + 1]
                parts, _ = ndimage.label(crop, self.structure)
                held = (y0, x0), parts, np.bincount(parts.ravel())
            self.parts[key] = held
            self.stats['labelled'] += 1
        else:
            self.stats['reused'] += 1
        return held

    def area_at(self, frame, owner, x, y, search_radius):
        """Pixel count of physical._component_at's component, or 0."""
        (origin, parts, sizes) = self.get(frame, owner)
        if parts is None:
            return 0
        height, width = self.labels.shape[1:]
        y0, x0 = origin
        cx, cy = int(round(x)), int(round(y))
        if 0 <= cy < height and 0 <= cx < width:
            local_y, local_x = cy - y0, cx - x0
            if (0 <= local_y < parts.shape[0] and 0 <= local_x < parts.shape[1]
                    and parts[local_y, local_x] > 0):
                return int(sizes[parts[local_y, local_x]])
        wy0, wy1 = max(0, cy - search_radius), min(height, cy + search_radius + 1)
        wx0, wx1 = max(0, cx - search_radius), min(width, cx + search_radius + 1)
        # The original slices with these bounds: a negative end counts from the far edge.
        wy0, wy1, _ = slice(wy0, wy1).indices(height)
        wx0, wx1, _ = slice(wx0, wx1).indices(width)
        # The window clipped to the box holds every mask pixel of the window.
        ly0, ly1 = max(wy0 - y0, 0), min(wy1 - y0, parts.shape[0])
        lx0, lx1 = max(wx0 - x0, 0), min(wx1 - x0, parts.shape[1])
        if ly0 >= ly1 or lx0 >= lx1:
            return 0
        window = parts[ly0:ly1, lx0:lx1]
        yy, xx = np.nonzero(window > 0)
        if not len(xx):
            return 0
        distances = (xx + (lx0 + x0) - x) ** 2 + (yy + (ly0 + y0) - y) ** 2
        best = int(np.argmin(distances))
        return int(sizes[window[yy[best], xx[best]]])


def expected_area(module, memo):
    def _expected_area(labels, group, owner, before):
        if labels is not memo.labels:
            memo.stats['fallback'] += 1
            return original(labels, group, owner, before)
        areas = []
        for point in group[
                group.frame.astype(int).lt(int(before))
                & group.physically_visible.map(module._truth)
                & group.accepted_owner.astype(int).eq(int(owner))].itertuples(
                    index=False):
            area = memo.area_at(int(point.frame), int(owner), float(point.x), float(point.y),
                                max(2, int(math.ceil(float(point.radius_px)))))
            if area > 0:
                areas.append(area)
        return float(np.median(areas)) if areas else 0.0
    original = module._expected_area
    return _expected_area


@contextmanager
def install(module):
    stats = dict(supported=False, calls=0, labelled=0, reused=0, fallback=0)
    original = module.discover_and_apply
    expected = module._expected_area
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return

    def discover_and_apply(labels, raw, points, params):
        plain = (isinstance(labels, np.ndarray) and labels.ndim == 3
                 and np.issubdtype(labels.dtype, np.integer))
        if not plain:
            return original(labels, raw, points, params)
        memo = OwnerParts(labels, module.partition.STRUCTURE, stats)
        module._expected_area = expected_area(module, memo)
        stats['calls'] += 1
        try:
            return original(labels, raw, points, params)
        finally:
            module._expected_area = expected
            memo.parts.clear()
    discover_and_apply.__wrapped__ = original
    module.discover_and_apply = discover_and_apply
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.discover_and_apply = original
        module._expected_area = expected
