"""Exact connected-component accounting shared by accepted-history rules."""
from __future__ import annotations

from collections import OrderedDict
import threading
import zlib

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as label_values


STRUCTURE = np.ones((3, 3), np.uint8)


def component_excess(frame: np.ndarray, identity: int) -> int:
    """Return components beyond the first for one identity in one frame."""
    count = int(ndi.label(frame == int(identity), STRUCTURE)[1])
    return max(0, count - 1)


def new_duplicate_components(baseline: np.ndarray,
                             candidate: np.ndarray) -> int:
    """Count new same-identity components using only changed owner-frames.

    An unchanged identity mask has exactly the same component count. Restricting
    the calculation to identities whose pixels changed is therefore
    mathematically identical to scanning every identity in every frame.
    """
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate label stacks must align")
    added = 0
    changed_frames = np.flatnonzero(np.any(
        baseline != candidate, axis=tuple(range(1, baseline.ndim))))
    for frame in changed_frames:
        before = baseline[int(frame)]
        after = candidate[int(frame)]
        changed = before != after
        owners = np.unique(np.concatenate((before[changed], after[changed])))
        for identity in owners:
            if int(identity) <= 0:
                continue
            added += max(
                0,
                component_excess(after, int(identity))
                - component_excess(before, int(identity)),
            )
    return int(added)


def _label_value_components_whole(
        frame: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label equal-valued 8-connected regions in one image pass."""
    components = label_values(frame, background=0, connectivity=2)
    count = int(components.max())
    if count == 0:
        return (components, np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64))
    indices = np.arange(1, count + 1)
    areas = np.bincount(components.ravel(), minlength=count + 1)[1:]
    if type(frame) is np.ndarray and np.issubdtype(frame.dtype, np.integer):
        # Equal-valued labeling guarantees every pixel in a component has the
        # same input value. Repeated writes therefore agree; no reduction or
        # pixel sorting is needed. Keep original behavior for other inputs.
        owner_values = np.empty(count + 1, dtype=frame.dtype)
        owner_values[components] = frame
        owners = owner_values[1:].astype(int)
    else:
        owners = ndi.maximum(frame, labels=components, index=indices).astype(int)
    return components, owners, areas.astype(np.int64, copy=False)


# Speed (folded from plain_component_frames, component_accounting_view and
# component_table_memo): an integer frame of an ndarray subclass (a disk-backed
# movie frame) is labelled through a plain view of the same memory, which reads
# owners in one indexed assignment (every pixel of a component holds the same
# value, so the owners are the maximum the subclass branch took); and a plain,
# memory-mapped or memory-pool two-dimensional integer frame seen before gets
# its table back from a bounded memo. A kept table is the frame's nonzero
# positions and values, the component number at each and the owner and area
# columns; the Adler-32 checksum only finds the candidate, which is used only
# when the frame holds the same values at the same nonzero positions, and is
# rebuilt as new arrays so callers that edit what they receive cannot reach it.
def _label_value_components_plain(frame):
    if (isinstance(frame, np.ndarray) and type(frame) is not np.ndarray
            and np.issubdtype(frame.dtype, np.integer)):
        return _label_value_components_whole(frame.view(np.ndarray))
    return _label_value_components_whole(frame)


def _memo_supported(kind):
    if kind is np.ndarray or kind is np.memmap:
        return True
    return (kind.__name__ == 'WorkingArray' and kind.__module__.rsplit('.', 1)[-1] == 'array_storage'
            and kind.__bases__ == (np.memmap,))


class _ComponentMemo:
    # About 0.1 MB a frame on A1, so two whole movies' tables fit.
    def __init__(self, compute, *, entries=4096, ram_bytes=96 << 20):
        self.compute = compute
        self.entries, self.ram_bytes = int(entries), int(ram_bytes)
        self.cache = OrderedDict()
        self.ram = 0
        self.lock = threading.Lock()

    def __call__(self, frame):
        if not _memo_supported(type(frame)) or frame.ndim != 2 or frame.dtype.kind not in 'iu':
            return self.compute(frame)
        data = np.ascontiguousarray(frame.view(np.ndarray))
        key = (data.dtype.str, data.shape, zlib.adler32(memoryview(data).cast('B')))
        with self.lock:
            entry = self.cache.get(key)
            if entry is not None:
                self.cache.move_to_end(key)
        flat = data.reshape(-1)
        if entry is not None:
            index, values, numbers, components_dtype, owners, areas = entry[:6]
            if np.count_nonzero(flat) == len(index) and np.array_equal(flat[index], values):
                components = np.zeros(flat.size, dtype=components_dtype)
                components[index] = numbers
                return components.reshape(data.shape), owners.copy(), areas.copy()
        result = self.compute(frame)
        self._store(key, flat, result)
        return result

    def _store(self, key, flat, result):
        if not (type(result) is tuple and len(result) == 3
                and all(type(value) is np.ndarray for value in result)):
            return
        components, owners, areas = result
        if (components.shape != tuple(key[1]) or components.dtype.kind not in 'iu'
                or owners.ndim != 1 or areas.ndim != 1):
            return
        numbered = components.reshape(-1)
        foreground = flat != 0
        if not np.array_equal(numbered != 0, foreground):
            return
        where = np.flatnonzero(foreground)
        index = where.astype(np.int32) if flat.size < (1 << 31) else where
        entry = (index, flat[where].copy(), numbered[where].copy(), components.dtype,
                 owners.copy(), areas.copy())
        size = sum(part.nbytes for part in entry if isinstance(part, np.ndarray))
        with self.lock:
            if key in self.cache:
                return
            self.cache[key] = entry + (size,)
            self.ram += size
            while self.cache and (len(self.cache) > self.entries or self.ram > self.ram_bytes):
                _, old = self.cache.popitem(last=False)
                self.ram -= old[-1]


_COMPONENT_MEMO = _ComponentMemo(_label_value_components_plain)


def label_value_components(
        frame: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label equal-valued 8-connected regions in one image pass."""
    return _COMPONENT_MEMO(frame)


def component_areas_by_identity(frame: np.ndarray) -> dict[int, np.ndarray]:
    """Return all 8-connected component areas after one value-aware pass."""
    _, owners, areas = label_value_components(frame)
    result: dict[int, list[int]] = {}
    for owner, area in zip(owners, areas):
        result.setdefault(int(owner), []).append(int(area))
    return {owner: np.asarray(values, dtype=np.int64)
            for owner, values in result.items() if owner > 0}


def total_component_excess(stack: np.ndarray,
                           minimum_area: int = 1) -> int:
    """Count components beyond the first for every owner and frame."""
    total = 0
    for frame in stack:
        for areas in component_areas_by_identity(frame).values():
            substantial = int(np.count_nonzero(areas >= int(minimum_area)))
            total += max(0, substantial - 1)
    return int(total)


def duplicate_owner_frames(stack: np.ndarray,
                           minimum_area: int = 1) -> int:
    """Count owner-frames containing more than one substantial component."""
    rows = 0
    for frame in stack:
        for areas in component_areas_by_identity(frame).values():
            rows += int(np.count_nonzero(areas >= int(minimum_area)) > 1)
    return int(rows)
