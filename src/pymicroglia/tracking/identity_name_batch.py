"""Count name overlaps with one lookup pass per frame and measure objects in their boxes.

reuse_retired_identity_names sorted every frame's (temporary, accepted) pixel
pairs with a row-wise unique and relabelled the whole movie once per reused
name. The same integer overlap matrix comes from one bincount per frame over
lookup-table indices, and the reuse is one lookup per frame.

restore_deferred_identity_layer compared every physical object against the
whole frame. Each object's pixels all lie inside its find_objects box, so the
same row-major values reach np.unique when the comparison runs in that box.
"""
import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

LEDGER = 'b56dcf690a131bb3526ee6e6d2f1e54cfbd0ebfadab7ec45f6b5fb1f1734531d'
CONFIDENCE = '30f9cef93954b2843bc964de7bd9a916fc41c148d06c0a13895215b91a09e552'
LARGEST_ID = 1 << 24
LARGEST_TABLE = 1 << 26


def _plain_ids(values):
    ids = sorted(set(map(int, values)) - {0})
    return ids, (not ids or (ids[0] > 0 and ids[-1] < LARGEST_ID))


def reuse_retired_identity_names(original, stats):
    def reuse(candidate, accepted_labels, minimum_overlap_px=1):
        if (np.asarray(candidate[:0]).dtype.kind not in 'iu'
                or np.asarray(accepted_labels[:0]).dtype.kind not in 'iu'
                or np.shape(candidate) != np.shape(accepted_labels) or np.ndim(candidate) != 3):
            stats['fallback'] += 1
            return original(candidate, accepted_labels, minimum_overlap_px)
        candidate_ids, plain_candidate = _plain_ids(np.unique(candidate))
        accepted_ids, plain_accepted = _plain_ids(np.unique(accepted_labels))
        novel = sorted(set(candidate_ids) - set(accepted_ids))
        retired = sorted(set(accepted_ids) - set(candidate_ids))
        if not novel or not retired or not (plain_candidate and plain_accepted) or (
                len(novel) * len(retired) > LARGEST_TABLE):
            stats['fallback'] += 1
            return original(candidate, accepted_labels, minimum_overlap_px)
        novel_index = {identity: index for index, identity in enumerate(novel)}
        retired_index = {identity: index for index, identity in enumerate(retired)}
        novel_lut = np.full(candidate_ids[-1] + 1, -1, np.int64)
        novel_lut[novel] = np.arange(len(novel))
        retired_lut = np.full(accepted_ids[-1] + 1, -1, np.int64)
        retired_lut[retired] = np.arange(len(retired))
        width = len(retired)
        overlap = np.zeros(len(novel) * width, np.int64)
        # Only pixels holding a novel name can count; a one-byte lookup finds
        # them and the two index lookups run on those pixels alone. Integer
        # counts do not depend on the order the codes reach bincount.
        is_novel = novel_lut >= 0
        for t in range(len(candidate)):
            plane = np.asarray(candidate[t]).ravel()
            where = np.flatnonzero(is_novel[plane])
            if not where.size:
                continue
            rows = novel_lut[plane[where]]
            columns = retired_lut[np.asarray(accepted_labels[t]).ravel()[where]]
            keep = columns >= 0
            if keep.any():
                overlap += np.bincount(rows[keep] * width + columns[keep], minlength=overlap.size)
        overlap = overlap.reshape(len(novel), width)
        rows, columns = linear_sum_assignment(-overlap)
        mapping = {
            novel[int(row)]: retired[int(column)]
            for row, column in zip(rows, columns)
            if int(overlap[int(row), int(column)]) >= minimum_overlap_px}
        result = candidate.copy()
        dtype = np.asarray(result[:0]).dtype
        if mapping:
            if any(int(dtype.type(reused)) != reused for reused in mapping.values()):
                stats['fallback'] += 1
                return original(candidate, accepted_labels, minimum_overlap_px)
            lut = np.arange(candidate_ids[-1] + 1, dtype=dtype)
            lut[list(mapping)] = list(mapping.values())
            # A lookup returns the table's dtype, so the check needs no frame.
            if lut.dtype != dtype:
                raise TypeError('lookup changed the label dtype')
            for t in range(len(candidate)):
                result[t] = lut[np.asarray(candidate[t])]
        final_ids = set(map(int, np.unique(result))) - {0}
        if len(final_ids) != len(candidate_ids):
            raise AssertionError("retired-name reuse changed the identity count")
        audit = pd.DataFrame([
            {
                "temporary_identity": temporary,
                "reused_identity": reused,
                "overlap_px": int(overlap[
                    novel_index[temporary], retired_index[reused]]),
            }
            for temporary, reused in sorted(mapping.items())
        ])
        stats['batched'] += 1
        return result, audit
    return reuse


def _boxes(frame):
    """(local, box) in ascending local order, or None when find_objects cannot stand in."""
    if frame.dtype.kind not in 'iu' or not frame.size:
        return None
    low, high = int(frame.min()), int(frame.max())
    if low < 0 or high >= LARGEST_ID:
        return None
    return [(local, box) for local, box in enumerate(ndi.find_objects(frame), start=1) if box is not None]


def _majority(values, excluded=None):
    found, counts = np.unique(values, return_counts=True)
    choices = [(int(count), int(value)) for value, count in zip(found, counts)
               if int(value) > 0 and (excluded is None or int(value) not in excluded)]
    return max(choices)[1] if choices else None


SHORT_SPAN = 1 << 16


def _choices(values):
    """_majority's (count, value) pairs for every positive value.

    Integer values within a short range are counted with one bincount, whose
    counts equal np.unique's; other inputs use np.unique.
    """
    if values.dtype.kind in 'iu' and values.size:
        low, high = int(values.min()), int(values.max())
        if high - low < SHORT_SPAN and -2**62 < low and high < 2**62:
            if low == high:
                return [(int(values.size), low)] if low > 0 else []
            counts = np.bincount((values.astype(np.int64) - low).ravel())
            return [(int(counts[k]), k + low) for k in np.flatnonzero(counts).tolist() if k + low > 0]
    found, counts = np.unique(values, return_counts=True)
    return [(int(count), int(value)) for value, count in zip(found, counts) if int(value) > 0]


def _chosen(choices, excluded=None):
    kept = [item for item in choices if excluded is None or item[1] not in excluded]
    return max(kept)[1] if kept else None


def restore_deferred_identity_layer(original, stats):
    def restore(tracked, physical_objects, reference_labels, high_identity_ids):
        if tracked.shape != physical_objects.shape or tracked.shape != reference_labels.shape:
            raise ValueError("tracked, physical and reference labels are incompatible")
        planes = [np.asarray(frame) for frame in physical_objects]
        boxes = [_boxes(frame) for frame in planes]
        if any(item is None for item in boxes) or np.ndim(tracked) != 3:
            stats['fallback'] += 1
            return original(tracked, physical_objects, reference_labels, high_identity_ids)
        result = tracked.copy()
        later = [[] for _ in planes]
        rows: list[dict] = []
        high_ids = {int(value) for value in high_identity_ids}
        for t, frame in enumerate(planes):
            records = []
            current = result[t]
            used_high = set(map(int, np.unique(current))) & high_ids
            reference = np.asarray(reference_labels[t])
            for local, box in boxes[t]:
                mask = frame[box] == local
                tracked_identity = _chosen(_choices(np.asarray(current[box])[mask])) or 0
                reference_choices = _choices(reference[box][mask])
                reference_identity = _chosen(reference_choices) or 0
                records.append((local, box, mask, tracked_identity, reference_identity))
                later[t].append([mask, tracked_identity, reference_choices])
            for (local, box, mask, tracked_identity, reference_identity), kept in zip(records, later[t]):
                if (reference_identity not in high_ids
                        or tracked_identity in high_ids
                        or reference_identity in used_high):
                    continue
                result[t][box][mask] = reference_identity
                # Objects are disjoint, so this object now holds only its new name.
                kept[1] = _chosen([(1, reference_identity)]) or 0
                used_high.add(reference_identity)
                rows.append({
                    "t": t, "imagej_frame": t + 1, "local_object": local,
                    "tracked_identity": tracked_identity,
                    "restored_identity": reference_identity,
                    "pixels": int(mask.sum()),
                    "restoration_layer": "unclaimed_established_name",
                })
        for t, frame in enumerate(planes):
            # Each object's first-pass counts still hold: later writes touch only other objects.
            for (local, box), (mask, tracked_identity, reference_choices) in zip(boxes[t], later[t]):
                if tracked_identity in high_ids:
                    continue
                restored = _chosen(reference_choices, high_ids)
                if restored is None:
                    continue
                result[t][box][mask] = restored
                rows.append({
                    "t": t, "imagej_frame": t + 1, "local_object": local,
                    "tracked_identity": tracked_identity,
                    "restored_identity": restored,
                    "pixels": int(mask.sum()),
                    "restoration_layer": "deferred_small_or_dim_name",
                })
        reference_ids = set(map(int, np.unique(reference_labels))) - {0}
        novel_ids = (set(map(int, np.unique(result))) - {0}) - reference_ids
        if novel_ids:
            for t in range(len(result)):
                present = (set(map(int, np.unique(result[t]))) - {0}) & novel_ids
                if not present:
                    continue
                changed = int(np.count_nonzero(result[t] != reference_labels[t]))
                result[t] = reference_labels[t]
                rows.append({
                    "t": t, "imagej_frame": t + 1, "local_object": 0,
                    "tracked_identity": ";".join(map(str, sorted(present))),
                    "restored_identity": 0,
                    "pixels": changed,
                    "restoration_layer": "ambiguous_frame_reverted_no_new_name",
                })
        stats['batched'] += 1
        return result, pd.DataFrame(rows)
    return restore


@contextmanager
def install(ledger, confidence, *aliases):
    stats = dict(supported=False, batched=0, fallback=0)
    pinned = [(ledger, 'reuse_retired_identity_names', LEDGER, reuse_retired_identity_names),
              (confidence, 'restore_deferred_identity_layer', CONFIDENCE, restore_deferred_identity_layer)]
    changed = []
    for module, name, digest, build in pinned:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            continue
        old = getattr(module, name)
        new = build(old, stats)
        for holder in (module, *aliases):
            if holder is not None and getattr(holder, name, None) is old:
                setattr(holder, name, new)
                changed.append((holder, name, old))
    stats['supported'] = bool(changed)
    try:
        yield stats
    finally:
        for holder, name, old in reversed(changed):
            setattr(holder, name, old)
