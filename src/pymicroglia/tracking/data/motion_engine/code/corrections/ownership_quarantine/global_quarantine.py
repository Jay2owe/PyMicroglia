"""Apply hard, whole-movie same-territory ownership reservations."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402
from merge_partition_speed import BoundedNdimage, slices as occupied_slices
from array_bounds import occupied_coordinates
from skimage.measure import label
ndi = BoundedNdimage()

STRUCTURE = np.ones((3, 3), np.uint8)


def objects(frame: np.ndarray) -> list[dict]:
    result = []
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        parts, count = ndi.label(frame == identity, structure=STRUCTURE)
        for component in range(1, count + 1):
            mask = parts == component
            yy, xx = occupied_coordinates(mask)
            result.append({"identity": identity, "mask": mask,
                           "area": int(mask.sum()),
                           "x": float(xx.mean()), "y": float(yy.mean())})
    return result


def _following_component_all_objects(frame: np.ndarray, observed_identity: int,
                                     previous: np.ndarray) -> dict | None:
    choices = []
    py, px = ndi.center_of_mass(previous)
    for item in objects(frame):
        if item["identity"] != observed_identity:
            continue
        overlap = int(np.count_nonzero(item["mask"] & previous))
        distance = math.hypot(item["x"] - px, item["y"] - py)
        choices.append(((overlap, -distance), item))
    if not choices:
        return None
    score, item = max(choices, key=lambda row: row[0])
    return item if score[0] >= 10 or -score[1] <= 15.0 else None


def following_component(frame: np.ndarray, observed_identity: int,
                        previous: np.ndarray) -> dict | None:
    if not (type(frame) is np.ndarray and frame.dtype.kind in "iu"
            and type(observed_identity) is int and observed_identity
            and np.iinfo(frame.dtype).min <= observed_identity
            <= np.iinfo(frame.dtype).max):
        return _following_component_all_objects(frame, observed_identity, previous)
    # Only the observed identity's components can be chosen, and an absent
    # identity labels nothing, as the original loop skips it.
    choices = []
    py, px = ndi.center_of_mass(previous)
    parts, count = ndi.label(frame == observed_identity, structure=STRUCTURE)
    for component in range(1, count + 1):
        mask = parts == component
        yy, xx = occupied_coordinates(mask)
        item = {"identity": observed_identity, "mask": mask,
                "area": int(mask.sum()),
                "x": float(xx.mean()), "y": float(yy.mean())}
        overlap = int(np.count_nonzero(item["mask"] & previous))
        distance = math.hypot(item["x"] - px, item["y"] - py)
        choices.append(((overlap, -distance), item))
    if not choices:
        return None
    score, item = max(choices, key=lambda row: row[0])
    return item if score[0] >= 10 or -score[1] <= 15.0 else None


def _frame_objects(frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """objects(frame) without whole-frame masks, and the frame's numbering.

    One labelling of every identity at once numbers each identity's
    8-connected components in the same raster order as labelling it alone.
    """
    numbered = label(frame, background=0, connectivity=2)
    positions = np.flatnonzero(numbered)
    values = numbered.flat[positions]
    order = np.argsort(values, kind="stable")
    positions = positions[order]
    values = values[order]
    cuts = np.flatnonzero(values[1:] != values[:-1]) + 1
    result = []
    for pixels in np.split(positions, cuts):
        if not len(pixels):
            continue
        yy, xx = np.divmod(pixels, frame.shape[1])
        number = int(numbered.flat[pixels[0]])
        # The whole-frame mask is drawn only for the few components an event keeps.
        result.append({"identity": int(frame.flat[pixels[0]]), "area": len(pixels),
                       "x": float(xx.mean()), "y": float(yy.mean()),
                       "_number": number, "_pixels": pixels, "_numbered": numbered})
    result.sort(key=lambda item: (item["identity"], item["_number"]))
    return result, numbered


def _mask(item: dict) -> np.ndarray:
    return item["_numbered"] == item["_number"]


def _frame_pair(labels: np.ndarray, t: int, memo: dict) -> tuple:
    # Labels are only read here, so frame t's components serve again as the
    # next pair's previous frame.
    previous = memo.pop(t - 1, None)
    if previous is None:
        previous = _frame_objects(labels[t - 1])
    current = memo[t] = _frame_objects(labels[t])
    return previous, current


def _overlaps(previous: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    width = int(current.max(initial=0)) + 1
    size = (int(previous.max(initial=0)) + 1) * width
    # The bounded histogram is an implementation choice, never an event gate.
    if size * 8 > 32 * 1024 ** 2:
        return None
    return np.bincount(previous.ravel().astype(np.int64) * width + current.ravel(),
                       minlength=size).reshape(-1, width)


def _overlap(source: dict, destination: dict, counts: np.ndarray | None) -> int:
    if counts is None:
        return int(np.count_nonzero(_mask(source) & _mask(destination)))
    return int(counts[source["_number"], destination["_number"]])


def _present(frame: np.ndarray) -> np.ndarray:
    # A count per value lists the same values in the same ascending order as
    # np.unique, without sorting.
    if (frame.dtype.kind == "u" and frame.dtype.itemsize <= 4
            or frame.dtype.kind == "i" and (not frame.size or frame.min() >= 0)):
        if int(frame.max(initial=0)) < 1 << 22:
            return np.flatnonzero(np.bincount(frame.ravel(), minlength=1))
    return np.unique(frame)


def _presence(labels: np.ndarray) -> dict[int, np.ndarray]:
    result: dict[int, list[int]] = {}
    for t, frame in enumerate(labels):
        for identity in _present(frame):
            if identity:
                result.setdefault(int(identity), []).append(t)
    return {identity: np.asarray(frames) for identity, frames in result.items()}


def _count_presence(index: dict, identity: int, t: int, before: bool) -> int:
    frames = index.get(identity, ())
    split = int(np.searchsorted(frames, t))
    return split if before else len(frames) - split


def _history_hits(history: np.ndarray, identity: int, destination: dict) -> int:
    # Boolean indexing visits exactly the destination pixels, in the same order.
    pixels = destination["_pixels"]
    yy, xx = np.divmod(pixels, history.shape[2])
    return int(np.count_nonzero(history[:, yy, xx] == identity))


def _detect_events_whole_frame(labels: np.ndarray, cfg: dict,
                               source_offset: int) -> list[dict]:
    events = []
    for t in range(1, len(labels)):
        previous = objects(labels[t - 1])
        current = objects(labels[t])
        for destination in current:
            candidates = []
            for source in previous:
                if source["identity"] == destination["identity"]:
                    continue
                overlap = int(np.count_nonzero(source["mask"]
                                               & destination["mask"]))
                union = source["area"] + destination["area"] - overlap
                iou = overlap / union if union else 0.0
                shift = math.hypot(source["x"] - destination["x"],
                                   source["y"] - destination["y"])
                if (source["area"] >= int(cfg["large_area_px"])
                        and destination["area"] >= int(cfg["large_area_px"])
                        and overlap >= int(cfg["minimum_overlap_px"])
                        and iou >= float(cfg["minimum_iou"])
                        and shift <= float(cfg["maximum_centroid_shift_px"])):
                    candidates.append(((overlap, iou, -shift), source))
            if not candidates:
                continue
            _, source = max(candidates, key=lambda row: row[0])
            history = labels[max(0, t - int(cfg["history_frames"])):t]
            source_hits = int(np.count_nonzero(
                (history == source["identity"]) & destination["mask"]))
            destination_hits = int(np.count_nonzero(
                (history == destination["identity"]) & destination["mask"]))
            if (source_hits < int(cfg["minimum_owner_history_pixels"])
                    or source_hits < destination_hits
                    + int(cfg["minimum_history_margin_pixels"])):
                continue
            prior_identity_frames = int(np.count_nonzero(np.any(
                labels[:t] == source["identity"], axis=(1, 2))))
            if prior_identity_frames < int(cfg["minimum_prior_identity_frames"]):
                continue
            if destination_hits > int(cfg["maximum_replacement_history_pixels"]):
                continue
            persistence = int(np.count_nonzero(np.any(
                labels[t:] == destination["identity"], axis=(1, 2))))
            if persistence < int(cfg["minimum_destination_persistence_frames"]):
                continue
            current_old_areas = [item["area"] for item in current
                                 if item["identity"] == source["identity"]]
            if max(current_old_areas, default=0) > int(
                    cfg["maximum_old_identity_current_area_px"]):
                continue
            events.append({
                "output_frame": t + 1,
                "source_imagej_frame": t + 1 + source_offset,
                "established_identity": source["identity"],
                "replacement_identity": destination["identity"],
                "source_mask": source["mask"],
                "destination_mask": destination["mask"],
                "overlap_px": int(np.count_nonzero(
                    source["mask"] & destination["mask"])),
                "iou": float(max(row[0][1] for row in candidates)),
                "centroid_shift_px": float(math.hypot(
                    source["x"] - destination["x"],
                    source["y"] - destination["y"])),
                "source_history_pixels": source_hits,
                "replacement_history_pixels": destination_hits,
                "prior_identity_frames": prior_identity_frames,
                "old_identity_current_max_area_px": max(current_old_areas,
                                                          default=0),
            })
    return events




def detect_events(labels: np.ndarray, cfg: dict,
                  source_offset: int) -> list[dict]:
    if (labels.ndim != 3 or labels.dtype.kind not in "iu"
            or (labels.dtype.kind == "u" and labels.size
                and int(labels.max()) > np.iinfo(np.int64).max)):
        return _detect_events_whole_frame(labels, cfg, source_offset)
    events = []
    presence_index = _presence(labels)
    frame_memo = {}
    for t in range(1, len(labels)):
        (previous, previous_numbers), (current, current_numbers) = _frame_pair(labels, t, frame_memo)
        counts = _overlaps(previous_numbers, current_numbers)
        for destination in current:
            candidates = []
            for source in previous:
                if source["identity"] == destination["identity"]:
                    continue
                overlap = _overlap(source, destination, counts)
                union = source["area"] + destination["area"] - overlap
                iou = overlap / union if union else 0.0
                shift = math.hypot(source["x"] - destination["x"],
                                   source["y"] - destination["y"])
                if (source["area"] >= int(cfg["large_area_px"])
                        and destination["area"] >= int(cfg["large_area_px"])
                        and overlap >= int(cfg["minimum_overlap_px"])
                        and iou >= float(cfg["minimum_iou"])
                        and shift <= float(cfg["maximum_centroid_shift_px"])):
                    candidates.append(((overlap, iou, -shift), source))
            if not candidates:
                continue
            _, source = max(candidates, key=lambda row: row[0])
            history = labels[max(0, t - int(cfg["history_frames"])):t]
            source_hits = _history_hits(history, source["identity"], destination)
            destination_hits = _history_hits(history, destination["identity"], destination)
            if (source_hits < int(cfg["minimum_owner_history_pixels"])
                    or source_hits < destination_hits
                    + int(cfg["minimum_history_margin_pixels"])):
                continue
            prior_identity_frames = _count_presence(presence_index, source["identity"], t, True)
            if prior_identity_frames < int(cfg["minimum_prior_identity_frames"]):
                continue
            if destination_hits > int(cfg["maximum_replacement_history_pixels"]):
                continue
            persistence = _count_presence(presence_index, destination["identity"], t, False)
            if persistence < int(cfg["minimum_destination_persistence_frames"]):
                continue
            current_old_areas = [item["area"] for item in current
                                 if item["identity"] == source["identity"]]
            if max(current_old_areas, default=0) > int(
                    cfg["maximum_old_identity_current_area_px"]):
                continue
            events.append({
                "output_frame": t + 1,
                "source_imagej_frame": t + 1 + source_offset,
                "established_identity": source["identity"],
                "replacement_identity": destination["identity"],
                "source_mask": _mask(source),
                "destination_mask": _mask(destination),
                "overlap_px": _overlap(source, destination, counts),
                "iou": float(max(row[0][1] for row in candidates)),
                "centroid_shift_px": float(math.hypot(
                    source["x"] - destination["x"],
                    source["y"] - destination["y"])),
                "source_history_pixels": source_hits,
                "replacement_history_pixels": destination_hits,
                "prior_identity_frames": prior_identity_frames,
                "old_identity_current_max_area_px": max(current_old_areas,
                                                          default=0),
            })
    return events


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("trimmed inputs are required")
    labels = load_stack(upstream_dir / f"{params['stem']}.tif")
    candidate = labels.copy()
    events = detect_events(labels, params["quarantine"],
                           int(params["source_frame_offset"]))
    enabled = bool(params.get("enable_quarantine", False))
    rows = []
    occupied = set()
    for event_number, event in enumerate(events, 1):
        row = {key: value for key, value in event.items()
               if not key.endswith("_mask")}
        row.update({"event_id": f"Q{event_number:04d}", "enabled": enabled,
                    "applied_frames": 0})
        if enabled:
            t = int(event["output_frame"]) - 1
            key = (t, int(event["replacement_identity"]))
            if key not in occupied:
                owner = int(event["established_identity"])
                observed = int(event["replacement_identity"])
                previous = event["destination_mask"]
                for frame in range(t, len(candidate)):
                    if frame == t:
                        item = {"mask": previous}
                    else:
                        item = following_component(labels[frame], observed, previous)
                        if item is None:
                            break
                    candidate[frame][item["mask"]] = owner
                    previous = item["mask"]
                    row["applied_frames"] += 1
                occupied.add(key)
        rows.append(row)
    label_path = out.out / f"{params['stem']}.tif"
    event_path = out.out / "quarantine_events.csv"
    save_stack(label_path, candidate, params["frame_interval_min"])
    pd.DataFrame(rows).to_csv(event_path, index=False)
    return {
        "outputs": {"labels": label_path, "events": event_path},
        "summary": {
            "enabled": enabled, "detected_events": int(len(rows)),
            "applied_events": int(sum(row["applied_frames"] > 0 for row in rows)),
            "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
            "foreground_changed_pixels": int(np.count_nonzero(
                (candidate > 0) != (labels > 0))),
        },
    }


