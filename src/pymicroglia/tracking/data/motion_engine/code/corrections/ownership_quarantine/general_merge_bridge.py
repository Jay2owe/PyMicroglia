"""Find and bridge every qualified motion-supported temporary merge."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import ndimage as scipy_ndimage

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402
from merge_partition_speed import BoundedNdimage, slices as occupied_slices
from array_bounds import occupied_coordinates
ndi = BoundedNdimage()
from bridge_merges import motion_partition, partition_host  # noqa: E402

STRUCTURE = np.ones((3, 3), np.uint8)
LARGEST_ID = 1 << 24


def _identity_objects_whole_frame(frame: np.ndarray,
                                  identity: int | None = None) -> list[dict]:
    result = []
    identities = ([identity] if identity is not None
                  else sorted(set(map(int, np.unique(frame))) - {0}))
    for owner in identities:
        parts, count = ndi.label(frame == owner, structure=STRUCTURE)
        for component in range(1, count + 1):
            mask = parts == component
            yy, xx = occupied_coordinates(mask)
            result.append({"owner": int(owner), "mask": mask,
                           "area": int(mask.sum()),
                           "x": float(xx.mean()), "y": float(yy.mean())})
    return result


def _component_rows(frame: np.ndarray, owner: int, box: tuple) -> tuple[list[dict], list[tuple]]:
    """identity_objects' rows for one identity, labelled inside its box, and
    each component's tight box (y0, y1, x0, x1)."""
    rows, boxes = [], []
    y0, x0 = box[0].start, box[1].start
    parts, count = scipy_ndimage.label(frame[box] == owner, structure=STRUCTURE)
    for component in range(1, count + 1):
        local = parts == component
        mask = np.zeros(frame.shape, bool)
        mask[box] = local
        ly, lx = np.nonzero(local)
        yy, xx = ly + y0, lx + x0
        rows.append({"owner": int(owner), "mask": mask,
                     "area": int(np.count_nonzero(local)),
                     "x": float(xx.mean()), "y": float(yy.mean())})
        boxes.append((int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1))
    return rows, boxes


def _plain(frame):
    # Disk-backed frames (memmaps, working arrays) are read through a plain
    # ndarray view of the same memory, so they take the boxed path too.
    frame = np.asarray(frame)
    return frame.view(np.ndarray) if isinstance(frame, np.ndarray) else frame


def _plain_labels(frame) -> bool:
    return (type(frame) is np.ndarray and frame.ndim == 2
            and frame.dtype.kind in "iu" and frame.size > 0)


def _supported(frame) -> bool:
    return (_plain_labels(frame) and int(frame.min()) >= 0
            and int(frame.max()) < LARGEST_ID)


def _frame_objects(frame: np.ndarray) -> tuple[list[dict], list[tuple] | None]:
    """identity_objects(frame) and each component's tight box (None when the
    frame takes the whole-frame route)."""
    frame = _plain(frame)
    if not _supported(frame):
        return _identity_objects_whole_frame(frame), None
    rows, boxes = [], []
    for owner, box in enumerate(scipy_ndimage.find_objects(frame), start=1):
        if box is None:
            continue
        found, found_boxes = _component_rows(frame, owner, box)
        rows += found
        boxes += found_boxes
    return rows, boxes


def identity_objects(frame: np.ndarray, identity: int | None = None) -> list[dict]:
    """Each identity's components, labelled inside its box.

    Each identity's pixels lie inside its box, and labelling a box numbers
    components in the same raster order, so the same masks, areas and
    coordinate means come from the box.
    """
    plain = _plain(frame)
    if identity is None:
        return _frame_objects(plain)[0]
    if not _plain_labels(plain) or not isinstance(identity, (int, np.integer)):
        return _identity_objects_whole_frame(frame, identity)
    owner = int(identity)
    if owner <= 0 or owner >= LARGEST_ID:
        return _identity_objects_whole_frame(frame, identity)
    present = plain == owner
    rows = np.flatnonzero(present.any(axis=1))
    if not len(rows):
        return []
    columns = np.flatnonzero(present.any(axis=0))
    box = (slice(int(rows[0]), int(rows[-1]) + 1),
           slice(int(columns[0]), int(columns[-1]) + 1))
    return _component_rows(plain, owner, box)[0]


def binary_objects(frame: np.ndarray) -> list[dict]:
    parts, count = ndi.label(frame > 0, structure=STRUCTURE)
    result = []
    for component in range(1, count + 1):
        mask = parts == component
        yy, xx = occupied_coordinates(mask)
        values, counts = np.unique(frame[mask], return_counts=True)
        owner = int(values[int(np.argmax(counts))])
        result.append({"owner": owner, "mask": mask, "area": int(mask.sum()),
                       "x": float(xx.mean()), "y": float(yy.mean())})
    return result


def mask_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(a & b):
        return 0.0
    if (type(a) is np.ndarray and type(b) is np.ndarray
            and a.ndim == b.ndim == 2 and a.shape == b.shape
            and a.dtype == b.dtype == np.bool_ and a.any() and b.any()):
        # Retain every source zero and every queried pixel. No outside region
        # can supply a nearer source; empty inputs keep original EDT behavior.
        region = occupied_slices(a | b)
        return float(ndi.distance_transform_edt(~a[region])[b[region]].min())
    return float(ndi.distance_transform_edt(~a)[b].min())


def _near(target: dict, item: dict, boxes: dict | None, cfg: dict) -> float:
    """mask_distance, or infinity for a pair whose boxes lie farther apart than
    the entry gap in rows or columns (a Euclidean distance is at least its row
    or column separation), so it cannot be within the gap."""
    gap = float(cfg["maximum_entry_gap_px"])
    if boxes is not None:
        a, b = boxes.get(id(target)), boxes.get(id(item))
        if a is not None and b is not None and a[0] is target and b[0] is item:
            a, b = a[1], b[1]
            rows = max(0, b[0] - (a[1] - 1), a[0] - (b[1] - 1))
            columns = max(0, b[2] - (a[3] - 1), a[2] - (b[3] - 1))
            if max(rows, columns) > gap:
                return float("inf")
            return _boxed_distance(target["mask"], item["mask"], a, b)
    return mask_distance(target["mask"], item["mask"])


def _boxed_distance(first: np.ndarray, second: np.ndarray, a: tuple, b: tuple) -> float:
    """mask_distance from the components' tight boxes (y0, y1, x0, x1).

    Shared pixels lie in both boxes; the union of the two tight boxes is the
    tight box of first | second, the region mask_distance crops to.
    """
    y0, y1, x0, x1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    if y0 < y1 and x0 < x1 and np.any(first[y0:y1, x0:x1] & second[y0:y1, x0:x1]):
        return 0.0
    region = (slice(min(a[0], b[0]), max(a[1], b[1])),
              slice(min(a[2], b[2]), max(a[3], b[3])))
    return float(ndi.distance_transform_edt(~first[region])[second[region]].min())


def _entry_box(target: dict, boxes: dict | None) -> tuple | None:
    entry = None if boxes is None else boxes.get(id(target))
    return entry[1] if entry is not None and entry[0] is target else None


def _edge(target: dict, boxes: dict | None, shape: tuple) -> int:
    """The smallest distance from the target to a frame edge; its tight box
    gives the same smallest and largest row and column as its mask."""
    box = _entry_box(target, boxes)
    if box is None:
        yy, xx = occupied_coordinates(target["mask"])
        return min(xx.min(), yy.min(), shape[2] - 1 - xx.max(), shape[1] - 1 - yy.max())
    y0, y1, x0, x1 = box
    return min(x0, y0, shape[2] - 1 - (x1 - 1), shape[1] - 1 - (y1 - 1))


def _blue(target: dict, boxes: dict | None, frame: np.ndarray) -> float:
    # The mask's pixels inside its tight box are all of its pixels, in the
    # same row-major order, so the same values are averaged.
    box = _entry_box(target, boxes)
    if box is None:
        return float(np.mean(frame[target["mask"]] <= -1.5))
    y0, y1, x0, x1 = box
    return float(np.mean(frame[y0:y1, x0:x1][target["mask"][y0:y1, x0:x1]] <= -1.5))


def continuing_target(labels: np.ndarray, lag: np.ndarray, t: int,
                      target: dict, cfg: dict) -> bool:
    for item in identity_objects(labels[t + 1], target["owner"]):
        distance = math.hypot(item["x"] - target["x"],
                              item["y"] - target["y"])
        ratio = item["area"] / max(target["area"], 1)
        red = float(np.mean(lag[t][item["mask"]] >= 1.5))
        if (distance <= float(cfg["maximum_radial_distance_px"])
                and 0.25 <= ratio <= 4.0 and red >= 0.4):
            return True
    return False


def detect_events(labels: np.ndarray, lag: np.ndarray, cfg: dict,
                  source_offset: int) -> list[dict]:
    events = []
    minimum_area = int(cfg["minimum_target_area_px"])
    for t in range(len(labels) - 2):
        current_objects, current_boxes = _frame_objects(labels[t])
        current_boxes = None if current_boxes is None else {
            id(item): (item, box) for item, box in zip(current_objects, current_boxes)}
        for target in current_objects:
            if target["area"] < minimum_area:
                continue
            if _edge(target, current_boxes, labels.shape) <= 2:
                continue
            blue = _blue(target, current_boxes, lag[t])
            if blue < float(cfg["minimum_source_blue_coverage"]):
                continue
            if continuing_target(labels, lag, t, target, cfg):
                continue
            nearby = [item for item in current_objects
                      if item is not target and item["owner"] != target["owner"]
                      and _near(target, item, current_boxes, cfg)
                      <= float(cfg["maximum_entry_gap_px"])]
            resident_options = []
            for resident in nearby:
                next_parts = identity_objects(labels[t + 1], resident["owner"])
                if not next_parts:
                    continue
                following = max(next_parts, key=lambda item: (
                    int(np.count_nonzero(item["mask"] & resident["mask"])),
                    item["area"]))
                gain = following["area"] - resident["area"]
                red_host = float(np.mean(lag[t][following["mask"]] >= 1.5))
                if (gain >= 0.5 * target["area"]
                        and red_host >= float(cfg["minimum_host_red_coverage"])):
                    resident_options.append(((gain, red_host), resident, following))
            if not resident_options:
                continue
            _, resident, first_host = max(resident_options,
                                          key=lambda row: row[0])
            merge_frames = [t + 1]
            hosts = {t + 1: first_host["mask"]}
            previous_host = first_host
            split_t = resident_exit = target_exit = None
            for frame in range(t + 2, min(len(labels), t + 2
                                         + int(cfg["maximum_merge_frames"]))):
                resident_parts = identity_objects(labels[frame], resident["owner"])
                if not resident_parts:
                    break
                resident_current = max(resident_parts, key=lambda item: (
                    int(np.count_nonzero(item["mask"] & previous_host["mask"])),
                    -math.hypot(item["x"] - previous_host["x"],
                                item["y"] - previous_host["y"])))
                if resident_current["area"] >= 0.55 * previous_host["area"]:
                    merge_frames.append(frame)
                    hosts[frame] = resident_current["mask"]
                    previous_host = resident_current
                    continue
                exit_options = []
                for item in binary_objects(labels[frame]):
                    if item["owner"] == resident["owner"]:
                        continue
                    distance = math.hypot(item["x"] - previous_host["x"],
                                          item["y"] - previous_host["y"])
                    ratio = item["area"] / max(target["area"], 1)
                    red = float(np.mean(lag[frame - 1][item["mask"]] >= 1.5))
                    if (distance <= float(cfg["maximum_radial_distance_px"])
                            and 0.25 <= ratio <= 4.0):
                        exit_options.append(((red, -distance), item))
                qualified_exits = [row for row in exit_options
                                   if row[0][0] >= float(
                                       cfg["minimum_exit_red_coverage"])]
                if qualified_exits:
                    # Red first qualifies a destination; distance then selects
                    # which qualified red object continues the lost source.
                    exit_score, departing = min(
                        qualified_exits, key=lambda row: (-row[0][1], -row[0][0]))
                    resident_red = float(np.mean(
                        lag[frame - 1][resident_current["mask"]] >= 1.5))
                    if (resident_red <= float(
                                cfg["maximum_resident_red_coverage"])):
                        split_t, resident_exit, target_exit = (
                            frame, resident_current, departing)
                break
            if split_t is None or target_exit is None or resident_exit is None:
                continue
            events.append({
                "target_identity": target["owner"],
                "resident_identity": resident["owner"],
                "entry_t": t,
                "merge_frames": merge_frames,
                "host_masks": hosts,
                "split_t": split_t,
                "target_entry": target,
                "resident_entry": resident,
                "target_exit": target_exit,
                "resident_exit": resident_exit,
                "source_blue_coverage": blue,
                "host_red_coverage": float(max(row[0][1]
                                                for row in resident_options)),
                "exit_red_coverage": float(np.mean(
                    lag[split_t - 1][target_exit["mask"]] >= 1.5)),
                "resident_red_coverage": float(np.mean(
                    lag[split_t - 1][resident_exit["mask"]] >= 1.5)),
                "source_entry_imagej_frame": t + 1 + source_offset,
                "source_split_imagej_frame": split_t + 1 + source_offset,
            })
    # Keep the strongest event when repeated target/resident windows overlap.
    selected = []
    occupied: dict[int, np.ndarray] = {}
    for event in sorted(events, key=lambda row: (
            row["source_blue_coverage"] + row["exit_red_coverage"]), reverse=True):
        conflict = any(frame in occupied and np.any(
            occupied[frame] & event["host_masks"][frame])
            for frame in event["merge_frames"])
        if conflict:
            continue
        selected.append(event)
        for frame in event["merge_frames"]:
            occupied[frame] = event["host_masks"][frame]
    return sorted(selected, key=lambda row: row["entry_t"])


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("quarantine-stage labels are required")
    labels = load_stack(upstream_dir / f"{params['stem']}.tif")
    trim_dir = Path(params["trim_dir"])
    raw = load_stack(trim_dir / f"{params['stem']}_registered_raw.tif")
    lag = load_stack(trim_dir / f"{params['stem']}_lag01_float.tif")
    candidate = labels.copy()
    events = detect_events(labels, lag, params["merge"],
                           int(params["source_frame_offset"]))
    enabled = bool(params.get("enable_merges", False))
    rows = []
    for index, event in enumerate(events, 1):
        applied = False
        if enabled:
            target = event["target_entry"]
            resident = event["resident_entry"]
            target_exit = event["target_exit"]
            resident_exit = event["resident_exit"]
            span = event["split_t"] - event["entry_t"]
            for frame in event["merge_frames"]:
                alpha = (frame - event["entry_t"]) / span
                target_point = (target["x"] * (1 - alpha)
                                + target_exit["x"] * alpha,
                                target["y"] * (1 - alpha)
                                + target_exit["y"] * alpha)
                resident_point = (resident["x"] * (1 - alpha)
                                  + resident_exit["x"] * alpha,
                                  resident["y"] * (1 - alpha)
                                  + resident_exit["y"] * alpha)
                divided = motion_partition(
                    event["host_masks"][frame], lag, frame, 1.5,
                    int(params["merge"]["minimum_partition_px"]))
                if divided is None:
                    divided = partition_host(
                        raw[frame], event["host_masks"][frame],
                        target_point, resident_point,
                        int(params["merge"]["minimum_partition_px"]))
                target_mask, resident_mask = divided
                candidate[frame][target_mask] = int(event["target_identity"])
                candidate[frame][resident_mask] = int(event["resident_identity"])
            observed_exit = int(target_exit["owner"])
            previous = target_exit["mask"]
            for frame in range(event["split_t"], len(candidate)):
                if frame == event["split_t"]:
                    selected = target_exit
                else:
                    choices = identity_objects(labels[frame], observed_exit)
                    if not choices:
                        break
                    py, px = ndi.center_of_mass(previous)
                    selected = max(choices, key=lambda item: (
                        int(np.count_nonzero(item["mask"] & previous)),
                        -math.hypot(item["x"] - px, item["y"] - py)))
                    if (not np.any(selected["mask"] & previous)
                            and math.hypot(selected["x"] - px,
                                           selected["y"] - py) > 30.0):
                        break
                candidate[frame][selected["mask"]] = int(event["target_identity"])
                previous = selected["mask"]
            applied = True
        rows.append({
            "event_id": f"M{index:04d}",
            "target_identity": event["target_identity"],
            "resident_identity": event["resident_identity"],
            "source_entry_imagej_frame": event["source_entry_imagej_frame"],
            "source_split_imagej_frame": event["source_split_imagej_frame"],
            "merge_source_frames": ";".join(str(frame + 1
                                                  + int(params["source_frame_offset"]))
                                               for frame in event["merge_frames"]),
            "source_blue_coverage": event["source_blue_coverage"],
            "host_red_coverage": event["host_red_coverage"],
            "exit_red_coverage": event["exit_red_coverage"],
            "resident_red_coverage": event["resident_red_coverage"],
            "applied": applied,
        })
    label_path = out.out / f"{params['stem']}.tif"
    event_path = out.out / "merge_events.csv"
    save_stack(label_path, candidate, params["frame_interval_min"])
    pd.DataFrame(rows).to_csv(event_path, index=False)
    return {
        "outputs": {"labels": label_path, "events": event_path},
        "summary": {
            "enabled": enabled, "detected_events": int(len(rows)),
            "applied_events": int(sum(row["applied"] for row in rows)),
            "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
            "foreground_changed_pixels": int(np.count_nonzero(
                (candidate > 0) != (labels > 0))),
        },
    }


