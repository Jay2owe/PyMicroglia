"""Optional post-tracking photon measurements in short, internal cell gaps.

The accepted identity table and sparse labels are read-only evidence. This
action writes separate proposal rows; it never changes tracking or an observed
measurement. The nearest endpoint outline is the retained orange method from
the native mask-to-Motion handoff tuning round.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

METHOD_VERSION = "2026-09-22-nearest-outline-gap-measurement-v1"
MAX_TESTED_GAP_FRAMES = 15

AUDIT_FIELDS = ("gap_id", "identity", "first_frame", "last_frame",
                "left_frame", "right_frame", "gap_frames", "eligible",
                "reason", "endpoint_iou", "centroid_shift_px",
                "left_area_px", "right_area_px")
ADDITION_FIELDS = ("identity", "frame_index", "hours", "corrected_mean",
                   "signal_mean", "background_median", "area_px",
                   "measurement_kind", "method", "gap_id", "left_frame",
                   "right_frame")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(anchor: np.ndarray, components: np.ndarray,
             identities: list[int]) -> dict[int, int]:
    """Recover the native component for each accepted sparse identity."""
    present = [identity for identity in identities if np.any(anchor == identity)]
    candidate = np.unique(components)
    candidate = candidate[candidate > 0]
    if not present or not len(candidate):
        return {}
    stride = int(components.max()) + 1
    joint = np.bincount((anchor.astype(np.int64) * stride + components).ravel(),
                        minlength=(int(anchor.max()) + 1) * stride)
    overlap = joint.reshape(-1, stride)[np.ix_(present, candidate)]
    anchor_areas = np.bincount(anchor.ravel().astype(np.int64),
                               minlength=int(anchor.max()) + 1)[present]
    support = overlap / np.maximum(anchor_areas[:, None], 1)
    row, col = linear_sum_assignment(-support)
    return {int(present[r]): int(candidate[c]) for r, c in zip(row, col)
            if int(overlap[r, c]) >= 5 and float(support[r, c]) >= 0.08}


def _endpoint_mask(row: dict[str, str], *, sparse: np.ndarray,
                   cells: np.ndarray, anchors: dict[int, int],
                   identities: list[int], cache: dict[tuple[int, int], dict[int, int]]
                   ) -> np.ndarray:
    t, identity = int(row["frame_index"]), int(row["identity"])
    anchor_t = int(row["anchor_frame"])
    anchor = sparse[anchors[anchor_t]]
    if row["anchor_kind"] == "accepted":
        mask = anchor == identity
    elif row["anchor_kind"] == "matched":
        key = t, anchor_t
        if key not in cache:
            cache[key] = _matches(anchor, cells[t], identities)
        component = cache[key].get(identity, 0)
        mask = cells[t] == component if component else np.zeros(cells.shape[1:], bool)
    else:
        raise ValueError(f"frame {t}, identity {identity}: endpoint is not observed")
    if not mask.any():
        raise ValueError(f"frame {t}, identity {identity}: observed endpoint has no outline")
    return mask


def _classify(left: np.ndarray, right: np.ndarray,
              intermediate: np.ndarray) -> tuple[str, float, float]:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    iou = intersection / max(union, 1)
    ly, lx = np.nonzero(left)
    ry, rx = np.nonzero(right)
    shift = float(np.linalg.norm(np.array([ly.mean(), lx.mean()]) -
                                 np.array([ry.mean(), rx.mean()])))
    areas = int(left.sum()), int(right.sum())
    if iou < 0.25 or shift > 8.0 or max(areas) / max(min(areas), 1) > 2.5:
        return "unstable_endpoints", iou, shift
    corridor = ndi.binary_dilation(left | right, iterations=2)
    if np.any(intermediate[:, corridor] > 0):
        return "native_component_in_corridor", iou, shift
    return "eligible", iou, shift


def _measure(raw: np.ndarray, occupancy: np.ndarray, mask: np.ndarray
             ) -> tuple[float, float, float, int]:
    yy, xx = np.nonzero(mask)
    bounds = (slice(int(yy.min()), int(yy.max()) + 1),
              slice(int(xx.min()), int(xx.max()) + 1))
    values = raw[bounds][mask[bounds]]
    distance = ndi.distance_transform_edt(~mask)
    ring = (distance > 4) & (distance <= 12) & ~occupancy
    if not np.any(ring):
        raise ValueError("no unoccupied local background pixels")
    signal = float(values.mean())
    background = float(np.median(raw[ring]))
    return signal - background, signal, background, int(values.size)


def measure_missing_gaps(source, *, accepted_sparse_labels,
                         cells_native, photons_native, merge_events,
                         frame_interval_h: float, excluded_identities: Sequence[int],
                         output_dir, max_gap_frames: int = 5) -> dict:
    """Write measurement-only proposals for stable disappear-and-reappear gaps.

    Invoking this action is the opt-in. ``max_gap_frames`` is a maximum *number
    of missing native frames*, not elapsed hours or anchor spacing. The method
    is fixed to the retained nearest-outline proposal; all other eligibility
    thresholds and the photon convention are fixed.
    """
    if isinstance(max_gap_frames, bool) or not isinstance(max_gap_frames, int) \
            or not 1 <= max_gap_frames <= MAX_TESTED_GAP_FRAMES:
        raise ValueError("max_gap_frames must be a whole number from 1 to 15")
    try:
        interval_h = float(frame_interval_h)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_interval_h must be positive") from exc
    if isinstance(frame_interval_h, bool) or not np.isfinite(interval_h) or interval_h <= 0:
        raise ValueError("frame_interval_h must be positive")
    if not isinstance(excluded_identities, Sequence) or isinstance(
            excluded_identities, (str, bytes)):
        raise ValueError("excluded_identities must be an explicit list, even if empty")
    blocked = {int(identity) for identity in excluded_identities}
    paths = {"baseline_cell_frame": Path(source),
             "accepted_sparse_labels": Path(accepted_sparse_labels),
             "cells_native": Path(cells_native),
             "photons_native": Path(photons_native),
             "merge_events": Path(merge_events)}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if "DISPLAY_ONLY" in paths["photons_native"].name.upper():
        raise ValueError("display-only photons cannot be measured")
    destination = Path(output_dir)
    outputs = {"gap_audit": destination / "gap_audit.csv",
               "measurement_additions": destination / "measurement_additions.csv",
               "settings": destination / "settings.json"}
    if any(path.exists() for path in outputs.values()):
        raise ValueError(f"gap measurement output already exists in {destination}")

    baseline = _rows(paths["baseline_cell_frame"])
    if not baseline:
        raise ValueError("baseline_cell_frame is empty")
    events = _rows(paths["merge_events"])
    if events and not {"identity_a", "identity_b"} <= set(events[0]):
        raise ValueError("merge_events must have identity_a and identity_b columns")
    for event in events:
        for key in ("identity_a", "identity_b"):
            if event.get(key):
                blocked.add(int(event[key]))
    cells = tifffile.imread(paths["cells_native"])
    sparse = tifffile.imread(paths["accepted_sparse_labels"])
    photons = tifffile.imread(paths["photons_native"]).astype(np.float32) * 1000
    if cells.shape != photons.shape or cells.ndim != 3 or sparse.ndim != 3 \
            or cells.shape[1:] != sparse.shape[1:]:
        raise ValueError("native cells, original photons and sparse labels do not align")
    by_identity: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in baseline:
        by_identity[int(row["identity"])].append(row)
    identities = sorted(by_identity)
    exact = sorted({int(row["frame_index"]) for row in baseline
                    if int(row["frame_index"]) == int(row["anchor_frame"])})
    if len(exact) != len(sparse):
        raise ValueError("accepted sparse labels do not match exact anchor frames")
    anchors = {frame: index for index, frame in enumerate(exact)}
    if any(int(row["anchor_frame"]) not in anchors for row in baseline):
        raise ValueError("baseline refers to an unknown accepted anchor frame")
    cache: dict[tuple[int, int], dict[int, int]] = {}
    audit: list[dict] = []
    additions: list[dict] = []
    for identity, series in by_identity.items():
        series.sort(key=lambda row: int(row["frame_index"]))
        frames = [int(row["frame_index"]) for row in series]
        if frames != list(range(frames[0], frames[-1] + 1)) or frames[-1] >= len(cells):
            raise ValueError(f"identity {identity}: baseline frames are not contiguous native frames")
        index = 0
        while index < len(series):
            if series[index]["anchor_kind"] != "unmatched":
                index += 1
                continue
            first_index = index
            while index < len(series) and series[index]["anchor_kind"] == "unmatched":
                index += 1
            if first_index == 0 or index == len(series):
                continue  # A missing beginning or end is not reappearance.
            first, last = frames[first_index], frames[index - 1]
            left_row, right_row = series[first_index - 1], series[index]
            left_t, right_t = frames[first_index - 1], frames[index]
            if right_t - left_t != last - first + 2:
                raise ValueError(f"identity {identity}: nonconsecutive gap endpoints")
            left = _endpoint_mask(left_row, sparse=sparse, cells=cells,
                                  anchors=anchors, identities=identities, cache=cache)
            right = _endpoint_mask(right_row, sparse=sparse, cells=cells,
                                   anchors=anchors, identities=identities, cache=cache)
            length = last - first + 1
            if identity in blocked:
                reason, iou, shift = "merged_identity", 0.0, 0.0
            elif length > max_gap_frames:
                reason, iou, shift = "gap_too_long", 0.0, 0.0
            else:
                reason, iou, shift = _classify(left, right, cells[first:last + 1])
            gap_id = f"G{len(audit):05d}"
            audit.append({"gap_id": gap_id, "identity": identity,
                          "first_frame": first, "last_frame": last,
                          "left_frame": left_t, "right_frame": right_t,
                          "gap_frames": length, "eligible": int(reason == "eligible"),
                          "reason": reason, "endpoint_iou": round(iou, 6),
                          "centroid_shift_px": round(shift, 6),
                          "left_area_px": int(left.sum()),
                          "right_area_px": int(right.sum())})
            if reason != "eligible":
                continue
            for t in range(first, last + 1):
                weight = (t - left_t) / (right_t - left_t)
                region = left if weight <= 0.5 else right
                if np.any(region & (cells[t] > 0)):
                    raise ValueError(f"identity {identity}, frame {t}: region overlaps a native cell")
                corrected, signal, background, area = _measure(
                    photons[t], cells[t] > 0, region)
                additions.append({"identity": identity, "frame_index": t,
                                  "hours": round(t * interval_h, 9),
                                  "corrected_mean": round(corrected, 6),
                                  "signal_mean": round(signal, 6),
                                  "background_median": round(background, 6),
                                  "area_px": area,
                                  "measurement_kind": "measurement_only",
                                  "method": "nearest", "gap_id": gap_id,
                                  "left_frame": left_t, "right_frame": right_t})
    destination.mkdir(parents=True, exist_ok=True)
    _write_rows(outputs["gap_audit"], audit, AUDIT_FIELDS)
    _write_rows(outputs["measurement_additions"], additions, ADDITION_FIELDS)
    outputs["settings"].write_text(json.dumps({
        "method_version": METHOD_VERSION, "method": "nearest",
        "max_gap_frames": max_gap_frames, "frame_interval_h": interval_h,
        "excluded_identities": sorted(blocked),
        "inputs_sha256": {key: _sha256(path) for key, path in paths.items()},
        "scope": "measurement_only_after_tracking",
    }, indent=2) + "\n", encoding="utf-8")
    return {"outputs": {key: str(path) for key, path in outputs.items()},
            "gap_runs": len(audit),
            "eligible_runs": sum(row["eligible"] for row in audit),
            "measurement_only_frames": len(additions),
            "max_gap_frames": max_gap_frames}
