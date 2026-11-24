"""Replace unsteady refined outlines with a morph of the cell's good outlines either side.

After a refiner (for example ``lv200_red``) a cell's outline can go wrong in single
frames in two ways, and both put one-frame dips into its total-light trace:

    fallback   the refiner did not outline the cell in that frame, so it kept the
               small tracking outline
    dip        the refined outline shrank to under ``dip_ratio`` of the cell's
               median refined area in the ``window`` frames either side
    surge      it grew past 1 / ``dip_ratio`` of that median (usually swallowing a
               neighbour's light for one frame)

Each such cell-frame is replaced by the signed-distance morph between the nearest
good outline before and after it (each at most ``max_gap`` frames away), weighted by
time, or by the one good outline if only one side has one. This is the "morph between
outlines" region from development/native_mask_motion_handoff_tuning (round 1: held-out
outline overlap 0.886 against 0.874 for the nearest outline). The cell's own tracking
pixels are always kept, and no pixel another cell owns in that frame is taken, so
identities never change and no pixel moves between cells.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from .contract import sha256_of

METHOD_VERSION = "2026-11-13-signed-distance-outline-morph-v2"
MAX_GAP = 5
DIP_RATIO = 0.5
WINDOW = 3
PAD_PX = 18


def signed_distance(mask: np.ndarray) -> np.ndarray:
    return ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)


def morph(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    """Pixels inside the time-weighted signed-distance blend of two outlines (weight 0 = left)."""
    union = left | right
    if not union.any():
        return np.zeros_like(left)
    yy, xx = np.nonzero(union)
    y = slice(max(0, int(yy.min()) - PAD_PX), min(left.shape[0], int(yy.max()) + PAD_PX + 1))
    x = slice(max(0, int(xx.min()) - PAD_PX), min(left.shape[1], int(xx.max()) + PAD_PX + 1))
    out = np.zeros_like(left)
    out[y, x] = ((1 - weight) * signed_distance(left[y, x]) + weight * signed_distance(right[y, x])) > 0
    return out


def unsteady(refined: np.ndarray, tracked: np.ndarray, *, dip_ratio: float = DIP_RATIO,
             window: int = WINDOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame x identity: present, fallback (refined == tracking outline) and dip flags."""
    frames, n = len(refined), int(max(refined.max(), tracked.max())) + 1
    area = np.zeros((frames, n))
    present = np.zeros((frames, n), bool)
    fallback = np.zeros((frames, n), bool)
    for t in range(frames):
        area[t] = np.bincount(refined[t].ravel(), minlength=n)
        own = np.bincount(tracked[t].ravel(), minlength=n)
        present[t] = own > 0
        kept = np.bincount(refined[t][(refined[t] == tracked[t]) & (tracked[t] > 0)].ravel(), minlength=n)
        fallback[t] = present[t] & (kept == own) & (area[t] == own)
    present[:, 0] = fallback[:, 0] = False
    dip = np.zeros_like(present)
    for cell in np.flatnonzero(present.any(0)):
        good = present[:, cell] & ~fallback[:, cell]
        for t in np.flatnonzero(good):
            lo, hi = max(0, t - window), min(frames, t + window + 1)
            near = [area[u, cell] for u in range(lo, hi) if u != t and good[u]]
            if len(near) >= 2 and not dip_ratio * np.median(near) <= area[t, cell] <= np.median(near) / dip_ratio:
                dip[t, cell] = True
    return present, fallback, dip


def smooth(refined: np.ndarray, tracked: np.ndarray, *, max_gap: int = MAX_GAP,
           dip_ratio: float = DIP_RATIO, window: int = WINDOW) -> tuple[np.ndarray, list[dict]]:
    """Return (labels with unsteady cell-frames morphed, one audit row per replaced cell-frame)."""
    refined, tracked = np.asarray(refined), np.asarray(tracked)
    if refined.shape != tracked.shape or refined.ndim != 3:
        raise ValueError("refined and tracking labels must be matching (time, height, width) stacks")
    present, fallback, dip = unsteady(refined, tracked, dip_ratio=dip_ratio, window=window)
    bad = fallback | dip
    good = present & ~bad
    out = refined.copy()
    audit = []
    for t, cell in zip(*np.nonzero(bad)):
        before = [u for u in range(t - 1, max(-1, t - max_gap - 1), -1) if good[u, cell]]
        after = [u for u in range(t + 1, min(len(refined), t + max_gap + 1)) if good[u, cell]]
        if not before and not after:
            continue
        if before and after:
            left, right = before[0], after[0]
            region = morph(refined[left] == cell, refined[right] == cell, (t - left) / (right - left))
        else:
            left = right = (before or after)[0]
            region = refined[left] == cell
        frame = out[t]
        free = ((frame == 0) & ((refined[t] == 0) | (refined[t] == cell))) | (frame == cell)
        region = (region & free & ~((tracked[t] > 0) & (tracked[t] != cell))) | (tracked[t] == cell)
        old_area = int((frame == cell).sum())
        frame[frame == cell] = 0
        frame[region] = cell
        neighbour_area = np.mean([(refined[u] == cell).sum() for u in (left, right)])
        reason = "fallback" if fallback[t, cell] else ("dip" if old_area < neighbour_area else "surge")
        audit.append({"frame_index": int(t), "identity": int(cell), "reason": reason,
                      "left_frame": int(left), "right_frame": int(right),
                      "area_before_px": old_area, "area_after_px": int(region.sum())})
    changed = out != refined
    if np.any(changed & (out > 0) & (tracked > 0) & (tracked != out)):
        raise RuntimeError("outline morph gave a cell a pixel another cell is tracked in")
    if np.any(changed & (refined > 0) & (out > 0) & (out != refined)):
        raise RuntimeError("outline morph took a pixel from another cell's outline")
    for t in range(len(out)):
        if set(np.unique(out[t])) != set(np.unique(refined[t])):
            raise RuntimeError(f"outline morph changed the identity set in frame {t}")
    return out, audit


def run(refined_labels, motion_labels, *, output_dir, max_gap: int = MAX_GAP,
        dip_ratio: float = DIP_RATIO, window: int = WINDOW) -> dict:
    """Write morphed_labels.tif, morph_audit.csv and morph_settings.json; inputs are read-only."""
    destination = Path(output_dir)
    outputs = {"morphed_labels": destination / "morphed_labels.tif",
               "morph_audit": destination / "morph_audit.csv",
               "settings": destination / "morph_settings.json"}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"outline morph already exists in {destination}")
    paths = {"refined_labels": Path(refined_labels), "motion_labels": Path(motion_labels)}
    hashes = {key: sha256_of(path) for key, path in paths.items()}
    refined = np.asarray(tifffile.imread(paths["refined_labels"]))
    tracked = np.asarray(tifffile.imread(paths["motion_labels"]))
    morphed, audit = smooth(refined, tracked, max_gap=max_gap, dip_ratio=dip_ratio, window=window)
    destination.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(outputs["morphed_labels"], morphed, compression="zlib", photometric="minisblack")
    fields = ("frame_index", "identity", "reason", "left_frame", "right_frame", "area_before_px", "area_after_px")
    with outputs["morph_audit"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    present, fallback, dip = unsteady(refined, tracked, dip_ratio=dip_ratio, window=window)
    settings = {"method_version": METHOD_VERSION, "max_gap": max_gap, "dip_ratio": dip_ratio, "window": window,
                "cell_frames": int(present.sum()), "fallback_cell_frames": int(fallback.sum()),
                "dip_or_surge_cell_frames": int(dip.sum()), "morphed_cell_frames": len(audit),
                "left_unchanged_no_good_neighbour": int((fallback | dip).sum()) - len(audit),
                "inputs": {k: str(p.resolve()) for k, p in paths.items()}, "inputs_sha256": hashes,
                "outputs_sha256": {"morphed_labels": sha256_of(outputs["morphed_labels"]),
                                   "morph_audit": sha256_of(outputs["morph_audit"])}}
    outputs["settings"].write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return {"outputs": {k: str(p) for k, p in outputs.items()}, "settings": settings}
