"""Keep two identity owners through a connected merge and restore them at split."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from project_paths import find_project_root


TUNING = Path(__file__).resolve().parents[1]
ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import Config, load_stack, save_stack  # noqa: E402
from pipeline import _pinned_paths  # noqa: E402

STRUCTURE = np.ones((3, 3), np.uint8)


def components(frame: np.ndarray, identity: int | None = None) -> list[dict]:
    selector = frame > 0 if identity is None else frame == identity
    labels, count = ndi.label(selector, structure=STRUCTURE)
    result: list[dict] = []
    for component in range(1, count + 1):
        mask = labels == component
        yy, xx = np.nonzero(mask)
        result.append({"mask": mask, "area": int(mask.sum()),
                       "x": float(xx.mean()), "y": float(yy.mean()),
                       "owner": int(np.bincount(frame[mask]).argmax())})
    return result


def mask_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(a & b):
        return 0.0
    return float(ndi.distance_transform_edt(~a)[b].min())


def select_near(frame: np.ndarray, previous: np.ndarray,
                exclude: set[int] | None = None) -> dict | None:
    excluded = exclude or set()
    py, px = ndi.center_of_mass(previous)
    choices = []
    for item in components(frame):
        if item["owner"] in excluded:
            continue
        overlap = int(np.count_nonzero(item["mask"] & previous))
        distance = math.hypot(item["x"] - px, item["y"] - py)
        choices.append(((overlap, -distance, item["area"]), item))
    return max(choices, key=lambda row: row[0])[1] if choices else None


def nearest_peak(raw: np.ndarray, host: np.ndarray,
                 point: tuple[float, float], radius: float = 12.0) -> tuple[int, int]:
    x, y = point
    yy, xx = np.nonzero(host)
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    if not np.any(inside):
        index = int(np.argmin((xx - x) ** 2 + (yy - y) ** 2))
        return int(yy[index]), int(xx[index])
    cy, cx = yy[inside], xx[inside]
    values = ndi.gaussian_filter(raw.astype(np.float32), 1.0)[cy, cx]
    index = int(np.argmax(values))
    return int(cy[index]), int(cx[index])


def partition_host(raw: np.ndarray, host: np.ndarray,
                   target_point: tuple[float, float],
                   resident_point: tuple[float, float],
                   minimum_px: int) -> tuple[np.ndarray, np.ndarray]:
    target_seed = nearest_peak(raw, host, target_point)
    resident_seed = nearest_peak(raw, host, resident_point)
    if target_seed == resident_seed:
        yy, xx = np.nonzero(host)
        distances = (xx - resident_point[0]) ** 2 + (yy - resident_point[1]) ** 2
        index = int(np.argmin(distances))
        resident_seed = int(yy[index]), int(xx[index])
    markers = np.zeros(host.shape, np.int16)
    markers[target_seed] = 1
    markers[resident_seed] = 2
    smooth = ndi.gaussian_filter(raw.astype(np.float32), 1.1)
    divided = watershed(-smooth, markers=markers, mask=host, compactness=0.01)
    target = divided == 1
    resident = divided == 2
    if target.sum() < minimum_px or resident.sum() < minimum_px:
        yy, xx = np.indices(host.shape)
        td = (xx - target_point[0]) ** 2 + (yy - target_point[1]) ** 2
        rd = (xx - resident_point[0]) ** 2 + (yy - resident_point[1]) ** 2
        target = host & (td <= rd)
        resident = host & ~target
    return target, resident


def motion_partition(host: np.ndarray, lag: np.ndarray, frame: int,
                     threshold: float, minimum_px: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Use red arrival and blue departure as the fast cell's merged territory."""
    red_arrival = lag[frame - 1] >= threshold
    blue_departure = (lag[frame] <= -threshold
                      if frame < len(lag) else np.zeros(host.shape, bool))
    evidence = host & (red_arrival | blue_departure)
    parts, count = ndi.label(evidence, structure=STRUCTURE)
    if count == 0:
        return None
    areas = np.bincount(parts.ravel())
    target = parts == int(np.argmax(areas[1:]) + 1)
    resident = host & ~target
    if target.sum() < minimum_px or resident.sum() < minimum_px:
        return None
    return target, resident

