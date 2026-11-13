"""Original measurements; array inputs, no feedback lookup or file I/O."""
from __future__ import annotations
import heapq
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import separable_merge_recovery as physical
from separable_merge_recovery import attach_owners
STRUCTURE = np.ones((3, 3), bool)
OFFSETS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy*dy+dx*dx <= 4]


def measure(labels: np.ndarray, row) -> dict:
    t = int(row.frame) - 1
    small = labels[t] == int(row.small_identity)
    large = labels[t] == int(row.large_identity)
    sy, sx = np.where(small)
    small_area, large_area = len(sy), int(large.sum())
    if not small_area or not large_area:
        return {"valid_current_pair": False}
    y0, y1 = max(0, int(sy.min()) - 12), min(labels.shape[1], int(sy.max()) + 13)
    x0, x1 = max(0, int(sx.min()) - 12), min(labels.shape[2], int(sx.max()) + 13)
    patch_small = small[y0:y1, x0:x1]
    patch_large = large[y0:y1, x0:x1]
    proximity = ndi.binary_dilation(patch_large, iterations=2)
    if not np.any(proximity & patch_small):
        return {"valid_current_pair": False}
    result = {
        "valid_current_pair": True,
        "accepted_small_area": small_area,
        "accepted_large_area": large_area,
        "small_footprint_at_large_edge_2px": round(float(np.mean(proximity[patch_small])), 4),
    }
    for lag in (1, 2, 3, 5):
        fractions = []
        still_small = []
        for other_t in (t - lag, t + lag):
            if 0 <= other_t < len(labels):
                other = labels[other_t, y0:y1, x0:x1]
                fractions.append(float(np.mean(other[patch_small] == int(row.large_identity))))
                still_small.append(float(np.mean(other[patch_small] == int(row.small_identity))))
        result[f"large_occupancy_lag_{lag}"] = round(max(fractions, default=np.nan), 4)
        result[f"small_occupancy_lag_{lag}"] = round(max(still_small, default=np.nan), 4)
    return result

def scan(movies: dict, source: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for well, group in source.groupby("well", sort=True):
        labels = movies[well]["embedded_labels"]
        for pair in group.itertuples(index=False):
            rows.append({**pair._asdict(), **measure(labels, pair)})
        print(well, len(group), "pairs", flush=True)

    return pd.DataFrame(rows)
