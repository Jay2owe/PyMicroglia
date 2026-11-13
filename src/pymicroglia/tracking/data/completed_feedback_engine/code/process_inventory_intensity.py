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


def measure(labels: np.ndarray, raw: np.ndarray, pair) -> dict:
    t = int(pair.frame) - 1
    large = labels[t] == int(pair.large_identity)
    small = labels[t] == int(pair.small_identity)
    if not np.any(large) or not np.any(small):
        return {}
    sy, sx = np.where(small)
    y0, y1 = max(0, sy.min()-6), min(labels.shape[1], sy.max()+7)
    x0, x1 = max(0, sx.min()-6), min(labels.shape[2], sx.max()+7)
    large_patch = large[y0:y1,x0:x1]
    small_patch = small[y0:y1,x0:x1]
    distance, inds = ndi.distance_transform_edt(~large_patch,return_indices=True)
    border_distance = distance[small_patch]
    shortest = float(border_distance.min())
    close = small_patch & (distance <= shortest+.1)
    nearest_y = inds[0][close]
    nearest_x = inds[1][close]
    contact_y, contact_x = np.where(close)
    bridge_y = ((contact_y+nearest_y)/2).round().astype(int)
    bridge_x = ((contact_x+nearest_x)/2).round().astype(int)
    signal = raw[t]
    small_signal = signal[small]
    large_signal = signal[large]
    local = signal[y0:y1,x0:x1]
    outer = local[(~small_patch)&(~large_patch)]
    background = float(np.percentile(outer,50)) if len(outer) else 0.
    small_q95 = float(np.percentile(small_signal,95))
    large_q95 = float(np.percentile(large_signal,95))
    bridge = local[bridge_y,bridge_x].astype(float)
    return {
        "boundary_gap_px": round(shortest,3),
        "contact_count_at_nearest_gap": len(contact_y),
        "small_q95_to_large_q95": round(small_q95/max(large_q95,1.),3),
        "small_median_to_large_median": round(float(np.median(small_signal))/max(float(np.median(large_signal)),1.),3),
        "small_excess_q95_to_large_excess_q95": round((small_q95-background)/max(large_q95-background,1.),3),
        "boundary_bridge_excess_to_large_q95": round(float(np.percentile(bridge,75)-background)/max(large_q95-background,1.),3),
    }

def scan(movies: dict, source: pd.DataFrame | None = None) -> pd.DataFrame:
    rows=[]
    for well, group in source.groupby("well",sort=True):
        labels=movies[well]["embedded_labels"]
        raw=movies[well]["raw"]
        for pair in group.itertuples(index=False):
            row=pair._asdict()
            if bool(row.get("valid_current_pair",False)):
                row.update(measure(labels,raw,pair))
            rows.append(row)
        print(well,len(group),flush=True)

    return pd.DataFrame(rows)
