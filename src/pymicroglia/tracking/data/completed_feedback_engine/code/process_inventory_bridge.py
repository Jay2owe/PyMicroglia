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


def component(frame: np.ndarray, name: int, point: pd.Series) -> np.ndarray | None:
    y, x = round(float(point.y)), round(float(point.x))
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return None
    if int(frame[y, x]) != name:
        return None
    parts, _ = ndi.label(frame == name, STRUCTURE)
    return parts == parts[y, x]

def widest_bridge(raw: np.ndarray, labels: np.ndarray, small: np.ndarray,
                  large: np.ndarray, contact_small: np.ndarray,
                  contact_large: np.ndarray, gap: float, radius: float) -> float:
    """Highest minimum raw value along an 8-connected short gap crossing."""
    pad = int(np.ceil(max(4., gap + radius + 2.)))
    y0 = max(0, int(min(contact_small[0], contact_large[0])) - pad)
    y1 = min(raw.shape[0], int(max(contact_small[0], contact_large[0])) + pad + 1)
    x0 = max(0, int(min(contact_small[1], contact_large[1])) - pad)
    x1 = min(raw.shape[1], int(max(contact_small[1], contact_large[1])) + pad + 1)
    local = raw[y0:y1, x0:x1].astype(float)
    yy, xx = np.indices(local.shape)
    coords = np.stack((yy + y0, xx + x0), axis=-1).astype(float)
    delta = contact_large.astype(float) - contact_small.astype(float)
    unit = delta / max(float(np.linalg.norm(delta)), 1e-6)
    projection = np.clip(np.sum((coords - contact_small) * unit, axis=-1), 0., gap)
    nearest = contact_small + projection[..., None] * unit
    corridor = np.linalg.norm(coords - nearest, axis=-1) <= radius
    sm = small[y0:y1, x0:x1]
    lg = large[y0:y1, x0:x1]
    forbidden = (labels[y0:y1, x0:x1] > 0) & ~(sm | lg)
    allowed = corridor & ~forbidden
    # Endpoints are the facing boundaries, not distant bright pixels in either outline.
    ds = np.linalg.norm(coords - contact_small, axis=-1)
    dl = np.linalg.norm(coords - contact_large, axis=-1)
    starts = np.argwhere(allowed & sm & (ds <= max(1.5, radius)))
    goals = allowed & lg & (dl <= max(1.5, radius))
    if not len(starts) or not np.any(goals):
        return np.nan
    capacity = np.full(local.shape, -np.inf)
    queue = []
    for y, x in starts:
        capacity[y, x] = np.inf
        heapq.heappush(queue, (-np.inf, int(y), int(x)))
    while queue:
        neg, y, x = heapq.heappop(queue)
        value = -neg
        if value < capacity[y, x]:
            continue
        if goals[y, x]:
            return value
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < local.shape[0] and 0 <= nx < local.shape[1] and allowed[ny, nx]):
                    continue
                new = min(value, local[ny, nx])
                if new > capacity[ny, nx]:
                    capacity[ny, nx] = new
                    heapq.heappush(queue, (-new, ny, nx))
    return np.nan

def measure(frame: np.ndarray, raw: np.ndarray, pair, lp: pd.Series, sp: pd.Series) -> dict:
    large = component(frame, int(pair.large_identity), lp)
    small = component(frame, int(pair.small_identity), sp)
    if large is None or small is None or np.any(large & small):
        return {}
    large_edge = np.argwhere(large & ~ndi.binary_erosion(large, structure=STRUCTURE))
    small_edge = np.argwhere(small & ~ndi.binary_erosion(small, structure=STRUCTURE))
    if not len(large_edge) or not len(small_edge):
        return {}
    gaps, indexes = cKDTree(large_edge).query(small_edge)
    which = int(np.argmin(gaps))
    gap = float(gaps[which])
    if gap > 6.:  # Frozen inventory targets nearby pairs; long paths are another class.
        return {}
    contact_small = small_edge[which]
    contact_large = large_edge[indexes[which]]
    q95 = float(np.percentile(raw[large], 95))
    # Use the same local median background convention as the prior bridge feature.
    y0 = max(0, int(min(contact_small[0], contact_large[0])) - 6)
    y1 = min(raw.shape[0], int(max(contact_small[0], contact_large[0])) + 7)
    x0 = max(0, int(min(contact_small[1], contact_large[1])) - 6)
    x1 = min(raw.shape[1], int(max(contact_small[1], contact_large[1])) + 7)
    bg = raw[y0:y1, x0:x1][frame[y0:y1, x0:x1] == 0]
    background = float(np.median(bg)) if len(bg) else 0.
    result = {"current_gap_px": round(gap, 3)}
    for radius in (2., 4.):
        value = widest_bridge(raw, frame, small, large, contact_small, contact_large, gap, radius)
        result[f"geodesic_bridge_{int(radius)}"] = round((value - background) / max(q95 - background, 1.), 4)
    return result

def scan(movies: dict, source: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for well, group in source.groupby("well", sort=True):
        labels = movies[well]["union_labels"]
        raw = movies[well]["raw"]
        points = movies[well]["points"].set_index(["track_id", "frame"])
        for pair in group.itertuples(index=False):
            if pd.isna(pair.large_track) or pd.isna(pair.small_track):
                continue
            t = int(pair.frame) - 1
            lk, sk = (int(pair.large_track), t), (int(pair.small_track), t)
            if lk not in points.index or sk not in points.index:
                continue
            lp, sp = points.loc[lk], points.loc[sk]
            if isinstance(lp, pd.DataFrame) or isinstance(sp, pd.DataFrame):
                continue
            measured = measure(labels[t], raw[t], pair, lp, sp)
            if measured:
                rows.append({**pair._asdict(), **measured})
        print("measured", well, sum(r["well"] == well for r in rows), flush=True)

    return pd.DataFrame(rows)
