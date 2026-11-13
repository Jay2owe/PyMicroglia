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


def geometry(frame: np.ndarray, name: int, y: int, x: int) -> dict:
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return {}
    if int(frame[y, x]) != name:
        return {}
    parts, _ = ndi.label(frame == name, STRUCTURE)
    mask = parts == parts[y, x]
    ys, xs = np.where(mask)
    area = len(xs)
    if area < 2:
        return {}
    xy = np.column_stack((xs, ys)).astype(float)
    covariance = np.cov(xy, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    perimeter = int(np.count_nonzero(mask & ~ndi.binary_erosion(mask, structure=STRUCTURE)))
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    return {"local_fragment_area": area,
            "fragment_elongation": round(float(np.sqrt(max(eigenvalues[-1], 0) /
                                                 max(eigenvalues[0], .05))), 3),
            "fragment_box_fill": round(area / (width * height), 3),
            "fragment_boundary_fraction": round(perimeter / area, 3),
            "fragment_width": width, "fragment_height": height,
            "fragment_centroid_distance_to_track": round(float(np.hypot(
                xs.mean() - x, ys.mean() - y)), 3)}

def scan(movies: dict, source: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    selected = source[source.valid_current_pair == True].copy()  # noqa: E712
    for well, group in selected.groupby("well", sort=True):
        labels = movies[well]["embedded_labels"]
        points = movies[well]["points"].set_index(["track_id", "frame"])
        for pair in group.itertuples(index=False):
            row = pair._asdict()
            key = (int(pair.small_track), int(pair.frame) - 1) if pd.notna(pair.small_track) else None
            if key is not None and key in points.index:
                point = points.loc[key]
                if isinstance(point, pd.Series):
                    row.update(geometry(labels[key[1]], int(pair.small_identity),
                                        round(float(point.y)), round(float(point.x))))
            rows.append(row)
        print(well, len(group), flush=True)

    return pd.DataFrame(rows)
