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


def _component(frame: np.ndarray, name: int, point: pd.Series) -> np.ndarray | None:
    y, x = round(float(point.y)), round(float(point.x))
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return None
    if int(frame[y, x]) != name:
        return None
    parts, _ = ndi.label(frame == name, STRUCTURE)
    return parts == parts[y, x]

def _axis(mask: np.ndarray, near: np.ndarray, radius: float) -> np.ndarray | None:
    y0 = max(0, int(np.floor(near[0] - radius)))
    y1 = min(mask.shape[0], int(np.ceil(near[0] + radius + 1)))
    x0 = max(0, int(np.floor(near[1] - radius)))
    x1 = min(mask.shape[1], int(np.ceil(near[1] + radius + 1)))
    coords = np.argwhere(mask[y0:y1, x0:x1]).astype(float)
    if len(coords) < 5:
        return None
    coords += [y0, x0]
    coords = coords[np.linalg.norm(coords-near, axis=1) <= radius]
    if len(coords) < 5:
        return None
    centered = coords - coords.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(centered.T @ centered / len(coords))
    if eigvals[-1] < .5 or eigvals[-1] < 1.2 * max(eigvals[0], .01):
        return None
    return eigvecs[:, -1]

def interface(frame: np.ndarray, large_name: int, small_name: int,
              large_point: pd.Series, small_point: pd.Series) -> dict:
    large = _component(frame, large_name, large_point)
    small = _component(frame, small_name, small_point)
    if large is None or small is None:
        return {}
    large_edge = np.argwhere(large & ~ndi.binary_erosion(large, structure=STRUCTURE))
    small_edge = np.argwhere(small & ~ndi.binary_erosion(small, structure=STRUCTURE))
    if not len(large_edge) or not len(small_edge):
        return {}
    gap, index = cKDTree(large_edge).query(small_edge)
    which = int(np.argmin(gap))
    small_contact = small_edge[which].astype(float)
    large_contact = large_edge[index[which]].astype(float)
    direction = large_contact - small_contact
    if np.linalg.norm(direction) == 0:
        return {}
    direction /= np.linalg.norm(direction)
    small_radius = np.sqrt(float(small.sum()) / np.pi)
    large_radius = np.sqrt(float(large.sum()) / np.pi)
    local_radius = max(6., 2.0 * small_radius)
    small_axis = _axis(small, small_contact, local_radius)
    large_axis = _axis(large, large_contact, local_radius)
    small_track_yx = np.array([float(small_point.y), float(small_point.x)])
    large_track_yx = np.array([float(large_point.y), float(large_point.x)])
    return {
        "interface_gap_px": round(float(gap[which]), 3),
        "small_contact_to_track_radii": round(float(np.linalg.norm(
            small_contact-small_track_yx) / max(small_radius, 1.)), 3),
        "large_contact_to_track_radii": round(float(np.linalg.norm(
            large_contact-large_track_yx) / max(large_radius, 1.)), 3),
        "small_faces_gap": round(abs(float(small_axis @ direction)), 3)
            if small_axis is not None else np.nan,
        "large_faces_gap": round(abs(float(large_axis @ direction)), 3)
            if large_axis is not None else np.nan,
        "axes_parallel": round(abs(float(small_axis @ large_axis)), 3)
            if small_axis is not None and large_axis is not None else np.nan,
    }

def scan(movies: dict, source: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for well, group in source[source.valid_current_pair == True].groupby("well", sort=True):
        labels = movies[well]["interface_labels"]
        points = movies[well]["points"].set_index(["track_id", "frame"])
        for pair in group.itertuples(index=False):
            row = pair._asdict()
            if pd.isna(pair.large_track) or pd.isna(pair.small_track):
                continue
            t = int(pair.frame)-1
            large_key = (int(pair.large_track), t)
            small_key = (int(pair.small_track), t)
            if large_key not in points.index or small_key not in points.index:
                continue
            large_point = points.loc[large_key]
            small_point = points.loc[small_key]
            if isinstance(large_point, pd.DataFrame) or isinstance(small_point, pd.DataFrame):
                continue
            row.update(interface(labels[t], int(pair.large_identity),
                                 int(pair.small_identity), large_point, small_point))
            if "interface_gap_px" in row:
                rows.append(row)
        print(well, "measured", len([r for r in rows if r["well"] == well]), flush=True)

    return pd.DataFrame(rows)
