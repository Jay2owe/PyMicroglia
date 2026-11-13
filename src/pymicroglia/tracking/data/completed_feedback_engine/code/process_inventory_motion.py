"""Original motion-aligned measurements without historical feedback annotations."""
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

def sample_occupancy(image: np.ndarray, name: int,
                     yy: np.ndarray, xx: np.ndarray,
                     dy: int, dx: int) -> tuple[float, float]:
    y, x = yy+dy, xx+dx
    inside = (y >= 0) & (y < image.shape[0]) & (x >= 0) & (x < image.shape[1])
    if not inside.any():
        return 0.0, 0.0
    direct = np.zeros(len(yy), bool)
    direct[inside] = image[y[inside], x[inside]] == name
    near = direct.copy()
    for oy, ox in OFFSETS:
        if not oy and not ox:
            continue
        sy, sx = y+oy, x+ox
        valid = (sy >= 0) & (sy < image.shape[0]) & (sx >= 0) & (sx < image.shape[1])
        near[valid] |= image[sy[valid], sx[valid]] == name
    return float(direct.mean()), float(near.mean())

# Speed (folded from process_inventory_lookups): screen_well looks points up
# by (track, frame) with `in` and `.loc` and reads only y, x and state from a
# unique row. A dict from the same (track_id, frame) values to row positions
# answers both: equal keys match as they do in the index, and a key returns
# that row's y, x and state as the column values. A table with any repeated
# key keeps the MultiIndex, because a non-unique MultiIndex makes .loc return
# a DataFrame for every key; so do tables without those columns or with
# missing keys.
class _Row:
    __slots__ = ('y', 'x', 'state')

    def __init__(self, y, x, state):
        self.y, self.x, self.state = y, x, state


class _Keys:
    __slots__ = ('positions',)

    def __init__(self, positions):
        self.positions = positions

    def __contains__(self, key):
        return key in self.positions


class _Lookup:
    __slots__ = ('_owner',)

    def __init__(self, owner):
        self._owner = owner

    def __getitem__(self, key):
        return self._owner.row(key)


class _MotionPoints:
    def __init__(self, points):
        self.positions = {}
        for position, key in enumerate(zip(points['track_id'].tolist(), points['frame'].tolist())):
            self.positions.setdefault(key, []).append(position)
        self.columns = (points['y'].tolist(), points['x'].tolist(), points['state'].tolist())
        self.index = _Keys(self.positions)
        self.loc = _Lookup(self)

    def row(self, key):
        position = self.positions[key][0]
        y, x, state = self.columns
        return _Row(y[position], x[position], state[position])


def _motion_points(points):
    columns = list(points.columns)
    if (any(name not in columns for name in ('track_id', 'frame', 'y', 'x', 'state'))
            or len(set(columns)) != len(columns)
            or points['track_id'].isna().any() or points['frame'].isna().any()):
        return points.set_index(["track_id", "frame"])
    lookup = _MotionPoints(points)
    if len(lookup.positions) != len(points):
        return points.set_index(["track_id", "frame"])
    return lookup

def screen_well(well: str, source: pd.DataFrame, movies: dict) -> list[dict]:
    labels = movies[well]["branch_labels"]
    points = movies[well]["points"]
    point_index = _motion_points(points)
    components: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    for pair in source.itertuples(index=False):
        if pd.isna(pair.large_track) or pd.isna(pair.small_track):
            continue
        large_track, small_track = int(pair.large_track), int(pair.small_track)
        t = int(pair.frame)-1
        lk, sk = (large_track, t), (small_track, t)
        if lk not in point_index.index or sk not in point_index.index:
            continue
        large, small = point_index.loc[lk], point_index.loc[sk]
        if isinstance(large, pd.DataFrame) or isinstance(small, pd.DataFrame):
            continue
        ly, lx = round(float(large.y)), round(float(large.x))
        sy, sx = round(float(small.y)), round(float(small.x))
        if (not 0 <= t < len(labels)
                or not 0 <= ly < labels.shape[1] or not 0 <= lx < labels.shape[2]
                or not 0 <= sy < labels.shape[1] or not 0 <= sx < labels.shape[2]):
            continue
        large_name, small_name = int(labels[t, ly, lx]), int(labels[t, sy, sx])
        if not large_name or not small_name or large_name == small_name:
            continue
        key = t, small_name
        if key not in components:
            components[key], _ = ndi.label(labels[t] == small_name, STRUCTURE)
        component = int(components[key][sy, sx])
        if not component:
            continue
        yy, xx = np.where(components[key] == component)
        if len(yy) > 500:
            continue
        direct, near = [], []
        for other in range(max(0, t-7), min(len(labels), t+8)):
            if other == t or (large_track, other) not in point_index.index:
                continue
            other_point = point_index.loc[(large_track, other)]
            if isinstance(other_point, pd.DataFrame) or str(other_point.state) == "dormant":
                continue
            shift_y = round(float(other_point.y)-float(large.y))
            shift_x = round(float(other_point.x)-float(large.x))
            a, b = sample_occupancy(labels[other], large_name,
                                    yy, xx, shift_y, shift_x)
            direct.append(a)
            near.append(b)
        if not direct:
            continue
        rows.append({
            "well": well, "frame": t+1, "large_track": large_track,
            "small_track": small_track, "large_name": large_name,
            "small_name": small_name, "small_pixels": len(yy),
            "neighbour_frames": len(direct),
            "aligned_exact_max": max(direct),
            "aligned_exact_top3_mean": float(np.mean(sorted(direct, reverse=True)[:3])),
            "aligned_near_max": max(near),
            "aligned_near_top3_mean": float(np.mean(sorted(near, reverse=True)[:3])),
            "aligned_near_half_frames": sum(value >= .5 for value in near),
            "unaligned_absorption_old": float(pair.large_occupancy_lag_5)
                                        if not pd.isna(pair.large_occupancy_lag_5) else 0.0,
            "median_large_area_old": float(pair.accepted_large_area),
            "fragment_elongation_old": float(pair.fragment_elongation),
            "soma_blobness_old": np.nan,
        })
    print("aligned", well, len(rows), flush=True)
    return rows


def scan(movies: dict, source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, group in source.groupby("well", sort=True):
        rows.extend(screen_well(well, group, movies))
    return pd.DataFrame(rows)
