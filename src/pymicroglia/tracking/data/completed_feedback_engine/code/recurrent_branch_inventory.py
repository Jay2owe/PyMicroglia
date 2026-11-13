"""Original feedback-free branch inventory with array inputs.

Geometry, rounding, qualification and ordering are copied unchanged from the
recorded screen; only file loading and current-parent globals were removed.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import label as equal_value_components

STRUCTURE = np.ones((3, 3), bool)


def _coords(components: np.ndarray, boxes: list, component: int,
            cache: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if component not in cache:
        box = boxes[component-1]
        local = components[box] == component
        yy, xx = np.nonzero(local)
        cache[component] = (yy+box[0].start, xx+box[1].start)
    return cache[component]


def _edge(coords: tuple[np.ndarray, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    yy, xx = coords
    y0, y1 = max(0, int(yy.min())-1), min(shape[0], int(yy.max())+2)
    x0, x1 = max(0, int(xx.min())-1), min(shape[1], int(xx.max())+2)
    local = np.zeros((y1-y0, x1-x0), bool)
    local[yy-y0, xx-x0] = True
    edge = local & ~ndi.binary_erosion(local, structure=STRUCTURE)
    ey, ex = np.nonzero(edge)
    return np.column_stack((ey+y0, ex+x0))


def _elongation(coords: tuple[np.ndarray, np.ndarray]) -> float:
    yy, xx = coords
    if len(yy) < 5:
        return 1.0
    matrix = np.column_stack((yy, xx)).astype(float)
    eigen = np.linalg.eigvalsh(np.cov(matrix, rowvar=False))
    return float(np.sqrt(max(eigen[-1], 0.01)/max(eigen[0], 0.01)))


def _absorption_whole(labels: np.ndarray, frame: int, coords: tuple[np.ndarray, np.ndarray],
                large_name: int) -> float:
    yy, xx = coords
    fractions = [float(np.mean(labels[other, yy, xx] == large_name))
                 for other in range(max(0, frame-5), min(len(labels), frame+6))
                 if other != frame]
    return max(fractions, default=0.0)


# Speed (folded from recurrent_branch_speed): the KD-tree of one component's
# edge and the gap of one (large, small) pair are the same whenever a frame's
# point pairs name them again, so scan keeps both for the frame. Edge pixels are
# integer coordinates: when two edges' boxes are more than 4 px apart along an
# axis every distance is at least 5, so the pair is skipped without measuring,
# exactly as the original skips it after measuring.
def _edge_box(edge):
    return int(edge[:, 0].min()), int(edge[:, 0].max()), int(edge[:, 1].min()), int(edge[:, 1].max())


def _branch_gap(edge_cache, large_id, small_id, trees, gaps):
    key = (large_id, small_id)
    held = gaps.get(key)
    if held is not None:
        return held
    large, small = edge_cache[large_id], edge_cache[small_id]
    if (type(large) is np.ndarray and type(small) is np.ndarray and large.ndim == 2 and small.ndim == 2
            and large.shape[1] == 2 and small.shape[1] == 2 and len(large) and len(small)
            and large.dtype.kind in 'iu' and small.dtype.kind in 'iu'):
        ly0, ly1, lx0, lx1 = _edge_box(large)
        sy0, sy1, sx0, sx1 = _edge_box(small)
        if max(sy0 - ly1, ly0 - sy1) > 4 or max(sx0 - lx1, lx0 - sx1) > 4:
            gaps[key] = math.inf
            return math.inf
    tree = trees.get(large_id)
    if tree is None:
        tree = trees[large_id] = cKDTree(large)
    value = float(tree.query(small)[0].min())
    gaps[key] = value
    return value


# Speed (folded from recurrent_branch_speed): one read of the neighbouring
# frames' block gives the same pixels as one fancy read per frame, and each
# row's mean of a Boolean comparison is an exact count over the same length.
# Inputs other than 3-D arrays with non-empty 1-D integer pixel lists keep the
# original.
def _absorption(labels: np.ndarray, frame: int, coords: tuple[np.ndarray, np.ndarray],
                large_name: int) -> float:
    yy, xx = coords
    if (not isinstance(labels, np.ndarray) or labels.ndim != 3 or type(yy) is not np.ndarray
            or type(xx) is not np.ndarray or yy.ndim != 1 or xx.ndim != 1 or len(yy) != len(xx)
            or not len(yy) or yy.dtype.kind not in 'iu' or xx.dtype.kind not in 'iu'):
        return _absorption_whole(labels, frame, coords, large_name)
    lo, hi = max(0, frame-5), min(len(labels), frame+6)
    rows = [other - lo for other in range(lo, hi) if other != frame]
    if not rows:
        return 0.0
    block = labels.view(np.ndarray)[lo:hi][:, yy, xx][rows]
    return max(float(value) for value in np.mean(block == large_name, axis=1))


def scan(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strong = points[points.state.astype(str).eq("observed")
                    & points.strong.astype(bool)].copy()
    rows = []
    for frame, frame_points in strong.groupby("frame"):
        frame = int(frame)
        if not 0 <= frame < len(labels):
            continue
        image = labels[frame]
        component_map = equal_value_components(image, connectivity=2, background=0)
        boxes = ndi.find_objects(component_map)
        sizes = np.bincount(component_map.ravel())
        frame_points = frame_points.copy()
        ys = np.rint(frame_points.y.to_numpy(float)).astype(int)
        xs = np.rint(frame_points.x.to_numpy(float)).astype(int)
        inside = ((ys >= 0) & (ys < image.shape[0])
                  & (xs >= 0) & (xs < image.shape[1]))
        frame_points = frame_points.loc[inside].copy()
        ys, xs = ys[inside], xs[inside]
        frame_points["component"] = component_map[ys, xs]
        frame_points["owner"] = image[ys, xs]
        frame_points = frame_points[frame_points.component.gt(0)
                                    & frame_points.owner.gt(0)]
        if len(frame_points) < 2:
            continue
        coords = frame_points[["y", "x"]].to_numpy(float)
        pairs = cKDTree(coords).query_pairs(r=45.0, output_type="ndarray")
        if not len(pairs):
            continue
        points_list = list(frame_points.itertuples(index=False))
        pixel_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        edge_cache: dict[int, np.ndarray] = {}
        elong_cache: dict[int, float] = {}
        _trees = {}
        _gaps = {}
        for ia, ib in pairs:
            a, b = points_list[int(ia)], points_list[int(ib)]
            ca, cb = int(a.component), int(b.component)
            if ca == cb or int(a.owner) == int(b.owner):
                continue
            if sizes[ca] >= sizes[cb]:
                large, small, large_id, small_id = a, b, ca, cb
            else:
                large, small, large_id, small_id = b, a, cb, ca
            large_area, small_area = int(sizes[large_id]), int(sizes[small_id])
            if (large_area < 180 or small_area < 10 or small_area > 150
                    or small_area/large_area > .6):
                continue
            for component in (large_id, small_id):
                if component not in pixel_cache:
                    pixel_cache[component] = _coords(component_map, boxes,
                                                      component, pixel_cache)
                if component not in edge_cache:
                    edge_cache[component] = _edge(pixel_cache[component], image.shape)
            gap = _branch_gap(edge_cache, large_id, small_id, _trees, _gaps)
            if gap > 4.0:
                continue
            if small_id not in elong_cache:
                elong_cache[small_id] = _elongation(pixel_cache[small_id])
            sy, sx = pixel_cache[small_id]
            ly, lx = pixel_cache[large_id]
            raw_frame = raw[frame]
            small_q95 = float(np.percentile(raw_frame[sy, sx], 95))
            large_q95 = float(np.percentile(raw_frame[ly, lx], 95))
            rows.append({
                "frame": frame+1,
                "large_track": int(large.track_id),
                "small_track": int(small.track_id),
                "large_name": int(large.owner),
                "small_name": int(small.owner),
                "large_area": large_area,
                "small_area": small_area,
                "gap_px": round(gap, 3),
                "small_elongation": round(elong_cache[small_id], 4),
                "small_to_large_q95": round(small_q95/max(large_q95, 1.0), 4),
                "temporal_absorption": round(_absorption(labels, frame,
                                                           pixel_cache[small_id],
                                                           int(large.owner)), 4),
            })
    frame_rows = pd.DataFrame(rows)
    if frame_rows.empty:
        return frame_rows, pd.DataFrame()

    # Each pair's large name must be stable while the small track alternates
    # between thin detached masks and the large cell's own name.
    paired = frame_rows.groupby(["large_track", "small_track"]).agg(
        qualifying_frames=("frame", "nunique"),
        qualifying_pixels=("small_area", "sum"),
        elongated_frames=("small_elongation", lambda values: int((values >= 2.4).sum())),
        median_small_area=("small_area", "median"),
        median_large_area=("large_area", "median"),
        median_q95_ratio=("small_to_large_q95", "median"),
        maximum_absorption=("temporal_absorption", "max"),
        large_name_count=("large_name", "nunique"),
        small_name_count=("small_name", "nunique"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
    ).reset_index()
    track_index = points.set_index(["track_id", "frame"])
    owner_rows = []
    for pair in paired.itertuples(index=False):
        subset = frame_rows[(frame_rows.large_track.eq(pair.large_track))
                            & frame_rows.small_track.eq(pair.small_track)]
        large_name = int(subset.large_name.mode().iat[0])
        large_points = points[points.track_id.eq(int(pair.large_track))]
        small_points = points[points.track_id.eq(int(pair.small_track))]
        both = sorted(set(large_points.frame.astype(int))
                      & set(small_points.frame.astype(int)))
        same_name = 0
        for frame in both:
            point = track_index.loc[(int(pair.small_track), frame)]
            y, x = round(float(point.y)), round(float(point.x))
            if (0 <= frame < len(labels) and 0 <= y < labels.shape[1]
                    and 0 <= x < labels.shape[2]
                    and int(labels[frame, y, x]) == large_name):
                same_name += 1
        owner_rows.append({"large_track": int(pair.large_track),
                           "small_track": int(pair.small_track),
                           "same_name_frames": same_name,
                           "coobserved_frames": len(both)})
    paired = paired.merge(pd.DataFrame(owner_rows), on=["large_track", "small_track"])
    paired["eligible"] = (
        paired.qualifying_frames.ge(5)
        & paired.elongated_frames.ge(5)
        & paired.median_small_area.le(60)
        & paired.maximum_absorption.ge(.6)
        & paired.median_q95_ratio.le(.6)
        & paired.large_name_count.eq(1)
        & paired.small_name_count.ge(2)
        & paired.same_name_frames.ge(5)
    )
    return frame_rows, paired

