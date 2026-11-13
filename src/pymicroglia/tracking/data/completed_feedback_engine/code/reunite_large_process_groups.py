"""Rejoin recurring, thin satellites of a process-rich large cell.

The pair inventory is a feedback-blind fieldwide neighbour enumeration.  Only
current masks, raw intensities, temporal occupancy and protected edits decide
whether a satellite is eligible.  No recording, coordinate or identity is a
rule parameter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import label as equal_value_components


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_large_area: int = 350
    minimum_small_area: int = 10
    maximum_small_area: int = 250
    maximum_gap_px: float = 4.0
    minimum_aligned_absorption: float = .4
    maximum_raw_q95_ratio: float = 1.05
    minimum_group_frames: int = 10
    minimum_group_satellites: int = 4
    minimum_group_pixels: int = 700
    minimum_group_elongated_frames: int = 5
    elongated_ratio: float = 2.4
    minimum_satellite_pixels: int = 100
    minimum_satellite_elongated_frames: int = 2
    satellite_elongation: float = 1.8
    minimum_single_frame_area: int = 100


def _elongation(yy: np.ndarray, xx: np.ndarray) -> float:
    if len(yy) < 5:
        return 1.0
    eigen = np.linalg.eigvalsh(np.cov(np.column_stack((yy, xx)), rowvar=False))
    return float(np.sqrt(max(eigen[-1], .01) / max(eigen[0], .01)))


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    ae = np.argwhere(a & ~ndi.binary_erosion(a, structure=STRUCTURE))
    be = np.argwhere(b & ~ndi.binary_erosion(b, structure=STRUCTURE))
    if not len(ae) or not len(be):
        return float("inf")
    return float(cKDTree(ae).query(be)[0].min())


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, neighbours: pd.DataFrame,
            params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return candidate, fieldwide frame evidence, group audit, applications."""
    if labels.shape != protected.shape or labels.ndim != 3 or len(raw) < len(labels):
        raise ValueError("labels, raw and protected edits must align")
    candidate = labels.copy()
    if neighbours.empty:
        return candidate, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    needed = {"frame", "large_track", "small_track", "aligned_exact_max"}
    if not needed.issubset(neighbours):
        raise ValueError("incomplete feedback-blind neighbour inventory")
    point_index = points.set_index(["track_id", "frame"])
    component_cache: dict[int, np.ndarray] = {}
    rows: list[dict] = []
    masks: dict[tuple[int, int], np.ndarray] = {}
    seen: set[tuple[int, int, int]] = set()
    for pair in neighbours.itertuples(index=False):
        t = int(pair.frame) - 1
        lk, sk = (int(pair.large_track), t), (int(pair.small_track), t)
        if not 0 <= t < len(labels) or lk not in point_index.index or sk not in point_index.index:
            continue
        large_point, small_point = point_index.loc[lk], point_index.loc[sk]
        if isinstance(large_point, pd.DataFrame) or isinstance(small_point, pd.DataFrame):
            continue
        ly, lx = round(float(large_point.y)), round(float(large_point.x))
        sy, sx = round(float(small_point.y)), round(float(small_point.x))
        if not (0 <= ly < labels.shape[1] and 0 <= lx < labels.shape[2]
                and 0 <= sy < labels.shape[1] and 0 <= sx < labels.shape[2]):
            continue
        if t not in component_cache:
            component_cache[t] = equal_value_components(labels[t], connectivity=2,
                                                        background=0)
        components = component_cache[t]
        large_id, small_id = int(components[ly, lx]), int(components[sy, sx])
        if not large_id or not small_id or large_id == small_id:
            continue
        large_name, small_name = int(labels[t, ly, lx]), int(labels[t, sy, sx])
        if not large_name or not small_name or large_name == small_name:
            continue
        key = t, large_id, small_id
        if key in seen:
            continue
        seen.add(key)
        large, small = components == large_id, components == small_id
        large_area, small_area = int(large.sum()), int(small.sum())
        if (large_area < params.minimum_large_area
                or not params.minimum_small_area <= small_area <= params.maximum_small_area
                or float(pair.aligned_exact_max) < params.minimum_aligned_absorption
                or np.any(protected[t] & small)):
            continue
        gap = _gap(large, small)
        if gap > params.maximum_gap_px:
            continue
        small_q95 = float(np.percentile(raw[t][small], 95))
        large_q95 = float(np.percentile(raw[t][large], 95))
        ratio = small_q95 / max(large_q95, 1.0)
        if ratio > params.maximum_raw_q95_ratio:
            continue
        yy, xx = np.where(small)
        row = {"frame": t+1, "large_name": large_name,
               "small_name": small_name, "large_track": int(pair.large_track),
               "small_track": int(pair.small_track), "large_area": large_area,
               "small_area": small_area, "gap_px": gap,
               "aligned_absorption": float(pair.aligned_exact_max),
               "raw_q95_ratio": ratio, "elongation": _elongation(yy, xx),
               "component": small_id}
        rows.append(row)
        masks[(t, small_id)] = small
    frame_evidence = pd.DataFrame(rows)
    if frame_evidence.empty:
        return candidate, frame_evidence, pd.DataFrame(), pd.DataFrame()
    audits, applications = [], []
    for large_name, group in frame_evidence.groupby("large_name", sort=True):
        group_pixels = int(group.small_area.sum())
        group_elongated = int(group.elongation.ge(params.elongated_ratio).sum())
        eligible = (group.frame.nunique() >= params.minimum_group_frames
                    and group.small_name.nunique() >= params.minimum_group_satellites
                    and group_pixels >= params.minimum_group_pixels
                    and group_elongated >= params.minimum_group_elongated_frames)
        audit = {"large_name": int(large_name), "frames": int(group.frame.nunique()),
                 "satellites": int(group.small_name.nunique()),
                 "eligible_pixels": group_pixels,
                 "elongated_frames": group_elongated,
                 "eligible": bool(eligible), "applied_pixels": 0}
        if eligible:
            for small_name, path in group.groupby("small_name", sort=True):
                long_path = (int(path.small_area.sum()) >= params.minimum_satellite_pixels
                             and int(path.elongation.ge(params.satellite_elongation).sum())
                             >= params.minimum_satellite_elongated_frames)
                single_large_branch = (len(path) == 1
                                       and int(path.small_area.iloc[0])
                                       >= params.minimum_single_frame_area
                                       and float(path.elongation.iloc[0])
                                       >= params.satellite_elongation)
                if not (long_path or single_large_branch):
                    continue
                for row in path.itertuples(index=False):
                    t = int(row.frame)-1
                    mask = masks[t, int(row.component)]
                    if np.any(candidate[t][mask] != int(row.small_name)):
                        raise AssertionError("overlapping process proposals")
                    candidate[t][mask] = int(large_name)
                    pixels = int(mask.sum())
                    audit["applied_pixels"] += pixels
                    applications.append({"frame": int(row.frame),
                                         "old_name": int(row.small_name),
                                         "new_name": int(large_name),
                                         "changed_pixels": pixels,
                                         "small_track": int(row.small_track),
                                         "gap_px": float(row.gap_px),
                                         "aligned_absorption": float(row.aligned_absorption),
                                         "raw_q95_ratio": float(row.raw_q95_ratio),
                                         "elongation": float(row.elongation)})
        audits.append(audit)
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels)))):
        raise AssertionError("process-group correction crossed an accepted boundary")
    return candidate, frame_evidence, pd.DataFrame(audits), pd.DataFrame(applications)
