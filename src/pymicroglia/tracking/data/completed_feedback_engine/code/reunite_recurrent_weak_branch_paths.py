"""Rejoin recurring weak fragments using fieldwide mask and raw evidence.

The caller supplies an inventory of nearby components. This rule never sees
feedback marks or recording, identity, track or coordinate selectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as value_components


@dataclass(frozen=True)
class Params:
    maximum_spatial_link_px: float = 10.0
    maximum_temporal_link_frames: int = 12
    minimum_path_frames: int = 5
    minimum_path_pixels: int = 180
    maximum_strong_soma_fraction: float = .21
    maximum_median_raw_q95_ratio: float = .3
    minimum_median_absorption: float = .5
    minimum_unique_parent_fraction: float = .5


EVIDENCE_COLUMNS = (
    "frame", "large_name", "small_name", "large_component",
    "small_component", "small_area", "x", "y", "q95_ratio",
    "absorption", "strong_count", "protected_pixels",
)


def scan_nearby(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
                protected: np.ndarray) -> pd.DataFrame:
    """Build feedback-free evidence on any aligned movie and raw image stack."""
    if labels.shape != protected.shape or raw.shape[0] < len(labels):
        raise ValueError("movie, raw signal and protected mask must align")
    strong = points[points.state.astype(str).eq("observed")
                    & points.strong.astype(bool)]
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    rows: list[dict] = []
    for t, frame in enumerate(labels):
        parts = value_components(frame, connectivity=2, background=0)
        sizes = np.bincount(parts.ravel())
        boxes = ndi.find_objects(parts)
        component_coords: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def coords(component: int) -> tuple[np.ndarray, np.ndarray]:
            if component not in component_coords:
                box = boxes[component - 1]
                yy, xx = np.nonzero(parts[box] == component)
                component_coords[component] = yy + box[0].start, xx + box[1].start
            return component_coords[component]

        strong_counts: dict[int, int] = {}
        for point in by_frame.get(t, points.iloc[0:0]).itertuples(index=False):
            y, x = round(float(point.y)), round(float(point.x))
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                component = int(parts[y, x])
                if component:
                    strong_counts[component] = strong_counts.get(component, 0) + 1
        for lid in np.flatnonzero(sizes >= 180):
            lid = int(lid)
            if lid == 0 or lid > len(boxes) or boxes[lid - 1] is None:
                continue
            box = boxes[lid - 1]
            y0, y1 = max(0, box[0].start - 5), min(frame.shape[0], box[0].stop + 5)
            x0, x1 = max(0, box[1].start - 5), min(frame.shape[1], box[1].stop + 5)
            local = parts[y0:y1, x0:x1]
            nearby = ndi.binary_dilation(local == lid, iterations=4)
            ly, lx = coords(lid)
            large_name = int(frame[ly[0], lx[0]])
            large_q95 = float(np.percentile(raw[t, ly, lx], 95))
            for sid in np.unique(local[nearby]):
                sid = int(sid)
                if sid in (0, lid) or not 10 <= int(sizes[sid]) <= 150:
                    continue
                sy, sx = coords(sid)
                small_name = int(frame[sy[0], sx[0]])
                if small_name == large_name:
                    continue
                absorption = max((
                    float(np.mean(labels[other, sy, sx] == large_name))
                    for other in range(max(0, t - 5), min(len(labels), t + 6))
                    if other != t
                ), default=0.0)
                rows.append({
                    "frame": t + 1, "large_name": large_name,
                    "small_name": small_name, "large_component": lid,
                    "small_component": sid, "small_area": int(sizes[sid]),
                    "x": float(sx.mean()), "y": float(sy.mean()),
                    "q95_ratio": float(np.percentile(raw[t, sy, sx], 95))
                    / max(large_q95, 1.0),
                    "absorption": absorption,
                    "strong_count": strong_counts.get(sid, 0),
                    "protected_pixels": int(np.count_nonzero(protected[t, sy, sx])),
                })
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def _clusters(group: pd.DataFrame, params: Params) -> np.ndarray:
    """Connected temporal-spatial paths, independent of temporary names elsewhere."""
    data = group[["frame", "x", "y"]].to_numpy(float)
    parent = list(range(len(data)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if (abs(data[i, 0] - data[j, 0]) <= params.maximum_temporal_link_frames
                    and np.hypot(*(data[i, 1:] - data[j, 1:]))
                    <= params.maximum_spatial_link_px):
                parent[root(j)] = root(i)
    roots = [root(i) for i in range(len(data))]
    names = {value: index for index, value in enumerate(sorted(set(roots)))}
    return np.array([names[value] for value in roots], dtype=int)


def select_paths(inventory: pd.DataFrame, params: Params = Params()
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select recurring physical paths; return audit and eligible frame rows."""
    if inventory.empty:
        return pd.DataFrame(), pd.DataFrame(columns=(*EVIDENCE_COLUMNS, "cluster"))
    missing = set(EVIDENCE_COLUMNS) - set(inventory)
    if missing:
        raise ValueError(f"missing inventory fields: {sorted(missing)}")
    # Drop any optional mark or recording columns at the rule boundary.
    evidence = inventory.loc[:, EVIDENCE_COLUMNS].copy()
    claimant_count = evidence.groupby(
        ["frame", "small_component"]).large_name.transform("nunique")
    evidence["unique_parent"] = claimant_count.eq(1)
    audits: list[dict] = []
    selected: list[pd.DataFrame] = []
    for (large_name, small_name), group in evidence.groupby(
            ["large_name", "small_name"], sort=True):
        if group.frame.nunique() < params.minimum_path_frames:
            continue
        group = group.copy()
        group["cluster"] = _clusters(group, params)
        for cluster, path in group.groupby("cluster", sort=True):
            frames = int(path.frame.nunique())
            pixels = int(path.small_area.sum())
            strong_fraction = float(path.groupby("frame").strong_count.max().gt(0).mean())
            unique_fraction = float(path.unique_parent.mean())
            protected = int(path.protected_pixels.sum())
            eligible = (
                frames >= params.minimum_path_frames
                and pixels >= params.minimum_path_pixels
                and strong_fraction <= params.maximum_strong_soma_fraction
                and float(path.q95_ratio.median()) <= params.maximum_median_raw_q95_ratio
                and float(path.absorption.median()) >= params.minimum_median_absorption
                and unique_fraction >= params.minimum_unique_parent_fraction
                and protected == 0
            )
            audits.append({
                "large_name": int(large_name), "small_name": int(small_name),
                "cluster": int(cluster), "frames": frames, "pixels": pixels,
                "strong_fraction": strong_fraction,
                "unique_parent_fraction": unique_fraction,
                "median_raw_q95_ratio": float(path.q95_ratio.median()),
                "median_absorption": float(path.absorption.median()),
                "protected_pixels": protected,
                "outcome": "eligible" if eligible else "rejected",
            })
            if eligible:
                selected.append(path)
    audit = pd.DataFrame(audits)
    if not selected:
        return audit, pd.DataFrame(columns=(*EVIDENCE_COLUMNS, "cluster"))
    proposed = pd.concat(selected, ignore_index=True)
    # A component can have two plausible large parents even if each path's
    # unique-frame fraction passes. Reject both complete paths on conflict.
    keys = ["frame", "small_component"]
    conflict = proposed.groupby(keys).large_name.transform("nunique").gt(1)
    if conflict.any():
        bad = proposed.loc[conflict, ["large_name", "small_name", "cluster"]].drop_duplicates()
        for row in bad.itertuples(index=False):
            mask = (audit.large_name.eq(row.large_name)
                    & audit.small_name.eq(row.small_name)
                    & audit.cluster.eq(row.cluster))
            audit.loc[mask, "outcome"] = "competing_eligible_parent"
        proposed = proposed.merge(
            bad.assign(rejected=True),
            on=["large_name", "small_name", "cluster"], how="left")
        proposed = proposed[proposed.rejected.isna()].drop(columns="rejected")
    return audit, proposed


def correct(labels: np.ndarray, protected: np.ndarray,
            inventory: pd.DataFrame, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != protected.shape:
        raise ValueError("labels and protected mask must align")
    candidate = labels.copy()
    audit, proposed = select_paths(inventory, params)
    applications: list[dict] = []
    by_frame = {int(t): group for t, group in proposed.groupby("frame")}
    for frame, rows in sorted(by_frame.items()):
        t = frame - 1
        if not 0 <= t < len(labels):
            raise ValueError("inventory frame outside label movie")
        components = value_components(labels[t], connectivity=2, background=0)
        for row in rows.itertuples(index=False):
            mask = components == int(row.small_component)
            if (int(mask.sum()) != int(row.small_area)
                    or not np.all(labels[t][mask] == int(row.small_name))
                    or np.any(protected[t] & mask)
                    or np.any(candidate[t][mask] != labels[t][mask])):
                raise AssertionError("stale, protected or overlapping fragment evidence")
            candidate[t][mask] = int(row.large_name)
            applications.append({
                "frame": frame, "old_name": int(row.small_name),
                "new_name": int(row.large_name),
                "cluster": int(row.cluster), "changed_pixels": int(mask.sum()),
                "raw_q95_ratio": float(row.q95_ratio),
                "temporal_absorption": float(row.absorption),
                "strong_count": int(row.strong_count),
            })
    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or int(changed.sum()) != sum(row["changed_pixels"] for row in applications)):
        raise AssertionError("branch path crossed accepted boundary")
    return candidate, audit, pd.DataFrame(applications)
