"""Extend approved process ownership through matching detached branch frames.

Approved process corrections are structural seeds. Nearby components of the
same former name may join the same large cell only when their motion-aligned
footprints overlap a seed, their raw signal remains bounded by the cell core,
and no earlier accepted pixel is touched. No feedback, movie, cell, coordinate
or region selector enters the decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    maximum_seed_distance_frames: int = 12
    minimum_component_area: int = 20
    maximum_component_area: int = 250
    maximum_edge_distance_px: float = 10.0
    minimum_exact_seed_overlap: float = .2
    minimum_near_seed_overlap: float = .75
    maximum_raw_q95_ratio: float = 1.1
    seed_dilation_radius_px: int = 3


def _largest_centre(image: np.ndarray, name: int) -> np.ndarray | None:
    parts, count = ndi.label(image == name, STRUCTURE)
    if not count:
        return None
    sizes = np.bincount(parts.ravel())
    yy, xx = np.where(parts == int(np.argmax(sizes[1:])+1))
    return np.array((float(yy.mean()), float(xx.mean())))


def _overlap(yy: np.ndarray, xx: np.ndarray, seed: np.ndarray,
             dy: int, dx: int) -> float:
    sy, sx = yy-dy, xx-dx
    valid = (sy >= 0) & (sy < seed.shape[0]) & (sx >= 0) & (sx < seed.shape[1])
    if not valid.any():
        return 0.0
    return float(seed[sy[valid], sx[valid]].sum()/len(yy))


def correct(labels: np.ndarray, raw: np.ndarray, protected: np.ndarray,
            seeds: list[dict], params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != protected.shape or len(raw) < len(labels):
        raise ValueError("complete aligned masks, raw frames and accepted protection required")
    candidate = labels.copy()
    if not seeds:
        return candidate, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    groups: dict[tuple[int, int], list[dict]] = {}
    for seed in seeds:
        t, large_name, old_name = (int(seed[key]) for key in
                                   ("frame", "large_name", "old_name"))
        mask = np.asarray(seed["mask"], bool)
        if (not 0 <= t < len(labels) or mask.shape != labels.shape[1:]
                or not mask.any() or np.any(labels[t][mask] != large_name)):
            raise ValueError("accepted process seed does not match current labels")
        centre = _largest_centre(labels[t], large_name)
        if centre is None:
            raise ValueError("accepted process seed has no large-cell core")
        radius = params.seed_dilation_radius_px
        near = ndi.binary_dilation(mask, structure=np.ones((2*radius+1, 2*radius+1), bool))
        groups.setdefault((large_name, old_name), []).append(
            {"frame": t, "mask": mask, "near": near, "centre": centre})
    evidence = []
    proposed: dict[tuple[int, int, int], list[dict]] = {}
    for (large_name, old_name), group in sorted(groups.items()):
        seed_frames = [int(seed["frame"]) for seed in group]
        for t in range(len(labels)):
            seed_distance = min(abs(t-frame) for frame in seed_frames)
            if seed_distance > params.maximum_seed_distance_frames:
                continue
            centre = _largest_centre(labels[t], large_name)
            if centre is None:
                continue
            large = labels[t] == large_name
            distance = ndi.distance_transform_edt(~large)
            parts, count = ndi.label(labels[t] == old_name, STRUCTURE)
            for component in range(1, count+1):
                small = parts == component
                yy, xx = np.where(small)
                area = len(yy)
                if not params.minimum_component_area <= area <= params.maximum_component_area:
                    continue
                gap = float(distance[yy, xx].min())
                if gap > params.maximum_edge_distance_px:
                    continue
                exact, near = 0.0, 0.0
                for seed in group:
                    if abs(t-int(seed["frame"])) > params.maximum_seed_distance_frames:
                        continue
                    dy, dx = np.rint(centre-seed["centre"]).astype(int)
                    exact = max(exact, _overlap(yy, xx, seed["mask"], int(dy), int(dx)))
                    near = max(near, _overlap(yy, xx, seed["near"], int(dy), int(dx)))
                raw_ratio = (float(np.percentile(raw[t, yy, xx], 95))
                             / max(float(np.percentile(raw[t][large], 95)), 1.0))
                protected_pixels = int(np.count_nonzero(protected[t] & small))
                eligible = (exact >= params.minimum_exact_seed_overlap
                            and near >= params.minimum_near_seed_overlap
                            and raw_ratio <= params.maximum_raw_q95_ratio
                            and not protected_pixels)
                row = {"frame": t+1, "large_name": large_name,
                       "old_name": old_name, "component": component,
                       "pixels": area, "gap_px": gap,
                       "seed_frame_distance": seed_distance,
                       "exact_seed_overlap": exact, "near_seed_overlap": near,
                       "raw_q95_ratio": raw_ratio,
                       "protected_pixels": protected_pixels,
                       "eligible": bool(eligible)}
                evidence.append(row)
                if eligible:
                    proposed.setdefault((t, old_name, component), []).append(
                        {"row": row, "mask": small})
    applications = []
    conflicts = 0
    for key, options in sorted(proposed.items()):
        if len({int(item["row"]["large_name"]) for item in options}) != 1:
            conflicts += 1
            continue
        selected = options[0]
        t, old_name, _ = key
        mask, row = selected["mask"], selected["row"]
        if np.any(candidate[t][mask] != old_name):
            raise AssertionError("overlapping proposed process extensions")
        candidate[t][mask] = int(row["large_name"])
        applications.append({"frame": t+1, "old_name": old_name,
                             "new_name": int(row["large_name"]),
                             "changed_pixels": int(mask.sum()),
                             "gap_px": float(row["gap_px"]),
                             "seed_frame_distance": int(row["seed_frame_distance"]),
                             "exact_seed_overlap": float(row["exact_seed_overlap"]),
                             "near_seed_overlap": float(row["near_seed_overlap"]),
                             "raw_q95_ratio": float(row["raw_q95_ratio"])})
    audits = []
    for (large_name, old_name), group in sorted(groups.items()):
        matched = [row for row in applications if row["new_name"] == large_name
                   and row["old_name"] == old_name]
        audits.append({"large_name": large_name, "old_name": old_name,
                       "seed_frames": len({int(seed["frame"]) for seed in group}),
                       "new_frames": len({row["frame"] for row in matched}),
                       "changed_pixels": sum(row["changed_pixels"] for row in matched),
                       "conflicting_components": conflicts})
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels)))):
        raise AssertionError("process extension crossed an approved boundary")
    return candidate, pd.DataFrame(evidence), pd.DataFrame(audits), pd.DataFrame(applications)
