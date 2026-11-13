"""Restore a vanished early name to supported basins on its continuous soma path.

Only accepted masks, registered raw images, physical point tracks and protected
pixels are inputs. Discovery has no well, feedback, identity or region selectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_path_frames: int = 40
    minimum_strong_frames: int = 25
    anchor_window_frames: int = 7
    minimum_anchor_observations: int = 4
    minimum_later_strong_named_frames: int = 20
    minimum_later_distinct_names: int = 2
    maximum_external_anchor_frames: int = 2
    maximum_external_anchor_pixels: int = 100
    maximum_step_p95_radii: float = 1.0
    maximum_single_step_radii: float = 1.75
    minimum_basin_area_ratio: float = .2
    maximum_basin_area_ratio: float = 1.75
    minimum_changed_path_frames: int = 12
    minimum_changed_path_pixels: int = 1000
    maximum_weak_bridge_frames: int = 2
    watershed_sigma_px: float = 1.0


def _point_owner(labels: np.ndarray, row) -> int:
    t = int(row.frame)
    y, x = round(float(row.y)), round(float(row.x))
    if 0 <= t < len(labels) and 0 <= y < labels.shape[1] and 0 <= x < labels.shape[2]:
        return int(labels[t, y, x])
    return 0


def _component(frame: np.ndarray, row) -> np.ndarray | None:
    y, x = round(float(row.y)), round(float(row.x))
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return None
    name = int(frame[y, x])
    if name == 0:
        return None
    parts, _ = ndi.label(frame == name, STRUCTURE)
    return parts == parts[y, x]


def _anchor(group: pd.DataFrame, labels: np.ndarray, params: Params) -> tuple[int, float] | None:
    if (len(group) < params.minimum_path_frames
            or int(group.strong.sum()) < params.minimum_strong_frames
            or not np.all(np.diff(group.frame.to_numpy(int)) == 1)):
        return None
    initial = group.head(params.anchor_window_frames)
    strong = initial[(initial.state == "observed") & initial.strong]
    if len(strong) < params.minimum_anchor_observations:
        return None
    names = strong.owner.to_numpy(int)
    name = int(names[0])
    if name <= 0 or int(np.count_nonzero(names == name)) < params.minimum_anchor_observations:
        return None
    later = group.iloc[params.anchor_window_frames:]
    later_strong = later[(later.state == "observed") & later.strong]
    if (len(later) < 30 or (later.owner == name).any()
            or int(later_strong.owner.gt(0).sum()) < params.minimum_later_strong_named_frames
            or later.loc[later.owner.gt(0), "owner"].nunique() < params.minimum_later_distinct_names):
        return None
    external = [int(np.count_nonzero(labels[int(t)] == name))
                for t in later.frame]
    if (sum(v > 0 for v in external) > params.maximum_external_anchor_frames
            or sum(external) > params.maximum_external_anchor_pixels):
        return None
    pos = group[["y", "x"]].to_numpy(float)
    radius = np.maximum(group.radius_px.to_numpy(float), 1.)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    scaled = steps / np.maximum((radius[:-1]+radius[1:])/2, 1.)
    if (np.percentile(scaled, 95) > params.maximum_step_p95_radii
            or np.max(scaled) > params.maximum_single_step_radii):
        return None
    areas = []
    for row in strong[strong.owner == name].itertuples(index=False):
        mask = _component(labels[int(row.frame)], row)
        if mask is not None:
            areas.append(int(mask.sum()))
    if len(areas) < params.minimum_anchor_observations:
        return None
    return name, float(np.median(areas))


def _basin(mask: np.ndarray, raw_frame: np.ndarray, point,
           companions: list, params: Params) -> np.ndarray | None:
    target_seed = physical._marker_pixel(mask, float(point.x), float(point.y))
    if target_seed is None:
        return None
    if not companions:
        return mask
    markers = np.zeros(mask.shape, np.uint16)
    markers[target_seed] = 1
    next_marker = 2
    for other in companions:
        seed = physical._marker_pixel(mask, float(other.x), float(other.y))
        if seed is None or markers[seed]:
            continue
        markers[seed] = next_marker
        next_marker += 1
    if next_marker == 2:
        return mask
    ys, xs = np.where(mask)
    y0, y1 = max(0, int(ys.min())-2), min(mask.shape[0], int(ys.max())+3)
    x0, x1 = max(0, int(xs.min())-2), min(mask.shape[1], int(xs.max())+3)
    box = np.s_[y0:y1, x0:x1]
    elevation = -ndi.gaussian_filter(raw_frame[box].astype(np.float32),
                                     params.watershed_sigma_px)
    selected = watershed(elevation, markers=markers[box], mask=mask[box],
                         connectivity=STRUCTURE) == 1
    result = np.zeros(mask.shape, bool)
    result[box] = selected
    return result


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or labels.shape != protected.shape
            or raw.shape[:3] != labels.shape):
        raise ValueError("labels, raw and protected pixels must align")
    scored = points.copy()
    scored["strong"] = scored.strong.astype(bool)
    scored["owner"] = [_point_owner(labels, row)
                       for row in scored.itertuples(index=False)]
    by_frame = {int(t): g for t, g in scored.groupby("frame")}
    candidate = labels.copy()
    audit, applications = [], []
    for track_id, group in scored.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        anchor = _anchor(group, labels, params)
        if anchor is None:
            continue
        name, reference_area = anchor
        proposals: list[tuple[int, np.ndarray, dict]] = []
        refusals: dict[int, str] = {}
        for point in group.itertuples(index=False):
            t = int(point.frame)
            old = int(point.owner)
            if old == 0 or old == name:
                continue
            strong = point.state == "observed" and bool(point.strong)
            if not strong:
                # A weak point needs strong support on both sides nearby.
                near = group[(group.frame >= t-params.maximum_weak_bridge_frames)
                             & (group.frame <= t+params.maximum_weak_bridge_frames)
                             & (group.state == "observed") & group.strong]
                if not (near.frame.lt(t).any() and near.frame.gt(t).any()):
                    continue
            if np.any(labels[t] == name):
                refusals[t] = "original_name_has_other_holder"
                continue
            mask = _component(labels[t], point)
            if mask is None:
                continue
            companions = []
            for other in by_frame[t].itertuples(index=False):
                if int(other.track_id) == int(track_id) or other.state == "dormant":
                    continue
                oy, ox = round(float(other.y)), round(float(other.x))
                if 0 <= oy < mask.shape[0] and 0 <= ox < mask.shape[1] and mask[oy, ox]:
                    companions.append(other)
            selected = _basin(mask, raw[t], point, companions, params)
            if selected is None:
                refusals[t] = "raw_basin_unavailable"
                continue
            area = int(selected.sum())
            ratio = area / reference_area
            if not params.minimum_basin_area_ratio <= ratio <= params.maximum_basin_area_ratio:
                refusals[t] = "implausible_basin_area"
                continue
            if np.any(protected[t] & selected):
                refusals[t] = "accepted_pixel_in_basin"
                continue
            if np.any(candidate[t][selected] != old):
                refusals[t] = "overlapping_prior_proposal"
                continue
            detail = {"frame": t+1, "track_id": int(track_id),
                      "old_name": old, "new_name": name,
                      "changed_pixels": area, "reference_area": reference_area,
                      "basin_area_ratio": ratio,
                      "companion_points": len(companions),
                      "strong_target": strong}
            proposals.append((t, selected, detail))
        pixels = sum(detail["changed_pixels"] for _, _, detail in proposals)
        eligible = (len(proposals) >= params.minimum_changed_path_frames
                    and pixels >= params.minimum_changed_path_pixels)
        audit.append({"track_id": int(track_id), "early_name": name,
                      "first_frame": int(group.frame.min())+1,
                      "last_frame": int(group.frame.max())+1,
                      "reference_area": reference_area,
                      "proposed_frames": len(proposals), "proposed_pixels": pixels,
                      "refused_frames": len(refusals),
                      "outcome": "applied" if eligible else "insufficient_supported_path"})
        if not eligible:
            continue
        for t, selected, detail in proposals:
            candidate[t][selected] = name
            applications.append(detail)
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("dormant-name path crossed an accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
