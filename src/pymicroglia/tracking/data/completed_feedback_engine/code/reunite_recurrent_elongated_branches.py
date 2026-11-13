"""Rejoin whole supported episodes of recurrent elongated cell branches.

The fieldwide pair inventory is produced without feedback by
screen_recurrent_elongated_branches.py.  This producer receives only those
structural measurements, accepted masks, raw images, tracker points and the
protected-pixel mask; no chosen identity, track, recording or region.
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
    maximum_gap_px: float = 4.0
    minimum_large_area: int = 180
    minimum_small_area: int = 10
    maximum_small_area: int = 150
    maximum_area_ratio: float = .6
    maximum_raw_q95_ratio: float = .6
    minimum_seed_absorption: float = .45
    minimum_seed_frames: int = 3
    maximum_seed_link_gap_frames: int = 12
    maximum_backward_extension_frames: int = 2
    maximum_forward_extension_frames: int = 2


def _mask_at(frame: np.ndarray, components: np.ndarray, point) -> np.ndarray | None:
    y, x = round(float(point.y)), round(float(point.x))
    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
        component = int(components[y, x])
        if component > 0:
            return components == component
    return None


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    ae = np.argwhere(a & ~ndi.binary_erosion(a, structure=STRUCTURE))
    be = np.argwhere(b & ~ndi.binary_erosion(b, structure=STRUCTURE))
    if not len(ae) or not len(be):
        return np.inf
    return float(cKDTree(ae).query(be)[0].min())


def _absorption(labels: np.ndarray, frame: int, small: np.ndarray,
                large_name: int) -> float:
    yy, xx = np.where(small)
    return max((float(np.mean(labels[other, yy, xx] == large_name))
                for other in range(max(0, frame-5), min(len(labels), frame+6))
                if other != frame), default=0.0)


def _frame_state(labels: np.ndarray, raw: np.ndarray, protected: np.ndarray,
                 frame: int, large_name: int, small_track: int,
                 large_point, small_point,
                 frame_points: pd.DataFrame, params: Params) -> dict:
    large_y, large_x = round(float(large_point.y)), round(float(large_point.x))
    small_y, small_x = round(float(small_point.y)), round(float(small_point.x))
    image = labels[frame]
    if not (0 <= large_y < image.shape[0] and 0 <= large_x < image.shape[1]
            and 0 <= small_y < image.shape[0] and 0 <= small_x < image.shape[1]):
        return {"state": "unsupported", "reason": "point_outside_field"}
    if (str(large_point.state) == "dormant"
            or str(small_point.state) == "dormant"):
        return {"state": "unsupported", "reason": "large_anchor_absent_or_dormant"}
    if int(image[large_y, large_x]) != large_name:
        radius = max(1.5, float(large_point.radius_px))
        y0, y1 = max(0, int(np.floor(large_point.y-radius))), min(image.shape[0], int(np.ceil(large_point.y+radius+1)))
        x0, x1 = max(0, int(np.floor(large_point.x-radius))), min(image.shape[1], int(np.ceil(large_point.x+radius+1)))
        nearby_y, nearby_x = np.where(image[y0:y1, x0:x1] == large_name)
        if not len(nearby_y):
            return {"state": "unsupported", "reason": "large_anchor_absent_or_dormant"}
        nearby_y, nearby_x = nearby_y+y0, nearby_x+x0
        distance = np.hypot(nearby_y-float(large_point.y), nearby_x-float(large_point.x))
        nearest = int(np.argmin(distance))
        if float(distance[nearest]) > radius:
            return {"state": "unsupported", "reason": "large_anchor_absent_or_dormant"}
        large_y, large_x = int(nearby_y[nearest]), int(nearby_x[nearest])
    small_name = int(image[small_y, small_x])
    if small_name == large_name:
        return {"state": "same", "reason": "already_under_large_name"}
    if small_name == 0:
        return {"state": "ownerless", "reason": "no_named_fragment"}
    components = equal_value_components(image, connectivity=2, background=0)
    large = components == int(components[large_y, large_x])
    small = _mask_at(image, components, small_point)
    if large is None or small is None or np.any(large & small):
        return {"state": "unsupported", "reason": "ambiguous_component"}
    large_area, small_area = int(large.sum()), int(small.sum())
    if (large_area < params.minimum_large_area
            or not params.minimum_small_area <= small_area <= params.maximum_small_area
            or small_area/large_area > params.maximum_area_ratio):
        return {"state": "unsupported", "reason": "component_area_outside_branch_range"}
    gap = _gap(large, small)
    if gap > params.maximum_gap_px:
        return {"state": "unsupported", "reason": "fragment_too_far"}
    if np.any(protected[frame] & small):
        return {"state": "unsupported", "reason": "protected_prior_change"}
    for point in frame_points.itertuples(index=False):
        if (int(point.track_id) == small_track
                or str(point.state) != "observed" or not bool(point.strong)):
            continue
        y, x = round(float(point.y)), round(float(point.x))
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1] and small[y, x]:
            return {"state": "unsupported", "reason": "independent_strong_soma_inside_fragment"}
    small_q95 = float(np.percentile(raw[frame][small], 95))
    large_q95 = float(np.percentile(raw[frame][large], 95))
    raw_ratio = small_q95/max(large_q95, 1.0)
    if raw_ratio > params.maximum_raw_q95_ratio:
        return {"state": "unsupported", "reason": "fragment_has_independent_bright_core"}
    absorption = _absorption(labels, frame, small, large_name)
    return {"state": "supported", "reason": "near_dim_branch",
            "small": small, "small_name": small_name, "small_area": small_area,
            "large_area": large_area, "gap_px": gap, "raw_q95_ratio": raw_ratio,
            "temporal_absorption": absorption}


def _episodes(states: dict[int, dict], params: Params) -> list[list[int]]:
    seeds = sorted(frame for frame, item in states.items()
                   if item["state"] == "supported"
                   and item["temporal_absorption"] >= params.minimum_seed_absorption)
    groups: list[list[int]] = []
    for frame in seeds:
        if not groups:
            groups.append([frame])
            continue
        previous = groups[-1][-1]
        intervening = range(previous+1, frame)
        if (frame-previous <= params.maximum_seed_link_gap_frames
                and all(states.get(t, {"state": "unsupported"})["state"]
                        in {"supported", "same", "ownerless"} for t in intervening)):
            groups[-1].append(frame)
        else:
            groups.append([frame])
    episodes = []
    for group in groups:
        if len(group) < params.minimum_seed_frames:
            continue
        start, end = group[0], group[-1]
        original_start, original_end = start, end
        for offset in range(1, params.maximum_backward_extension_frames+1):
            prior = original_start-offset
            if states.get(prior, {"state": "unsupported"})["state"] != "supported":
                break
            start = prior
        for offset in range(1, params.maximum_forward_extension_frames+1):
            later = original_end+offset
            if states.get(later, {"state": "unsupported"})["state"] != "supported":
                break
            end = later
        episodes.append([frame for frame in range(start, end+1)
                         if states[frame]["state"] == "supported"])
    return episodes


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, pair_frames: pd.DataFrame,
            pair_summary: pd.DataFrame, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != protected.shape or labels.ndim != 3 or raw.shape[0] < labels.shape[0]:
        raise ValueError("labels, protected mask and raw movie must align")
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    point_index = points.set_index(["track_id", "frame"])
    by_frame = {int(frame): group for frame, group in points.groupby("frame")}
    audits, applications = [], []
    if pair_summary.empty or "eligible" not in pair_summary.columns:
        return candidate, pd.DataFrame(), pd.DataFrame()
    for pair in pair_summary[pair_summary.eligible.astype(bool)].itertuples(index=False):
        large_track, small_track = int(pair.large_track), int(pair.small_track)
        frames = pair_frames[(pair_frames.large_track.astype(int).eq(large_track))
                             & (pair_frames.small_track.astype(int).eq(small_track))]
        if frames.empty or frames.large_name.nunique() != 1:
            raise AssertionError("eligible pair has no stable large name")
        large_name = int(frames.large_name.iloc[0])
        states: dict[int, dict] = {}
        for frame in range(int(pair.first_frame)-5, int(pair.last_frame)+2):
            t = frame-1
            if not 0 <= t < len(labels):
                continue
            lk, sk = (large_track, t), (small_track, t)
            if lk not in point_index.index or sk not in point_index.index:
                states[t] = {"state": "unsupported", "reason": "physical_track_missing"}
                continue
            large_point, small_point = point_index.loc[lk], point_index.loc[sk]
            if isinstance(large_point, pd.DataFrame) or isinstance(small_point, pd.DataFrame):
                states[t] = {"state": "unsupported", "reason": "duplicate_track_point"}
                continue
            states[t] = _frame_state(labels, raw, protected, t, large_name,
                                     small_track,
                                     large_point, small_point,
                                     by_frame.get(t, points.iloc[0:0]), params)
        episodes = _episodes(states, params)
        if not episodes:
            audits.append({"large_track": large_track, "small_track": small_track,
                           "large_name": large_name, "episode": 0,
                           "first_frame": 0, "last_frame": 0,
                           "supported_frames": 0, "changed_pixels": 0,
                           "outcome": "rejected", "reason": "no_complete_supported_episode"})
            continue
        for episode_index, episode in enumerate(episodes, start=1):
            if any(np.any(occupied[t] & states[t]["small"]) for t in episode):
                audits.append({"large_track": large_track, "small_track": small_track,
                               "large_name": large_name, "episode": episode_index,
                               "first_frame": episode[0]+1, "last_frame": episode[-1]+1,
                               "supported_frames": len(episode), "changed_pixels": 0,
                               "outcome": "rejected", "reason": "overlaps_another_proposal"})
                continue
            total = 0
            for t in episode:
                item = states[t]
                pixels = int(item["small_area"])
                candidate[t][item["small"]] = large_name
                occupied[t][item["small"]] = True
                total += pixels
                applications.append({
                    "large_track": large_track, "small_track": small_track,
                    "episode": episode_index, "frame": t+1,
                    "old_name": int(item["small_name"]),
                    "new_name": large_name, "changed_pixels": pixels,
                    "gap_px": float(item["gap_px"]),
                    "raw_q95_ratio": float(item["raw_q95_ratio"]),
                    "temporal_absorption": float(item["temporal_absorption"]),
                })
            audits.append({"large_track": large_track, "small_track": small_track,
                           "large_name": large_name, "episode": episode_index,
                           "first_frame": episode[0]+1, "last_frame": episode[-1]+1,
                           "supported_frames": len(episode), "changed_pixels": total,
                           "outcome": "applied", "reason": "complete_supported_branch_episode"})
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels)))):
        raise AssertionError("recurrent branch correction crossed an accepted boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
