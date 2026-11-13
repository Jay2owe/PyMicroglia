"""Remove transient weak tracker seats embedded in a dominant soma outline.

Selection is field-wide and uses only temporal coverage, scale-normalized
motion, raw intensity and shared-outline evidence. It has no recording,
identity, fixed track, coordinate, region, frame or feedback selector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as connected_labels

MIN_STRONG_FRACTION = .10
MAX_STRONG_FRACTION = .25
MIN_NAMED_MOVIE_FRACTION = .05
MAX_NAMED_FRACTION = .60
MAX_MEDIAN_STEP_RADII = 1.2
MAX_MEDIAN_RAW_FACTOR = 5.
MIN_DOMINANT_PEER_FRACTION = .5
MIN_PEER_RAW_MULTIPLE = 5.
MIN_PEER_PATH_MULTIPLE = 2.
MIN_MEDIAN_PEER_RAW_RATIO = 10.


def _measure_tracks(labels: np.ndarray, raw: np.ndarray,
                    points: pd.DataFrame) -> tuple[set[int], dict[int, set[int]], pd.DataFrame]:
    background = np.array([np.median(f[f > 0]) if np.any(f > 0) else 1.
                           for f in raw], dtype=float)
    strong = points[(points.state == "observed") & points.strong.astype(bool)].copy()
    strong["track_id"] = strong.track_id.astype(int)
    path_length = strong.groupby("track_id").size().astype(int).to_dict()
    observations = []
    for t, group in strong.groupby("frame"):
        t = int(t)
        component = connected_labels(labels[t], background=0, connectivity=2)
        frame_rows = []
        for point in group.itertuples(index=False):
            y, x = round(float(point.y)), round(float(point.x))
            index = int(component[y, x])
            frame_rows.append({"track_id": int(point.track_id), "frame": t,
                               "component": index, "owner": int(labels[t, y, x]),
                               "raw_factor": float(raw[t, y, x] / background[t])})
        table = pd.DataFrame(frame_rows)
        # Read the same peer columns once, preserving table order and float dtype.
        component_ids = table.component.to_numpy()
        track_ids = table.track_id.to_numpy()
        raw_factors = table.raw_factor.to_numpy(float)
        for row in table.itertuples(index=False):
            peers = ((component_ids == row.component) & (component_ids > 0)
                     & (track_ids != row.track_id))
            if np.any(peers):
                ratios = raw_factors[peers] / max(float(row.raw_factor), .01)
                peer_raw_ratio = float(np.max(ratios))
                peer_path_multiple = float(max(path_length[int(v)] for v in track_ids[peers])
                                           / max(path_length[int(row.track_id)], 1))
            else:
                peer_raw_ratio = 0.
                peer_path_multiple = 0.
            observations.append({"track_id": int(row.track_id), "frame": t,
                                 "owner": int(row.owner), "raw_factor": float(row.raw_factor),
                                 "peer_raw_ratio": peer_raw_ratio,
                                 "peer_path_multiple": peer_path_multiple})
    obs = pd.DataFrame(observations)
    eligible: set[int] = set()
    eligible_names: dict[int, set[int]] = {}
    audit = []
    for track_id, group in strong.groupby("track_id"):
        track_id = int(track_id)
        group = group.sort_values("frame")
        measured = obs[obs.track_id == track_id]
        named = measured[measured.owner > 0]
        radius = float(np.median(group.radius_px.to_numpy(float)))
        if len(group) > 1:
            xy = group[["x", "y"]].to_numpy(float)
            frame = group.frame.to_numpy(int)
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(frame).clip(min=1)
            step_radii = float(np.median(step) / max(radius, .01))
        else:
            step_radii = float("inf")
        strong_fraction = float(len(group) / len(labels))
        named_fraction = float(len(named) / len(group))
        median_raw = float(named.raw_factor.median()) if len(named) else float("inf")
        dominant = ((named.peer_raw_ratio >= MIN_PEER_RAW_MULTIPLE)
                    & (named.peer_path_multiple >= MIN_PEER_PATH_MULTIPLE))
        dominant_fraction = float(dominant.mean()) if len(named) else 0.
        shared = named[named.peer_raw_ratio > 0]
        median_peer_ratio = float(shared.peer_raw_ratio.median()) if len(shared) else 0.
        passes = (MIN_STRONG_FRACTION <= strong_fraction <= MAX_STRONG_FRACTION
                  and len(named) >= int(np.ceil(MIN_NAMED_MOVIE_FRACTION * len(labels)))
                  and named_fraction <= MAX_NAMED_FRACTION
                  and step_radii <= MAX_MEDIAN_STEP_RADII
                  and median_raw <= MAX_MEDIAN_RAW_FACTOR
                  and dominant_fraction >= MIN_DOMINANT_PEER_FRACTION
                  and median_peer_ratio >= MIN_MEDIAN_PEER_RAW_RATIO)
        if passes:
            eligible.add(track_id)
            eligible_names[track_id] = set(map(int, named.owner.unique()))
        audit.append({"track_id": track_id, "strong_frames": len(group),
                      "strong_fraction": strong_fraction, "named_frames": len(named),
                      "named_fraction": named_fraction, "median_step_radii": step_radii,
                      "median_raw_factor": median_raw,
                      "dominant_peer_fraction": dominant_fraction,
                      "median_peer_raw_ratio": median_peer_ratio,
                      "outcome": "eligible" if passes else "outside_gate"})
    return eligible, eligible_names, pd.DataFrame(audit)


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.shape != raw.shape or labels.shape != protected.shape
            or labels.ndim != 3):
        raise ValueError("aligned complete label, raw and protection stacks required")
    candidate = labels.copy()
    eligible, eligible_names, track_audit = _measure_tracks(labels, raw, points)
    applications = []
    if not eligible:
        return candidate, track_audit, pd.DataFrame(applications)
    episode_points = points[points.track_id.astype(int).isin(eligible)].copy()
    strong = points[(points.state == "observed") & points.strong.astype(bool)].copy()
    all_by_frame = {int(t): g for t, g in episode_points.groupby("frame")}
    strong_by_frame = {int(t): g for t, g in strong.groupby("frame")}
    for t, frame in enumerate(labels):
        targets = all_by_frame.get(t, episode_points.iloc[:0])
        if targets.empty:
            continue
        components = connected_labels(frame, background=0, connectivity=2)
        boxes = ndi.find_objects(components)
        selected: dict[int, list[int]] = {}
        for point in targets.itertuples(index=False):
            y, x = round(float(point.y)), round(float(point.x))
            track_id = int(point.track_id)
            if (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]
                    and int(frame[y, x]) in eligible_names[track_id]):
                selected.setdefault(int(components[y, x]), []).append(track_id)
        frame_strong = strong_by_frame.get(t, strong.iloc[:0])
        for index, target_tracks in sorted(selected.items()):
            if index <= 0:
                continue
            box = boxes[index - 1]
            source = components[box] == index
            remove_seed = np.zeros(source.shape, bool)
            keep_seed = np.zeros(source.shape, bool)
            for point in targets.itertuples(index=False):
                if int(point.track_id) not in target_tracks:
                    continue
                y, x = round(float(point.y)), round(float(point.x))
                if box[0].start <= y < box[0].stop and box[1].start <= x < box[1].stop:
                    yy, xx = y - box[0].start, x - box[1].start
                    if source[yy, xx]:
                        remove_seed[yy, xx] = True
            protected_tracks = []
            for point in frame_strong.itertuples(index=False):
                if int(point.track_id) in eligible:
                    continue
                y, x = round(float(point.y)), round(float(point.x))
                if box[0].start <= y < box[0].stop and box[1].start <= x < box[1].stop:
                    yy, xx = y - box[0].start, x - box[1].start
                    if source[yy, xx]:
                        keep_seed[yy, xx] = True
                        protected_tracks.append(int(point.track_id))
            if not remove_seed.any() or not keep_seed.any():
                applications.append({"frame": t + 1,
                                     "source_name": int(frame[box][source][0]),
                                     "changed_pixels": 0,
                                     "eligible_track_count": len(set(target_tracks)),
                                     "protected_track_count": len(set(protected_tracks)),
                                     "outcome": "skipped_without_both_seats"})
                continue
            distance_to_remove = ndi.distance_transform_edt(~remove_seed)
            distance_to_keep = ndi.distance_transform_edt(~keep_seed)
            remove = source & (distance_to_remove < distance_to_keep)
            name = int(frame[box][source][0])
            if np.any(remove & protected[t][box]):
                applications.append({"frame": t + 1, "source_name": name,
                                     "changed_pixels": 0,
                                     "eligible_track_count": len(set(target_tracks)),
                                     "protected_track_count": len(set(protected_tracks)),
                                     "outcome": "rejected_prior_accepted_pixel"})
                continue
            view = candidate[t][box]
            if np.any(view[remove] != name):
                raise AssertionError("weak-seat component changed before application")
            view[remove] = 0
            applications.append({"frame": t + 1, "source_name": name,
                                 "changed_pixels": int(remove.sum()),
                                 "eligible_track_count": len(set(target_tracks)),
                                 "protected_track_count": len(set(protected_tracks)),
                                 "outcome": "applied"})
    return candidate, track_audit, pd.DataFrame(applications)
