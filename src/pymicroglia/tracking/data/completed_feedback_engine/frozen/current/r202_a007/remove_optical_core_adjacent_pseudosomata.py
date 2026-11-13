"""Remove long-lived, nearly static pseudo-somata beside optical cores.

Eligibility uses raw intensity, scale-normalized motion, temporal coverage,
distance to image-defined optical cores and current foreground occupancy.
There are no recording, identity, coordinate, frame, region or feedback
selectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as connected_labels

import remove_persistent_overbright_sources as large_core
import remove_compact_stable_overbright_sources as compact_core
import remove_extreme_compact_overbright_sources as extreme_core

MIN_TRACK_FRACTION = .8
MIN_NAMED_FRACTION = .8
MAX_MEDIAN_STEP_RADII = .5
MIN_MEDIAN_RAW_FACTOR = 90.
MAX_MEDIAN_CORE_DISTANCE_RADII = 4.


def _optical_core_union(raw: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce([
        large_core._eligible_cores(raw)[0],
        compact_core._eligible_cores(raw)[0],
        extreme_core._eligible_cores(raw)[0],
    ])


def _eligible_tracks(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
                     core: np.ndarray) -> tuple[set[int], pd.DataFrame]:
    background = np.array([np.median(frame[frame > 0]) if np.any(frame > 0) else 1.
                           for frame in raw], dtype=float)
    core_distance = ndi.distance_transform_edt(~core)
    strong = points[(points.state == "observed") & points.strong.astype(bool)].copy()
    eligible: set[int] = set()
    rows = []
    for track_id, group in strong.groupby("track_id"):
        group = group.sort_values("frame")
        frame = group.frame.to_numpy(int)
        xy = group[["x", "y"]].to_numpy(float)
        yx = np.rint(group[["y", "x"]].to_numpy(float)).astype(int)
        in_bounds = ((frame >= 0) & (frame < len(labels))
                     & (yx[:, 0] >= 0) & (yx[:, 0] < labels.shape[1])
                     & (yx[:, 1] >= 0) & (yx[:, 1] < labels.shape[2]))
        if not in_bounds.all() or not len(group):
            continue
        radius = float(np.median(group.radius_px.to_numpy(float)))
        if not np.isfinite(radius) or radius <= 0:
            continue
        owner = labels[frame, yx[:, 0], yx[:, 1]]
        named_fraction = float(np.mean(owner > 0))
        if len(group) > 1:
            delta_frame = np.diff(frame).clip(min=1)
            step_per_frame = np.linalg.norm(np.diff(xy, axis=0), axis=1) / delta_frame
            step_radii = float(np.median(step_per_frame) / radius)
        else:
            step_radii = float("inf")
        raw_factor = raw[frame, yx[:, 0], yx[:, 1]] / background[frame]
        median_raw_factor = float(np.median(raw_factor))
        median_core_distance_radii = float(np.median(
            core_distance[yx[:, 0], yx[:, 1]]) / radius)
        passes = (len(group) >= int(np.ceil(MIN_TRACK_FRACTION * len(labels)))
                  and named_fraction >= MIN_NAMED_FRACTION
                  and step_radii <= MAX_MEDIAN_STEP_RADII
                  and median_raw_factor >= MIN_MEDIAN_RAW_FACTOR
                  and 0 < median_core_distance_radii <= MAX_MEDIAN_CORE_DISTANCE_RADII)
        if passes:
            eligible.add(int(track_id))
        rows.append({"track_id": int(track_id), "strong_frames": len(group),
                     "named_fraction": named_fraction,
                     "median_radius_px": radius,
                     "median_step_radii": step_radii,
                     "median_raw_factor": median_raw_factor,
                     "median_core_distance_radii": median_core_distance_radii,
                     "outcome": "eligible" if passes else "outside_gate"})
    return eligible, pd.DataFrame(rows)


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.shape != raw.shape or labels.shape != protected.shape
            or labels.ndim != 3):
        raise ValueError("aligned complete label, raw and protection stacks required")
    candidate = labels.copy()
    core = _optical_core_union(raw)
    eligible, track_audit = _eligible_tracks(labels, raw, points, core)
    applications = []
    if not eligible:
        return candidate, track_audit, pd.DataFrame(applications)
    strong = points[(points.state == "observed") & points.strong.astype(bool)].copy()
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    for t, frame in enumerate(labels):
        frame_points = by_frame.get(t, strong.iloc[:0])
        bad_points = frame_points[frame_points.track_id.astype(int).isin(eligible)]
        if bad_points.empty:
            continue
        components = connected_labels(frame, background=0, connectivity=2)
        boxes = ndi.find_objects(components)
        selected = set()
        for point in bad_points.itertuples(index=False):
            y, x = round(float(point.y)), round(float(point.x))
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and frame[y, x] > 0:
                selected.add(int(components[y, x]))
        for index in sorted(selected):
            if index <= 0:
                continue
            box = boxes[index - 1]
            source = components[box] == index
            remove_seed = np.zeros(source.shape, bool)
            keep_seed = np.zeros(source.shape, bool)
            removed_tracks, protected_tracks = [], []
            for point in frame_points.itertuples(index=False):
                y, x = round(float(point.y)), round(float(point.x))
                if not (box[0].start <= y < box[0].stop
                        and box[1].start <= x < box[1].stop):
                    continue
                yy, xx = y - box[0].start, x - box[1].start
                if not source[yy, xx]:
                    continue
                if int(point.track_id) in eligible:
                    remove_seed[yy, xx] = True
                    removed_tracks.append(int(point.track_id))
                else:
                    keep_seed[yy, xx] = True
                    protected_tracks.append(int(point.track_id))
            if not remove_seed.any():
                continue
            if keep_seed.any():
                distance_to_remove = ndi.distance_transform_edt(~remove_seed)
                distance_to_keep = ndi.distance_transform_edt(~keep_seed)
                remove = source & (distance_to_remove < distance_to_keep)
                method = "partition_around_other_somata"
            else:
                remove = source
                method = "entire_pseudosoma_outline"
            name = int(frame[box][source][0])
            if np.any(remove & protected[t][box]):
                applications.append({"frame": t + 1, "source_name": name,
                                     "changed_pixels": 0,
                                     "removed_track_count": len(set(removed_tracks)),
                                     "protected_track_count": len(set(protected_tracks)),
                                     "method": method,
                                     "outcome": "rejected_prior_accepted_pixel"})
                continue
            view = candidate[t][box]
            if np.any(view[remove] != name):
                raise AssertionError("pseudo-soma component changed before application")
            view[remove] = 0
            applications.append({"frame": t + 1, "source_name": name,
                                 "changed_pixels": int(remove.sum()),
                                 "removed_track_count": len(set(removed_tracks)),
                                 "protected_track_count": len(set(protected_tracks)),
                                 "method": method, "outcome": "applied"})
    return candidate, track_audit, pd.DataFrame(applications)
