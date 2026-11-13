"""Original raw-linker loop with one declared eligibility boundary before solving.

The source loop is retained from latent_body_tracks.link_observations. Helpers,
cost terms, prediction, assignment solver and birth handling remain shared.
The optional eligibility rule is isolated in measured_core_gap_eligibility.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from latent_body_tracks import Track, _predicted, _patch, _motion_support_plane
from measured_core_gap_eligibility import constrain_gap_links

def link_observations(
        hypotheses: pd.DataFrame, raw: np.ndarray,
        motion: np.ndarray | None, params: dict, audit=None) -> list[Track]:
    maximum_gap = int(params.get("maximum_track_gap_frames", 16))
    base_distance = float(params.get("base_link_distance_px", 7.0))
    gap_distance = float(params.get("additional_link_px_per_gap", 1.25))
    maximum_distance = float(params.get("maximum_link_distance_px", 18.0))
    damping = float(params.get("velocity_prediction_damping", 0.5))
    area_weight = float(params.get("area_cost_weight", 0.25))
    appearance_weight = float(params.get("appearance_cost_weight", 0.50))
    motion_weight = float(params.get("motion_cost_reward", 0.15))
    motion_extension = float(params.get("motion_link_extension_px", 0.0))
    maximum_gap_bridge_speed_radii = float(
        params.get("maximum_gap_bridge_speed_sum_radii_per_frame", np.inf))
    tracks: list[Track] = []
    next_id = 1
    motion_planes = (np.max(motion[:, 1:], axis=1) if motion is not None
                     and motion.ndim == 4 and motion.shape[1] > 1 else None)
    by_frame = {int(frame): group.to_dict("records")
                for frame, group in hypotheses.groupby("frame")}
    for frame in range(len(raw)):
        detections = by_frame.get(frame, [])
        motion_plane = (motion_planes[min(frame, len(motion_planes) - 1)]
                        if motion_planes is not None and len(motion_planes)
                        else None)
        active = [index for index, track in enumerate(tracks)
                  if 0 < frame - int(track.observations[-1]["frame"])
                  <= maximum_gap]
        cost = np.full((len(active), len(detections)), 1e6, np.float64)
        for ai, track_index in enumerate(active):
            track = tracks[track_index]
            last = track.observations[-1]
            gap = frame - int(last["frame"])
            px, py = _predicted(track, frame, damping)
            allowed = min(maximum_distance,
                          base_distance + gap_distance * max(0, gap - 1))
            last_patch = _patch(raw[int(last["frame"])], last["x"], last["y"])
            for di, detection in enumerate(detections):
                endpoint_distance = float(np.hypot(
                    float(detection["x"]) - float(last["x"]),
                    float(detection["y"]) - float(last["y"])))
                endpoint_radii = max(
                    float(last["radius_px"])
                    + float(detection["radius_px"]), 1e-6)
                if (gap > 1
                        and endpoint_distance / (endpoint_radii * gap)
                        > maximum_gap_bridge_speed_radii):
                    continue
                distance = float(np.hypot(
                    float(detection["x"]) - px,
                    float(detection["y"]) - py))
                if distance > maximum_distance:
                    continue
                motion_support = _motion_support_plane(
                    motion_plane, last["x"], last["y"],
                    detection["x"], detection["y"])
                adaptive_allowed = min(
                    maximum_distance, allowed + motion_extension * motion_support)
                if distance > adaptive_allowed:
                    continue
                area_change = abs(np.log(
                    max(float(detection["core_area_px"]), 1.0)
                    / max(float(last["core_area_px"]), 1.0)))
                current_patch = _patch(
                    raw[frame], detection["x"], detection["y"])
                correlation = float(np.clip(
                    np.dot(last_patch, current_patch), -1.0, 1.0))
                cost[ai, di] = (
                    distance / max(adaptive_allowed, 1e-6)
                    + area_weight * area_change
                    + appearance_weight * (1.0 - correlation)
                    - motion_weight * motion_support)
        cost = constrain_gap_links(cost, tracks, active, detections, frame, params, audit)
        used_detections: set[int] = set()
        if cost.size:
            rows, columns = linear_sum_assignment(cost)
            for ai, di in zip(rows, columns):
                if cost[ai, di] >= 1e5:
                    continue
                tracks[active[int(ai)]].observations.append(detections[int(di)])
                used_detections.add(int(di))
        for di, detection in enumerate(detections):
            if di in used_detections or not bool(detection["strong"]):
                continue
            tracks.append(Track(next_id, [detection]))
            next_id += 1
    return tracks

