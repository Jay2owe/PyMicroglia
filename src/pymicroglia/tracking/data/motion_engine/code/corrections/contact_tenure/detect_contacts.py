"""Detect conserved two-cell contacts from uninterrupted local episodes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))


def _position(frame: np.ndarray, identity: int) -> np.ndarray | None:
    points = np.column_stack(np.nonzero(frame == identity))
    return points.mean(axis=0) if len(points) else None


def _episode_motion(labels: np.ndarray, t: int, identity: int,
                    lookback: int) -> dict | None:
    start = t - lookback
    if start < 0:
        return None
    positions = []
    for frame in range(start, t):
        position = _position(labels[frame], identity)
        if position is None:
            return None
        positions.append(position)
    positions = np.asarray(positions)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return {
        "median_step_px": float(np.median(steps)),
        "p75_step_px": float(np.percentile(steps, 75)),
        "net_displacement_px": float(np.linalg.norm(
            positions[-1] - positions[0])),
    }


def _tenure_seed_size(labels: np.ndarray, t: int, identity: int,
                      host: np.ndarray, lookback: int,
                      minimum_frames: int) -> int:
    tenure = np.count_nonzero(
        labels[t - lookback:t] == identity, axis=0)
    return int(np.count_nonzero(host & (tenure >= minimum_frames)))


def detect_contacts(labels: np.ndarray, params: dict) -> pd.DataFrame:
    """Find static identities lost inside an area-conserving mobile host."""
    lookback = int(params["lookback_frames"])
    structure = np.ones((3, 3), np.uint8)
    rows: list[dict] = []
    event_number = 0
    for t in range(1, len(labels)):
        previous, current = labels[t - 1], labels[t]
        components, count = ndi.label(current > 0, structure=structure)
        for component in range(1, count + 1):
            host = components == component
            values, counts = np.unique(previous[host], return_counts=True)
            predecessors = []
            for value, intersection in zip(values, counts):
                identity = int(value)
                if identity == 0:
                    continue
                area = int(np.count_nonzero(previous == identity))
                coverage = int(intersection) / max(area, 1)
                if (int(intersection) >= int(params["minimum_overlap_px"])
                        and coverage >= float(params["minimum_source_coverage"])):
                    predecessors.append({
                        "identity": identity,
                        "intersection_px": int(intersection),
                        "area_px": area,
                        "source_coverage": float(coverage),
                    })
            if len(predecessors) < 2:
                continue
            predecessors.sort(
                key=lambda row: row["intersection_px"], reverse=True)
            first, second = predecessors[:2]
            area_ratio = int(host.sum()) / max(
                first["area_px"] + second["area_px"], 1)
            if not (float(params["minimum_combined_area_ratio"])
                    <= area_ratio
                    <= float(params["maximum_combined_area_ratio"])):
                continue
            owners = sorted(set(map(int, np.unique(current[host]))) - {0})
            if len(owners) != 1:
                continue
            owner = owners[0]
            pair = [first, second]
            motion = {
                row["identity"]: _episode_motion(
                    labels, t, row["identity"], lookback)
                for row in pair
            }
            if any(value is None for value in motion.values()):
                continue
            for static in pair:
                static_id = static["identity"]
                if static_id == owner:
                    continue
                mobile = second if static is first else first
                mobile_id = mobile["identity"]
                static_motion = motion[static_id]
                mobile_motion = motion[mobile_id]
                assert static_motion is not None and mobile_motion is not None
                static_allowed = bool(
                    static_motion["p75_step_px"]
                    <= float(params["static_maximum_p75_step_px"])
                    and static_motion["net_displacement_px"]
                    <= float(params["static_maximum_net_displacement_px"]))
                mobile_allowed = bool(
                    mobile_motion["p75_step_px"]
                    >= float(params["mobile_minimum_p75_step_px"])
                    or mobile_motion["net_displacement_px"]
                    >= float(params["mobile_minimum_net_displacement_px"]))
                if not static_allowed or not mobile_allowed:
                    continue
                static_seed = _tenure_seed_size(
                    labels, t, static_id, host, lookback,
                    int(params["minimum_tenure_frames"]))
                mobile_seed = _tenure_seed_size(
                    labels, t, mobile_id, host, lookback,
                    int(params["minimum_tenure_frames"]))
                if min(static_seed, mobile_seed) < int(
                        params["minimum_tenure_seed_px"]):
                    continue
                event_number += 1
                rows.append({
                    "event_id": f"TT{event_number:04d}",
                    "t": t,
                    "imagej_frame": t + 1,
                    "static_identity": static_id,
                    "mobile_identity": mobile_id,
                    "input_owner": owner,
                    "host_area_px": int(host.sum()),
                    "static_previous_area_px": int(static["area_px"]),
                    "mobile_previous_area_px": int(mobile["area_px"]),
                    "combined_area_ratio": float(area_ratio),
                    "static_source_coverage": float(
                        static["source_coverage"]),
                    "mobile_source_coverage": float(
                        mobile["source_coverage"]),
                    "static_p75_step_px": float(
                        static_motion["p75_step_px"]),
                    "static_net_displacement_px": float(
                        static_motion["net_displacement_px"]),
                    "mobile_p75_step_px": float(
                        mobile_motion["p75_step_px"]),
                    "mobile_net_displacement_px": float(
                        mobile_motion["net_displacement_px"]),
                    "static_tenure_seed_px": static_seed,
                    "mobile_tenure_seed_px": mobile_seed,
                    "reason": "locally static established territory and mobile neighbour survive in one area-conserving host",
                })
    return pd.DataFrame(rows)

