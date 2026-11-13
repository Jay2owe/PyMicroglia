"""Freeze each numerical identity from its earliest movie-wide observation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import find_project_root
import lineage_hierarchy as local


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack  # noqa: E402


LAYER_ORDER = {
    "large_static_isolated": 0,
    "large_static_near_smaller": 1,
    "large_mobile": 2,
    "small_or_dim": 3,
}


def earliest_tracklet_by_identity(tracklets: list[local.Tracklet]
                                  ) -> dict[int, local.Tracklet]:
    result: dict[int, local.Tracklet] = {}
    for tracklet in tracklets:
        prior = result.get(tracklet.identity)
        if prior is None or (
                tracklet.birth.frame, tracklet.number) < (
                    prior.birth.frame, prior.number):
            result[tracklet.identity] = tracklet
    return result


def movie_authority(labels: np.ndarray, cfg: dict
                    ) -> tuple[pd.DataFrame, list[list[local.Component]]]:
    frames = [local.frame_components(
        labels[frame], frame, int(cfg["minimum_component_area_px"]))
        for frame in range(len(labels))]
    tracklet_cfg = {
        "maximum_tracklet_gap_frames": 2,
        "same_identity_tracklet_radius_px": 30.0,
    }
    tracklets = local.build_tracklets(frames, tracklet_cfg)
    earliest = earliest_tracklet_by_identity(tracklets)
    rows = []
    for identity in sorted(earliest):
        tracklet = earliest[identity]
        count = int(cfg["classification_observations"])
        observations = tracklet.nodes[:count]
        birth = tracklet.birth
        birth_small = birth.area <= int(cfg["small_birth_area_px"])
        if birth_small:
            layer = "small_or_dim"
            p75 = np.nan
            near_fraction = 0.0
        else:
            steps = [local.distance(first, second)
                     for first, second in zip(observations,
                                              observations[1:])]
            p75 = float(np.percentile(steps, 75)) if steps else 0.0
            near = 0
            for node in observations:
                smaller = [item for item in frames[node.frame]
                           if item.identity != identity
                           and item.area <= int(cfg["small_birth_area_px"])]
                if smaller and min(local.approximate_edge_gap(node, item)
                                   for item in smaller) <= float(
                                       cfg["near_smaller_edge_gap_px"]):
                    near += 1
            near_fraction = near / max(len(observations), 1)
            if p75 > float(cfg["static_maximum_p75_step_px"]):
                layer = "large_mobile"
            elif near_fraction >= float(cfg["near_smaller_frame_fraction"]):
                layer = "large_static_near_smaller"
            else:
                layer = "large_static_isolated"
        rows.append({
            "identity": identity,
            "birth_output_frame": birth.frame + 1,
            "birth_source_imagej_frame": birth.frame + 3,
            "birth_area_px": birth.area,
            "earliest_tracklet_id": tracklet.number,
            "classification_observations_used": len(observations),
            "early_median_area_px": float(np.median(
                [node.area for node in observations])),
            "early_p75_step_px": p75,
            "early_near_smaller_fraction": near_fraction,
            "confidence_layer": layer,
            "priority_rank": LAYER_ORDER[layer],
            "permanently_birth_small": birth_small,
            "later_tracklets_can_promote": False,
        })
    return pd.DataFrame(rows), frames

