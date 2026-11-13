"""Detect independent red destinations that compete with a forced host carry."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))


def _load(path: Path) -> np.ndarray:
    import tifffile
    return tifffile.imread(path)


def _objects(frame: np.ndarray, minimum_px: int) -> list[dict]:
    result: list[dict] = []
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        components, count = ndi.label(frame == identity,
                                      structure=np.ones((3, 3), np.uint8))
        for component in range(1, count + 1):
            mask = components == component
            area = int(mask.sum())
            if area < minimum_px:
                continue
            yy, xx = np.nonzero(mask)
            result.append({
                "identity": identity,
                "component": component,
                "mask": mask,
                "area_px": area,
                "centroid_x": float(xx.mean()),
                "centroid_y": float(yy.mean()),
            })
    return result


def _overlap_identity(frame: np.ndarray, mask: np.ndarray) -> int:
    values, counts = np.unique(frame[mask], return_counts=True)
    choices = [(int(count), int(value)) for value, count in zip(values, counts)
               if int(value) != 0]
    return max(choices)[1] if choices else 0


def _area_similarity(a: int, b: int) -> float:
    return abs(math.log2(max(a, 1) / max(b, 1)))


def detect(accepted: np.ndarray, pre: np.ndarray, lag: np.ndarray,
           audit: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(cfg["motion_log2_threshold"])
    minimum_px = int(cfg["minimum_object_px"])
    max_distance = float(cfg["maximum_radial_distance_px"])
    min_loss = float(cfg["minimum_source_loss_coverage"])
    min_gain = float(cfg["minimum_destination_gain_coverage"])
    history_frames = int(cfg["history_frames"])

    events: list[dict] = []
    candidates: list[dict] = []
    for (target, host), group in audit.groupby(
            ["identity", "host_identity"], sort=False):
        first = group.sort_values("t").iloc[0]
        event_id = str(first.event_id)
        current = int(first.t)
        previous = current - 1
        source_reference = accepted[previous] == int(target)
        source_identity = _overlap_identity(pre[previous], source_reference)
        source_parts = [item for item in _objects(pre[previous], minimum_px)
                        if item["identity"] == source_identity
                        and np.any(item["mask"] & source_reference)]
        if not source_parts:
            continue
        source = max(source_parts,
                     key=lambda item: int(np.count_nonzero(
                         item["mask"] & source_reference)))
        transition = lag[previous]
        source_loss = float(np.mean(transition[source["mask"]] <= -threshold))
        wrong_piece = accepted[current] == int(target)
        wrong_history = pre[max(0, current - history_frames):current]
        wrong_source_history = int(np.count_nonzero(
            np.any(wrong_history == source_identity, axis=0) & wrong_piece))
        wrong_host_history = int(np.count_nonzero(
            np.any(wrong_history == int(host), axis=0) & wrong_piece))

        blue_sources = []
        for prior in _objects(pre[previous], minimum_px):
            loss = float(np.mean(transition[prior["mask"]] <= -threshold))
            if loss >= min_loss:
                blue_sources.append((prior, loss))

        recent = pre[max(0, current - history_frames):current]
        false_host_frames = int(len(group))
        event = {
            "event_id": str(event_id),
            "target_identity": int(target),
            "host_identity": int(host),
            "source_frame": previous,
            "source_imagej_frame": previous + 1,
            "destination_frame": current,
            "destination_imagej_frame": current + 1,
            "source_local_identity": source_identity,
            "source_area_px": source["area_px"],
            "source_centroid_x": source["centroid_x"],
            "source_centroid_y": source["centroid_y"],
            "source_loss_coverage": source_loss,
            "forced_piece_area_px": int(wrong_piece.sum()),
            "forced_area_ratio": float(wrong_piece.sum() / source["area_px"]),
            "forced_gain_host_overlap": float(first.gain_host_overlap),
            "wrong_source_history_px": wrong_source_history,
            "wrong_host_history_px": wrong_host_history,
            "false_host_frames": false_host_frames,
        }
        events.append(event)

        for destination in _objects(pre[current], minimum_px):
            mask = destination["mask"]
            gain = float(np.mean(transition[mask] >= threshold))
            distance = float(math.hypot(
                destination["centroid_x"] - source["centroid_x"],
                destination["centroid_y"] - source["centroid_y"]))
            if gain < min_gain or distance > max_distance:
                continue
            source_options = []
            for prior, loss in blue_sources:
                source_distance = float(math.hypot(
                    destination["centroid_x"] - prior["centroid_x"],
                    destination["centroid_y"] - prior["centroid_y"]))
                if source_distance > max_distance:
                    continue
                cost = (source_distance / max_distance
                        + 0.2 * _area_similarity(
                            destination["area_px"], prior["area_px"]))
                source_options.append((cost, prior["identity"], loss))
            source_options.sort()
            best = source_options[0] if source_options else (math.inf, 0, 0.0)
            second_cost = source_options[1][0] if len(source_options) > 1 else math.inf
            recent_source = int(np.count_nonzero(
                np.any(recent == source_identity, axis=0) & mask))
            competing = []
            for other in set(map(int, np.unique(recent))) - {0, source_identity}:
                competing.append(int(np.count_nonzero(
                    np.any(recent == other, axis=0) & mask)))
            candidates.append({
                "event_id": str(event_id),
                "destination_local_identity": destination["identity"],
                "destination_component": destination["component"],
                "destination_area_px": destination["area_px"],
                "destination_centroid_x": destination["centroid_x"],
                "destination_centroid_y": destination["centroid_y"],
                "destination_gain_coverage": gain,
                "radial_distance_px": distance,
                "area_log2_change": _area_similarity(
                    destination["area_px"], source["area_px"]),
                "recent_source_territory_px": recent_source,
                "maximum_competing_territory_px": max(competing, default=0),
                "best_blue_source_identity": int(best[1]),
                "best_blue_source_loss_coverage": float(best[2]),
                "red_to_blue_cost": float(best[0]),
                "red_to_blue_margin": float(second_cost - best[0]),
                "red_source_is_target": bool(int(best[1]) == source_identity),
                "independent_of_forced_host": bool(
                    destination["identity"] != int(host)
                    and not np.any(mask & wrong_piece)),
            })
    return pd.DataFrame(events), pd.DataFrame(candidates)

