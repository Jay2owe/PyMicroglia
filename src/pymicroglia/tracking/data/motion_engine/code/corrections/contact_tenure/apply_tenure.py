"""Keep locally static territory and its mobile neighbour through contact."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))

import sys
sys.path.insert(0, str(ROOT / "code"))

from persistent_merges import (  # noqa: E402
    _candidate_seeds,
    _identity_shape,
    _nearest_host_pixel,
    _partition_score,
)
from tracking import motion_support  # noqa: E402


STRUCTURE = np.ones((3, 3), np.uint8)


def _observation(mask: np.ndarray) -> dict:
    points = np.column_stack(np.nonzero(mask))
    if not len(points):
        raise ValueError("empty observation")
    return {
        "mask": mask,
        "position": points.mean(axis=0),
        "area": int(len(points)),
    }


def _component_with_most_overlap(mask: np.ndarray,
                                 reference: np.ndarray) -> np.ndarray | None:
    components, count = ndi.label(mask, structure=STRUCTURE)
    if not count:
        return None
    overlap = np.bincount(
        components[reference].ravel(), minlength=count + 1)
    if len(overlap) <= 1 or int(overlap[1:].max()) == 0:
        return None
    return components == int(np.argmax(overlap[1:]) + 1)


def _label_observations(frame: np.ndarray,
                        excluded: np.ndarray) -> list[dict]:
    """Label each identity inside its bounding box, from one find_objects pass.

    Boxes come in ascending identity order, as np.unique lists them; components
    keep their raster order inside a box, a component overlaps the excluded
    pixels exactly when its crop does, and its coordinates are the crop's
    shifted by the box origin, so the rows, keys and means are the whole-frame
    ones. Negative or very large identities and other excluded masks take the
    whole-frame route.
    """
    if not (isinstance(frame, np.ndarray) and frame.ndim == 2 and frame.size and frame.dtype.kind in "iu"
            and isinstance(excluded, np.ndarray) and excluded.dtype == np.bool_
            and excluded.shape == frame.shape):
        return _label_observations_whole_frame(frame, excluded)
    if int(frame.min()) < 0 or int(frame.max()) >= 1 << 20:
        return _label_observations_whole_frame(frame, excluded)
    rows: list[dict] = []
    for index, box in enumerate(ndi.find_objects(frame)):
        if box is None:
            continue
        identity = index + 1
        components, count = ndi.label(frame[box] == identity, structure=STRUCTURE)
        outside = excluded[box]
        y0, x0 = box[0].start, box[1].start
        for component in range(1, count + 1):
            local = components == component
            if np.any(local & outside):
                continue
            yy, xx = np.nonzero(local)
            mask = np.zeros(frame.shape, bool)
            mask[box] = local
            points = np.column_stack((yy + y0, xx + x0))
            row = {"mask": mask, "position": points.mean(axis=0), "area": int(len(points))}
            row["input_identity"] = identity
            rows.append(row)
    return rows


def _label_observations_whole_frame(frame: np.ndarray,
                                    excluded: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        components, count = ndi.label(frame == identity, structure=STRUCTURE)
        for component in range(1, count + 1):
            mask = components == component
            if np.any(mask & excluded):
                continue
            row = _observation(mask)
            row["input_identity"] = identity
            rows.append(row)
    return rows


def _tenure_scores(labels: np.ndarray, t: int, identities: tuple[int, int],
                   lookback: int, decay: float) -> dict[int, np.ndarray]:
    scores = {identity: np.zeros(labels.shape[1:], np.float32)
              for identity in identities}
    for age, frame in enumerate(range(t - 1, max(-1, t - lookback - 1), -1)):
        weight = decay ** age
        for identity in identities:
            scores[identity] += weight * (labels[frame] == identity)
    return scores


def _expected_reference(labels: np.ndarray, start_t: int, identity: int,
                        lookback: int) -> tuple[float, np.ndarray]:
    masks = [labels[t] == identity
             for t in range(max(0, start_t - lookback), start_t)
             if np.any(labels[t] == identity)]
    return (
        float(np.median([mask.sum() for mask in masks])),
        np.mean([_identity_shape(mask) for mask in masks], axis=0),
    )


def _partition_contact(host: np.ndarray, raw_frame: np.ndarray,
                       lag_frame: np.ndarray, working: np.ndarray, t: int,
                       static: dict, mobile: dict,
                       expected_static_area: float,
                       expected_mobile_area: float,
                       expected_static_shape: np.ndarray,
                       expected_mobile_shape: np.ndarray,
                       mobile_velocity: np.ndarray, params: dict,
                       ) -> dict | None:
    static_id = int(static["identity"])
    mobile_id = int(mobile["identity"])
    scores = _tenure_scores(
        working, t, (static_id, mobile_id),
        int(params["lookback_frames"]), float(params["decay"]))
    predicted_static = static["position"]
    # Both zero-motion and velocity-following positions enter the radial seed list.
    predicted_mobile = mobile["position"] + mobile_velocity
    seeds = _candidate_seeds(
        host, raw_frame, predicted_static, predicted_mobile, params)
    for identity in (static_id, mobile_id):
        territory = host & (scores[identity] > scores[
            mobile_id if identity == static_id else static_id])
        if np.any(territory):
            point = np.average(
                np.column_stack(np.nonzero(territory)), axis=0,
                weights=scores[identity][territory])
            seeds.append(_nearest_host_pixel(host, point))
    seeds = list(dict.fromkeys(seeds))
    candidates: list[dict] = []
    for seed_static, seed_mobile in itertools.permutations(seeds, 2):
        scored = _partition_score(
            host, raw_frame, static, mobile,
            predicted_static, predicted_mobile,
            expected_static_area, expected_mobile_area,
            expected_static_shape, expected_mobile_shape,
            lag_frame, seed_static, seed_mobile, params)
        if scored is None:
            continue
        static_total = max(float(scores[static_id][host].sum()), 1.0)
        mobile_total = max(float(scores[mobile_id][host].sum()), 1.0)
        static_tenure = float(
            scores[static_id][scored["part_a"]].sum() / static_total)
        mobile_tenure = float(
            scores[mobile_id][scored["part_b"]].sum() / mobile_total)
        scored["static_tenure_fraction"] = static_tenure
        scored["mobile_tenure_fraction"] = mobile_tenure
        scored["adjusted_cost"] = float(
            scored["cost"] - float(params["tenure_weight"])
            * (static_tenure + mobile_tenure))
        candidates.append(scored)
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["adjusted_cost"])


def _select_mobile(frame: np.ndarray, lag_frame: np.ndarray,
                   previous: dict, velocity: np.ndarray, gap: int,
                   excluded: np.ndarray, params: dict) -> dict | None:
    candidates = []
    predicted = previous["position"] + velocity * (gap + 1)
    for observation in _label_observations(frame, excluded):
        ratio = observation["area"] / max(previous["area"], 1)
        if not (float(params["mobile_minimum_area_ratio"])
                <= ratio <= float(params["mobile_maximum_area_ratio"])):
            continue
        zero_distance = float(np.linalg.norm(
            observation["position"] - previous["position"]))
        velocity_distance = float(np.linalg.norm(
            observation["position"] - predicted))
        distance = min(zero_distance, velocity_distance)
        if distance > float(params["mobile_maximum_search_px"]):
            continue
        support = motion_support(
            previous["mask"], observation["mask"], lag_frame)
        support_value = float(support["motion_support"]) if gap == 0 else 0.0
        expanded = ndi.binary_dilation(
            observation["mask"], structure=STRUCTURE, iterations=2)
        previous_overlap = float(np.mean(expanded[previous["mask"]]))
        cost = (
            distance / float(params["mobile_position_scale_px"])
            + float(params["mobile_area_weight"])
            * abs(float(np.log2(max(ratio, 1e-6))))
            - float(params["mobile_motion_weight"]) * support_value
            - float(params["mobile_overlap_weight"]) * previous_overlap
        )
        observation.update({
            "cost": float(cost),
            "distance_px": distance,
            "zero_motion_distance_px": zero_distance,
            "velocity_distance_px": velocity_distance,
            "area_ratio": float(ratio),
            "motion_support": support_value,
            "previous_overlap": previous_overlap,
        })
        candidates.append(observation)
    if not candidates:
        return None
    best = min(candidates, key=lambda row: row["cost"])
    # After a gap, admit a wider radial link but do not invent directional certainty.
    maximum = float(params["mobile_maximum_cost"]) + min(gap, 3) * 0.6
    return best if best["cost"] <= maximum else None


def _majority_identity(frame: np.ndarray, mask: np.ndarray) -> int:
    values, counts = np.unique(frame[mask], return_counts=True)
    rows = [(int(count), int(value)) for value, count in zip(values, counts)
            if int(value) != 0]
    return max(rows)[1] if rows else 0


def _apply_event(base: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                 working: np.ndarray, event: pd.Series,
                 params: dict) -> list[dict]:
    start_t = int(event.t)
    static_id = int(event.static_identity)
    mobile_id = int(event.mobile_identity)
    lookback = int(params["lookback_frames"])
    history = base[max(0, start_t - lookback):start_t]
    anchor = np.count_nonzero(history == static_id, axis=0) >= 3
    static_area, static_shape = _expected_reference(
        base, start_t, static_id, lookback)
    mobile_area, mobile_shape = _expected_reference(
        base, start_t, mobile_id, lookback)
    previous_static = _observation(working[start_t - 1] == static_id)
    previous_static["identity"] = static_id
    previous_mobile = _observation(working[start_t - 1] == mobile_id)
    previous_mobile["identity"] = mobile_id
    earlier = _observation(working[start_t - 2] == mobile_id)
    velocity = previous_mobile["position"] - earlier["position"]
    gap = 0
    rows: list[dict] = []

    for t in range(start_t, len(working)):
        input_frame = base[t]
        binary_host = _component_with_most_overlap(input_frame > 0, anchor)
        input_static = _component_with_most_overlap(
            input_frame == _majority_identity(
                input_frame, binary_host) if binary_host is not None
                else np.zeros_like(input_frame, bool), anchor)
        if binary_host is None:
            gap += 1
            rows.append({
                "event_id": event.event_id, "imagej_frame": t + 1,
                "status": "static_anchor_not_detected",
                "static_identity": static_id, "mobile_identity": mobile_id,
            })
            continue
        # The accepted label observation containing the anchor is safer than taking
        # every touching foreground object after the original two-cell merge ends.
        anchor_owner = _majority_identity(input_frame, binary_host & anchor)
        input_static = _component_with_most_overlap(
            input_frame == anchor_owner, anchor)
        if input_static is None:
            input_static = binary_host

        separate_mobile = _select_mobile(
            input_frame, lag[t - 1], previous_mobile, velocity, gap,
            input_static, params)
        previous_mobile_host_coverage = float(np.mean(
            input_static[previous_mobile["mask"]]))
        host_motion = motion_support(
            previous_mobile["mask"], input_static, lag[t - 1])
        host_pixels = np.column_stack(np.nonzero(input_static))
        host_reach = float(np.min(np.linalg.norm(
            host_pixels - previous_mobile["position"][None, :], axis=1)))
        radial_motion_arrival = bool(
            host_reach <= float(params["merge_motion_maximum_reach_px"])
            and host_motion["source_loss_coverage"]
            >= float(params["merge_minimum_source_loss_coverage"]))
        merged = bool(
            t == start_t
            or (input_static.sum() / max(static_area, 1.0)
                >= float(params["merge_minimum_static_area_ratio"])
                and (previous_mobile_host_coverage >= 0.20
                     or radial_motion_arrival)))

        output_frame = input_frame.copy()
        target_owner = _majority_identity(input_frame, input_static)
        mobile_mask = np.zeros_like(input_frame, bool)
        mobile_owner = 0
        if merged:
            partition = _partition_contact(
                input_static, raw[t], lag[t - 1], working, t,
                previous_static, previous_mobile,
                static_area, mobile_area, static_shape, mobile_shape,
                velocity, params)
            if partition is None:
                rows.append({
                    "event_id": event.event_id, "imagej_frame": t + 1,
                    "status": "partition_not_resolved",
                    "static_identity": static_id,
                    "mobile_identity": mobile_id,
                })
                gap += 1
                continue
            static_mask = partition["part_a"]
            mobile_mask = partition["part_b"]
            output_frame[input_static] = 0
            output_frame[static_mask] = static_id
            output_frame[mobile_mask] = mobile_id
            status = "connected_host_partitioned"
            partition_cost = float(partition["adjusted_cost"])
            mobile_owner = target_owner
        else:
            static_mask = input_static
            output_frame[static_mask] = static_id
            partition_cost = np.nan
            status = "static_territory_reserved"
            if separate_mobile is not None:
                mobile_mask = separate_mobile["mask"]
                mobile_owner = int(separate_mobile["input_identity"])
                output_frame[mobile_mask] = mobile_id
                status = "static_reserved_mobile_relinked"

        # Complete the local name displacement cycle so the reserved static name is
        # never duplicated on an unrelated component in the same frame.
        displaced_static = ((input_frame == static_id)
                            & ~static_mask & ~mobile_mask)
        if np.any(displaced_static):
            if mobile_owner not in {0, static_id, mobile_id}:
                replacement = mobile_owner
            elif target_owner not in {0, static_id}:
                replacement = target_owner
            else:
                replacement = mobile_id
            output_frame[displaced_static] = replacement

        if not np.array_equal(output_frame > 0, input_frame > 0):
            raise AssertionError("territory tenure changed foreground support")
        working[t] = output_frame
        old_mobile_position = previous_mobile["position"]
        previous_static = _observation(static_mask)
        previous_static["identity"] = static_id
        if np.any(mobile_mask):
            previous_mobile = _observation(mobile_mask)
            previous_mobile["identity"] = mobile_id
            measured_velocity = (
                previous_mobile["position"] - old_mobile_position) / (gap + 1)
            velocity = 0.5 * velocity + 0.5 * measured_velocity
            gap = 0
        else:
            velocity *= 0.8
            gap += 1
        rows.append({
            "event_id": event.event_id,
            "imagej_frame": t + 1,
            "status": status,
            "static_identity": static_id,
            "mobile_identity": mobile_id,
            "input_static_owner": target_owner,
            "input_mobile_owner": mobile_owner,
            "static_area_px": int(static_mask.sum()),
            "mobile_area_px": int(mobile_mask.sum()),
            "partition_cost": partition_cost,
            "mobile_gap_frames": gap,
        })
        if gap > int(params["maximum_gap_frames"]):
            break
    return rows


def apply_tenure(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                 events: pd.DataFrame, params: dict,
                 enabled: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    working = labels.copy()
    rows: list[dict] = []
    if enabled:
        for event in events.itertuples(index=False):
            rows.extend(_apply_event(
                labels, raw, lag, working, pd.Series(event._asdict()), params))
    return working, pd.DataFrame(rows)

