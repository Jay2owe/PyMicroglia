"""Apply additive previous-frame ownership, extension, merge, and U rules."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import label as connected_labels, regionprops

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402


STRUCTURE = np.ones((3, 3), bool)


@dataclass
class Observation:
    identity: int
    mask: np.ndarray | None
    area: int
    centroid: np.ndarray
    pixels: np.ndarray


def observations(frame: np.ndarray, with_masks: bool = False) -> list[Observation]:
    rows: list[Observation] = []
    numbered = connected_labels(frame, background=0, connectivity=2)
    for region in regionprops(numbered):
        points = region.coords
        mask = numbered == int(region.label) if with_masks else None
        identity = int(frame[tuple(points[0])])
        rows.append(Observation(
            identity=identity, mask=mask, area=int(len(points)),
            centroid=np.asarray(region.centroid, float), pixels=points))
    return rows


def observation_mask(item: Observation, shape: tuple[int, int]) -> np.ndarray:
    if item.mask is not None:
        return item.mask
    mask = np.zeros(shape, bool)
    mask[tuple(item.pixels.T)] = True
    return mask


def identity_points(frame: np.ndarray) -> dict[int, np.ndarray]:
    pixels = np.column_stack(np.nonzero(frame))
    if not len(pixels):
        return {}
    identities = frame[tuple(pixels.T)].astype(int)
    order = np.argsort(identities, kind="stable")
    pixels, identities = pixels[order], identities[order]
    boundaries = np.flatnonzero(np.diff(identities)) + 1
    groups = np.split(pixels, boundaries)
    values = np.split(identities, boundaries)
    return {int(value[0]): group for value, group in zip(values, groups)}


def binary_components(mask: np.ndarray) -> list[np.ndarray]:
    numbered, count = ndi.label(mask, structure=STRUCTURE)
    return [numbered == component for component in range(1, count + 1)]


def centroid(mask: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.nonzero(mask))
    return points.mean(axis=0) if len(points) else np.array([np.nan, np.nan])


def source_metrics(previous: np.ndarray, item: Observation, identity: int,
                   lag: np.ndarray, dilation: int,
                   source_lookup: dict[int, np.ndarray]) -> dict | None:
    source_pixels = source_lookup.get(identity)
    source_area = int(len(source_pixels)) if source_pixels is not None else 0
    if source_area == 0:
        return None
    destination_values = previous[tuple(item.pixels.T)]
    overlap = int(np.count_nonzero(destination_values == identity))
    if dilation > 0:
        distances = cKDTree(source_pixels).query(item.pixels, k=1)[0]
        destination_dilation = float(np.mean(distances <= dilation * np.sqrt(2)))
    else:
        destination_dilation = overlap / max(item.area, 1)
    source_fraction = overlap / source_area
    destination_fraction = overlap / max(item.area, 1)
    overlap_coefficient = overlap / max(min(source_area, item.area), 1)
    new_pixels = item.pixels[destination_values != identity]
    red = (float(np.mean(lag[tuple(new_pixels.T)] >= 1.5))
           if len(new_pixels) else 0.0)
    blue = float(np.mean(lag[tuple(source_pixels.T)] <= -1.5))
    distance = float(np.linalg.norm(source_pixels.mean(axis=0) - item.centroid))
    return {
        "identity": identity,
        "source_area": source_area,
        "overlap_px": overlap,
        "source_fraction": source_fraction,
        "destination_fraction": destination_fraction,
        "overlap_coefficient": overlap_coefficient,
        "destination_dilation_fraction": destination_dilation,
        "distance_px": distance,
        "red_extension_fraction": red,
        "blue_release_fraction": blue,
    }


def nearby_identities(previous: np.ndarray, item: Observation,
                      dilation: int) -> list[int]:
    if dilation <= 0:
        values = previous[tuple(item.pixels.T)]
        return sorted(set(map(int, np.unique(values))) - {0})
    y0 = max(0, int(item.pixels[:, 0].min()) - dilation)
    y1 = min(previous.shape[0], int(item.pixels[:, 0].max()) + dilation + 1)
    x0 = max(0, int(item.pixels[:, 1].min()) - dilation)
    x1 = min(previous.shape[1], int(item.pixels[:, 1].max()) + dilation + 1)
    crop = np.zeros((y1 - y0, x1 - x0), bool)
    crop[(item.pixels[:, 0] - y0, item.pixels[:, 1] - x0)] = True
    search = ndi.binary_dilation(crop, structure=STRUCTURE,
                                 iterations=dilation)
    return sorted(set(map(int, np.unique(previous[y0:y1, x0:x1][search]))) - {0})


def candidate_sources(previous: np.ndarray, item: Observation,
                      lag: np.ndarray, dilation: int,
                      source_lookup: dict[int, np.ndarray]) -> list[dict]:
    identities = nearby_identities(previous, item, dilation)
    rows = []
    for identity in identities:
        values = source_metrics(previous, item, identity, lag, dilation,
                                source_lookup)
        if values is not None:
            rows.append(values)
    return rows


def substantial_duplicate(frame: np.ndarray, identity: int,
                          minimum_area: int, minimum_distance: float,
                          maximum_fragment_gap: float) -> bool:
    components = [mask for mask in binary_components(frame == identity)
                  if int(mask.sum()) >= minimum_area]
    if len(components) < 2:
        return False
    primary = max(components, key=lambda mask: int(mask.sum()))
    primary_distance = ndi.distance_transform_edt(~primary)
    other_identities = (frame > 0) & (frame != identity)
    for part in components:
        if part is primary or float(primary_distance[part].min()) < minimum_distance:
            continue
        if np.any(other_identities):
            neighbour_distance = ndi.distance_transform_edt(~part)
            if float(neighbour_distance[other_identities].min()) \
                    <= maximum_fragment_gap:
                return True
    return False


def substantial_soma_duplicate(frame: np.ndarray, identity: int,
                               minimum_core_area: int = 8) -> bool:
    bodies = 0
    for mask in binary_components(frame == identity):
        eroded = ndi.binary_erosion(mask, structure=STRUCTURE, iterations=1)
        numbered, count = ndi.label(eroded, structure=STRUCTURE)
        bodies += sum(int(np.count_nonzero(numbered == component))
                      >= minimum_core_area
                      for component in range(1, count + 1))
    return bodies > 1


def apply_proposals(frame: np.ndarray, proposals: list[dict], method: str,
                    cfg: dict, source_frame: int) -> tuple[np.ndarray, list[dict]]:
    if not proposals:
        return frame, []
    original = frame.copy()
    rejected: set[int] = set()
    minimum_area = int(cfg.get("duplicate_minimum_area_px", 48))
    minimum_distance = float(cfg.get("duplicate_minimum_distance_px", 12.0))
    maximum_fragment_gap = float(cfg.get(
        "same_owner_maximum_fragment_gap_px", 8.0))
    minimum_core_area = int(cfg.get("duplicate_minimum_core_area_px", 8))
    # Rebuild after each rejection. A rejected swap restores its old label and
    # can expose a second duplicate that the all-proposals preview had hidden.
    while True:
        tentative = frame.copy()
        for proposal in proposals:
            if int(proposal["owner"]) in rejected:
                continue
            pixels = proposal["pixels"]
            tentative[tuple(pixels.T)] = int(proposal["owner"])
        new_rejections = {
            owner for owner in ({int(item["owner"]) for item in proposals}
                                - rejected)
            if (substantial_duplicate(tentative, owner, minimum_area,
                                      minimum_distance, maximum_fragment_gap)
                or (substantial_soma_duplicate(
                    tentative, owner, minimum_core_area)
                    and not (method == "R" and any(
                        int(item["owner"]) == owner
                        and len(item["pixels"]) <= int(cfg.get(
                            "motion_core_override_maximum_area_px", 80))
                        and float(item.get("red_extension_fraction", 0.0))
                        >= float(cfg.get(
                            "motion_core_override_minimum_red_fraction", 0.85))
                        and float(item.get("blue_release_fraction", 0.0))
                        >= float(cfg.get(
                            "motion_core_override_minimum_blue_fraction", 0.35))
                        and float(item.get("blue_release_fraction", 1.0))
                        <= float(cfg.get(
                            "motion_core_override_maximum_blue_fraction", 0.6))
                        for item in proposals))))
        }
        if not new_rejections:
            break
        rejected.update(new_rejections)
    applied: list[dict] = []
    result = original.copy()
    for proposal in proposals:
        owner = int(proposal["owner"])
        if owner in rejected:
            continue
        pixels = proposal["pixels"]
        changed = int(np.count_nonzero(result[tuple(pixels.T)] != owner))
        result[tuple(pixels.T)] = owner
        if changed:
            applied.append({
                "method": method,
                "action": proposal["action"],
                "source_imagej_frame": source_frame,
                "prior_identity": owner,
                "original_identity": int(proposal["original_identity"]),
                "assigned_identity": owner,
                "changed_pixels": changed,
                "host_identity": int(proposal.get("host_identity", 0)),
                "overlap_px": int(proposal.get("overlap_px", 0)),
                "source_fraction": float(proposal.get("source_fraction", 0.0)),
                "destination_fraction": float(
                    proposal.get("destination_fraction", 0.0)),
                "overlap_coefficient": float(
                    proposal.get("overlap_coefficient", 0.0)),
                "distance_px": float(proposal.get("distance_px", np.nan)),
                "red_extension_fraction": float(
                    proposal.get("red_extension_fraction", np.nan)),
                "blue_release_fraction": float(
                    proposal.get("blue_release_fraction", np.nan)),
                "decision": "applied",
            })
    return result, applied


def overlap_proposals(previous: np.ndarray, current: np.ndarray,
                      lag: np.ndarray, cfg: dict,
                      source_lookup: dict[int, np.ndarray]) -> list[dict]:
    proposals: list[dict] = []
    dilation = int(cfg["extension_dilation_px"])
    for item in observations(current):
        rows = candidate_sources(previous, item, lag, 0, source_lookup)
        merge_sources = [row for row in rows
                         if row["overlap_px"] >= int(cfg["minimum_overlap_px"])
                         and row["source_fraction"] >= float(
                             cfg["merge_source_fraction"])]
        if len(merge_sources) > 1:
            strong_identities = {int(row["identity"]) for row in merge_sources}
            # A weak label covering several established previous owners is a relay,
            # not a new merged owner. Partition only that case; when the current
            # label is itself one established owner, M records the other as dormant.
            weak_label_area = len(source_lookup.get(
                item.identity, np.empty((0, 2), int)))
            if (item.identity not in strong_identities
                    and len(merge_sources) == 2
                    and weak_label_area <= int(cfg[
                        "weak_merge_label_maximum_previous_area_px"])):
                distances = np.column_stack([
                    cKDTree(source_lookup[int(row["identity"])]).query(
                        item.pixels, k=1)[0]
                    for row in merge_sources])
                assignment = np.argmin(distances, axis=1)
                for source_index, values in enumerate(merge_sources):
                    pixels = item.pixels[assignment == source_index]
                    if not len(pixels):
                        continue
                    proposals.append({
                        **values, "pixels": pixels,
                        "owner": int(values["identity"]),
                        "original_identity": item.identity,
                        "action": "multi_owner_previous_domain_partition",
                        "tenure_mode": "partition",
                    })
            continue
        eligible = []
        for row in rows:
            stationary_takeover = bool(
                row["overlap_coefficient"] >= 0.6
                and row["source_fraction"] >= 0.6
                and row["destination_fraction"] >= 0.6
                and row["distance_px"] <= float(
                    cfg["overlap_tenure_maximum_step_px"])
                and row["blue_release_fraction"] < 0.3)
            high_overlap_continuation = bool(
                row["overlap_coefficient"] >= 0.8
                and row["source_fraction"] >= 0.3
                and row["destination_fraction"] >= 0.8
                and row["distance_px"] <= 10.0
                and row["blue_release_fraction"] < 0.5)
            if (row["overlap_px"] >= int(cfg["minimum_overlap_px"])
                    and (stationary_takeover or high_overlap_continuation)):
                eligible.append({
                    **row,
                    "tenure_mode": "stationary" if stationary_takeover
                    else "high_overlap_continuation",
                })
        if not eligible:
            continue
        best = max(eligible, key=lambda row: (
            row["overlap_coefficient"] + row["source_fraction"]
            + row["destination_fraction"] - 0.05 * row["distance_px"]))
        if int(best["identity"]) == item.identity:
            continue
        proposals.append({
            **best, "pixels": item.pixels, "owner": int(best["identity"]),
            "original_identity": item.identity,
            "action": "previous_owner_first_bid",
        })
    return proposals


def red_extension_proposals(previous: np.ndarray, current: np.ndarray,
                            lag: np.ndarray, cfg: dict,
                            source_lookup: dict[int, np.ndarray]) -> list[dict]:
    proposals: list[dict] = []
    dilation = int(cfg["extension_dilation_px"])
    for item in observations(current):
        prior_name = source_lookup.get(item.identity)
        # Red extension is reserved for a label that truly appears in this frame.
        # If the same label existed anywhere one frame earlier, ordinary ownership
        # and merge rules must resolve it instead of a red-led takeover.
        if prior_name is not None:
            continue
        rows = candidate_sources(previous, item, lag, dilation, source_lookup)
        eligible = [row for row in rows if (
            row["overlap_px"] >= 1
            and row["destination_dilation_fraction"] >= float(
                cfg["minimum_connected_fraction"])
            and row["red_extension_fraction"] >= float(
                cfg["minimum_connected_red_extension_fraction"])
            and row["blue_release_fraction"] < float(
                cfg["minimum_blue_release_fraction"])
            and row["distance_px"] <= float(cfg["red_maximum_step_px"]))]
        if not eligible:
            continue
        eligible.sort(key=lambda row: (
            row["red_extension_fraction"]
            + row["destination_dilation_fraction"]
            + 0.5 * row["overlap_coefficient"]
            - 0.03 * row["distance_px"]), reverse=True)
        best = eligible[0]
        if len(eligible) > 1:
            first_score = (best["red_extension_fraction"]
                           + best["destination_dilation_fraction"])
            second_score = (eligible[1]["red_extension_fraction"]
                            + eligible[1]["destination_dilation_fraction"])
            if first_score - second_score < float(cfg["minimum_red_margin"]):
                continue
        if int(best["identity"]) == item.identity:
            continue
        proposals.append({
            **best, "pixels": item.pixels, "owner": int(best["identity"]),
            "original_identity": item.identity,
            "action": "connected_red_growth",
            "tenure_mode": "red_extension",
        })
    return proposals


def detect_new_dormancy(previous: np.ndarray, current: np.ndarray,
                        lag: np.ndarray, cfg: dict,
                        states: dict[int, dict], source_frame: int,
                        rows: list[dict],
                        source_lookup: dict[int, np.ndarray]) -> None:
    for item in observations(current):
        sources = []
        for identity in sorted(set(map(int, np.unique(
                previous[tuple(item.pixels.T)]))) - {0}):
            values = source_metrics(previous, item, identity, lag, 1,
                                    source_lookup)
            if values is None:
                continue
            if (values["overlap_px"] >= int(cfg["minimum_overlap_px"])
                    and values["source_fraction"] >= float(
                        cfg["merge_absorption_fraction"])):
                sources.append(values)
        if len(sources) < 2:
            continue
        host = item.identity if any(int(row["identity"]) == item.identity
                                    for row in sources) else max(
                                        sources, key=lambda row: row["overlap_px"]
                                    )["identity"]
        for values in sources:
            identity = int(values["identity"])
            if identity == int(host) or identity in states:
                continue
            current_area = int(np.count_nonzero(current == identity))
            if current_area > float(cfg["dormant_visible_fraction"]) * int(
                    values["source_area"]):
                continue
            states[identity] = {
                "identity": identity,
                "host_identity": int(host),
                "start_source_frame": source_frame,
                "age": 0,
                "reference_area": int(values["source_area"]),
                "anchor_mask": previous == identity,
                "host_mask": observation_mask(item, current.shape),
            }
            rows.append({
                "method": "M", "action": "dormancy_started",
                "source_imagej_frame": source_frame,
                "prior_identity": identity,
                "original_identity": identity,
                "assigned_identity": 0,
                "changed_pixels": 0,
                "host_identity": int(host),
                "overlap_px": int(values["overlap_px"]),
                "source_fraction": float(values["source_fraction"]),
                "destination_fraction": float(values["destination_fraction"]),
                "overlap_coefficient": float(values["overlap_coefficient"]),
                "distance_px": float(values["distance_px"]),
                "red_extension_fraction": float(values["red_extension_fraction"]),
                "blue_release_fraction": float(values["blue_release_fraction"]),
                "decision": "reserved_inside_host",
            })


def apply_dormancy(previous: np.ndarray, current: np.ndarray,
                   unclaimed: np.ndarray, lag: np.ndarray, cfg: dict,
                   states: dict[int, dict], source_frame: int,
                   rows: list[dict], source_lookup: dict[int, np.ndarray]
                   ) -> tuple[np.ndarray, np.ndarray]:
    result = current.copy()
    released: list[int] = []
    for identity, state in list(states.items()):
        state["age"] += 1
        host_candidates = binary_components(result > 0)
        host = None
        host_score = -1.0
        expanded = ndi.binary_dilation(state["host_mask"], structure=STRUCTURE,
                                       iterations=4)
        for mask in host_candidates:
            score = float(np.mean(expanded[mask]))
            if score > host_score:
                host_score, host = score, mask
        if host is not None and host_score > 0:
            state["host_mask"] = host.copy()

        for item in observations(result, with_masks=True):
            if item.identity != identity:
                continue
            near_host = bool(host is not None and np.any(
                ndi.binary_dilation(host, structure=STRUCTURE,
                                    iterations=int(cfg["split_radius_px"]))
                & item.mask))
            red = float(np.mean(lag[item.mask] >= 1.5))
            anchor_distance = float(np.linalg.norm(
                item.centroid - centroid(state["anchor_mask"])))
            supported_split = bool(
                near_host
                and item.area >= float(cfg["split_minimum_area_fraction"])
                * int(state["reference_area"])
                and (red >= float(cfg["split_minimum_red_fraction"])
                     or anchor_distance <= float(cfg["split_radius_px"])))
            if supported_split:
                released.append(identity)
                rows.append({
                    "method": "M", "action": "supported_split_release",
                    "source_imagej_frame": source_frame,
                    "prior_identity": identity,
                    "original_identity": identity,
                    "assigned_identity": identity,
                    "changed_pixels": 0,
                    "host_identity": int(state["host_identity"]),
                    "overlap_px": 0, "source_fraction": 0.0,
                    "destination_fraction": 0.0,
                    "overlap_coefficient": 0.0,
                    "distance_px": anchor_distance,
                    "red_extension_fraction": red,
                    "blue_release_fraction": np.nan,
                    "decision": "released_near_host",
                })
                break
            # A weak fragment still touching the host remains part of the recorded
            # merge. It is not an outside bid and must not be renamed elsewhere.
            if near_host:
                continue
            # The registry disables this identity as a future O/R bidder. The
            # accepted observation is not erased merely because no replacement is
            # yet proved; a later owner-first bid may restore it safely.
            replacement = identity
            changed = 0
            decision = "dormant_external_bid_disabled"
            rows.append({
                "method": "M", "action": "external_bid_blocked",
                "source_imagej_frame": source_frame,
                "prior_identity": identity,
                "original_identity": identity,
                "assigned_identity": replacement,
                "changed_pixels": changed,
                "host_identity": int(state["host_identity"]),
                "overlap_px": 0, "source_fraction": 0.0,
                "destination_fraction": 0.0,
                "overlap_coefficient": 0.0,
                "distance_px": anchor_distance,
                "red_extension_fraction": red,
                "blue_release_fraction": np.nan,
                "decision": decision,
            })
        if state["age"] > int(cfg["maximum_dormant_frames"]):
            released.append(identity)
    for identity in set(released):
        states.pop(identity, None)
    return result, unclaimed


def add_reciprocal_dormant_swaps(
        proposals: list[dict], current: np.ndarray, lag: np.ndarray,
        cfg: dict, states: dict[int, dict]) -> list[dict]:
    """Require a paired split before reclaiming an active dormant name.

    If an established lineage bids for an external observation carrying a name
    that is currently reserved inside a host, the displaced dormant lineage must
    simultaneously have a supported component beside that host. This turns a
    one-way relabel into an identity-preserving reciprocal swap.
    """
    result: list[dict] = []
    split_radius = int(cfg["split_radius_px"])
    for proposal in proposals:
        dormant_identity = int(proposal["original_identity"])
        state = states.get(dormant_identity)
        if state is None:
            result.append(proposal)
            continue
        established_owner = int(proposal["owner"])
        if established_owner in states:
            continue
        expanded_host = ndi.binary_dilation(
            state["host_mask"], structure=STRUCTURE,
            iterations=split_radius)
        candidates = []
        for item in observations(current, with_masks=True):
            if item.identity != established_owner:
                continue
            near_host = bool(np.any(expanded_host & item.mask))
            red = float(np.mean(lag[item.mask] >= 1.5))
            anchor_distance = float(np.linalg.norm(
                item.centroid - centroid(state["anchor_mask"])))
            supported = bool(
                near_host
                and item.area >= float(cfg["split_minimum_area_fraction"])
                * int(state["reference_area"])
                and (red >= float(cfg.get(
                    "reciprocal_split_minimum_red_fraction",
                    cfg["split_minimum_red_fraction"]))
                     or anchor_distance <= split_radius))
            if supported:
                candidates.append((red, -anchor_distance, item))
        result.append({
            **proposal,
            "action": "dormant_external_reclaim",
            "dormant_swap": True,
        })
        # A one-way reclaim is valid when the dormant lineage remains hidden in
        # its host. The soma-duplicate gate rejects it if the established owner
        # would occupy two cells; in that case a reciprocal component is required.
        if not candidates:
            continue
        chosen = max(candidates, key=lambda row: row[:2])
        _, _, displaced = chosen
        result.append({
            "pixels": displaced.pixels,
            "owner": dormant_identity,
            "original_identity": established_owner,
            "action": "dormant_reciprocal_split_swap",
            "tenure_mode": "reciprocal_split",
            "dormant_swap": True,
            "overlap_px": 0,
            "source_fraction": 0.0,
            "destination_fraction": 0.0,
            "overlap_coefficient": 0.0,
            "distance_px": float(-chosen[1]),
            "red_extension_fraction": float(chosen[0]),
            "blue_release_fraction": np.nan,
        })
    return result


def add_reciprocal_continuation_swaps(
        proposals: list[dict], previous: np.ndarray, current: np.ndarray,
        lag: np.ndarray, cfg: dict,
        source_lookup: dict[int, np.ndarray]) -> list[dict]:
    """Complete a two-label swap when the reverse side is smaller but overlaps."""
    result = list(proposals)
    existing = {(int(item["owner"]), int(item["original_identity"]))
                for item in proposals}
    for proposal in proposals:
        owner = int(proposal["owner"])
        original = int(proposal["original_identity"])
        if owner == original or (original, owner) in existing:
            continue
        choices = []
        for item in observations(current):
            if item.identity != owner:
                continue
            values = source_metrics(previous, item, original, lag, 0,
                                    source_lookup)
            if values is None:
                continue
            eligible = bool(
                values["overlap_coefficient"] >= 0.8
                and values["source_fraction"] >= 0.2
                and values["destination_fraction"] >= 0.8
                and values["distance_px"] <= 10.0
                and values["blue_release_fraction"] < 0.6)
            if eligible:
                choices.append((values["overlap_coefficient"], item, values))
        if not choices:
            continue
        _, displaced, values = max(choices, key=lambda row: row[0])
        proposal["reciprocal_swap"] = True
        result.append({
            **values,
            "pixels": displaced.pixels,
            "owner": original,
            "original_identity": owner,
            "action": "reciprocal_overlap_continuation",
            "tenure_mode": "reciprocal_continuation",
            "reciprocal_swap": True,
        })
        existing.add((original, owner))
    return result


def unclaimed_observations(frame: np.ndarray) -> list[Observation]:
    rows = []
    numbered, count = ndi.label(frame > 0, structure=STRUCTURE)
    for component in range(1, count + 1):
        mask = numbered == component
        points = np.column_stack(np.nonzero(mask))
        original_values, counts = np.unique(frame[mask], return_counts=True)
        identity = int(original_values[int(np.argmax(counts))])
        rows.append(Observation(identity, mask, int(len(points)),
                                points.mean(axis=0), points))
    return rows


def choose_bridge_match(state: dict, previous: np.ndarray,
                        current: np.ndarray, unclaimed: np.ndarray,
                        lag: np.ndarray, cfg: dict) -> tuple[str, Observation,
                                                             dict] | None:
    last_mask = state["last_mask"]
    last_area = int(state["last_area"])
    gap = int(state["gap"])
    last_centre = centroid(last_mask)
    blue = float(np.mean(lag[last_mask] <= -1.5)) if gap == 0 else np.nan
    choices: list[tuple[float, str, Observation, dict]] = []
    for item in unclaimed_observations(unclaimed):
        ratio = item.area / max(last_area, 1)
        distance = float(np.linalg.norm(item.centroid - last_centre))
        red = float(np.mean(lag[item.mask] >= 1.5))
        exact = int(np.count_nonzero(item.mask & last_mask))
        overlap_coefficient = exact / max(min(item.area, last_area), 1)
        if not (float(cfg["bridge_minimum_area_ratio"]) <= ratio
                <= float(cfg["bridge_maximum_area_ratio"])):
            continue
        if gap == 0:
            supported = bool(
                distance <= float(cfg["bridge_motion_radius_px"])
                and ((red >= float(cfg["bridge_minimum_red_fraction"])
                      and (blue >= float(cfg["bridge_minimum_blue_fraction"])
                           or exact > 0))
                     or overlap_coefficient >= 0.5))
        else:
            supported = bool(
                distance <= float(cfg["bridge_gap_radius_px"])
                and red >= float(cfg["bridge_minimum_red_fraction"]))
        if supported:
            score = red + (0.0 if np.isnan(blue) else blue) - 0.03 * distance
            choices.append((score, "unclaimed", item, {
                "distance_px": distance, "red": red, "blue": blue,
                "overlap_px": exact,
            }))
    previous_lookup = identity_points(previous)
    for item in observations(current, with_masks=True):
        prior_name_pixels = previous_lookup.get(item.identity)
        if prior_name_pixels is not None:
            local_name_distance = float(np.min(
                cKDTree(prior_name_pixels).query(item.pixels, k=1)[0]))
            if local_name_distance <= 2.0:
                continue
        ratio = item.area / max(last_area, 1)
        distance = float(np.linalg.norm(item.centroid - last_centre))
        red = float(np.mean(lag[item.mask] >= 1.5))
        exact = int(np.count_nonzero(item.mask & last_mask))
        overlap_coefficient = exact / max(min(item.area, last_area), 1)
        radius = (float(cfg["bridge_motion_radius_px"]) if gap == 0
                  else float(cfg["bridge_gap_radius_px"]))
        if (float(cfg["bridge_minimum_area_ratio"]) <= ratio
                <= float(cfg["bridge_maximum_area_ratio"])
                and distance <= radius
                and (red >= float(cfg["bridge_minimum_red_fraction"])
                     or overlap_coefficient >= 0.5)):
            score = red + overlap_coefficient - 0.03 * distance
            choices.append((score, "assigned_reappearance", item, {
                "distance_px": distance, "red": red, "blue": np.nan,
                "overlap_px": exact,
            }))
    if not choices:
        return None
    _, kind, item, metrics = max(choices, key=lambda entry: entry[0])
    return kind, item, metrics


def apply_unclaimed_priority(previous: np.ndarray, current: np.ndarray,
                             unclaimed: np.ndarray, lag: np.ndarray,
                             cfg: dict, bridges: dict[int, dict],
                             history: dict[int, int], source_frame: int,
                             rows: list[dict],
                             source_lookup: dict[int, np.ndarray]
                             ) -> tuple[np.ndarray, np.ndarray]:
    result = current.copy()
    # Continue lineages that were first proved through explicit unclaimed space.
    for identity, state in list(bridges.items()):
        existing = [item for item in observations(result, with_masks=True)
                    if item.identity == identity]
        if existing:
            item = min(existing, key=lambda entry: float(np.linalg.norm(
                entry.centroid - centroid(state["last_mask"]))))
            state.update(last_mask=item.mask.copy(), last_area=item.area, gap=0)
            continue
        matched = choose_bridge_match(state, previous, result, unclaimed,
                                      lag, cfg)
        if matched is None:
            state["gap"] += 1
            if state["gap"] > int(cfg["maximum_dormant_frames"]):
                bridges.pop(identity, None)
            continue
        kind, item, metrics = matched
        original = item.identity if kind == "assigned_reappearance" else 0
        result[item.mask] = identity
        if kind == "unclaimed":
            unclaimed[item.mask] = 0
        state.update(last_mask=item.mask.copy(), last_area=item.area, gap=0)
        rows.append({
            "method": "U", "action": "unclaimed_bridge_continuation"
            if kind == "unclaimed" else "unclaimed_bridge_reappearance",
            "source_imagej_frame": source_frame,
            "prior_identity": identity,
            "original_identity": int(original),
            "assigned_identity": identity,
            "changed_pixels": int(item.area),
            "host_identity": 0,
            "overlap_px": int(metrics["overlap_px"]),
            "source_fraction": 0.0,
            "destination_fraction": 0.0,
            "overlap_coefficient": 0.0,
            "distance_px": float(metrics["distance_px"]),
            "red_extension_fraction": float(metrics["red"]),
            "blue_release_fraction": float(metrics["blue"]),
            "decision": kind,
        })

    occupied = set(bridges)
    current_values, current_counts = np.unique(result, return_counts=True)
    current_count = {int(identity): int(count)
                     for identity, count in zip(current_values, current_counts)}
    for identity, source_pixels in sorted(source_lookup.items()):
        if identity in occupied or history.get(identity, 0) < int(
                cfg["minimum_owner_history_frames"]):
            continue
        source_area = int(len(source_pixels))
        if source_area < int(cfg["minimum_owner_area_px"]):
            continue
        if current_count.get(identity, 0) >= float(
                cfg["minimum_owner_retained_fraction"]) * source_area:
            continue
        blue = float(np.mean(lag[tuple(source_pixels.T)] <= -1.5))
        if blue >= float(cfg["maximum_stable_blue_fraction"]):
            continue
        choices = []
        for item in unclaimed_observations(unclaimed):
            overlap = int(np.count_nonzero(
                item.mask[tuple(source_pixels.T)]))
            source_fraction = overlap / max(source_area, 1)
            ratio = item.area / max(source_area, 1)
            if (source_fraction >= float(cfg["minimum_unclaimed_claim_fraction"])
                    and ratio <= float(cfg["initial_claim_maximum_area_ratio"])):
                distance = float(np.linalg.norm(
                    item.centroid - source_pixels.mean(axis=0)))
                red = float(np.mean(lag[item.mask] >= 1.5))
                choices.append((source_fraction + red - 0.02 * distance,
                                item, source_fraction, distance, red))
        if not choices:
            continue
        _, item, source_fraction, distance, red = max(
            choices, key=lambda entry: entry[0])
        result[item.mask] = identity
        unclaimed[item.mask] = 0
        bridges[identity] = {
            "identity": identity, "last_mask": item.mask.copy(),
            "last_area": item.area, "gap": 0,
        }
        rows.append({
            "method": "U", "action": "stable_owner_claimed_unclaimed",
            "source_imagej_frame": source_frame,
            "prior_identity": identity,
            "original_identity": 0,
            "assigned_identity": identity,
            "changed_pixels": int(item.area),
            "host_identity": 0,
            "overlap_px": int(overlap),
            "source_fraction": float(source_fraction),
            "destination_fraction": float(
                np.count_nonzero(item.mask[tuple(source_pixels.T)])
                / max(item.area, 1)),
            "overlap_coefficient": float(
                overlap / max(min(source_area, item.area), 1)),
            "distance_px": distance,
            "red_extension_fraction": red,
            "blue_release_fraction": blue,
            "decision": "assigned_before_U",
        })
    return result, unclaimed


def reconcile(labels: np.ndarray, unclaimed: np.ndarray, lag: np.ndarray,
              methods: set[str], cfg: dict, source_offset: int,
              protection: np.ndarray | None = None,
              ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    actions: list[dict] = []
    dormant: dict[int, dict] = {}
    merge_lineages: set[int] = set()
    bridges: dict[int, dict] = {}
    history: dict[int, int] = {}
    registry_rows: list[dict] = []
    for identity in set(map(int, np.unique(candidate[0]))) - {0}:
        history[identity] = 1
    for frame in range(1, len(candidate)):
        previous = candidate[frame - 1]
        current = candidate[frame].copy()
        current_unclaimed = candidate_unclaimed[frame].copy()
        transition = lag[frame - 1]
        protected_frame = (protection[frame] if protection is not None
                           else np.zeros_like(current, bool))
        source_frame = frame + source_offset + 1
        source_lookup = identity_points(previous)
        if "M" in methods:
            current, current_unclaimed = apply_dormancy(
                previous, current, current_unclaimed, transition, cfg,
                dormant, source_frame, actions, source_lookup)
            detect_new_dormancy(previous, current, transition, cfg,
                                dormant, source_frame, actions, source_lookup)
            merge_lineages.update(dormant)
        # A lineage already proved through explicit U territory has first claim.
        # This prevents a later red/overlap rule from assigning its reappearance to
        # a nearby host before the bridge can bid.
        if "U" in methods:
            current, current_unclaimed = apply_unclaimed_priority(
                previous, current, current_unclaimed, transition, cfg,
                bridges, history, source_frame, actions, source_lookup)
        if "O" in methods:
            proposals = overlap_proposals(previous, current, transition, cfg,
                                          source_lookup)
            if "M" in methods:
                proposals = add_reciprocal_continuation_swaps(
                    proposals, previous, current, transition, cfg,
                    source_lookup)
                proposals = add_reciprocal_dormant_swaps(
                    proposals, current, transition, cfg, dormant)
                protected = []
                for item in proposals:
                    if (bool(item.get("dormant_swap", False))
                            or bool(item.get("reciprocal_swap", False))):
                        protected.append(item)
                        continue
                    original = int(item["original_identity"])
                    local_merge_continuation = False
                    if original in merge_lineages and original in source_lookup:
                        local_merge_continuation = bool(np.min(
                            cKDTree(source_lookup[original]).query(
                                item["pixels"], k=1)[0])
                            <= float(cfg["local_name_continuation_radius_px"]))
                    if (int(item["owner"]) not in dormant
                            and (not local_merge_continuation
                                 or item.get("tenure_mode") == "stationary")):
                        protected.append(item)
                proposals = protected
            if "U" in methods:
                proposals = [item for item in proposals
                             if int(item["original_identity"]) not in bridges]
            proposals = [item for item in proposals
                         if (item.get("dormant_swap", False)
                             or item.get("reciprocal_swap", False)
                             or item.get("tenure_mode") == "stationary"
                             or not np.any(protected_frame[
                                 tuple(item["pixels"].T)]))]
            current, rows = apply_proposals(
                current, proposals,
                "O", cfg, source_frame)
            actions.extend(rows)
            for row in rows:
                if row["action"] == "dormant_reciprocal_split_swap":
                    dormant.pop(int(row["assigned_identity"]), None)
        if "R" in methods:
            proposals = red_extension_proposals(
                previous, current, transition, cfg, source_lookup)
            if "M" in methods:
                proposals = add_reciprocal_dormant_swaps(
                    proposals, current, transition, cfg, dormant)
                proposals = [item for item in proposals
                             if bool(item.get("dormant_swap", False))
                             or (int(item["owner"]) not in dormant
                                 and int(item["original_identity"])
                                 not in merge_lineages)]
            if "U" in methods:
                proposals = [item for item in proposals
                             if int(item["original_identity"]) not in bridges]
            proposals = [item for item in proposals
                         if (item.get("dormant_swap", False)
                             or not np.any(protected_frame[
                                 tuple(item["pixels"].T)]))]
            current, rows = apply_proposals(
                current, proposals,
                "R", cfg, source_frame)
            actions.extend(rows)
            for row in rows:
                if row["action"] == "dormant_reciprocal_split_swap":
                    dormant.pop(int(row["assigned_identity"]), None)
        if not np.array_equal(
                (current > 0) | (current_unclaimed > 0),
                (labels[frame] > 0) | (unclaimed[frame] > 0)):
            raise AssertionError(
                f"foreground accounting changed at source frame {source_frame}")
        if np.any((current > 0) & (current_unclaimed > 0)):
            raise AssertionError(
                f"assigned and unclaimed overlap at source frame {source_frame}")
        candidate[frame] = current
        candidate_unclaimed[frame] = current_unclaimed
        for identity in set(map(int, np.unique(current))) - {0}:
            history[identity] = history.get(identity, 0) + 1
        for state in dormant.values():
            registry_rows.append({
                "source_imagej_frame": source_frame,
                "identity": int(state["identity"]),
                "host_identity": int(state["host_identity"]),
                "start_source_imagej_frame": int(state["start_source_frame"]),
                "age_frames": int(state["age"]),
                "status": "dormant_inside_host",
            })
    return (candidate, candidate_unclaimed, pd.DataFrame(actions),
            pd.DataFrame(registry_rows))

