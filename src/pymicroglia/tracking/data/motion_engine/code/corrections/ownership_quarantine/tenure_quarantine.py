"""Apply component-local ownership quarantine and three additive safeguards."""
from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root
from component_tenure_audit import (
    Component, Tracklet, best_component, build_tracklets, component_mask,
    detect_takeovers, distance, frame_components, overlap_pixels,
)


TUNING = Path(__file__).resolve().parents[1]
ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402
from merge_partition_speed import BoundedNdimage, slices as occupied_slices
from array_bounds import occupied_coordinates
ndi = BoundedNdimage()

STRUCTURE = np.ones((3, 3), np.uint8)


@dataclass
class Branch:
    identity: int
    reference: Component
    last_frame: int
    missing_frames: int = 0


def translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(mask)
    y0s, y1s = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
    x0s, x1s = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
    if y1s <= y0s or x1s <= x0s:
        return result
    result[y0s + dy:y1s + dy, x0s + dx:x1s + dx] = mask[y0s:y1s, x0s:x1s]
    return result


def component_from_mask(mask: np.ndarray, frame: int, identity: int,
                        local_component: int = 0) -> Component:
    pixels = np.flatnonzero(mask.ravel())
    yy, xx = np.where(mask)
    return Component(frame, identity, local_component, pixels, int(len(pixels)),
                     float(xx.mean()), float(yy.mean()))


def find_event_components(row: pd.Series, frames: list[list[Component]],
                          ) -> tuple[Component, Component]:
    source = int(row.source_output_frame) - 1
    destination = int(row.destination_output_frame) - 1
    owner = next(item for item in frames[source]
                 if item.observed_identity == int(row.owner_identity)
                 and item.local_component == int(row.owner_local_component))
    target = next(item for item in frames[destination]
                  if item.observed_identity == int(row.challenger_identity)
                  and item.local_component == int(row.destination_local_component))
    return owner, target


def recent_components(track: Tracklet, frame: int, window: int) -> list[Component]:
    eligible = [item for item in track.components if item.frame <= frame]
    return eligible[-window:]


def protected_core(track: Tracklet, owner: Component, shape: tuple[int, int],
                   cfg: dict) -> np.ndarray:
    history = recent_components(track, owner.frame,
                                int(cfg["history_window_frames"]))
    aligned = []
    for item in history:
        mask = component_mask(item, shape)
        aligned.append(translate_mask(
            mask, int(round(owner.y - item.y)), int(round(owner.x - item.x))))
    occupancy = np.mean(np.stack(aligned), axis=0)
    core = occupancy >= float(cfg["core_occupancy_fraction"])
    erosion = int(cfg["core_erosion_px"])
    if erosion:
        eroded = ndi.binary_erosion(core, structure=STRUCTURE, iterations=erosion)
        if int(eroded.sum()) >= int(cfg["minimum_core_px"]):
            core = eroded
    if int(core.sum()) < int(cfg["minimum_core_px"]):
        core = component_mask(owner, shape)
    return core


def tracklet_tier(track: Tracklet, owner: Component,
                  frame_items: list[Component], cfg: dict) -> str:
    history = recent_components(track, owner.frame,
                                int(cfg["history_window_frames"]))
    median_area = float(np.median([item.area for item in history]))
    if median_area < float(cfg["large_area_px"]):
        return "established_small" if len(track.components) >= 4 else "provisional_small"
    steps = [distance(first, second) for first, second in zip(history, history[1:])]
    mobile = bool(steps and np.percentile(steps, 75) > float(cfg["static_step_px"]))
    if mobile:
        return "large_mobile"
    owner_radius = np.sqrt(owner.area / np.pi)
    nearby = False
    for other in frame_items:
        if other is owner:
            continue
        other_radius = np.sqrt(other.area / np.pi)
        edge_gap = distance(owner, other) - owner_radius - other_radius
        if edge_gap <= float(cfg["nearby_cell_edge_px"]):
            nearby = True
            break
    return "large_static_neighboured" if nearby else "large_static_isolated"


def current_components(frame: np.ndarray, frame_index: int,
                       minimum_area: int = 1) -> list[Component]:
    return frame_components(frame, frame_index, minimum_area)


def assign_component(candidate: np.ndarray, component: Component,
                     new_identity: int, cfg: dict, method: str,
                     actions: list[dict], event_id: str,
                     allow_swap: bool = True) -> int:
    """Assign one local component and swap a displaced nearby owner, never the field."""
    frame = candidate[component.frame]
    mask = component_mask(component, frame.shape)
    values, counts = np.unique(frame[mask], return_counts=True)
    old_identity = int(values[np.argmax(counts)]) if len(values) else 0
    if old_identity == new_identity and bool(np.all(frame[mask] == new_identity)):
        return 0
    existing_parts, count = ndi.label(frame == new_identity, structure=STRUCTURE)
    swapped = 0
    for number in range(1, count + 1):
        existing = existing_parts == number
        if np.any(existing & mask):
            continue
        area = int(existing.sum())
        if area < int(cfg["minimum_independent_component_px"]):
            continue
        yy, xx = np.where(existing)
        separation = float(np.hypot(xx.mean() - component.x,
                                    yy.mean() - component.y))
        edge_gap = float(ndi.distance_transform_edt(~mask)[existing].min())
        if (allow_swap and edge_gap > 1.5
                and separation <= float(cfg["maximum_swap_distance_px"])
                and old_identity not in (0, new_identity)):
            frame[existing] = old_identity
            swapped += area
    changed = int(np.count_nonzero(frame[mask] != new_identity))
    frame[mask] = new_identity
    if changed or swapped:
        actions.append({
            "event_id": event_id, "method": method,
            "output_frame": component.frame + 1,
            "assigned_identity": int(new_identity),
            "displaced_identity": old_identity,
            "assigned_pixels": changed, "swapped_pixels": swapped,
            "x": component.x, "y": component.y,
        })
    return changed + swapped


def independent_challenger(row: pd.Series, target: Component,
                           frames: list[list[Component]], lag: np.ndarray,
                           cfg: dict) -> Component | None:
    destination = target.frame
    threshold = 1.5
    transition = lag[destination - 1] if destination else np.zeros(lag.shape[1:])
    choices = []
    for item in frames[destination]:
        if item is target or item.observed_identity != int(row.challenger_identity):
            continue
        if item.area < int(cfg["minimum_independent_component_px"]):
            continue
        if distance(item, target) > float(cfg["maximum_radial_distance_px"]):
            continue
        mask = component_mask(item, transition.shape)
        red = float(np.mean(transition[mask] >= threshold))
        choices.append(((red, -distance(item, target), item.area), item))
    return max(choices, key=lambda value: value[0])[1] if choices else None


def apply_foundation(candidate: np.ndarray, events: pd.DataFrame,
                     frames: list[list[Component]], tracklets: dict[int, Tracklet],
                     lag: np.ndarray, cfg: dict, actions: list[dict]) -> None:
    del lag
    ordered = events.sort_values(
        ["destination_output_frame", "owner_local_age_frames"],
        ascending=[True, False])
    claimed: set[tuple[int, int]] = set()
    for _, row in ordered.iterrows():
        owner, target = find_event_components(row, frames)
        key = (target.frame, int(target.pixels[0]))
        if key in claimed:
            continue
        core = protected_core(tracklets[owner.tracklet], owner,
                              candidate.shape[1:], cfg)
        if int(core.sum()) < int(cfg["minimum_core_px"]):
            continue
        tier = tracklet_tier(tracklets[owner.tracklet], owner,
                             frames[owner.frame], cfg)
        before = len(actions)
        assign_component(candidate, target, int(row.owner_identity), cfg,
                         "mandatory_component_tenure", actions,
                         str(row.event_id), allow_swap=False)
        if len(actions) > before:
            actions[-1]["owner_tier"] = tier
            actions[-1]["protected_core_px"] = int(core.sum())
        claimed.add(key)


def apply_volume_rollback(candidate: np.ndarray, events: pd.DataFrame,
                          frames: list[list[Component]],
                          tracklets: dict[int, Tracklet], cfg: dict,
                          volume_cfg: dict, actions: list[dict]) -> None:
    shape = candidate.shape[1:]
    for _, row in events.iterrows():
        owner, target = find_event_components(row, frames)
        owner_track = tracklets[owner.tracklet]
        challenger_track = tracklets[target.tracklet]
        owner_history = recent_components(
            owner_track, owner.frame, int(cfg["history_window_frames"]))
        territory = np.zeros(shape, bool)
        for item in owner_history:
            territory |= component_mask(item, shape)
        territory = ndi.binary_dilation(
            territory, structure=STRUCTURE,
            iterations=int(volume_cfg["core_dilation_px"]))
        start = target.frame - int(volume_cfg["window_frames"])
        for item in challenger_track.components:
            if item.frame < start or item.frame >= target.frame:
                continue
            mask = component_mask(item, shape)
            overlap_fraction = float(np.mean(territory[mask]))
            if overlap_fraction < float(volume_cfg["minimum_territory_fraction"]):
                continue
            assign_component(candidate, item, int(row.owner_identity), cfg,
                             "progressive_volume_rollback", actions,
                             str(row.event_id), allow_swap=False)


def nearest_partition(host: np.ndarray, owner_seed: np.ndarray,
                      challenger_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    owner_distance = ndi.distance_transform_edt(~owner_seed)
    challenger_distance = ndi.distance_transform_edt(~challenger_seed)
    challenger = host & (challenger_distance < owner_distance)
    owner = host & ~challenger
    return owner, challenger


def apply_merge_partition(candidate: np.ndarray, events: pd.DataFrame,
                          frames: list[list[Component]], lag: np.ndarray,
                          cfg: dict, merge_cfg: dict,
                          actions: list[dict]) -> None:
    shape = candidate.shape[1:]
    for _, row in events.iterrows():
        _, target = find_event_components(row, frames)
        challenger = independent_challenger(row, target, frames, lag, cfg)
        if challenger is None:
            continue
        owner_ref = target
        challenger_ref = challenger
        for frame_index in range(target.frame + 1, min(
                len(candidate), target.frame + 1 + int(merge_cfg["maximum_frames"]))):
            foreground, count = ndi.label(candidate[frame_index] > 0,
                                          structure=STRUCTURE)
            owner_seed = component_mask(owner_ref, shape)
            challenger_seed = component_mask(challenger_ref, shape)
            host_choices = []
            for number in range(1, count + 1):
                host = foreground == number
                host_distance = ndi.distance_transform_edt(~host)
                owner_distance = float(host_distance[owner_seed].min())
                challenger_distance = float(host_distance[challenger_seed].min())
                maximum = float(merge_cfg["maximum_host_seed_distance_px"])
                if owner_distance <= maximum and challenger_distance <= maximum:
                    host_choices.append(((max(owner_distance, challenger_distance),
                                          owner_distance + challenger_distance), host))
            if not host_choices:
                break
            host = min(host_choices, key=lambda value: value[0])[1]
            owner_part, challenger_part = nearest_partition(
                host, owner_seed, challenger_seed)
            minimum = int(merge_cfg["minimum_seed_px"])
            if int(owner_part.sum()) < minimum or int(challenger_part.sum()) < minimum:
                break
            before = candidate[frame_index].copy()
            candidate[frame_index][owner_part] = int(row.owner_identity)
            candidate[frame_index][challenger_part] = int(row.challenger_identity)
            changed = int(np.count_nonzero(before != candidate[frame_index]))
            actions.append({
                "event_id": str(row.event_id),
                "method": "explicit_multi_owner_partition",
                "output_frame": frame_index + 1,
                "assigned_identity": int(row.owner_identity),
                "displaced_identity": int(row.challenger_identity),
                "assigned_pixels": changed, "swapped_pixels": 0,
                "x": float(np.where(host)[1].mean()),
                "y": float(np.where(host)[0].mean()),
                "owner_partition_px": int(owner_part.sum()),
                "challenger_partition_px": int(challenger_part.sum()),
            })
            owner_ref = component_from_mask(owner_part, frame_index,
                                            int(row.owner_identity))
            challenger_ref = component_from_mask(challenger_part, frame_index,
                                                 int(row.challenger_identity))


def best_any(candidates: list[Component], reference: Component,
             maximum_distance: float, gap: int,
             excluded: set[int]) -> Component | None:
    choices = []
    for item in candidates:
        if int(item.pixels[0]) in excluded:
            continue
        shared = overlap_pixels(item, reference)
        separation = distance(item, reference)
        if shared == 0 and separation > maximum_distance * max(gap, 1):
            continue
        ratio_penalty = abs(np.log(max(item.area, 1) / max(reference.area, 1)))
        choices.append(((shared, -separation, -ratio_penalty, item.area), item))
    return max(choices, key=lambda value: value[0])[1] if choices else None


def apply_bidirectional_bracket(candidate: np.ndarray, events: pd.DataFrame,
                                frames: list[list[Component]], lag: np.ndarray,
                                cfg: dict, bracket_cfg: dict,
                                actions: list[dict]) -> None:
    del lag
    for _, row in events.iterrows():
        _, target = find_event_components(row, frames)
        challenger_ref = independent_challenger(row, target, frames,
                                                np.zeros((len(candidate) - 1,
                                                          *candidate.shape[1:])),
                                                cfg)
        owner_branch = Branch(int(row.owner_identity), target, target.frame)
        challenger_branch = (Branch(int(row.challenger_identity), challenger_ref,
                                    target.frame) if challenger_ref is not None else None)
        for frame_index in range(target.frame + 1, min(
                len(candidate), target.frame + 1 + int(bracket_cfg["maximum_frames"]))):
            items = current_components(candidate[frame_index], frame_index,
                                       int(cfg["minimum_independent_component_px"]))
            used: set[int] = set()
            owner_gap = frame_index - owner_branch.last_frame
            owner_match = best_any(
                items, owner_branch.reference,
                float(bracket_cfg["maximum_step_px"]), owner_gap, used)
            if owner_match is not None:
                assign_component(candidate, owner_match, owner_branch.identity, cfg,
                                 "bidirectional_bracket_owner", actions,
                                 str(row.event_id))
                used.add(int(owner_match.pixels[0]))
                owner_branch.reference = component_from_mask(
                    component_mask(owner_match, candidate.shape[1:]), frame_index,
                    owner_branch.identity)
                owner_branch.last_frame = frame_index
                owner_branch.missing_frames = 0
            else:
                owner_branch.missing_frames += 1
            if challenger_branch is not None:
                challenger_gap = frame_index - challenger_branch.last_frame
                challenger_match = best_any(
                    items, challenger_branch.reference,
                    float(bracket_cfg["maximum_step_px"]), challenger_gap, used)
                if challenger_match is not None:
                    assign_component(candidate, challenger_match,
                                     challenger_branch.identity, cfg,
                                     "bidirectional_bracket_challenger", actions,
                                     str(row.event_id))
                    challenger_branch.reference = component_from_mask(
                        component_mask(challenger_match, candidate.shape[1:]),
                        frame_index, challenger_branch.identity)
                    challenger_branch.last_frame = frame_index
                    challenger_branch.missing_frames = 0
                else:
                    challenger_branch.missing_frames += 1
            if owner_branch.missing_frames > int(bracket_cfg["maximum_gap_frames"]):
                break


def quarantine_labels(labels: np.ndarray, lag: np.ndarray, params: dict,
                      enabled: set[str]) -> tuple[np.ndarray, pd.DataFrame,
                                                  pd.DataFrame]:
    events, frames, tracklets = detect_takeovers(
        labels, lag, params["tenure_audit"],
        int(params.get("source_frame_offset", 0)))
    # Attempt 020 already handles abrupt one-frame replacements.  This layer is
    # intentionally reserved for measured, progressive encroachment: the same
    # local owner/challenger relationship must be observed at least twice.
    events["owner_confidence_layer"] = ""
    if events.empty:
        events["actionable_progressive_pair"] = pd.Series(dtype=bool)
        events["addon_representative"] = pd.Series(dtype=bool)
        return labels.copy(), events, pd.DataFrame()
    for index, row in events.iterrows():
        owner, _ = find_event_components(row, frames)
        events.at[index, "owner_confidence_layer"] = tracklet_tier(
            tracklets[owner.tracklet], owner, frames[owner.frame],
            params["foundation"])
    selected: list[int] = []
    addon_selected: list[int] = []
    separation = int(params["foundation"]["maximum_pair_separation_frames"])
    for _, group in events.groupby(["owner_identity", "challenger_identity"]):
        group = group.sort_values("source_output_frame")
        cluster: list[int] = []
        previous: int | None = None
        for index, row in group.iterrows():
            frame = int(row.source_output_frame)
            if previous is not None and frame - previous > separation:
                if len(cluster) >= 2:
                    layers = events.loc[cluster, "owner_confidence_layer"]
                    if bool(layers.str.startswith("large_static_").all()):
                        selected.extend(cluster)
                        addon_selected.append(cluster[-1])
                cluster = []
            cluster.append(index)
            previous = frame
        if len(cluster) >= 2:
            layers = events.loc[cluster, "owner_confidence_layer"]
            if bool(layers.str.startswith("large_static_").all()):
                selected.extend(cluster)
                addon_selected.append(cluster[-1])
    actionable = events.loc[selected].copy()
    addon_events = events.loc[addon_selected].copy()
    candidate = labels.copy()
    actions: list[dict] = []
    apply_foundation(candidate, actionable, frames, tracklets, lag,
                     params["foundation"], actions)
    if "V" in enabled:
        apply_volume_rollback(candidate, addon_events, frames, tracklets,
                              params["foundation"], params["volume_rollback"],
                              actions)
    if "M" in enabled:
        apply_merge_partition(candidate, addon_events, frames, lag,
                              params["foundation"], params["merge_partition"],
                              actions)
    if "B" in enabled:
        apply_bidirectional_bracket(candidate, addon_events, frames, lag,
                                    params["foundation"], params["bracket"],
                                    actions)
    events["actionable_progressive_pair"] = events.index.isin(selected)
    events["addon_representative"] = events.index.isin(addon_selected)
    return candidate, events, pd.DataFrame(actions)
