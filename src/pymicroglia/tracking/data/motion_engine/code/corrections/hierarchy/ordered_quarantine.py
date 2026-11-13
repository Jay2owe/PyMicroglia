"""Apply permanent hierarchy-ordered ownership and gap-safe motion recovery."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import find_project_root
import lineage_hierarchy as m1


ROOT = find_project_root(Path(__file__))
TUNING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402
from duplicate_cleanup import clean_duplicates  # noqa: E402


@dataclass
class Event:
    number: int
    method: str
    source_frame: int
    owner: m1.Component
    destination: m1.Component
    companion_source: m1.Component
    companion_destination: m1.Component | None
    owner_identity: int
    companion_identity: int
    owner_layer: str
    priority_rank: int
    owner_birth_frame: int
    horizon: int
    metrics: dict


@dataclass
class Branch:
    canonical_identity: int
    reference: m1.Component
    last_seen_frame: int
    aliases: set[int] = field(default_factory=set)
    missing_frames: int = 0


@dataclass
class Lock:
    number: int
    canonical_identity: int
    layer: str
    priority_rank: int
    birth_frame: int
    reference: m1.Component
    last_seen_frame: int
    owner_aliases: set[int]
    missing_frames: int = 0
    companions: list[Branch] = field(default_factory=list)


def comp_from_mask(mask: np.ndarray, frame: int,
                   identity: int) -> m1.Component:
    yy, xx = np.where(mask)
    return m1.Component(
        frame, identity, 0, np.flatnonzero(mask.ravel()), int(mask.sum()),
        float(xx.mean()), float(yy.mean()))


def meta_maps(tracklets: list[m1.Tracklet], classes: pd.DataFrame
              ) -> tuple[dict[tuple[int, int, int], m1.Tracklet], dict[int, dict]]:
    node_map = {}
    for tracklet in tracklets:
        for node in tracklet.nodes:
            node_map[(node.frame, node.identity, node.number)] = tracklet
    class_map = {int(row.tracklet_id): row._asdict()
                 for row in classes.itertuples(index=False)}
    return node_map, class_map


def authority_key(classification: dict) -> tuple[int, int, int]:
    """Lower keys reserve territory first and can never be promoted later."""
    return (
        int(classification["priority_rank"]),
        int(classification["birth_output_frame"]),
        int(classification["tracklet_id"]),
    )


def owner_has_priority(owner: dict, challenger: dict) -> bool:
    """Apply permanent quarantine both between and within all four layers."""
    return authority_key(owner) < authority_key(challenger)


def transition_metrics(owner: m1.Component, destination: m1.Component,
                       lag: np.ndarray, threshold: float,
                       shape: tuple[int, int]) -> dict:
    shared = m1.overlap(owner, destination)
    union = owner.area + destination.area - shared
    owner_mask = m1.component_mask(owner, shape)
    destination_mask = m1.component_mask(destination, shape)
    return {
        "overlap_px": shared,
        "previous_area_overlap_fraction": shared / max(owner.area, 1),
        "intersection_over_union": shared / max(union, 1),
        "centroid_shift_px": m1.distance(owner, destination),
        "blue_departure_coverage": float(np.mean(
            lag[owner_mask] <= -threshold)),
        "red_arrival_coverage": float(np.mean(
            lag[destination_mask] >= threshold)),
    }


def direct_pass(metrics: dict, cfg: dict) -> bool:
    return bool(
        metrics["previous_area_overlap_fraction"]
        >= float(cfg["minimum_previous_overlap_fraction"])
        and metrics["intersection_over_union"]
        >= float(cfg["minimum_intersection_over_union"])
        and metrics["centroid_shift_px"]
        <= float(cfg["maximum_centroid_shift_px"])
        and metrics["blue_departure_coverage"]
        < float(cfg["maximum_blue_departure_coverage"]))


def tracklet_node_index(tracklet: m1.Tracklet,
                        component: m1.Component) -> int:
    for index, node in enumerate(tracklet.nodes):
        if (node.frame, node.identity, node.number) == (
                component.frame, component.identity, component.number):
            return index
    raise KeyError("component is not in tracklet")


def detect_direct(frames: list[list[m1.Component]], lag: np.ndarray,
                  node_map: dict, class_map: dict, hierarchy: dict,
                  cfg: dict, shape: tuple[int, int]) -> list[Event]:
    events = []
    threshold = float(hierarchy["motion_threshold"])
    for frame in range(len(frames) - 1):
        for owner in frames[frame]:
            owner_track = node_map[(frame, owner.identity, owner.number)]
            owner_class = class_map[owner_track.number]
            if tracklet_node_index(owner_track, owner) + 1 < int(
                    hierarchy["minimum_owner_tenure_frames"]):
                continue
            destination = m1.best_match(
                frames[frame + 1], owner, 20.0,
                minimum_area_ratio=0.1, maximum_area_ratio=10.0)
            if destination is None or destination.identity == owner.identity:
                continue
            challenger_track = node_map[(
                destination.frame, destination.identity, destination.number)]
            challenger_class = class_map[challenger_track.number]
            if not owner_has_priority(owner_class, challenger_class):
                continue
            prior_challengers = [node for node in challenger_track.nodes
                                 if node.frame <= frame]
            if not prior_challengers:
                continue
            companion_source = min(
                prior_challengers,
                key=lambda node: (frame - node.frame,
                                  m1.distance(node, destination)))
            metrics = transition_metrics(
                owner, destination, lag[frame], threshold, shape)
            if not direct_pass(metrics, cfg):
                continue
            remnants = [node for node in frames[frame + 1]
                        if node.identity == owner.identity
                        and node is not destination]
            remnant = m1.best_match(
                remnants, companion_source, 60.0,
                minimum_area_ratio=0.05, maximum_area_ratio=20.0)
            events.append(Event(
                0, "direct_overlap", frame, owner, destination,
                companion_source, remnant,
                int(owner_class["canonical_identity"]),
                int(challenger_class["canonical_identity"]),
                str(owner_class["confidence_layer"]),
                int(owner_class["priority_rank"]),
                int(owner_class["birth_output_frame"]) - 1,
                1, metrics))
    return events


def delayed_support(raw: np.ndarray, source_frame: int,
                    source: m1.Component, tracklet: m1.Tracklet,
                    first_destination: m1.Component, hierarchy: dict,
                    cfg: dict, shape: tuple[int, int]) -> tuple[int, dict] | None:
    threshold = float(hierarchy["motion_threshold"])
    source_mask = m1.component_mask(source, shape)
    future = [node for node in tracklet.nodes
              if node.frame >= first_destination.frame]
    by_frame = {node.frame: node for node in future}
    best = None
    for horizon in range(1, int(cfg["maximum_horizon_frames"]) + 1):
        frame = source_frame + horizon
        destination = by_frame.get(frame)
        if destination is None or frame >= len(raw):
            continue
        delayed = np.log2((raw[frame].astype(np.float32) + 1.0)
                          / (raw[source_frame].astype(np.float32) + 1.0))
        destination_mask = m1.component_mask(destination, shape)
        blue = float(np.mean(delayed[source_mask] <= -threshold))
        red = float(np.mean(delayed[destination_mask] >= threshold))
        score = min(blue, red)
        if best is None or score > best[0]:
            best = (score, horizon, {
                "delayed_blue_coverage": blue,
                "delayed_red_coverage": red,
                "delayed_destination_area_px": destination.area,
            })
    if best is None:
        return None
    _, horizon, metrics = best
    if (metrics["delayed_blue_coverage"] < float(cfg["minimum_delayed_blue"])
            or metrics["delayed_red_coverage"]
            < float(cfg["minimum_delayed_red"])):
        return None
    return horizon, metrics


def detect_delayed(frames: list[list[m1.Component]], raw: np.ndarray,
                   node_map: dict, class_map: dict, hierarchy: dict,
                   cfg: dict, shape: tuple[int, int]) -> list[Event]:
    events = []
    for frame in range(len(frames) - 1):
        for owner in frames[frame]:
            owner_track = node_map[(frame, owner.identity, owner.number)]
            owner_class = class_map[owner_track.number]
            if tracklet_node_index(owner_track, owner) + 1 < int(
                    hierarchy["minimum_owner_tenure_frames"]):
                continue
            remnants = [node for node in frames[frame + 1]
                        if node.identity == owner.identity]
            remnant = m1.best_match(
                remnants, owner, 20.0,
                minimum_area_ratio=0.05, maximum_area_ratio=10.0)
            if remnant is None:
                continue
            retained = m1.overlap(owner, remnant) / max(owner.area, 1)
            if retained > float(cfg["maximum_owner_retained_fraction"]):
                continue
            if remnant.area / max(owner.area, 1) > float(
                    cfg["maximum_remnant_area_ratio"]):
                continue
            for destination in frames[frame + 1]:
                if (destination.identity == owner.identity
                        or m1.distance(owner, destination)
                        > float(cfg["maximum_destination_radius_px"])):
                    continue
                challenger_track = node_map[(destination.frame,
                                             destination.identity,
                                             destination.number)]
                challenger_class = class_map[challenger_track.number]
                if not owner_has_priority(owner_class, challenger_class):
                    continue
                prior_nodes = [node for node in challenger_track.nodes
                               if node.frame <= frame]
                if not prior_nodes:
                    continue
                companion_source = max(prior_nodes, key=lambda node: node.frame)
                prior_area = companion_source.area
                growth = destination.area - prior_area
                ratio = destination.area / max(prior_area, 1)
                if (growth < int(cfg["minimum_challenger_growth_px"])
                        and ratio < float(cfg["minimum_challenger_area_ratio"])):
                    continue
                supported = delayed_support(
                    raw, frame, owner, challenger_track, destination,
                    hierarchy, cfg, shape)
                if supported is None:
                    continue
                horizon, delayed = supported
                metrics = {
                    "owner_retained_fraction": retained,
                    "remnant_area_ratio": remnant.area / max(owner.area, 1),
                    "challenger_growth_px": growth,
                    "challenger_area_ratio": ratio,
                    "destination_distance_px": m1.distance(owner, destination),
                    **delayed,
                }
                events.append(Event(
                    0, "delayed_remnant", frame, owner, destination,
                    companion_source, remnant,
                    int(owner_class["canonical_identity"]),
                    int(challenger_class["canonical_identity"]),
                    str(owner_class["confidence_layer"]),
                    int(owner_class["priority_rank"]),
                    int(owner_class["birth_output_frame"]) - 1,
                    horizon, metrics))
    return events


def motion_destination(reference: m1.Component,
                       candidates: list[m1.Component], lag: np.ndarray,
                       hierarchy: dict, shape: tuple[int, int],
                       used: set[tuple[int, int]]) -> m1.Component | None:
    source_mask = m1.component_mask(reference, shape)
    threshold = float(hierarchy["motion_threshold"])
    blue = float(np.mean(lag[source_mask] <= -threshold))
    if blue < float(hierarchy["minimum_motion_blue"]):
        return None
    choices = []
    for item in candidates:
        key = (item.identity, item.number)
        if key in used or m1.distance(reference, item) > float(
                hierarchy["continuous_motion_radius_px"]):
            continue
        red = float(np.mean(
            lag[m1.component_mask(item, shape)] >= threshold))
        if red >= float(hierarchy["minimum_motion_red"]):
            choices.append(((red, -m1.distance(reference, item), item.area), item))
    return max(choices, key=lambda value: value[0])[1] if choices else None


def choose_destination(lock: Lock, frame: int,
                       items: list[m1.Component], lag: np.ndarray,
                       hierarchy: dict, shape: tuple[int, int],
                       used: set[tuple[int, int]]) -> tuple[m1.Component | None, str]:
    continuous = lock.last_seen_frame == frame
    if continuous and lock.missing_frames <= int(
            hierarchy["maximum_missing_frames_for_large_jump"]):
        moved = motion_destination(
            lock.reference, items, lag, hierarchy, shape, used)
        if moved is not None:
            return moved, "continuous_motion_jump"
    if lock.missing_frames >= 2:
        radius = float(hierarchy["post_gap_local_radius_px"])
        action = "post_gap_local_reacquisition"
    else:
        radius = 20.0
        action = "ordered_local_continuation"
    match = m1.best_match(
        items, lock.reference, radius, used,
        minimum_area_ratio=0.05, maximum_area_ratio=20.0)
    return match, action


def assign(labels: np.ndarray, frame: int, component: m1.Component,
           identity: int) -> tuple[m1.Component, int, int]:
    """Assign through the component's own flat pixel indices.

    When they are strictly increasing and inside a writable C-ordered frame, the
    whole-frame mask's True positions in row-major order are exactly those
    indices: the same values are read and written, and the new component has
    the same pixels, area and coordinates. Anything else takes the mask route.
    """
    pixels = component.pixels
    plane = labels[frame] if isinstance(labels, np.ndarray) and labels.ndim == 3 else None
    if plane is not None and type(plane) is not np.ndarray:
        plane = plane.view(np.ndarray)
    if (plane is None or not plane.flags.c_contiguous or not plane.flags.writeable
            or not isinstance(pixels, np.ndarray) or pixels.ndim != 1
            or pixels.dtype.kind not in "iu" or not len(pixels)
            or int(pixels[0]) < 0 or int(pixels[-1]) >= plane.size
            or (len(pixels) > 1 and not bool(np.all(pixels[1:] > pixels[:-1])))):
        return _assign_mask(labels, frame, component, identity)
    flat = plane.reshape(-1)
    values = flat[pixels]
    observed_values, counts = np.unique(values, return_counts=True)
    observed = int(observed_values[np.argmax(counts)])
    changed = int(np.count_nonzero(values != identity))
    flat[pixels] = identity
    positions = pixels.astype(np.intp, copy=True)
    yy, xx = np.divmod(positions, plane.shape[1])
    return m1.Component(
        frame, identity, 0, positions, int(len(positions)),
        float(xx.mean()), float(yy.mean())), observed, changed


def _assign_mask(labels: np.ndarray, frame: int, component: m1.Component,
                 identity: int) -> tuple[m1.Component, int, int]:
    mask = m1.component_mask(component, labels.shape[1:])
    values, counts = np.unique(labels[frame][mask], return_counts=True)
    observed = int(values[np.argmax(counts)])
    changed = int(np.count_nonzero(labels[frame][mask] != identity))
    labels[frame][mask] = identity
    return comp_from_mask(mask, frame, identity), observed, changed


def propagate_branch(labels: np.ndarray, frame: int, branch: Branch,
                     items: list[m1.Component], used: set[tuple[int, int]],
                     lag: np.ndarray, hierarchy: dict,
                     event_id: str, actions: list[dict]) -> None:
    eligible = [item for item in items
                if item.identity in branch.aliases
                | {branch.canonical_identity}]
    moved = None
    if branch.last_seen_frame == frame - 1 and branch.missing_frames <= 1:
        moved = motion_destination(
            branch.reference, eligible, lag, hierarchy,
            labels.shape[1:], used)
    match = moved or m1.best_match(
        eligible, branch.reference,
        float(hierarchy["post_gap_local_radius_px"])
        if branch.missing_frames >= 2 else 40.0,
        used, minimum_area_ratio=0.05, maximum_area_ratio=20.0)
    if match is None:
        branch.missing_frames += 1
        return
    used.add((match.identity, match.number))
    reference, observed, changed = assign(
        labels, frame, match, branch.canonical_identity)
    actions.append({
        "event_id": event_id, "action": "propagate_lower_layer_branch",
        "output_frame": frame + 1,
        "canonical_identity": branch.canonical_identity,
        "observed_identity": observed,
        "changed_pixels": changed, "area_px": match.area,
        "x": match.x, "y": match.y,
    })
    branch.reference = reference
    branch.last_seen_frame = frame
    branch.missing_frames = 0


def apply_events(labels: np.ndarray, lag: np.ndarray, events: list[Event],
                 frames: list[list[m1.Component]], hierarchy: dict
                 ) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    actions = []
    locks: list[Lock] = []
    events_by_source: dict[int, list[Event]] = {}
    for event in events:
        events_by_source.setdefault(event.source_frame, []).append(event)
    shape = labels.shape[1:]
    for frame in range(len(labels) - 1):
        transition = lag[frame]
        next_items = frames[frame + 1]
        used: set[tuple[int, int]] = set()
        for lock in sorted(locks, key=lambda item: (
                item.priority_rank, item.birth_frame, item.number)):
            match, action = choose_destination(
                lock, frame, next_items, transition, hierarchy, shape, used)
            if match is None:
                lock.missing_frames += 1
                continue
            used.add((match.identity, match.number))
            event_id = f"H{lock.number:04d}"
            reference, observed, changed = assign(
                candidate, frame + 1, match, lock.canonical_identity)
            actions.append({
                "event_id": event_id, "action": action,
                "output_frame": frame + 2,
                "canonical_identity": lock.canonical_identity,
                "observed_identity": observed,
                "changed_pixels": changed, "area_px": match.area,
                "x": match.x, "y": match.y,
            })
            lock.reference = reference
            lock.last_seen_frame = frame + 1
            lock.missing_frames = 0
            lock.owner_aliases.add(match.identity)
            for branch in lock.companions:
                propagate_branch(
                    candidate, frame + 1, branch, next_items, used,
                    transition, hierarchy,
                    event_id, actions)

        for event in sorted(events_by_source.get(frame, []),
                            key=lambda item: (item.priority_rank,
                                              item.owner_birth_frame,
                                              item.number)):
            destination_key = (event.destination.identity,
                               event.destination.number)
            if destination_key in used:
                actions.append({
                    "event_id": f"E{event.number:04d}",
                    "action": "activation_blocked_by_higher_layer",
                    "output_frame": frame + 2,
                    "canonical_identity": event.owner_identity,
                    "observed_identity": event.destination.identity,
                    "changed_pixels": 0, "area_px": event.destination.area,
                    "x": event.destination.x, "y": event.destination.y,
                })
                continue
            used.add(destination_key)
            reference, observed, changed = assign(
                candidate, frame + 1, event.destination,
                event.owner_identity)
            branches = []
            if event.companion_destination is not None:
                companion_key = (event.companion_destination.identity,
                                 event.companion_destination.number)
                if companion_key not in used:
                    used.add(companion_key)
                    branch_reference, old, altered = assign(
                        candidate, frame + 1,
                        event.companion_destination,
                        event.companion_identity)
                    branches.append(Branch(
                        event.companion_identity, branch_reference,
                        frame + 1, {
                            event.owner.identity,
                            event.destination.identity,
                            event.companion_identity,
                        }))
                    actions.append({
                        "event_id": f"E{event.number:04d}",
                        "action": "restore_birth_small_branch",
                        "output_frame": frame + 2,
                        "canonical_identity": event.companion_identity,
                        "observed_identity": old,
                        "changed_pixels": altered,
                        "area_px": event.companion_destination.area,
                        "x": event.companion_destination.x,
                        "y": event.companion_destination.y,
                    })
            locks.append(Lock(
                event.number, event.owner_identity, event.owner_layer,
                event.priority_rank, event.owner_birth_frame,
                reference, frame + 1, {
                    event.owner.identity, event.destination.identity,
                    event.owner_identity}, companions=branches))
            actions.append({
                "event_id": f"E{event.number:04d}",
                "action": f"activate_{event.method}",
                "output_frame": frame + 2,
                "canonical_identity": event.owner_identity,
                "observed_identity": observed,
                "changed_pixels": changed,
                "area_px": event.destination.area,
                "x": event.destination.x, "y": event.destination.y,
            })
    return candidate, pd.DataFrame(actions)


def event_table(events: list[Event], source_offset: int) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append({
            "event_id": f"E{event.number:04d}",
            "method": event.method,
            "source_output_frame": event.source_frame + 1,
            "source_imagej_frame": event.source_frame + source_offset + 1,
            "destination_output_frame": event.source_frame + 2,
            "destination_imagej_frame": event.source_frame + source_offset + 2,
            "owner_identity": event.owner_identity,
            "challenger_identity": event.companion_identity,
            "owner_layer": event.owner_layer,
            "priority_rank": event.priority_rank,
            "owner_birth_output_frame": event.owner_birth_frame + 1,
            "owner_area_px": event.owner.area,
            "challenger_birth_area_px": event.companion_source.area,
            "destination_area_px": event.destination.area,
            "horizon_frames": event.horizon,
            **event.metrics,
        })
    return pd.DataFrame(rows)


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    labels = load_stack(ROOT / params["baseline_labels"])
    trim_dir = ROOT / params["trim_dir"]
    lag = load_stack(trim_dir / f"{params['stem']}_lag01_float.tif")
    raw = load_stack(trim_dir / f"{params['stem']}_registered_raw.tif")
    _, classes, frames, tracklets = m1.hierarchy_tables(
        labels, params["hierarchy"])
    node_map, class_map = meta_maps(tracklets, classes)
    events = []
    enabled = set(params["enabled_methods"])
    if "D" in enabled:
        events.extend(detect_direct(
            frames, lag, node_map, class_map, params["hierarchy"],
            params["direct"], labels.shape[1:]))
    if "M" in enabled:
        events.extend(detect_delayed(
            frames, raw, node_map, class_map, params["hierarchy"],
            params["delayed"], labels.shape[1:]))
    events.sort(key=lambda event: (
        event.source_frame, event.priority_rank, event.owner_birth_frame,
        event.method))
    for number, event in enumerate(events, 1):
        event.number = number
    candidate, actions = apply_events(
        labels, lag, events, frames, params["hierarchy"])
    candidate, duplicates = clean_duplicates(
        candidate, params["duplicates"], int(params["source_frame_offset"]))
    label_path = out.out / f"{params['stem']}.tif"
    event_path = out.out / "ordered_events.csv"
    action_path = out.out / "ordered_actions.csv"
    duplicate_path = out.out / "duplicate_reassignments.csv"
    metrics_path = out.out / "ordered_metrics.json"
    save_stack(label_path, candidate, params["frame_interval_min"])
    event_table(events, int(params["source_frame_offset"])).to_csv(
        event_path, index=False)
    actions.to_csv(action_path, index=False)
    duplicates.to_csv(duplicate_path, index=False)
    summary = {
        "enabled_methods": sorted(enabled),
        "detected_events": int(len(events)),
        "direct_events": int(sum(event.method == "direct_overlap"
                                 for event in events)),
        "delayed_events": int(sum(event.method == "delayed_remnant"
                                  for event in events)),
        "changed_identity_pixels": int(np.count_nonzero(candidate != labels)),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "identities_before": int(len(set(map(int, np.unique(labels))) - {0})),
        "identities_after": int(len(set(map(int, np.unique(candidate))) - {0})),
        "future_area_used_for_birth_priority": False,
        "permanent_quarantine_all_layers": True,
        "large_jump_requires_continuous_detection": True,
        "post_two_missing_frames_maximum_reacquisition_px": float(
            params["hierarchy"]["post_gap_local_radius_px"]),
        "registration_added": False,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"outputs": {
        "labels": label_path, "events": event_path, "actions": action_path,
        "duplicates": duplicate_path, "metrics": metrics_path},
        "summary": summary}

