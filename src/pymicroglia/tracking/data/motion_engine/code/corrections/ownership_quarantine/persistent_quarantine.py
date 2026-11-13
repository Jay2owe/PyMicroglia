"""Persistent local small-cell quarantine with earliest-frame canonical identity."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import find_project_root
from component_tenure_audit import (
    Component, component_mask, distance, frame_components, overlap_pixels,
)


TUNING = Path(__file__).resolve().parents[1]
ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402


@dataclass
class CompanionBranch:
    canonical_identity: int
    reference: Component
    last_seen_frame: int
    missing_frames: int = 0


@dataclass
class OwnershipLock:
    number: int
    canonical_identity: int
    reference: Component
    birth_frame: int
    birth_area: int
    activated_frame: int
    last_seen_frame: int
    owner_aliases: set[int]
    missing_frames: int = 0
    companions: list[CompanionBranch] = field(default_factory=list)


def comp_from_mask(mask: np.ndarray, frame: int, identity: int) -> Component:
    pixels = np.flatnonzero(mask.ravel())
    yy, xx = np.where(mask)
    return Component(frame, identity, 0, pixels, int(len(pixels)),
                     float(xx.mean()), float(yy.mean()))


def best_match(candidates: list[Component], reference: Component,
               radius: float, used: set[int] | None = None) -> Component | None:
    used = used or set()
    choices = []
    for item in candidates:
        key = int(item.pixels[0])
        if key in used:
            continue
        shared = overlap_pixels(reference, item)
        separation = distance(reference, item)
        if shared == 0 and separation > radius:
            continue
        ratio = abs(np.log(max(item.area, 1) / max(reference.area, 1)))
        choices.append(((shared, -separation, -ratio, item.area), item))
    return max(choices, key=lambda value: value[0])[1] if choices else None


def physical_chain_backward(frames: list[list[Component]], frame: int,
                            start: Component,
                            cfg: dict) -> list[Component]:
    """Trace geometry backward without trusting the current identity name."""
    chain = [start]
    reference = start
    for prior_frame in range(frame - 1, -1, -1):
        items = frames[prior_frame]
        match = best_match(items, reference, float(cfg["owner_match_radius_px"]))
        if match is None:
            break
        area_ratio = match.area / max(reference.area, 1)
        if area_ratio < 0.25 or area_ratio > 4.0:
            break
        chain.append(match)
        reference = match
    return chain


def eligible_large_static(chain: list[Component], cfg: dict) -> bool:
    if len(chain) < int(cfg["minimum_large_tenure_frames"]):
        return False
    # Jamie's rule is sticky: one genuinely small observation keeps this local
    # lineage out of the large-owner layer.
    if any(item.area <= int(cfg["small_area_px"]) for item in chain):
        return False
    recent = list(reversed(chain[:int(cfg["minimum_large_tenure_frames"])]))
    if float(np.median([item.area for item in recent])) < float(cfg["large_area_px"]):
        return False
    steps = [distance(first, second) for first, second in zip(recent, recent[1:])]
    return not steps or float(np.percentile(steps, 75)) <= float(
        cfg["static_step_p75_px"])


def isolated_edge_gap(owner: Component, items: list[Component]) -> float:
    owner_radius = np.sqrt(owner.area / np.pi)
    gaps = []
    for item in items:
        if item is owner:
            continue
        item_radius = np.sqrt(item.area / np.pi)
        gaps.append(distance(owner, item) - owner_radius - item_radius)
    return float(min(gaps)) if gaps else float("inf")


def prior_local_small(frames: list[list[Component]],
                      challenger_identity: int,
                      destination: Component, before_frame: int,
                      cfg: dict) -> tuple[bool, int | None, int | None]:
    radius = float(cfg["small_history_radius_px"])
    maximum = int(cfg["small_area_px"])
    earliest_frame = None
    earliest_area = None
    for frame in range(0, before_frame + 1):
        for item in frames[frame]:
            if item.observed_identity != challenger_identity:
                continue
            if item.area <= maximum and distance(item, destination) <= radius:
                if earliest_frame is None:
                    earliest_frame, earliest_area = frame, item.area
    return earliest_frame is not None, earliest_frame, earliest_area


def transition_metrics(owner: Component, destination: Component,
                       transition: np.ndarray, threshold: float,
                       shape: tuple[int, int]) -> dict:
    shared = overlap_pixels(owner, destination)
    union = owner.area + destination.area - shared
    owner_mask = component_mask(owner, shape)
    destination_mask = component_mask(destination, shape)
    return {
        "overlap_px": shared,
        "previous_area_overlap_fraction": shared / max(owner.area, 1),
        "intersection_over_union": shared / max(union, 1),
        "centroid_shift_px": distance(owner, destination),
        "blue_departure_coverage": float(np.mean(
            transition[owner_mask] <= -threshold)),
        "red_arrival_coverage": float(np.mean(
            transition[destination_mask] >= threshold)),
    }


def motion_destination(owner: Component, candidates: list[Component],
                       transition: np.ndarray, cfg: dict,
                       shape: tuple[int, int]) -> Component | None:
    owner_mask = component_mask(owner, shape)
    threshold = float(cfg["motion_threshold"])
    blue = float(np.mean(transition[owner_mask] <= -threshold))
    if blue < float(cfg["minimum_blue_departure_for_motion"]):
        return None
    choices = []
    for item in candidates:
        if distance(owner, item) > float(cfg["red_destination_radius_px"]):
            continue
        mask = component_mask(item, shape)
        red = float(np.mean(transition[mask] >= threshold))
        if red >= float(cfg["minimum_red_destination_for_motion"]):
            choices.append(((red, -distance(owner, item), item.area), item))
    return max(choices, key=lambda value: value[0])[1] if choices else None


def restore_displaced_branches(labels: np.ndarray, frame: int,
                               owner_reference: Component,
                               destination: Component, canonical: int,
                               owner_aliases: set[int],
                               cfg: dict, event_id: str,
                               actions: list[dict]) -> list[CompanionBranch]:
    current_items = frame_components(labels[frame - 1], frame - 1,
                                     int(cfg["minimum_component_area_px"]))
    next_items = frame_components(labels[frame], frame,
                                  int(cfg["minimum_component_area_px"]))
    owner_mask = component_mask(owner_reference, labels.shape[1:])
    restored = []
    for extra in next_items:
        if extra.observed_identity not in owner_aliases | {canonical}:
            continue
        if overlap_pixels(extra, destination):
            continue
        alternatives = [item for item in current_items
                        if not np.any(component_mask(item, labels.shape[1:])
                                      & owner_mask)]
        previous = best_match(alternatives, extra,
                              float(cfg["branch_match_radius_px"]))
        if previous is None or previous.observed_identity == canonical:
            continue
        mask = component_mask(extra, labels.shape[1:])
        changed = int(np.count_nonzero(labels[frame][mask]
                                       != previous.observed_identity))
        labels[frame][mask] = previous.observed_identity
        actions.append({
            "event_id": event_id, "action": "restore_displaced_branch",
            "output_frame": frame + 1,
            "canonical_identity": canonical,
            "observed_identity": extra.observed_identity,
            "assigned_identity": previous.observed_identity,
            "changed_pixels": changed, "area_px": extra.area,
            "x": extra.x, "y": extra.y,
        })
        restored.append(CompanionBranch(
            int(previous.observed_identity),
            comp_from_mask(mask, frame, int(previous.observed_identity)), frame))
    return restored


def assign_owner(labels: np.ndarray, frame: int, owner_reference: Component,
                 destination: Component, canonical: int,
                 owner_aliases: set[int], cfg: dict,
                 event_id: str, action: str, actions: list[dict]) -> Component:
    del owner_aliases
    mask = component_mask(destination, labels.shape[1:])
    values, counts = np.unique(labels[frame][mask], return_counts=True)
    observed = int(values[np.argmax(counts)])
    changed = int(np.count_nonzero(labels[frame][mask] != canonical))
    labels[frame][mask] = canonical
    actions.append({
        "event_id": event_id, "action": action,
        "output_frame": frame + 1,
        "canonical_identity": canonical,
        "observed_identity": observed,
        "assigned_identity": canonical,
        "changed_pixels": changed, "area_px": destination.area,
        "x": destination.x, "y": destination.y,
    })
    return comp_from_mask(mask, frame, canonical)


def propagate_companions(labels: np.ndarray, frame: int,
                         companions: list[CompanionBranch],
                         candidates: list[Component], used: set[int],
                         allowed_observed_identities: set[int],
                         transition: np.ndarray,
                         cfg: dict, event_id: str,
                         actions: list[dict]) -> None:
    """Keep displaced lower-priority physical branches without duplicating owner."""
    for branch in companions:
        allowed = allowed_observed_identities | {branch.canonical_identity}
        eligible = [item for item in candidates
                    if item.observed_identity in allowed]
        motion_choices = [item for item in eligible
                          if int(item.pixels[0]) not in used]
        match = motion_destination(branch.reference, motion_choices,
                                   transition, cfg, labels.shape[1:])
        if match is None:
            match = best_match(eligible, branch.reference,
                               float(cfg["branch_match_radius_px"]), used)
        if match is None:
            branch.missing_frames += 1
            continue
        used.add(int(match.pixels[0]))
        mask = component_mask(match, labels.shape[1:])
        values, counts = np.unique(labels[frame][mask], return_counts=True)
        observed = int(values[np.argmax(counts)])
        changed = int(np.count_nonzero(
            labels[frame][mask] != branch.canonical_identity))
        labels[frame][mask] = branch.canonical_identity
        actions.append({
            "event_id": event_id, "action": "propagate_displaced_branch",
            "output_frame": frame + 1,
            "canonical_identity": branch.canonical_identity,
            "observed_identity": observed,
            "assigned_identity": branch.canonical_identity,
            "changed_pixels": changed, "area_px": match.area,
            "x": match.x, "y": match.y,
        })
        branch.reference = comp_from_mask(
            mask, frame, branch.canonical_identity)
        branch.last_seen_frame = frame
        branch.missing_frames = 0


def reassign_owner_alias_extras(labels: np.ndarray, frame: int,
                                lock: OwnershipLock,
                                candidates: list[Component], used: set[int],
                                cfg: dict, event_id: str,
                                actions: list[dict]) -> None:
    """Give every spare owner-labelled piece to its tracked physical branch."""
    if not lock.companions:
        return
    owner_names = lock.owner_aliases | {lock.canonical_identity}
    radius = float(cfg["dormant_reacquisition_radius_px"])
    for extra in candidates:
        key = int(extra.pixels[0])
        if key in used or extra.observed_identity not in owner_names:
            continue
        choices = []
        for branch in lock.companions:
            separation = distance(branch.reference, extra)
            shared = overlap_pixels(branch.reference, extra)
            if shared == 0 and separation > radius:
                continue
            choices.append(((shared, -separation, branch.reference.area), branch))
        if not choices:
            continue
        branch = max(choices, key=lambda value: value[0])[1]
        mask = component_mask(extra, labels.shape[1:])
        changed = int(np.count_nonzero(
            labels[frame][mask] != branch.canonical_identity))
        labels[frame][mask] = branch.canonical_identity
        used.add(key)
        actions.append({
            "event_id": event_id,
            "action": "quarantine_owner_alias_extra",
            "output_frame": frame + 1,
            "canonical_identity": lock.canonical_identity,
            "observed_identity": extra.observed_identity,
            "assigned_identity": branch.canonical_identity,
            "changed_pixels": changed,
            "area_px": extra.area,
            "x": extra.x,
            "y": extra.y,
        })


def discover_displaced_extras(labels: np.ndarray, frame: int,
                              lock: OwnershipLock,
                              candidates: list[Component], used: set[int],
                              cfg: dict, event_id: str,
                              actions: list[dict]) -> None:
    """Restore newly appearing owner-name pieces to their previous local branch."""
    current_items = frame_components(labels[frame - 1], frame - 1,
                                     int(cfg["minimum_component_area_px"]))
    excluded = [lock.reference]
    excluded.extend(branch.reference for branch in lock.companions
                    if branch.reference.frame == frame - 1)
    alternatives = []
    for item in current_items:
        if any(overlap_pixels(item, other) > 0 for other in excluded):
            continue
        alternatives.append(item)
    for extra in candidates:
        key = int(extra.pixels[0])
        if key in used:
            continue
        if extra.observed_identity not in lock.owner_aliases | {
                lock.canonical_identity}:
            continue
        previous = best_match(alternatives, extra,
                              float(cfg["branch_match_radius_px"]))
        if (previous is None
                or previous.observed_identity in lock.owner_aliases | {
                    lock.canonical_identity}):
            continue
        mask = component_mask(extra, labels.shape[1:])
        changed = int(np.count_nonzero(
            labels[frame][mask] != previous.observed_identity))
        labels[frame][mask] = previous.observed_identity
        used.add(key)
        branch = CompanionBranch(
            int(previous.observed_identity),
            comp_from_mask(mask, frame, int(previous.observed_identity)), frame)
        lock.companions.append(branch)
        actions.append({
            "event_id": event_id, "action": "discover_displaced_branch",
            "output_frame": frame + 1,
            "canonical_identity": lock.canonical_identity,
            "observed_identity": extra.observed_identity,
            "assigned_identity": previous.observed_identity,
            "changed_pixels": changed, "area_px": extra.area,
            "x": extra.x, "y": extra.y,
        })


def relabel_backward_chain(labels: np.ndarray, chain: list[Component],
                           canonical: int, event_id: str,
                           actions: list[dict]) -> None:
    for item in chain:
        mask = component_mask(item, labels.shape[1:])
        changed = int(np.count_nonzero(labels[item.frame][mask] != canonical))
        if not changed:
            continue
        observed = int(np.bincount(labels[item.frame][mask].astype(int)).argmax())
        labels[item.frame][mask] = canonical
        actions.append({
            "event_id": event_id, "action": "earliest_identity_backpropagation",
            "output_frame": item.frame + 1,
            "canonical_identity": canonical,
            "observed_identity": observed, "assigned_identity": canonical,
            "changed_pixels": changed, "area_px": item.area,
            "x": item.x, "y": item.y,
        })


def persistent_quarantine(labels: np.ndarray, lag: np.ndarray, cfg: dict,
                          activation: dict) -> tuple[np.ndarray, pd.DataFrame,
                                                     pd.DataFrame]:
    candidate = labels.copy()
    locks: list[OwnershipLock] = []
    actions: list[dict] = []
    events: list[dict] = []
    next_lock = 1
    shape = candidate.shape[1:]
    threshold = float(cfg["motion_threshold"])
    # Geometry is immutable because quarantine changes identities only.  Caching
    # component instances makes whole-lineage backward tracing inexpensive.
    history_frames = [frame_components(
        labels[frame], frame, int(cfg["minimum_component_area_px"]))
        for frame in range(len(labels))]
    current_frames = [list(items) for items in history_frames]
    for frame in range(len(candidate) - 1):
        transition = lag[frame] if frame < len(lag) else np.zeros(shape)
        next_items = current_frames[frame + 1]
        used: set[int] = set()

        # Existing ownership locks never expire. Missing frames preserve the state;
        # they do not return ownership to the upstream labels.
        for lock in sorted(locks, key=lambda item: -item.reference.area):
            red = motion_destination(lock.reference, next_items, transition,
                                     cfg, shape)
            match = red or best_match(
                next_items, lock.reference,
                float(cfg["owner_match_radius_px"]), used)
            if match is None:
                match = best_match(
                    next_items, lock.reference,
                    float(cfg["dormant_reacquisition_radius_px"]), used)
            if match is None:
                lock.missing_frames += 1
                continue
            key = int(match.pixels[0])
            used.add(key)
            event_id = f"P{lock.number:04d}"
            propagate_companions(candidate, frame + 1, lock.companions,
                                 next_items, used, lock.owner_aliases,
                                 transition, cfg, event_id, actions)
            reassign_owner_alias_extras(
                candidate, frame + 1, lock, next_items, used, cfg,
                event_id, actions)
            lock.reference = assign_owner(
                candidate, frame + 1, lock.reference, match,
                lock.canonical_identity, lock.owner_aliases, cfg, event_id,
                "propagate_motion_destination" if red is not None
                else "propagate_permanent_tenure", actions)
            lock.last_seen_frame = frame + 1
            lock.missing_frames = 0

        current_frames[frame + 1] = frame_components(
            candidate[frame + 1], frame + 1,
            int(cfg["minimum_component_area_px"]))
        current_items = current_frames[frame]
        next_items = frame_components(candidate[frame + 1], frame + 1,
                                      int(cfg["minimum_component_area_px"]))
        for owner in sorted(current_items, key=lambda item: -item.area):
            if owner.area < int(cfg["large_area_px"]):
                continue
            if any(lock.reference.frame == frame
                   and overlap_pixels(lock.reference, owner) > 0 for lock in locks):
                continue
            destination = best_match(next_items, owner,
                                     float(cfg["owner_match_radius_px"]), used)
            if destination is None or destination.observed_identity == owner.observed_identity:
                continue
            chain = physical_chain_backward(history_frames, frame, owner, cfg)
            if not eligible_large_static(chain, cfg):
                continue
            if isolated_edge_gap(owner, current_items) < float(
                    cfg["minimum_isolated_edge_gap_px"]):
                continue
            canonical = int(chain[-1].observed_identity)
            if destination.observed_identity == canonical:
                continue
            sticky, small_frame, small_area = prior_local_small(
                history_frames, int(destination.observed_identity), destination,
                frame, cfg)
            if not sticky:
                continue
            metrics = transition_metrics(owner, destination, transition,
                                         threshold, shape)
            passed = bool(
                metrics["previous_area_overlap_fraction"]
                >= float(activation["minimum_previous_overlap_fraction"])
                and metrics["intersection_over_union"]
                >= float(activation["minimum_intersection_over_union"])
                and metrics["centroid_shift_px"]
                <= float(activation["maximum_centroid_shift_px"])
                and metrics["blue_departure_coverage"]
                < float(activation["maximum_blue_departure_coverage"]))
            event_id = f"P{next_lock:04d}"
            events.append({
                "event_id": event_id,
                "source_output_frame": frame + 1,
                "destination_output_frame": frame + 2,
                "owner_observed_identity": owner.observed_identity,
                "canonical_earliest_identity": canonical,
                "challenger_identity": destination.observed_identity,
                "earliest_lineage_output_frame": chain[-1].frame + 1,
                "earliest_lineage_area_px": chain[-1].area,
                "challenger_first_small_output_frame": (
                    small_frame + 1 if small_frame is not None else None),
                "challenger_first_small_area_px": small_area,
                "owner_tenure_frames": len(chain),
                **metrics, "activated": passed,
            })
            if not passed:
                continue
            relabel_backward_chain(candidate, chain, canonical, event_id, actions)
            aliases = {int(item.observed_identity) for item in chain}
            companions = restore_displaced_branches(
                candidate, frame + 1, owner, destination, canonical, aliases,
                cfg, event_id, actions)
            mask = component_mask(destination, shape)
            observed = int(np.bincount(
                candidate[frame + 1][mask].astype(int)).argmax())
            changed = int(np.count_nonzero(
                candidate[frame + 1][mask] != canonical))
            candidate[frame + 1][mask] = canonical
            actions.append({
                "event_id": event_id,
                "action": "activate_permanent_small_quarantine",
                "output_frame": frame + 2,
                "canonical_identity": canonical,
                "observed_identity": observed,
                "assigned_identity": canonical,
                "changed_pixels": changed, "area_px": destination.area,
                "x": destination.x, "y": destination.y,
            })
            reference = comp_from_mask(mask, frame + 1, canonical)
            locks.append(OwnershipLock(
                next_lock, canonical, reference, chain[-1].frame,
                chain[-1].area, frame + 1, frame + 1,
                aliases | {canonical}, companions=companions))
            used.add(int(destination.pixels[0]))
            next_lock += 1
            current_frames[frame + 1] = frame_components(
                candidate[frame + 1], frame + 1,
                int(cfg["minimum_component_area_px"]))
    return candidate, pd.DataFrame(events), pd.DataFrame(actions)

