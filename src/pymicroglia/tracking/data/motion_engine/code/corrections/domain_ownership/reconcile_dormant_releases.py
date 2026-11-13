"""Restore established names on complete physical component trajectories."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402
from detect_dormant_releases import components, largest  # noqa: E402


def select_events(events: pd.DataFrame, candidates: pd.DataFrame,
                  cfg: dict) -> pd.DataFrame:
    enabled = cfg.get("enabled_event_types",
                      ["weak_name_takeover", "true_dormancy"])
    selected: list[dict] = []
    for event in events.itertuples(index=False):
        if event.event_type not in enabled:
            continue
        if event.event_type == "weak_name_takeover":
            row = event._asdict()
            row.update({
                "release_source_imagej_frame": int(event.start_source_imagej_frame),
                "candidate_identity": int(event.host_identity),
                "selection_score": 1.0,
            })
            selected.append(row)
            continue
        # Permanent quarantine is based on the identity's earliest established
        # history, not on a large territory it may acquire much later.
        if (float(event.established_area_px) < float(cfg.get(
                "minimum_dormant_established_area_px", 80))
                or int(event.identity_large_history_frames) < int(cfg.get(
                    "minimum_dormant_history_frames", 3))):
            continue
        if candidates.empty:    # no release candidate anywhere: nothing to restore
            continue
        group = candidates[candidates.event_id == event.event_id].copy()
        if group.empty:
            continue
        group = group[
            (group.host_loss_to_established_ratio >= float(
                cfg["minimum_host_loss_ratio"]))
            & (group.distance_from_host_px <= float(cfg["release_radius_px"]))
            & (group.red_arrival_fraction >= float(
                cfg["minimum_red_arrival_fraction"]))
            & (group.persistence_frames >= int(cfg.get(
                "minimum_persistence_frames", 2)))
            & (group.mass_match >= float(cfg.get("minimum_mass_match", 0.5)))
            & (group.prior_local_to_established_ratio <= float(cfg.get(
                "maximum_prior_local_area_ratio", 0.5)))
        ].copy()
        if group.empty:
            continue
        group["selection_score"] = (
            1.5 * group.mass_match + group.red_arrival_fraction
            + 0.25 * np.minimum(group.persistence_frames, 3)
            - 0.005 * group.distance_from_host_px
            - 0.25 * group.prior_local_to_established_ratio)
        # A reservation exits at the first causally complete split. Waiting for
        # a later, numerically neater object would let the same identity remain
        # trapped in a host after its visible body has already departed.
        first_release = int(group.release_source_imagej_frame.min())
        best = group[group.release_source_imagej_frame == first_release].sort_values(
            "selection_score", ascending=False).iloc[0].to_dict()
        best.update({key: getattr(event, key) for key in event._fields
                     if key not in best})
        selected.append(best)
    return pd.DataFrame(selected)


def _all_components(frame: np.ndarray, minimum_area: int) -> list[dict]:
    result = []
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        for item in components(frame, identity, minimum_area):
            item = dict(item)
            item["identity"] = identity
            result.append(item)
    return result


def _frame_key(frame) -> tuple:
    return (frame.__array_interface__["data"][0], frame.shape, frame.strides,
            frame.dtype.str)


class _OutlineCache:
    """Components of the input labels, found once per reconcile call.

    reconcile reads labels and writes a copy, so each frame's components serve
    every track. They are labelled inside the identity's box, which numbers
    them in the same raster order, and kept as crops; a whole-frame mask is
    drawn only for a component whose mask is read. The byte budgets limit
    execution memory, never which components are found.
    """

    def __init__(self, labels: np.ndarray, budget: int = 32 * 1024 ** 2):
        self.labels = labels  # Holding labels keeps the pointer keys alive.
        self.allowed = {_frame_key(frame) for frame in labels}
        self.cache: OrderedDict = OrderedDict()
        self.bounds: OrderedDict = OrderedDict()
        self.budget, self.retained = budget, 0
        self.listed: OrderedDict = OrderedDict()
        self.listed_budget, self.listed_retained = 64 * 1024 ** 2, 0

    def geometry(self, frame: np.ndarray, identity: int) -> list[dict]:
        frame_key = _frame_key(frame)
        cache_key = (frame_key, int(identity))
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key][0]
        region = self.frame_bounds(frame, frame_key).get(int(identity))
        parts = []
        if region is not None:
            numbered, _ = ndi.label(frame[region] == identity,
                                    structure=np.ones((3, 3), bool))
            for number, local_region in enumerate(ndi.find_objects(numbered), 1):
                if local_region is None:
                    continue
                mask = numbered[local_region] == number
                points = np.argwhere(mask)
                y = region[0].start + local_region[0].start
                x = region[1].start + local_region[1].start
                points += np.array([y, x])
                centre = points.mean(axis=0)  # The same ordered integer sums.
                mask.flags.writeable = False
                parts.append({"component": number, "area": len(points),
                              "y": float(centre[0]), "x": float(centre[1]),
                              "region": (slice(y, y + mask.shape[0]),
                                         slice(x, x + mask.shape[1])),
                              "crop": mask})
        cost = 512 + sum(1536 + part["crop"].nbytes for part in parts)
        if cost <= self.budget:
            while self.cache and self.retained + cost > self.budget:
                _, (_, old_cost) = self.cache.popitem(last=False)
                self.retained -= old_cost
            self.cache[cache_key] = (parts, cost)
            self.retained += cost
        return parts

    def frame_bounds(self, frame: np.ndarray, frame_key: tuple) -> dict:
        if frame_key not in self.bounds:
            plain = (frame.view(np.ndarray) if isinstance(frame, np.ndarray)
                     else np.asarray(frame))
            top = (int(plain.max()) if plain.size and plain.dtype.kind in "iu"
                   and (plain.dtype.kind == "u" or int(plain.min()) >= 0) else -1)
            if 0 <= top <= plain.size:
                # Non-negative labels: find_objects gives each present value's
                # box (None for absent ones), and the background box comes
                # from one comparison; the same keys and boxes as below.
                found = {value: box for value, box in
                         enumerate(ndi.find_objects(plain), 1) if box is not None}
                background = plain == 0
                rows = np.flatnonzero(background.any(axis=1))
                if rows.size:
                    columns = np.flatnonzero(background.any(axis=0))
                    found[0] = (slice(int(rows[0]), int(rows[-1]) + 1),
                                slice(int(columns[0]), int(columns[-1]) + 1))
                self.bounds[frame_key] = found
            else:
                # Compact inverse labels avoid arrays indexed by arbitrary identity.
                values, inverse = np.unique(frame, return_inverse=True)
                numbered = inverse.reshape(frame.shape) + 1
                self.bounds[frame_key] = dict(zip(map(int, values),
                                                  ndi.find_objects(numbered)))
            if len(self.bounds) > 8:
                self.bounds.popitem(last=False)
        self.bounds.move_to_end(frame_key)
        return self.bounds[frame_key]

    def pending_components(self, frame: np.ndarray, minimum_area: int) -> list[dict]:
        """_all_components with each mask left to ready(): the same identities
        (the frame's sorted nonzero values), components and order."""
        frame_key = _frame_key(frame)
        if frame_key not in self.allowed:
            return _all_components(frame, minimum_area)
        listed = self.listed.get((frame_key, minimum_area))
        if listed is not None:
            # Paths revisit frames; the step only reads these entries.
            self.listed.move_to_end((frame_key, minimum_area))
            return list(listed[0])
        found = []
        for identity in sorted(set(self.frame_bounds(frame, frame_key)) - {0}):
            for part in self.geometry(frame, identity):
                if part["area"] >= minimum_area:
                    found.append({"identity": identity, "area": part["area"],
                                  "y": part["y"], "x": part["x"],
                                  "_part": part, "_shape": frame.shape})
        cost = 512 + sum(1024 + item["_part"]["crop"].nbytes for item in found)
        if cost <= self.listed_budget:
            while self.listed and self.listed_retained + cost > self.listed_budget:
                _, (_, old) = self.listed.popitem(last=False)
                self.listed_retained -= old
            self.listed[(frame_key, minimum_area)] = (tuple(found), cost)
            self.listed_retained += cost
        return list(found)

    def ready(self, item: dict) -> dict:
        if "_part" not in item:
            return item
        made = self.materialize(item["_part"], item["_shape"])
        made["identity"] = item["identity"]
        return made

    @staticmethod
    def materialize(part: dict, shape: tuple) -> dict:
        mask = np.zeros(shape, bool)
        mask[part["region"]] = part["crop"]
        return {"component": part["component"], "area": part["area"],
                "y": part["y"], "x": part["x"], "mask": mask}

    def components(self, frame: np.ndarray, identity: int,
                   minimum_area: int = 8) -> list[dict]:
        if _frame_key(frame) not in self.allowed:
            return components(frame, identity, minimum_area)
        return [self.materialize(part, frame.shape)
                for part in self.geometry(frame, identity)
                if part["area"] >= minimum_area]


def _component_cost(item: dict, predicted: np.ndarray, previous_mask: np.ndarray,
                    target_area: float, target_raw: float, raw_frame: np.ndarray,
                    radius: float) -> float:
    centre = np.array([item["y"], item["x"]])
    distance = float(np.linalg.norm(centre - predicted))
    if distance > radius:
        return math.inf
    area_cost = abs(math.log(max(item["area"], 1) / max(target_area, 1)))
    mean_raw = float(raw_frame[item["mask"]].mean())
    raw_cost = abs(math.log(max(mean_raw, 1) / max(target_raw, 1)))
    near = ndi.binary_dilation(previous_mask, iterations=4)
    near_fraction = float(np.count_nonzero(near & item["mask"]) /
                          max(min(previous_mask.sum(), item["area"]), 1))
    return (1.8 * area_cost + raw_cost + 0.045 * distance
            + 0.8 * (1.0 - min(near_fraction, 1.0)))


def absorbed_neighbours(labels: np.ndarray, index: int, candidate: dict,
                        tracked_previous: dict | None, cfg: dict,
                        cache: _OutlineCache | None = None) -> dict[int, dict]:
    """Established cells whose whole body has just been swallowed by a component.

    When the tracked body passes behind another cell the two share one blob. The
    blob is still claimed - the baseline had already merged them - but the cell
    that vanished into it is remembered, so that when the blob separates again
    the name cannot walk out on the other cell's body.
    """
    if index == 0:
        return {}
    found = components if cache is None else cache.components
    minimum_body = int(cfg.get("minimum_foreign_body_px", 80))
    fraction = float(cfg.get("minimum_foreign_absorbed_fraction", 0.5))
    vanished_fraction = float(cfg.get("minimum_vanished_overlap_fraction", 0.25))
    previous_frame, frame = labels[index - 1], labels[index]
    region = ndi.binary_dilation(candidate["mask"], iterations=2)
    absorbed: dict[int, dict] = {}
    for identity in sorted(set(map(int, np.unique(previous_frame[region]))) - {0}):
        for body in found(previous_frame, identity, minimum_body):
            if tracked_previous is not None and overlap_is_tracked(
                    body, tracked_previous):
                continue
            present = found(frame, identity, minimum_body)
            separate = [part for part in present
                        if not np.any(part["mask"] & candidate["mask"])]
            if separate:
                continue
            share = int(np.count_nonzero(
                body["mask"] & candidate["mask"])) / max(body["area"], 1)
            # A cell that disappears has not died, it has been absorbed, so a
            # smaller shared area is enough evidence than for one still visible.
            needed = vanished_fraction if not present else fraction
            if share < needed:
                continue
            absorbed[identity] = body
    return absorbed


def overlap_is_tracked(body: dict, tracked_previous: dict) -> bool:
    shared = int(np.count_nonzero(body["mask"] & tracked_previous["mask"]))
    return shared >= 0.5 * min(body["area"], tracked_previous["area"])


def owns_its_own_body(labels: np.ndarray, index: int, item: dict,
                      claimed: dict[int, np.ndarray], cfg: dict,
                      cache: _OutlineCache | None = None) -> bool:
    """True when a component is its own label's established body, not a free one.

    A name may move onto a component that has just appeared, or onto one the
    track itself has been carrying. It may not move onto a cell that has been
    sitting there under its own name frame after frame.
    """
    frames = int(cfg.get("prehistory_frames", 3))
    radius = float(cfg.get("prehistory_radius_px", 20))
    share = float(cfg.get("prehistory_minimum_area_ratio", 0.5))
    minimum_area = int(cfg.get("minimum_component_area_px", 8))
    identity = int(item["identity"])
    centre = np.array([item["y"], item["x"]])
    found = components if cache is None else cache.components
    for back in range(1, frames + 1):
        past = index - back
        if past < 0:
            return False
        established = False
        for part in found(labels[past], identity, minimum_area):
            if part["area"] < share * item["area"]:
                continue
            if float(np.linalg.norm(
                    np.array([part["y"], part["x"]]) - centre)) > radius:
                continue
            carried = claimed.get(past)
            if carried is not None and np.any(carried & part["mask"]):
                continue
            established = True
            break
        if not established:
            return False
    return True


def still_inside_refused(frame: np.ndarray, previous: dict,
                        merged_with: dict[int, dict], cfg: dict,
                        cache: _OutlineCache | None = None) -> bool:
    """True while the last known body sits inside a refused cell's component.

    The component has to be large enough to hold both cells at once. Once it has
    shrunk back to about one cell it is the other cell alone, standing where the
    two used to overlap, and the name is free to look elsewhere.
    """
    if not merged_with:
        return False
    minimum_area = int(cfg.get("minimum_component_area_px", 8))
    fraction = float(cfg.get("minimum_still_merged_fraction", 0.5))
    ratio = float(cfg.get("minimum_still_merged_area_ratio", 1.5))
    found = components if cache is None else cache.components
    for other in merged_with:
        for part in found(frame, other, minimum_area):
            if part["area"] < ratio * max(previous["area"], 1):
                continue
            shared = int(np.count_nonzero(part["mask"] & previous["mask"]))
            if shared >= fraction * max(previous["area"], 1):
                return True
    return False


def track_component(labels: np.ndarray, raw: np.ndarray, identity: int,
                    start_source: int, start_identity: int, offset: int,
                    target_area: float, target_raw: float, cfg: dict,
                    cache: _OutlineCache | None = None
                    ) -> list[tuple[int, dict]]:
    minimum_area = int(cfg.get("minimum_component_area_px", 8))
    start_index = start_source - offset - 1
    found = components if cache is None else cache.components
    starts = found(labels[start_index], start_identity, minimum_area)
    if not starts:
        return []
    if start_source > offset + 1:
        prior = largest(labels[start_index - 1], identity, minimum_area,
                        cache=cache)
    else:
        prior = None
    if prior is not None:
        start = max(starts, key=lambda item: int(np.count_nonzero(
            ndi.binary_dilation(prior["mask"], iterations=4) & item["mask"])))
    else:
        start = max(starts, key=lambda item: item["area"])
    start = dict(start)
    start["identity"] = start_identity
    result = [(start_source, start)]
    previous = start
    velocity = np.zeros(2, float)
    gap = 0
    maximum_gap = int(cfg.get("maximum_track_gap_frames", 6))
    radius = float(cfg.get("component_search_radius_px", 60))
    # Cells this body has been merged with at any point. Their labels stay
    # refused for the rest of the track, so a blob that separates never hands the
    # name to the cell it was sharing pixels with.
    merged_with: dict[int, dict] = {}
    claimed: dict[int, np.ndarray] = {start_index: start["mask"]}
    for source in range(start_source + 1, offset + len(labels) + 1):
        index = source - offset - 1
        if still_inside_refused(labels[index], previous, merged_with, cfg,
                                cache=cache):
            # The body is back inside a cell it may not be named after, so the
            # name waits rather than moving to whatever else is nearby.
            gap += 1
            if gap > maximum_gap:
                break
            continue
        predicted = np.array([previous["y"], previous["x"]]) + velocity
        scored = []
        listed = (_all_components(labels[index], minimum_area) if cache is None
                  else cache.pending_components(labels[index], minimum_area))
        for item in listed:
            if int(item["identity"]) in merged_with:
                continue
            ratio = item["area"] / max(target_area, 1.0)
            if not float(cfg.get("minimum_track_area_ratio", 0.25)) <= ratio <= float(cfg.get("maximum_track_area_ratio", 2.5)):
                continue
            # _component_cost rejects a component beyond the search radius;
            # rejecting it first skips the pure prehistory test.
            centre = np.array([item["y"], item["x"]])
            if float(np.linalg.norm(centre - predicted)) > radius:
                continue
            if cache is not None:
                item = cache.ready(item)
            if owns_its_own_body(labels, index, item, claimed, cfg, cache=cache):
                continue
            cost = _component_cost(item, predicted, previous["mask"],
                                   target_area, target_raw, raw[index], radius)
            if np.isfinite(cost):
                scored.append((cost, item))
        if not scored:
            gap += 1
            if gap > maximum_gap:
                break
            continue
        cost, chosen = min(scored, key=lambda pair: pair[0])
        if cost > float(cfg.get("maximum_component_cost", 5.0)):
            gap += 1
            if gap > maximum_gap:
                break
            continue
        merged_with.update(
            absorbed_neighbours(labels, index, chosen, previous, cfg, cache=cache))
        claimed[index] = chosen["mask"]
        old_centre = np.array([previous["y"], previous["x"]])
        new_centre = np.array([chosen["y"], chosen["x"]])
        step = new_centre - old_centre
        velocity = 0.65 * velocity + 0.35 * step
        previous = chosen
        gap = 0
        result.append((source, chosen))
    return result


def reconcile(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
              selected: pd.DataFrame, offset: int, cfg: dict
              ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    result = labels.copy()
    result_unclaimed = unclaimed.copy()
    actions: list[dict] = []
    cache = _OutlineCache(labels)
    # Earlier established names win if independently selected tracks collide.
    ordered = selected.sort_values(
        ["start_source_imagej_frame", "established_area_px"],
        ascending=[True, False]) if len(selected) else selected
    for event in ordered.itertuples(index=False):
        identity = int(event.identity)
        start = int(event.release_source_imagej_frame)
        carrier = int(event.candidate_identity)
        track = track_component(
            labels, raw, identity, start, carrier, offset,
            float(event.established_area_px),
            float(event.established_raw_mean), cfg, cache=cache)
        for source, item in track:
            index = source - offset - 1
            mask = item["mask"]
            before = result[index][mask].copy()
            result[index][mask] = identity
            result_unclaimed[index][mask] = 0
            changed = int(np.count_nonzero(before != identity))
            actions.append({
                "event_id": event.event_id, "event_type": event.event_type,
                "source_imagej_frame": source,
                "action": "physical_component_name_restored",
                "from_identity": int(item["identity"]),
                "to_identity": identity,
                "host_identity": int(event.host_identity),
                "changed_pixels": changed,
                "component_area_px": int(item["area"]),
                "component_x": float(item["x"]),
                "component_y": float(item["y"]),
            })
            # A dim residual cannot keep the same established name elsewhere.
            numbered, count = ndi.label(result[index] == identity,
                                        structure=np.ones((3, 3), bool))
            keep_number = int(np.bincount(numbered[mask].ravel())[1:].argmax() + 1)
            for number in range(1, count + 1):
                if number == keep_number:
                    continue
                residual = numbered == number
                area = int(residual.sum())
                if area < int(cfg.get("maximum_residual_area_px", 80)):
                    result[index][residual] = 0
                    result_unclaimed[index][residual] = identity
                    actions.append({
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "source_imagej_frame": source,
                        "action": "duplicate_residual_made_unclaimed",
                        "from_identity": identity, "to_identity": 0,
                        "host_identity": int(event.host_identity),
                        "changed_pixels": area, "component_area_px": area,
                        "component_x": np.nan, "component_y": np.nan,
                    })
    if not np.array_equal((result > 0) | (result_unclaimed > 0),
                          (labels > 0) | (unclaimed > 0)):
        raise AssertionError("foreground accounting changed")
    if np.any((result > 0) & (result_unclaimed > 0)):
        raise AssertionError("assigned and unclaimed foreground overlap")
    return result, result_unclaimed, pd.DataFrame(actions)

