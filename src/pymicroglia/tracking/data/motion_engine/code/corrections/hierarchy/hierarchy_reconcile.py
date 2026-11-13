"""Reconcile every frame with movie-wide authority and three optional methods."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root
import lineage_hierarchy as geometry
import movie_authority as authority_stage


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402


@dataclass
class Authority:
    identity: int
    layer: str
    rank: int
    birth_frame: int
    birth_area: int
    birth_small: bool


def authority_map(table: pd.DataFrame) -> dict[int, Authority]:
    return {
        int(row.identity): Authority(
            int(row.identity), str(row.confidence_layer),
            int(row.priority_rank), int(row.birth_output_frame) - 1,
            int(row.birth_area_px), bool(row.permanently_birth_small))
        for row in table.itertuples(index=False)
    }


def _boxes_disjoint(first, second, extents):
    """True when the two components' pixel boxes cannot meet.

    With the frame width recorded under 'width', flat indices give each
    component's row and column range; boxes that do not intersect share no
    pixel. Without it, the flat index ranges are compared.
    """
    width = extents.get('width')
    ranges = []
    for component in (first, second):
        key = id(component)
        if key not in extents or extents[key][0] is not component:
            pixels = np.asarray(component.pixels)
            if not pixels.size:
                box = None
            elif width:
                rows, columns = pixels // width, pixels % width
                box = (int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max()))
            else:
                box = (int(pixels.min()), int(pixels.max()))
            extents[key] = (component, box)
        ranges.append(extents[key][1])
    a, b = ranges
    if a is None or b is None or a[1] < b[0] or b[1] < a[0]:
        return True
    return len(a) == 4 and (a[3] < b[2] or b[3] < a[2])


class _PairMemo:
    """One run's component pixel extents (within a byte budget) and per-frame motion fractions.

    Overlaps of components whose flat pixel indices are strictly increasing (as
    frame components list them) are counted by searching one sorted list in the
    other over their shared range: the size of the same intersection that
    np.intersect1d(..., assume_unique=True) forms by sorting both together.
    Components are held with what is remembered about them, so an id cannot be
    reused while its entry lives.
    """

    BUDGET = 32 * 1024 ** 2

    def __init__(self) -> None:
        self.bounds: OrderedDict = OrderedDict()
        self.retained = 0
        self.fractions: dict = {}
        self.frame_key = None

    def extent(self, component: geometry.Component):
        identity = id(component)
        if identity in self.bounds:
            self.bounds.move_to_end(identity)
            return self.bounds[identity][1]
        pixels = component.pixels
        value = ((int(pixels.min()), int(pixels.max()),
                  bool(pixels.ndim == 1 and pixels.dtype.kind in 'iu'
                       and (len(pixels) < 2 or bool(np.all(pixels[1:] > pixels[:-1])))))
                 if len(pixels) else None)
        cost = pixels.nbytes + 512
        if cost <= self.BUDGET:
            while self.bounds and self.retained + cost > self.BUDGET:
                _, (_, _, old_cost) = self.bounds.popitem(last=False)
                self.retained -= old_cost
            self.bounds[identity] = (component, value, cost)
            self.retained += cost
        return value

    def overlap(self, first: geometry.Component, second: geometry.Component) -> int:
        a, b = self.extent(first), self.extent(second)
        if a is None or b is None or a[1] < b[0] or b[1] < a[0]:
            return 0
        if a[2] and b[2] and first.pixels.dtype == second.pixels.dtype:
            low, high = max(a[0], b[0]), min(a[1], b[1])
            small, large = ((first.pixels, second.pixels) if len(first.pixels) <= len(second.pixels)
                            else (second.pixels, first.pixels))
            probe = small[np.searchsorted(small, low, 'left'):np.searchsorted(small, high, 'right')]
            if not len(probe):
                return 0
            return int(np.count_nonzero(large[np.searchsorted(large, probe)] == probe))
        return geometry.overlap(first, second)

    def coverage(self, component: geometry.Component, lag: np.ndarray,
                 threshold: float) -> tuple[float, float]:
        identity = id(component)
        if identity not in self.fractions:
            sampled = lag.ravel()[component.pixels]
            self.fractions[identity] = (component, float(np.mean(sampled <= -threshold)),
                                        float(np.mean(sampled >= threshold)))
        return self.fractions[identity][1:]

    def metrics(self, source: geometry.Component, destination: geometry.Component,
                lag: np.ndarray, threshold: float) -> dict:
        key = (lag.__array_interface__['data'][0], lag.shape, lag.strides, lag.dtype.str, threshold)
        if key != self.frame_key:
            self.fractions.clear()
            self.frame_key = key
        shared = self.overlap(source, destination)
        union = source.area + destination.area - shared
        return dict(overlap_px=shared,
                    previous_overlap_fraction=shared / max(source.area, 1),
                    destination_from_source_fraction=shared / max(destination.area, 1),
                    intersection_over_union=shared / max(union, 1),
                    centroid_shift_px=geometry.distance(source, destination),
                    source_blue_coverage=self.coverage(source, lag, threshold)[0],
                    destination_red_coverage=self.coverage(destination, lag, threshold)[1],
                    area_ratio=destination.area / max(source.area, 1))


def metrics(source: geometry.Component, destination: geometry.Component,
            lag: np.ndarray, threshold: float,
            shape: tuple[int, int], memo: _PairMemo | None = None) -> dict:
    if memo is not None:
        return memo.metrics(source, destination, lag, threshold)
    del shape
    shared = geometry.overlap(source, destination)
    union = source.area + destination.area - shared
    flat_lag = lag.ravel()
    return {
        "overlap_px": shared,
        "previous_overlap_fraction": shared / max(source.area, 1),
        "destination_from_source_fraction": shared / max(destination.area, 1),
        "intersection_over_union": shared / max(union, 1),
        "centroid_shift_px": geometry.distance(source, destination),
        "source_blue_coverage": float(np.mean(
            flat_lag[source.pixels] <= -threshold)),
        "destination_red_coverage": float(np.mean(
            flat_lag[destination.pixels] >= threshold)),
        "area_ratio": destination.area / max(source.area, 1),
    }


def frame_components(frame: np.ndarray, number: int,
                     minimum: int) -> list[geometry.Component]:
    return geometry.frame_components(frame, number, minimum)


def assign(frame: np.ndarray, component: geometry.Component,
           identity: int, shape: tuple[int, int]) -> int:
    mask = geometry.component_mask(component, shape)
    changed = int(np.count_nonzero(frame[mask] != identity))
    frame[mask] = identity
    return changed


def strong_pass(values: dict, cfg: dict) -> bool:
    return bool(
        values["previous_overlap_fraction"] >= float(
            cfg["minimum_strong_previous_overlap_fraction"])
        and values["intersection_over_union"] >= float(
            cfg["minimum_strong_intersection_over_union"])
        and values["centroid_shift_px"] <= float(
            cfg["maximum_strong_centroid_shift_px"])
        and values["source_blue_coverage"] <= float(
            cfg["maximum_strong_blue_departure"]))


def partition_by_previous(host: geometry.Component,
                          owners: list[geometry.Component],
                          shape: tuple[int, int], minimum: int
                          ) -> list[tuple[geometry.Component, np.ndarray]] | None:
    host_mask = geometry.component_mask(host, shape)
    distances = []
    for owner in owners:
        owner_mask = geometry.component_mask(owner, shape)
        distances.append(ndi.distance_transform_edt(~owner_mask))
    winner = np.argmin(np.stack(distances), axis=0)
    partitions = [(owner, host_mask & (winner == index))
                  for index, owner in enumerate(owners)]
    if any(int(mask.sum()) < minimum for _, mask in partitions):
        return None
    return partitions


def apply_ordered(candidate: np.ndarray, frame: int,
                  previous: list[geometry.Component],
                  current: list[geometry.Component], lag: np.ndarray,
                  authorities: dict[int, Authority], cfg: dict,
                  threshold: float, aliases: set[tuple[int, int]],
                  actions: list[dict], memo: _PairMemo | None = None
                  ) -> tuple[int, int]:
    overlap = geometry.overlap if memo is None else memo.overlap
    shape = candidate.shape[1:]
    _extents = {'width': int(shape[1]) if len(shape) == 2 else None}
    changed = 0
    partitions = 0
    minimum_owner = int(cfg.get("multiowner_minimum_owner_area_px", 48))
    for host in list(current):
        qualified = []
        for owner in previous:
            if owner.area < minimum_owner:
                continue
            # Disjoint pixel ranges share no pixel, so the previous-overlap
            # fraction is 0.0 and cannot reach a positive threshold.
            if (_boxes_disjoint(owner, host, _extents)
                    and float(cfg["multiowner_minimum_previous_overlap_fraction"]) > 0):
                continue
            values = metrics(owner, host, lag, threshold, shape, memo)
            if (owner.area >= minimum_owner
                    and values["previous_overlap_fraction"] >= float(
                        cfg["multiowner_minimum_previous_overlap_fraction"])
                    and values["source_blue_coverage"] <= float(
                        cfg["multiowner_maximum_blue_departure"])):
                qualified.append((owner, values))
        best_by_identity = {}
        for owner, values in qualified:
            prior = best_by_identity.get(owner.identity)
            if prior is None or values["overlap_px"] > prior[1]["overlap_px"]:
                best_by_identity[owner.identity] = (owner, values)
        owners = [row[0] for row in best_by_identity.values()]
        if len(owners) < 2:
            continue
        if (bool(cfg.get("multiowner_requires_established_alias", False))
                and not any((owner.identity, host.identity) in aliases
                            for owner in owners)):
            continue
        host_rank = authorities[host.identity].rank
        if not any(authorities[owner.identity].rank < host_rank
                   for owner in owners):
            continue
        divided = partition_by_previous(
            host, owners, shape, int(cfg["multiowner_minimum_partition_px"]))
        if divided is None:
            continue
        for owner, mask in divided:
            altered = int(np.count_nonzero(
                candidate[frame + 1][mask] != owner.identity))
            candidate[frame + 1][mask] = owner.identity
            changed += altered
            actions.append({
                "method": "O", "action": "multiowner_partition",
                "source_output_frame": frame + 1,
                "destination_output_frame": frame + 2,
                "owner_identity": owner.identity,
                "observed_identity": host.identity,
                "owner_layer": authorities[owner.identity].layer,
                "changed_pixels": altered,
                "host_area_px": host.area,
                "partition_area_px": int(mask.sum()),
            })
        partitions += 1

    current = frame_components(candidate[frame + 1], frame + 1,
                               int(cfg["minimum_owner_area_px"]))
    bids = []
    for owner in previous:
        owner_auth = authorities[owner.identity]
        for destination in current:
            # These exact authority requirements already veto every new
            # transfer below. Same-name and established-alias cases retain
            # their original evaluation, as do missing authority entries.
            if (destination.identity != owner.identity
                    and (owner.identity, destination.identity) not in aliases
                    and destination.identity in authorities):
                destination_auth = authorities[destination.identity]
                if owner_auth.rank >= destination_auth.rank or not destination_auth.birth_small:
                    continue
            # A different identity sharing no pixel has zero overlap fractions:
            # not a same-name continuation, not strong when either strong
            # overlap minimum is positive, and no alias continuation when both
            # alias overlap minima are positive.
            if (destination.identity != owner.identity
                    and _boxes_disjoint(owner, destination, _extents)
                    and (float(cfg["minimum_strong_previous_overlap_fraction"]) > 0
                         or float(cfg["minimum_strong_intersection_over_union"]) > 0)
                    and float(cfg.get("minimum_motion_alias_overlap_fraction", 0.10)) > 0
                    and float(cfg.get("minimum_alias_trace_destination_fraction", 0.80)) > 0):
                continue
            values = metrics(owner, destination, lag, threshold, shape, memo)
            same = destination.identity == owner.identity
            same_continuation = bool(same and (
                values["overlap_px"] > 0
                or (values["centroid_shift_px"] <= 12.0
                    and 0.2 <= values["area_ratio"] <= 5.0)))
            strong = strong_pass(values, cfg)
            alias = (owner.identity, destination.identity) in aliases
            motion_alias = bool(
                values["previous_overlap_fraction"] >= float(cfg.get(
                    "minimum_motion_alias_overlap_fraction", 0.10))
                and values["source_blue_coverage"] >= float(cfg.get(
                    "minimum_motion_alias_blue_departure", 0.30))
                and values["destination_red_coverage"] >= float(cfg.get(
                    "minimum_motion_alias_red_arrival", 0.45)))
            disappearing_trace = bool(
                values["destination_from_source_fraction"] >= float(cfg.get(
                    "minimum_alias_trace_destination_fraction", 0.80))
                and values["source_blue_coverage"] >= float(cfg.get(
                    "minimum_alias_trace_blue_departure", 0.70)))
            alias_continuation = bool(
                alias and (values["previous_overlap_fraction"] >= 0.35
                           or motion_alias or disappearing_trace)
                and values["centroid_shift_px"] <= 20.0
                and 0.1 <= values["area_ratio"] <= 10.0)
            if not same_continuation and not strong and not alias_continuation:
                continue
            if not same and not alias:
                destination_auth = authorities[destination.identity]
                prior_observed_fraction = max((
                    overlap(item, destination)
                    / max(destination.area, 1)
                    for item in previous
                    if item.identity == destination.identity), default=0.0)
                if (owner_auth.rank >= destination_auth.rank
                        or not destination_auth.birth_small
                        or prior_observed_fraction > float(cfg.get(
                            "maximum_prior_observed_destination_fraction",
                            0.10))):
                    continue
            evidence = (values["previous_overlap_fraction"]
                        + values["intersection_over_union"])
            motion_priority = 0 if alias and motion_alias else 1
            bids.append((owner_auth.rank, motion_priority, -evidence,
                         owner_auth.birth_frame, owner, destination,
                         values, strong, alias_continuation, motion_alias))
    used_owners = set()
    used_destinations = set()
    for (_, _, _, _, owner, destination, values, strong,
         alias_continuation, motion_alias) in sorted(
            bids, key=lambda row: (
                row[0], row[1], row[2], row[3],
                row[4].identity, row[4].number,
                row[5].identity, row[5].number)):
        owner_key = (owner.identity, owner.number)
        destination_key = (destination.identity, destination.number)
        if owner_key in used_owners or destination_key in used_destinations:
            continue
        used_owners.add(owner_key)
        used_destinations.add(destination_key)
        if (destination.identity == owner.identity
                or (not strong and not alias_continuation)):
            continue
        altered = assign(candidate[frame + 1], destination,
                         owner.identity, shape)
        changed += altered
        aliases.add((owner.identity, destination.identity))
        displaced_fragments = []
        destination_mask = geometry.component_mask(destination, shape)
        destination_distance = ndi.distance_transform_edt(~destination_mask)
        for item in current:
            item_key = (item.identity, item.number)
            if (item.identity != destination.identity
                    or item_key == destination_key
                    or item_key in used_destinations
                    or item.area > int(cfg.get(
                        "maximum_displaced_fragment_area_px", 48))):
                continue
            edge_gap = float(np.min(destination_distance.ravel()[item.pixels]))
            if edge_gap > float(cfg.get(
                    "maximum_displaced_fragment_gap_px", 3.0)):
                continue
            changed += assign(candidate[frame + 1], item,
                              owner.identity, shape)
            displaced_fragments.append(item)
            used_destinations.add(item_key)
        companion = None
        if motion_alias:
            companion_choices = []
            for item in current:
                item_key = (item.identity, item.number)
                if (item.identity != owner.identity
                        or item_key in used_destinations):
                    continue
                item_values = metrics(owner, item, lag, threshold, shape, memo)
                if (item_values["overlap_px"] <= 0
                        or item_values["destination_red_coverage"] > float(
                            cfg.get("maximum_motion_alias_companion_red", 0.20))
                        or item_values["area_ratio"] > float(cfg.get(
                            "maximum_motion_alias_companion_area_ratio", 0.75))
                        or (values["destination_red_coverage"]
                            - item_values["destination_red_coverage"]
                            < float(cfg.get(
                                "minimum_motion_alias_red_advantage", 0.25)))):
                    continue
                companion_choices.append((
                    item_values["overlap_px"], item, item_values))
            if companion_choices:
                _, companion, _ = max(
                    companion_choices, key=lambda row: row[0])
                changed += assign(candidate[frame + 1], companion,
                                  destination.identity, shape)
                aliases.add((destination.identity, owner.identity))
                used_destinations.add((companion.identity, companion.number))
        actions.append({
            "method": "O", "action": "strong_territory_reservation",
            "source_output_frame": frame + 1,
            "destination_output_frame": frame + 2,
            "owner_identity": owner.identity,
            "observed_identity": destination.identity,
            "owner_layer": authorities[owner.identity].layer,
            "changed_pixels": altered, **values,
            "motion_alias_priority": motion_alias,
            "alias_companion_swapped": companion is not None,
            "alias_companion_area_px": companion.area if companion else 0,
            "displaced_fragments_absorbed": len(displaced_fragments),
            "displaced_fragment_pixels": int(sum(
                item.area for item in displaced_fragments)),
        })
    return changed, partitions


def apply_partial(candidate: np.ndarray, frame: int,
                  previous: list[geometry.Component], lag: np.ndarray,
                  authorities: dict[int, Authority], cfg: dict,
                  threshold: float, minimum: int,
                  actions: list[dict], memo: _PairMemo | None = None) -> int:
    overlap = geometry.overlap if memo is None else memo.overlap
    shape = candidate.shape[1:]
    changed = 0
    _main = [None, None, 0.0]
    _prior = {}
    current = frame_components(candidate[frame + 1], frame + 1, minimum)
    for owner in sorted(previous, key=lambda item: (
            authorities[item.identity].rank, -item.area)):
        owner_auth = authorities[owner.identity]
        for destination in list(current):
            if destination.identity == owner.identity:
                continue
            destination_auth = authorities[destination.identity]
            if owner_auth.rank >= destination_auth.rank:
                continue
            if (owner.area < int(cfg.get("minimum_owner_area_px", 80))
                    or destination.area < int(
                        cfg.get("minimum_destination_area_px", 80))):
                continue
            if _main[0] is not owner or _main[1] is not current:
                _main[:] = [owner, current, max((
                    overlap(owner, item) / max(owner.area, 1)
                    for item in current if item.identity == owner.identity),
                    default=0.0)]
            main_continuation = _main[2]
            if main_continuation < float(
                    cfg.get("minimum_main_continuation_fraction", 0.35)):
                continue
            _held = _prior.get(id(destination))
            if _held is None or _held[0] is not destination:
                _held = _prior[id(destination)] = (destination, max((
                    overlap(item, destination) / max(item.area, 1)
                    for item in previous
                    if item.identity == destination.identity), default=0.0))
            prior_challenger = _held[1]
            if prior_challenger > float(
                    cfg.get("maximum_prior_challenger_fraction", 0.10)):
                continue
            values = metrics(owner, destination, lag, threshold, shape, memo)
            if not (
                    values["overlap_px"] >= int(cfg["minimum_overlap_px"])
                    and values["destination_from_source_fraction"] >= float(
                        cfg["minimum_destination_from_owner_fraction"])
                    and values["previous_overlap_fraction"] >= float(
                        cfg["minimum_previous_owner_fraction"])
                    and values["source_blue_coverage"] <= float(
                        cfg["maximum_owner_blue_departure"])
                    and values["destination_red_coverage"] <= float(
                        cfg["maximum_destination_red_arrival"])):
                continue
            altered = assign(candidate[frame + 1], destination,
                             owner.identity, shape)
            changed += altered
            actions.append({
                "method": "P", "action": "reclaim_partial_transfer",
                "source_output_frame": frame + 1,
                "destination_output_frame": frame + 2,
                "owner_identity": owner.identity,
                "observed_identity": destination.identity,
                "owner_layer": owner_auth.layer,
                "changed_pixels": altered, **values,
            })
            current = frame_components(candidate[frame + 1], frame + 1,
                                       minimum)
            break
    return changed


def apply_red_led(candidate: np.ndarray, frame: int,
                  previous: list[geometry.Component], lag: np.ndarray,
                  authorities: dict[int, Authority], cfg: dict,
                  threshold: float, minimum: int,
                  aliases: set[tuple[int, int]],
                  actions: list[dict],
                  source_tenures: dict[tuple[int, int], int] | None = None,
                  source_area_histories: dict[
                      tuple[int, int], tuple[int, ...]] | None = None,
                  memo: _PairMemo | None = None) -> int:
    overlap = geometry.overlap if memo is None else memo.overlap
    shape = candidate.shape[1:]
    changed = 0
    current = frame_components(candidate[frame + 1], frame + 1, minimum)
    for destination in list(current):
        if destination.area < int(cfg["minimum_destination_area_px"]):
            continue
        flat_lag = lag.ravel()
        red = float(np.mean(flat_lag[destination.pixels] >= threshold))
        if red < float(cfg["minimum_red_arrival"]):
            continue
        same_previous = [item for item in previous
                         if item.identity == destination.identity]
        same_fraction = max((
            overlap(item, destination) / max(item.area, 1)
            for item in same_previous), default=0.0)
        if same_fraction > float(
                cfg["maximum_same_identity_continuation_fraction"]):
            continue
        choices = []
        for source in previous:
            if source.identity == destination.identity:
                continue
            source_auth = authorities[source.identity]
            destination_auth = authorities[destination.identity]
            if (bool(cfg.get("require_birth_small_pair", False))
                    and (not source_auth.birth_small
                         or not destination_auth.birth_small)):
                continue
            if source_auth.rank > destination_auth.rank:
                continue
            source_tenure = (source_tenures or {}).get(
                (source.identity, source.number))
            if source_tenure is None:
                source_tenure = consecutive_source_tenure(
                    candidate, frame, source, minimum,
                    float(cfg.get("tenure_maximum_step_px", 12.0)), memo)
            if source_tenure < int(cfg.get(
                    "minimum_source_tenure_frames", 1)):
                continue
            blue = float(np.mean(flat_lag[source.pixels] <= -threshold))
            distance = geometry.distance(source, destination)
            area_history = (source_area_histories or {}).get(
                (source.identity, source.number), (source.area,))
            reference_area = float(np.median(area_history))
            ratio = destination.area / max(reference_area, 1.0)
            if (blue < float(cfg["minimum_source_blue_departure"])
                    or distance > float(cfg["maximum_source_distance_px"])
                    or ratio < float(cfg["minimum_area_ratio"])
                    or ratio > float(cfg["maximum_area_ratio"])):
                continue
            own_destinations = [item for item in current
                                if item.identity == source.identity]
            protected_trace = any(
                overlap(source, item) / max(item.area, 1)
                >= float(cfg.get(
                    "minimum_protected_trace_destination_fraction", 0.90))
                and float(np.mean(flat_lag[item.pixels] >= threshold))
                <= float(cfg.get("maximum_protected_trace_red", 0.20))
                for item in own_destinations)
            if protected_trace:
                continue
            own_fraction = max((
                overlap(source, item) / max(source.area, 1)
                for item in own_destinations), default=0.0)
            if own_fraction > float(
                    cfg["maximum_same_identity_continuation_fraction"]):
                continue
            score = (red + blue - 0.02 * distance
                     - 0.2 * abs(np.log(max(ratio, 1e-12))))
            choices.append((score, source, blue, distance, ratio,
                            own_destinations, source_tenure,
                            reference_area))
        if not choices:
            continue
        (_, source, blue, distance, ratio, own_destinations, source_tenure,
         reference_area) = max(choices, key=lambda row: row[0])
        observed = destination.identity
        altered = assign(candidate[frame + 1], destination,
                         source.identity, shape)
        changed += altered
        aliases.add((source.identity, observed))
        companion = None
        for item in own_destinations:
            source_overlap = overlap(source, item)
            item_red = float(np.mean(flat_lag[item.pixels] >= threshold))
            area_ratio = item.area / max(source.area, 1)
            if (source_overlap > 0
                    and area_ratio <= float(cfg["maximum_companion_area_ratio"])
                    and item_red <= float(cfg["maximum_companion_red_arrival"])):
                companion = item
                break
        if companion is not None:
            changed += assign(candidate[frame + 1], companion,
                              observed, shape)
        actions.append({
            "method": "R", "action": "red_destination_finds_blue_source",
            "source_output_frame": frame + 1,
            "destination_output_frame": frame + 2,
            "owner_identity": source.identity,
            "observed_identity": observed,
            "owner_layer": authorities[source.identity].layer,
            "changed_pixels": altered,
            "source_blue_coverage": blue,
            "destination_red_coverage": red,
            "centroid_shift_px": distance,
            "area_ratio": ratio,
            "source_reference_area_px": reference_area,
            "source_consecutive_tenure_frames": source_tenure,
            "companion_restored": companion is not None,
            "companion_area_px": companion.area if companion else 0,
        })
        current = frame_components(candidate[frame + 1], frame + 1, minimum)
    return changed


def consecutive_source_tenure(candidate: np.ndarray, frame: int,
                              source: geometry.Component, minimum: int,
                              maximum_step: float,
                              memo: _PairMemo | None = None) -> int:
    """Count an identity's immediately preceding continuous component path."""
    overlap = geometry.overlap if memo is None else memo.overlap
    tenure = 1
    current = source
    for prior_frame in range(frame - 1, -1, -1):
        choices = [item for item in frame_components(
            candidate[prior_frame], prior_frame, minimum)
            if item.identity == source.identity]
        if not choices:
            break
        previous = min(choices, key=lambda item: (
            -overlap(item, current),
            geometry.distance(item, current), item.number))
        if (overlap(previous, current) == 0
                and geometry.distance(previous, current) > maximum_step):
            break
        tenure += 1
        current = previous
    return tenure


def advance_component_tenures(
        previous: list[geometry.Component],
        previous_tenures: dict[tuple[int, int], int],
        current: list[geometry.Component], maximum_step: float,
        memo: _PairMemo | None = None) -> dict[tuple[int, int], int]:
    """Advance consecutive component tenure once for the next frame."""
    overlap = geometry.overlap if memo is None else memo.overlap
    advanced = {}
    for destination in current:
        choices = [source for source in previous
                   if source.identity == destination.identity]
        tenure = 1
        if choices:
            source = min(choices, key=lambda item: (
                -overlap(item, destination),
                geometry.distance(item, destination), item.number))
            if (overlap(source, destination) > 0
                    or geometry.distance(source, destination) <= maximum_step):
                tenure = previous_tenures.get(
                    (source.identity, source.number), 1) + 1
        advanced[(destination.identity, destination.number)] = tenure
    return advanced


def advance_component_area_histories(
        previous: list[geometry.Component],
        previous_histories: dict[tuple[int, int], tuple[int, ...]],
        current: list[geometry.Component], maximum_step: float,
        history_length: int, memo: _PairMemo | None = None
        ) -> dict[tuple[int, int], tuple[int, ...]]:
    """Carry a short robust area history along continuous identity paths."""
    overlap = geometry.overlap if memo is None else memo.overlap
    advanced = {}
    for destination in current:
        choices = [source for source in previous
                   if source.identity == destination.identity]
        history = (destination.area,)
        if choices:
            source = min(choices, key=lambda item: (
                -overlap(item, destination),
                geometry.distance(item, destination), item.number))
            if (overlap(source, destination) > 0
                    or geometry.distance(source, destination) <= maximum_step):
                prior = previous_histories.get(
                    (source.identity, source.number), (source.area,))
                history = tuple((*prior, destination.area)[-history_length:])
        advanced[(destination.identity, destination.number)] = history
    return advanced


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("movie-wide authority stage is required")
    labels = load_stack(ROOT / params["parent_labels"])
    trim = ROOT / params["trim_dir"]
    lag_stack = load_stack(trim / f"{params['stem']}_lag01_float.tif")
    authority_table, _ = authority_stage.movie_authority(
        labels, params["authority"])
    authorities = authority_map(authority_table)
    methods = set(params["enabled_methods"])
    candidate = labels.copy()
    actions: list[dict] = []
    aliases: set[tuple[int, int]] = set()
    minimum = int(params["authority"]["minimum_component_area_px"])
    considered = 0
    partitions = 0
    source_tenures: dict[tuple[int, int], int] = {}
    source_area_histories: dict[tuple[int, int], tuple[int, ...]] = {}
    tenure_step = float(params["red_led"].get(
        "tenure_maximum_step_px", 12.0))
    area_history_length = int(params["red_led"].get(
        "source_area_history_frames", 5))
    memo = _PairMemo()
    for frame in range(len(candidate) - 1):
        previous = frame_components(candidate[frame], frame, minimum)
        if frame == 0:
            source_tenures = {
                (item.identity, item.number): 1 for item in previous}
            source_area_histories = {
                (item.identity, item.number): (item.area,)
                for item in previous}
        current = frame_components(candidate[frame + 1], frame + 1, minimum)
        considered += len(previous)
        transition = lag_stack[frame]
        if "O" in methods:
            _, divided = apply_ordered(
                candidate, frame, previous, current, transition,
                authorities, params["ordered"],
                float(params["motion_threshold"]), aliases, actions, memo)
            partitions += divided
        if "P" in methods:
            apply_partial(
                candidate, frame, previous, transition, authorities,
                params["partial"], float(params["motion_threshold"]),
                minimum, actions, memo)
        if "R" in methods:
            apply_red_led(
                candidate, frame, previous, transition, authorities,
                params["red_led"], float(params["motion_threshold"]),
                minimum, aliases, actions, source_tenures,
                source_area_histories, memo)
        corrected_current = frame_components(
            candidate[frame + 1], frame + 1, minimum)
        source_tenures = advance_component_tenures(
            previous, source_tenures, corrected_current, tenure_step, memo)
        source_area_histories = advance_component_area_histories(
            previous, source_area_histories, corrected_current, tenure_step,
            area_history_length, memo)
    label_path = out.out / f"{params['stem']}.tif"
    action_path = out.out / "hierarchy_actions.csv"
    metrics_path = out.out / "hierarchy_metrics.json"
    save_stack(label_path, candidate, params["frame_interval_min"])
    pd.DataFrame(actions).to_csv(action_path, index=False)
    method_counts = {
        method: int(sum(action.get("method") == method
                        for action in actions))
        for method in sorted(methods)}
    summary = {
        "enabled_methods": sorted(methods),
        "method_action_counts": method_counts,
        "active_owner_components_considered": considered,
        "all_active_components_audited": bool(considered > 0),
        "multiowner_hosts_partitioned": partitions,
        "persistent_corrected_aliases": [
            [int(owner), int(observed)]
            for owner, observed in sorted(aliases)],
        "canonical_identity_rule": "earliest_movie_wide_identity_observation",
        "future_area_can_promote_identity": False,
        "large_jump_requires_immediately_previous_source": True,
        "maximum_missing_frames_for_large_jump": 0,
        "source_tenure_cached_once_per_transition": True,
        "source_area_history_frames": area_history_length,
        "changed_identity_pixels": int(np.count_nonzero(candidate != labels)),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "identities_before": int(len(set(map(int, np.unique(labels))) - {0})),
        "identities_after": int(len(set(map(int, np.unique(candidate))) - {0})),
        "registration_added": False,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"outputs": {"labels": label_path, "actions": action_path,
                        "metrics": metrics_path},
            "summary": summary}


