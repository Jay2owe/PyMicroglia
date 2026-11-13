"""Field-wide, area-consistent ownership of two touching soma seats.

The upstream accepted two-seat layer supplies target-free pair discoveries and
trusted frames where it recovered both bodies. This layer considers only the
gaps between those frames, leaving all established pixels immutable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import episode_consistent_fusion as episode
import owner_consensus


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    maximum_bookend_gap: int = 16
    minimum_interval_anomaly: float = 0.25
    maximum_seat_offset_radii: float = 1.2
    edit_cost: float = 0.035
    changed_fraction_cost: float = 0.12
    transition_cost: float = 0.22
    minimum_interval_gain: float = 0.035


def _area_cost(areas: tuple[int, int], expected: tuple[float, float]) -> float:
    return (abs(areas[0] - expected[0]) / max(expected[0], 1)
            + abs(areas[1] - expected[1]) / max(expected[1], 1))


def _transition_cost(first: tuple[int, int], second: tuple[int, int],
                     expected: tuple[float, float], weight: float) -> float:
    return weight * (
        abs(first[0] - second[0]) / max(expected[0], 1)
        + abs(first[1] - second[1]) / max(expected[1], 1))


def _seat_near(mask: np.ndarray, position: np.ndarray, area: float,
               params: Params) -> bool:
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return False
    distance = float(np.sqrt(np.min((yy - position[0]) ** 2
                                    + (xx - position[1]) ** 2)))
    radius = max(np.sqrt(area / np.pi), 1.0)
    return distance <= params.maximum_seat_offset_radii * radius


def _options(frame: np.ndarray, protected: np.ndarray,
             donor: int, host: int, donor_position: np.ndarray,
             host_position: np.ndarray, expected: tuple[float, float],
             params: Params) -> list[dict]:
    left = episode._largest(frame, donor)
    right = episode._largest(frame, host)
    if left is None or right is None:
        return []
    original = (int(left["area"]), int(right["area"]))
    baseline = {"areas": original, "changed": 0, "partitions": None,
                "method": "parent"}
    result = [baseline]
    if (not _seat_near(left["mask"], donor_position, expected[0], params)
            or not _seat_near(right["mask"], host_position, expected[1], params)):
        return result
    merged = left["mask"] | right["mask"]
    if ndi.label(merged, STRUCTURE)[1] != 1:
        return result
    partition = episode._geometry_partition(
        merged, donor_position, host_position, *expected)
    if partition is None:
        return result
    donor_part, host_part = partition
    changed = ((frame == donor) & host_part) | ((frame == host) & donor_part)
    changed_count = int(changed.sum())
    if not changed_count or np.any(changed & protected):
        return result
    result.append({"areas": (int(donor_part.sum()), int(host_part.sum())),
                   "changed": changed_count,
                   "partitions": (donor_part, host_part),
                   "method": "bookended_connected_area_partition"})
    return result


def _solve_interval(options: list[list[dict]], targets: list[tuple[float, float]],
                    left: tuple[int, int], right: tuple[int, int],
                    params: Params) -> tuple[list[int], float]:
    """Find the lowest-cost sequence, anchored by immutable endpoint areas."""
    if not options:
        return [], 0.0
    def unary(row: dict, target: tuple[float, float]) -> float:
        cost = _area_cost(row["areas"], target)
        if row["changed"]:
            cost += params.edit_cost + params.changed_fraction_cost * (
                row["changed"] / max(sum(row["areas"]), 1))
        return cost

    costs = []
    paths = []
    previous = [{"areas": left}]
    previous_costs = [0.0]
    for choices, target in zip(options, targets):
        current_costs = []
        current_paths = []
        for j, choice in enumerate(choices):
            predecessors = [
                previous_costs[k] + _transition_cost(
                    old["areas"], choice["areas"], target,
                    params.transition_cost)
                for k, old in enumerate(previous)]
            best = int(np.argmin(predecessors))
            current_costs.append(predecessors[best] + unary(choice, target))
            current_paths.append((paths[best] if paths else []) + [j])
        costs, paths = current_costs, current_paths
        previous, previous_costs = choices, current_costs
    totals = [cost + _transition_cost(
        choice["areas"], right, targets[-1], params.transition_cost)
        for cost, choice in zip(costs, options[-1])]
    winner = int(np.argmin(totals))
    # The all-parent path uses the same objective, including transition terms.
    parent_cost = 0.0
    previous_areas = left
    for choices, target in zip(options, targets):
        row = choices[0]
        parent_cost += _transition_cost(
            previous_areas, row["areas"], target, params.transition_cost)
        parent_cost += unary(row, target)
        previous_areas = row["areas"]
    parent_cost += _transition_cost(
        previous_areas, right, targets[-1], params.transition_cost)
    return paths[winner], parent_cost - totals[winner]


def correct(labels: np.ndarray, pair_audit: pd.DataFrame,
            trusted_applications: pd.DataFrame,
            protected: np.ndarray | None = None,
            params: Params = Params()) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return labels, every interval verdict, and selected frame changes."""
    if protected is None:
        protected = np.zeros_like(labels, bool)
    if protected.shape != labels.shape:
        raise ValueError("protected pixels must align with labels")
    candidate = labels.copy()
    audit, applications = [], []
    if pair_audit.empty or trusted_applications.empty:
        return candidate, pd.DataFrame(), pd.DataFrame()
    applied = pair_audit[pair_audit.outcome.eq("applied")]
    for pair in applied.itertuples(index=False):
        pair_id = str(pair.proposal_id)
        donor, host = int(pair.donor), int(pair.host)
        donor_position = np.array([pair.donor_y, pair.donor_x], float)
        host_position = np.array([pair.host_y, pair.host_x], float)
        trusted = trusted_applications[
            trusted_applications.proposal_id.astype(str).eq(pair_id)].sort_values("frame")
        rows = list(trusted.itertuples(index=False))
        for before, after in zip(rows, rows[1:]):
            low, high = int(before.frame), int(after.frame)
            base = {"proposal_id": pair_id, "donor": donor, "host": host,
                    "first_bookend": low, "last_bookend": high}
            if high - low < 2 or high - low > params.maximum_bookend_gap:
                continue
            left = (int(before.donor_pixels), int(before.host_pixels))
            right = (int(after.donor_pixels), int(after.host_pixels))
            options, targets = [], []
            for number in range(low + 1, high):
                fraction = (number - low) / (high - low)
                expected = ((1 - fraction) * left[0] + fraction * right[0],
                            (1 - fraction) * left[1] + fraction * right[1])
                targets.append(expected)
                choices = _options(
                    labels[number - 1], protected[number - 1], donor, host,
                    donor_position, host_position, expected, params)
                if not choices:
                    # No two named seats here. The parent state remains fixed.
                    choices = [{"areas": (0, 0), "changed": 0,
                                "partitions": None, "method": "parent"}]
                options.append(choices)
            anomaly = max(
                max(abs(row[0]["areas"][k] - target[k]) / max(target[k], 1)
                    for k in (0, 1))
                for row, target in zip(options, targets))
            if anomaly < params.minimum_interval_anomaly:
                audit.append({**base, "outcome": "rejected",
                              "reason": "no_substantial_area_anomaly",
                              "area_anomaly": anomaly, "changed_pixels": 0})
                continue
            selected, gain = _solve_interval(options, targets, left, right, params)
            if gain < params.minimum_interval_gain or not any(selected):
                audit.append({**base, "outcome": "rejected",
                              "reason": "no_consistent_better_partition",
                              "area_anomaly": anomaly, "objective_gain": gain,
                              "changed_pixels": 0})
                continue
            trial = candidate.copy()
            rows_out = []
            for number, choice_index, choices in zip(
                    range(low + 1, high), selected, options):
                choice = choices[choice_index]
                if choice["partitions"] is None:
                    continue
                donor_part, host_part = choice["partitions"]
                trial[number - 1][donor_part] = donor
                trial[number - 1][host_part] = host
                rows_out.append({**base, "frame": number,
                                 "method": choice["method"],
                                 "changed_pixels": choice["changed"],
                                 "donor_pixels": choice["areas"][0],
                                 "host_pixels": choice["areas"][1]})
            if (np.any((trial != candidate) & protected)
                    or not np.array_equal(trial > 0, candidate > 0)
                    or owner_consensus.count_new_duplicate_components(
                        candidate, trial)):
                audit.append({**base, "outcome": "rejected",
                              "reason": "protected_or_duplicate_component",
                              "area_anomaly": anomaly, "objective_gain": gain,
                              "changed_pixels": 0})
                continue
            changed = int(np.count_nonzero(trial != candidate))
            candidate = trial
            applications.extend(rows_out)
            audit.append({**base, "outcome": "applied",
                          "reason": "bookended_area_consistency",
                          "area_anomaly": anomaly, "objective_gain": gain,
                          "changed_pixels": changed})
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
