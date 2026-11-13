"""Conservative, target-free correction of crowded reciprocal label swaps.

The learned-link experiments ranked candidate-partner count, overlap pixels and
overlap margin among their strongest features.  This deterministic version uses
the same evidence without importing a fitted model or its imperfect labels.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


STRUCTURE = np.ones((3, 3), np.uint8)
PARAMETERS = {
    "minimum_overlap_px", "minimum_cross_coverage",
    "minimum_margin_fraction", "maximum_neighbour_sum_radii",
    "minimum_local_neighbours", "minimum_recurrent_exchanges",
    "maximum_recurrence_gap_movie_fraction",
}


def _objects(frame: np.ndarray) -> dict[int, dict]:
    result = {}
    for identity in np.unique(frame):
        identity = int(identity)
        if identity <= 0:
            continue
        mask = frame == identity
        yy, xx = np.nonzero(mask)
        result[identity] = {
            "mask": mask,
            "area": int(len(xx)),
            "y": float(yy.mean()),
            "x": float(xx.mean()),
            "radius": float(np.sqrt(len(xx) / np.pi)),
            "components": int(ndi.label(mask, STRUCTURE)[1]),
            "first_y": int(yy[0]),
            "first_x": int(xx[0]),
        }
    return result


def _overlaps(previous: np.ndarray, current: np.ndarray) -> dict[tuple[int, int], int]:
    valid = (previous > 0) & (current > 0)
    if not np.any(valid):
        return {}
    pairs = np.column_stack((previous[valid], current[valid])).astype(np.int64)
    values, counts = np.unique(pairs, axis=0, return_counts=True)
    return {(int(a), int(b)): int(n) for (a, b), n in zip(values, counts)}


def correct(labels: np.ndarray, params: dict | None = None):
    params = dict(params or {})
    unexpected = set(params) - PARAMETERS
    if unexpected:
        raise ValueError("Unknown or targeted parameters: " + str(sorted(unexpected)))
    minimum_overlap_px = int(params.get("minimum_overlap_px", 12))
    minimum_cross_coverage = float(params.get("minimum_cross_coverage", 0.55))
    minimum_margin_fraction = float(params.get("minimum_margin_fraction", 0.20))
    maximum_neighbour_sum_radii = float(
        params.get("maximum_neighbour_sum_radii", 4.0))
    minimum_local_neighbours = int(params.get("minimum_local_neighbours", 1))
    minimum_recurrent_exchanges = int(
        params.get("minimum_recurrent_exchanges", 1))
    maximum_recurrence_gap = max(1, int(math.ceil(
        float(params.get("maximum_recurrence_gap_movie_fraction", 1.0))
        * len(labels))))
    output = np.asarray(labels).copy()
    rows = []
    applied = []
    for frame in range(1, len(output)):
        previous, current = output[frame - 1], output[frame]
        old, new = _objects(previous), _objects(current)
        common = sorted(set(old) & set(new))
        overlap = _overlaps(previous, current)
        # Learned n_partners feature: any plausible overlap or nearby object is a
        # competitor. Radius is cell-relative, never a fixed image region.
        neighbours = {}
        for identity, item in old.items():
            neighbours[identity] = sum(
                other != identity and np.hypot(
                    item["y"] - candidate["y"], item["x"] - candidate["x"]
                ) <= maximum_neighbour_sum_radii * max(
                    item["radius"] + candidate["radius"], 1.0)
                for other, candidate in old.items())

        source_best = {}
        destination_best = {}
        for source in old:
            options = [(pixels, destination) for (candidate, destination), pixels
                       in overlap.items() if candidate == source]
            if options:
                best = max(pixels for pixels, _ in options)
                winners = [destination for pixels, destination in options
                           if pixels == best]
                if len(winners) == 1:
                    source_best[source] = winners[0]
        for destination in new:
            options = [(pixels, source) for (source, candidate), pixels
                       in overlap.items() if candidate == destination]
            if options:
                best = max(pixels for pixels, _ in options)
                winners = [source for pixels, source in options if pixels == best]
                if len(winners) == 1:
                    destination_best[destination] = winners[0]

        proposals = []
        for index, left in enumerate(common):
            for right in common[index + 1:]:
                cross_left = overlap.get((left, right), 0)
                cross_right = overlap.get((right, left), 0)
                same_left = overlap.get((left, left), 0)
                same_right = overlap.get((right, right), 0)
                cross = cross_left + cross_right
                same = same_left + same_right
                denominator = max(cross, same, 1)
                margin = (cross - same) / denominator
                coverage_left = cross_left / max(old[left]["area"], new[right]["area"], 1)
                coverage_right = cross_right / max(old[right]["area"], new[left]["area"], 1)
                mutual = (
                    source_best.get(left) == right
                    and source_best.get(right) == left
                    and destination_best.get(left) == right
                    and destination_best.get(right) == left
                )
                dense = min(neighbours.get(left, 0), neighbours.get(right, 0)) \
                    >= minimum_local_neighbours
                eligible = bool(
                    mutual and dense
                    and min(cross_left, cross_right) >= minimum_overlap_px
                    and min(coverage_left, coverage_right) >= minimum_cross_coverage
                    and margin >= minimum_margin_fraction
                    and old[left]["components"] == old[right]["components"] == 1
                    and new[left]["components"] == new[right]["components"] == 1
                )
                if cross > 0:
                    rows.append({
                        "frame": frame, "identity_a": left, "identity_b": right,
                        "local_neighbours_a": neighbours.get(left, 0),
                        "local_neighbours_b": neighbours.get(right, 0),
                        "same_overlap_px": same, "cross_overlap_px": cross,
                        "cross_coverage_a": coverage_left,
                        "cross_coverage_b": coverage_right,
                        "cross_margin_fraction": margin,
                        "mutual_best": mutual, "density_gate": dense,
                        "eligible": eligible,
                    })
                if eligible:
                    geometry = tuple(sorted((
                        (old[left]["y"], old[left]["x"], old[left]["area"],
                         old[left]["first_y"], old[left]["first_x"]),
                        (old[right]["y"], old[right]["x"], old[right]["area"],
                         old[right]["first_y"], old[right]["first_x"]),
                    )))
                    proposals.append((cross - same, geometry, left, right))

        used = set()
        for _, _, left, right in sorted(
                proposals, key=lambda item: (-item[0], item[1])):
            if left in used or right in used:
                continue
            left_mask = output[frame] == left
            right_mask = output[frame] == right
            output[frame][left_mask] = right
            output[frame][right_mask] = left
            used.update((left, right))
            applied.append((frame, left, right))

    audit = pd.DataFrame(rows)
    # A single crowded cross-over can be a real physical crossing. Repeated
    # reciprocal evidence for the same pair is a self-calibrating sign of label
    # oscillation, and was the distinction that separated the B2 false positive
    # from the repeated A2/B4 patterns without naming any well or cell.
    retained: set[tuple[int, int, int]] = set()
    if applied:
        by_pair: dict[tuple[int, int], list[int]] = {}
        for frame, left, right in applied:
            by_pair.setdefault((left, right), []).append(frame)
        for (left, right), frames in by_pair.items():
            cluster = [frames[0]]
            clusters = []
            for frame in frames[1:]:
                if frame - cluster[-1] <= maximum_recurrence_gap:
                    cluster.append(frame)
                else:
                    clusters.append(cluster)
                    cluster = [frame]
            clusters.append(cluster)
            for group in clusters:
                if len(group) >= minimum_recurrent_exchanges:
                    retained.update((frame, left, right) for frame in group)
        if len(retained) != len(applied):
            output = np.asarray(labels).copy()
            for frame, left, right in sorted(retained):
                left_mask = output[frame] == left
                right_mask = output[frame] == right
                output[frame][left_mask] = right
                output[frame][right_mask] = left
    if len(audit):
        provisional = audit.eligible.astype(bool).copy()
        audit["provisional_eligible"] = provisional
        audit["eligible"] = [
            bool(was and (int(row.frame), int(row.identity_a),
                          int(row.identity_b)) in retained)
            for was, row in zip(provisional, audit.itertuples())
        ]
        audit["minimum_recurrent_exchanges"] = minimum_recurrent_exchanges
        audit["maximum_recurrence_gap_frames"] = maximum_recurrence_gap
    summary = {
        "applied_exchanges": int(audit.eligible.sum()) if len(audit) else 0,
        "changed_pixels": int(np.count_nonzero(output != labels)),
        "changed_frames": int(np.count_nonzero(np.any(output != labels, axis=(1, 2)))),
        "foreground_changed_pixels": int(np.count_nonzero((output > 0) != (labels > 0))),
        "identity_set_changed_frames": int(sum(
            set(np.unique(output[t])) != set(np.unique(labels[t]))
            for t in range(len(labels)))),
    }
    return output, audit, summary
