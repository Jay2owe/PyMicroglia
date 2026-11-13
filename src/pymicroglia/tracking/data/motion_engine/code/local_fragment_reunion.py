"""Discover small untracked outline fragments beside tracked cell bodies.

This module uses only labels and physical tracks. Feedback identities, frames,
coordinates, and wells are never inputs to the producer.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math

import numpy as np
import pandas as pd
from skimage.measure import label as connected_labels


@dataclass(frozen=True)
class Params:
    maximum_gap_px: int = 3
    maximum_gap_sum_radii: float = 0.25
    maximum_small_large_area_ratio: float = 0.40
    minimum_recurrent_frames: int = 3
    maximum_relative_step_sum_radii: float = 0.50
    minimum_strong_large_track_points: int = 1
    minimum_pair_track_handoffs: int = 4
    minimum_shared_track_pair_purity: float = 0.50
    minimum_union_coverage: float = 1.0
    maximum_cooccurrence_movie_fraction: float = 0.60


def _components(frame: np.ndarray):
    components = connected_labels(frame, background=0, connectivity=2)
    count = int(components.max())
    areas = np.bincount(components.ravel(), minlength=count + 1)
    identities = np.zeros(count + 1, dtype=np.int32)
    if count:
        unique, first = np.unique(components, return_index=True)
        identities[unique] = frame.ravel()[first]
    yy, xx = np.nonzero(components)
    ids = components[yy, xx]
    cy = np.bincount(ids, weights=yy, minlength=count + 1) / np.maximum(areas, 1)
    cx = np.bincount(ids, weights=xx, minlength=count + 1) / np.maximum(areas, 1)
    return components, areas, identities, cy, cx


def _nearby_pairs(components: np.ndarray, maximum_gap_px: int):
    height, width = components.shape
    nearest: dict[tuple[int, int], float] = {}
    for dy in range(maximum_gap_px + 1):
        for dx in range(-maximum_gap_px, maximum_gap_px + 1):
            if dy == 0 and dx <= 0:
                continue
            distance = math.hypot(dy, dx)
            if distance > maximum_gap_px:
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            first = components[y0:y1, x0:x1]
            second = components[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            valid = (first > 0) & (second > 0) & (first != second)
            if not np.any(valid):
                continue
            first_ids, second_ids = first[valid], second[valid]
            pairs = np.unique(np.column_stack((np.minimum(first_ids, second_ids),
                                               np.maximum(first_ids, second_ids))), axis=0)
            for a, b in pairs:
                pair = int(a), int(b)
                nearest[pair] = min(nearest.get(pair, math.inf), distance)
    return nearest


def _point_components(components: np.ndarray, points: pd.DataFrame):
    strong = Counter()
    height, width = components.shape
    for point in points.itertuples(index=False):
        if str(point.state) not in ("observed", "latent_visible") or not bool(point.strong):
            continue
        x, y = float(point.x), float(point.y)
        limit = max(1.0, 0.75 * float(point.radius_px))
        y0, y1 = max(0, int(math.floor(y - limit))), min(height, int(math.ceil(y + limit)) + 1)
        x0, x1 = max(0, int(math.floor(x - limit))), min(width, int(math.ceil(x + limit)) + 1)
        patch = components[y0:y1, x0:x1]
        yy, xx = np.nonzero(patch)
        if not len(yy):
            continue
        distances = (yy + y0 - y) ** 2 + (xx + x0 - x) ** 2
        minimum = float(distances.min())
        if minimum > limit ** 2:
            continue
        nearest = np.unique(patch[yy[distances == minimum], xx[distances == minimum]])
        if len(nearest) == 1:
            strong[int(nearest[0])] += 1
    return strong


def discover_frame(frame: np.ndarray, points: pd.DataFrame, frame_index: int,
                   params: Params = Params()) -> tuple[pd.DataFrame, np.ndarray]:
    components, areas, identities, cy, cx = _components(frame)
    pairs = _nearby_pairs(components, params.maximum_gap_px)
    strong = _point_components(components, points)
    rows = []
    for (a, b), gap in pairs.items():
        if identities[a] == identities[b]:
            continue
        large, small = (a, b) if areas[a] >= areas[b] else (b, a)
        area_ratio = float(areas[small] / max(areas[large], 1))
        radius_sum = math.sqrt(areas[large] / math.pi) + math.sqrt(areas[small] / math.pi)
        gap_radii = gap / max(radius_sum, 1)
        preliminary = bool(
            area_ratio <= params.maximum_small_large_area_ratio
            and gap_radii <= params.maximum_gap_sum_radii
            and strong[small] == 0
            and strong[large] >= params.minimum_strong_large_track_points)
        rows.append({
            "frame": frame_index, "large_component": large, "small_component": small,
            "large_identity": int(identities[large]),
            "small_identity": int(identities[small]),
            "large_area": int(areas[large]), "small_area": int(areas[small]),
            "area_ratio": area_ratio, "gap_px": gap, "gap_sum_radii": gap_radii,
            "large_strong_points": int(strong[large]),
            "small_strong_points": int(strong[small]),
            "large_centroid_y": float(cy[large]), "large_centroid_x": float(cx[large]),
            "small_centroid_y": float(cy[small]), "small_centroid_x": float(cx[small]),
            "radius_sum": radius_sum,
            "preliminary": preliminary,
        })
    return pd.DataFrame(rows), components


def discover(labels: np.ndarray, points: pd.DataFrame, params: Params = Params()):
    if labels.ndim != 3:
        raise ValueError("labels must have frame, y, x axes")
    if params.maximum_gap_px < 1 or params.minimum_recurrent_frames < 2:
        raise ValueError("invalid fragment reunion thresholds")
    point_groups = {int(frame): group for frame, group in points.groupby("frame")}
    empty_points = points.iloc[:0]
    tables = []
    for frame in range(len(labels)):
        table, _ = discover_frame(labels[frame], point_groups.get(frame, empty_points),
                                  frame, params)
        if len(table):
            tables.append(table)
    if not tables:
        return pd.DataFrame()
    audit = pd.concat(tables, ignore_index=True)
    audit["eligible"] = False
    audit["reason"] = "preliminary_failed"
    preliminary = audit[audit.preliminary.astype(bool)].copy()
    if preliminary.empty:
        return audit
    # Each frame must have one unambiguous component pair for these identities.
    key = ["frame", "large_identity", "small_identity"]
    duplicates = preliminary.groupby(key).size()
    duplicate_keys = set(duplicates[duplicates > 1].index)
    small_counts = preliminary.groupby(["frame", "small_component"]).size()
    conflict_keys = set(small_counts[small_counts > 1].index)
    preliminary = preliminary[
        ~preliminary.apply(lambda row: (
            (row.frame, row.large_identity, row.small_identity) in duplicate_keys
            or (row.frame, row.small_component) in conflict_keys), axis=1)]
    for (_, _), group in preliminary.groupby(["large_identity", "small_identity"]):
        group = group.sort_values("frame")
        runs = np.split(group.index.to_numpy(),
                        np.where(np.diff(group.frame.to_numpy()) != 1)[0] + 1)
        for indices in runs:
            if len(indices) < params.minimum_recurrent_frames:
                continue
            run = audit.loc[indices]
            relative = np.column_stack((run.small_centroid_y - run.large_centroid_y,
                                        run.small_centroid_x - run.large_centroid_x))
            relative_steps = np.linalg.norm(np.diff(relative, axis=0), axis=1)
            normalizer = 0.5 * (run.radius_sum.to_numpy()[1:]
                                + run.radius_sum.to_numpy()[:-1])
            maximum_relative_step = float(np.max(relative_steps / np.maximum(normalizer, 1)))
            if maximum_relative_step > params.maximum_relative_step_sum_radii:
                audit.loc[indices, "reason"] = "relative_motion_disagrees"
                continue
            audit.loc[indices, "eligible"] = True
            audit.loc[indices, "reason"] = "persistent_untracked_adjacent_fragment"
    return audit


def correct(labels: np.ndarray, points: pd.DataFrame,
            lineage_audit: pd.DataFrame, params: Params = Params(),
            applied_unions: pd.DataFrame | None = None):
    """Attach only recurring, untracked components backed by exclusive lineage.

    ``lineage_audit`` is the frozen independent-track evidence from the
    preceding complete-lineage stage. A proposed pair that participated in an
    already applied whole-lineage union is excluded because its evidence then
    refers to different identity labels. No feedback enters this producer.
    """
    audit = discover(labels, points, params)
    candidate = np.asarray(labels).copy()
    if audit.empty:
        return candidate, audit, pd.DataFrame()
    required = {"canonical_identity", "alias_identity", "track_handoffs",
                "minimum_shared_track_pair_purity", "union_coverage",
                "cooccurrence_fraction"}
    if not required.issubset(lineage_audit.columns):
        raise ValueError("lineage audit lacks independent-track evidence")
    evidence = {}
    for row in lineage_audit.itertuples(index=False):
        key = frozenset((int(row.canonical_identity), int(row.alias_identity)))
        if key in evidence:
            raise ValueError("duplicate lineage evidence for one identity pair")
        evidence[key] = row
    previously_changed = set()
    if applied_unions is not None and not applied_unions.empty:
        for row in applied_unions.itertuples(index=False):
            previously_changed.update((int(row.canonical_identity),
                                       int(row.alias_identity)))
    audit["lineage_supported"] = False
    audit["applied"] = False
    for (large, small), group in audit[audit.eligible.astype(bool)].groupby(
            ["large_identity", "small_identity"]):
        key = frozenset((int(large), int(small)))
        row = evidence.get(key)
        supported = bool(
            row is not None
            and not (key & previously_changed)
            and int(row.track_handoffs) >= params.minimum_pair_track_handoffs
            and float(row.minimum_shared_track_pair_purity)
            >= params.minimum_shared_track_pair_purity
            and float(row.union_coverage) >= params.minimum_union_coverage
            and float(row.cooccurrence_fraction)
            <= params.maximum_cooccurrence_movie_fraction)
        if not supported:
            audit.loc[group.index, "eligible"] = False
            audit.loc[group.index, "reason"] = "independent_lineage_evidence_insufficient"
            continue
        audit.loc[group.index, "lineage_supported"] = True
    selected = audit[audit.eligible.astype(bool)]
    # An outline fragment must have just one claimant in its frame.
    conflicts = selected.groupby(["frame", "small_component"]).size()
    conflict_keys = set(conflicts[conflicts > 1].index)
    if conflict_keys:
        indices = selected.index[selected.apply(
            lambda row: (int(row.frame), int(row.small_component))
            in conflict_keys, axis=1)]
        audit.loc[indices, "eligible"] = False
        audit.loc[indices, "reason"] = "competing_fragment_claims"
    applications = []
    for frame, group in audit[audit.eligible.astype(bool)].groupby("frame"):
        t = int(frame)
        components = connected_labels(labels[t], background=0, connectivity=2)
        for index, row in group.iterrows():
            component_mask = components == int(row.small_component)
            if (int(component_mask.sum()) != int(row.small_area)
                    or not np.all(labels[t][component_mask] == int(row.small_identity))
                    or not np.any(labels[t][components == int(row.large_component)]
                                  == int(row.large_identity))):
                raise AssertionError("fragment component changed after discovery")
            candidate[t][component_mask] = int(row.large_identity)
            audit.at[index, "applied"] = True
            applications.append({"frame": t,
                                 "recipient_identity": int(row.large_identity),
                                 "fragment_identity": int(row.small_identity),
                                 "changed_pixels": int(component_mask.sum())})
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("fragment reunion changed foreground")
    return candidate, audit, pd.DataFrame(applications)
