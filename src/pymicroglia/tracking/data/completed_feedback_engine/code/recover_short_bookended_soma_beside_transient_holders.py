"""Restore a short bookended soma name beside transient distant holders.

The eligibility test uses only physical points and accepted labels. It has
no movie, identity, track, coordinate, frame, region or feedback selector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import complete_owner_handoff_episodes_r10_a002 as partition


@dataclass(frozen=True)
class Params:
    minimum_bookend_frames: int = 5
    minimum_middle_frames: int = 8
    maximum_step_p95_radii: float = 0.8
    minimum_basin_area_ratio: float = 0.25
    maximum_basin_area_ratio: float = 4.0
    maximum_individual_remnant_area_ratio: float = 1.0
    maximum_total_remnant_area_ratio: float = 0.20
    minimum_remnant_separation_sum_radii: float = 2.5
    minimum_large_remnant_separation_sum_radii: float = 5.0
    large_remnant_area_ratio: float = 0.35
    maximum_strong_holder_track_frames: int = 2
    maximum_strong_holder_frames: int = 3
    remnant_link_radius_multiplier: float = 4.0
    remnant_link_pixel_allowance: float = 2.0
    minimum_second_best_distance_ratio: float = 1.5
    minimum_prior_owner_frames: int = 3
    prior_owner_window_frames: int = 8
    minimum_linked_bookend_frames: int = 3
    maximum_linked_wrong_frames: int = 3
    minimum_temporal_companion_overlap_ratio: float = 0.15
    maximum_temporal_companion_area_ratio: float = 4.0


STRUCTURE = np.ones((3, 3), bool)


def _strong_points(points: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    strong = points[points.state.astype(str).eq("observed")
                    & points.strong.astype(bool)].copy()
    if strong.empty:
        return strong.assign(owner=pd.Series(dtype=int))
    t = strong.frame.to_numpy(int)
    y = np.rint(strong.y.to_numpy(float)).astype(int)
    x = np.rint(strong.x.to_numpy(float)).astype(int)
    valid = ((0 <= t) & (t < len(labels)) & (0 <= y)
             & (y < labels.shape[1]) & (0 <= x) & (x < labels.shape[2]))
    strong = strong.iloc[np.flatnonzero(valid)].copy()
    strong["owner"] = labels[t[valid], y[valid], x[valid]].astype(int)
    return strong


def _runs(group: pd.DataFrame) -> list[pd.DataFrame]:
    rows = group.sort_values("frame").reset_index(drop=True)
    chunks: list[pd.DataFrame] = []
    start = 0
    frames = rows.frame.to_numpy(dtype=int)
    owners = rows.owner.to_numpy(dtype=int)
    for i in range(1, len(rows) + 1):
        if (i == len(rows) or frames[i] != frames[i - 1] + 1
                or owners[i] != owners[i - 1]):
            chunks.append(rows.iloc[start:i])
            start = i
    return chunks


def discover(labels: np.ndarray, points: pd.DataFrame,
             params: Params = Params()) -> list[dict]:
    strong = _strong_points(points, labels)
    candidates = []
    for track_id, group in strong.groupby("track_id", sort=True):
        runs = _runs(group)
        for early, middle, later in zip(runs, runs[1:], runs[2:]):
            owner, wrong = int(early.owner.iloc[0]), int(middle.owner.iloc[0])
            if (owner <= 0 or wrong <= 0 or owner == wrong
                    or int(later.owner.iloc[0]) != owner):
                continue
            path = pd.concat([early.tail(4), middle, later.head(4)])
            jumps = np.hypot(np.diff(path.x), np.diff(path.y))
            radii = np.maximum(path.radius_px.to_numpy(float)[:-1], 1.0)
            step_p95 = float(np.percentile(jumps / radii, 95))
            if step_p95 > params.maximum_step_p95_radii:
                continue
            candidates.append({
                "track_id": int(track_id), "owner": owner, "wrong_owner": wrong,
                "early_first_frame": int(early.frame.min()) + 1,
                "early_last_frame": int(early.frame.max()) + 1,
                "early_frames": len(early),
                "middle_first_frame": int(middle.frame.min()) + 1,
                "middle_last_frame": int(middle.frame.max()) + 1,
                "middle_frames": len(middle),
                "later_first_frame": int(later.frame.min()) + 1,
                "later_last_frame": int(later.frame.max()) + 1,
                "later_frames": len(later),
                "step_p95_radii": step_p95,
                "early": early, "middle": middle, "later": later,
            })
    primaries = [proposal for proposal in candidates
                 if (proposal["early_frames"] >= params.minimum_bookend_frames
                     and proposal["middle_frames"] >= params.minimum_middle_frames
                     and proposal["later_frames"] >= params.minimum_bookend_frames)]
    linked = []
    for proposal in candidates:
        if (not 1 <= proposal["middle_frames"] <= params.maximum_linked_wrong_frames
                or proposal["early_frames"] < params.minimum_linked_bookend_frames
                or proposal["later_frames"] < params.minimum_linked_bookend_frames):
            continue
        for primary in primaries:
            if (proposal["track_id"] != primary["track_id"]
                    or proposal["owner"] != primary["owner"]
                    or proposal["wrong_owner"] != primary["wrong_owner"]
                    or proposal["middle_first_frame"] <= primary["middle_last_frame"]):
                continue
            between = strong[(strong.track_id.eq(proposal["track_id"]))
                             & (strong.frame.ge(primary["middle_last_frame"] - 1))
                             & (strong.frame.le(proposal["middle_first_frame"] - 1))]
            frames = between.frame.to_numpy(int)
            if (len(frames) < 2 or np.any(np.diff(frames) != 1)):
                continue
            jumps = np.hypot(np.diff(between.x), np.diff(between.y))
            radii = np.maximum(between.radius_px.to_numpy(float)[:-1], 1.0)
            if float(np.percentile(jumps/radii, 95)) > params.maximum_step_p95_radii:
                continue
            proposal["linked_short"] = True
            linked.append(proposal)
            break
    for primary in primaries:
        primary["linked_short"] = False
    return sorted(primaries + linked,
                  key=lambda proposal: (proposal["track_id"],
                                        proposal["middle_first_frame"]))


def _temporal_companion_partition(labels: np.ndarray, raw: np.ndarray,
                                  t: int, mask: np.ndarray, target,
                                  wrong: int, params: Params
                                  ) -> np.ndarray | None:
    """Split a missed second soma using its adjacent-frame named outline."""
    candidates = []
    for neighbor in (t-1, t+1):
        if not 0 <= neighbor < len(labels):
            continue
        parts, count = ndi.label(labels[neighbor] == wrong, STRUCTURE)
        for index in range(1, count + 1):
            prior = parts == index
            area = int(prior.sum())
            if area < partition.Params().minimum_partition_pixels:
                continue
            candidate_pixels = mask & ndi.binary_dilation(prior, STRUCTURE)
            if (int(candidate_pixels.sum())
                    < params.minimum_temporal_companion_overlap_ratio * area):
                continue
            yy, xx = np.where(candidate_pixels)
            distance = np.hypot(xx - float(target.x), yy - float(target.y))
            eligible = distance >= 4.0
            yy, xx = yy[eligible], xx[eligible]
            if len(xx):
                candidates.append((neighbor, area, yy, xx))
    if len(candidates) != 1:
        return None
    _, reference_area, yy, xx = candidates[0]
    smoothed = ndi.gaussian_filter(raw[t].astype(np.float32),
                                   partition.Params().watershed_sigma_px)
    order = np.argsort(smoothed[yy, xx])[::-1]
    for i in order:
        companion = type("SomaSeed", (), {"x": int(xx[i]), "y": int(yy[i])})()
        selected = partition._partition(mask, raw[t], target, companion,
                                        partition.Params())
        if selected is None:
            continue
        companion_area = int((mask & ~selected).sum())
        ratio = companion_area / reference_area
        if (1 / params.maximum_temporal_companion_area_ratio <= ratio
                <= params.maximum_temporal_companion_area_ratio):
            return selected
    return None


def _proposal_masks(labels: np.ndarray, raw: np.ndarray,
                    protected: np.ndarray, strong_by_frame: dict,
                    proposal: dict, params: Params) -> tuple[list[dict], str]:
    track = proposal["track_id"]
    owner, wrong = proposal["owner"], proposal["wrong_owner"]
    reference_areas = []
    for row in pd.concat([proposal["early"].tail(4),
                          proposal["later"].head(4)]).itertuples(index=False):
        mask = partition._at_component(labels[int(row.frame)], owner, row)
        if mask is None:
            return [], "bookend_component_missing"
        reference_areas.append(int(mask.sum()))
    reference_area = float(np.median(reference_areas))
    records = []
    for row in proposal["middle"].itertuples(index=False):
        t = int(row.frame)
        mask = partition._at_component(labels[t], wrong, row)
        if mask is None:
            return [], "middle_component_missing"
        others = [other for other in strong_by_frame[t].itertuples(index=False)
                  if int(other.track_id) != track]
        companions = [other for other in others
                      if partition._inside(mask, other)]
        if any(int(other.owner) != wrong for other in companions):
            return [], "shared_outline_has_mixed_existing_names"
        if len(companions) > 1:
            selected = partition._partition_many(
                mask, raw[t], row, companions, partition.Params())
            method = "multi_soma_raw_watershed"
        elif companions:
            selected = partition._partition(
                mask, raw[t], row, companions[0], partition.Params())
            method = "two_soma_raw_watershed"
        else:
            if proposal.get("linked_short"):
                selected = _temporal_companion_partition(
                    labels, raw, t, mask, row, wrong, params)
                method = "adjacent_outline_raw_watershed"
            else:
                if partition._unexpected_second_core(mask):
                    return [], "untracked_second_core"
                selected = mask
                method = "single_soma_component"
        if selected is None:
            return [], "raw_soma_partition_unsupported"
        size = int(selected.sum())
        if (not params.minimum_basin_area_ratio
                <= size / max(reference_area, 1.0)
                <= params.maximum_basin_area_ratio):
            return [], "middle_soma_area_inconsistent"
        if np.any(selected & protected[t]):
            return [], "accepted_pixel_in_middle_soma"
        records.append({"frame": t, "mask": selected, "method": method,
                        "changed_pixels": size})
    return records, "supported"


def _separate_transient_remnants(labels: np.ndarray, records: list[dict],
                              owner: int, strong_by_frame: dict,
                              params: Params) -> tuple[bool, int, int]:
    """Bound leftover names by area, distance and holder-track duration."""
    component_count, pixel_count = 0, 0
    target_pixels = sum(int(record["changed_pixels"]) for record in records)
    strong_holder_tracks: dict[int, set[int]] = {}
    strong_holder_frames: set[int] = set()
    for record in records:
        t = int(record["frame"])
        target = record["mask"]
        ty, tx = np.where(target)
        target_radius = np.sqrt(len(tx)/np.pi)
        parts, count = ndi.label(labels[t] == owner, STRUCTURE)
        for index in range(1, count + 1):
            remnant = parts == index
            yy, xx = np.where(remnant)
            area = len(xx)
            if not area:
                continue
            distance = np.hypot(float(xx.mean()-tx.mean()),
                                float(yy.mean()-ty.mean()))
            scale = target_radius + np.sqrt(area/np.pi)
            area_ratio = area / max(len(tx), 1)
            needed_separation = (params.minimum_large_remnant_separation_sum_radii
                                 if area_ratio > params.large_remnant_area_ratio
                                 else params.minimum_remnant_separation_sum_radii)
            if (area_ratio > params.maximum_individual_remnant_area_ratio
                    or distance < needed_separation*scale):
                return False, component_count, pixel_count
            for point in strong_by_frame.get(t, pd.DataFrame()).itertuples(index=False):
                x, y = round(float(point.x)), round(float(point.y))
                if 0 <= x < remnant.shape[1] and 0 <= y < remnant.shape[0] and remnant[y, x]:
                    strong_holder_tracks.setdefault(int(point.track_id),set()).add(t)
                    strong_holder_frames.add(t)
            component_count += 1
            pixel_count += area
    if (pixel_count > params.maximum_total_remnant_area_ratio*target_pixels
            or len(strong_holder_frames) > params.maximum_strong_holder_frames
            or any(len(frames) > params.maximum_strong_holder_track_frames
                   for frames in strong_holder_tracks.values())):
        return False, component_count, pixel_count
    return True, component_count, pixel_count


def _alternate_remnant_owner(labels: np.ndarray, strong: pd.DataFrame,
                             t: int, mask: np.ndarray, restored: int,
                             params: Params) -> tuple[int, str]:
    """Choose one established nearby or prior physical owner for a fragment."""
    frame = labels[t]
    current = strong[strong.frame.eq(t)]
    history_names = set()
    for point in current.itertuples(index=False):
        x, y = round(float(point.x)), round(float(point.y))
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]
                and mask[y, x]):
            continue
        past = strong[(strong.track_id.eq(int(point.track_id)))
                      & (strong.frame.ge(t-params.prior_owner_window_frames))
                      & (strong.frame.lt(t))]
        positive = past.loc[past.owner.gt(0), "owner"]
        if len(positive) >= params.minimum_prior_owner_frames:
            names = set(map(int, positive))
            if restored in names or len(names) != 1:
                return 0, "conflicting_prior_physical_owner"
            history_names |= names
    if len(history_names) > 1:
        return 0, "conflicting_prior_physical_owners"
    if history_names:
        return history_names.pop(), "sustained_prior_physical_owner"

    yy, xx = np.where(mask)
    radius = np.sqrt(len(xx) / np.pi)
    maximum_distance = (params.remnant_link_radius_multiplier*radius
                        + params.remnant_link_pixel_allowance)
    candidates = []
    for name in np.unique(frame):
        name = int(name)
        if name in (0, restored):
            continue
        other = frame == name
        if np.count_nonzero(other) < len(xx):
            continue
        distance = float(ndi.distance_transform_edt(~other)[mask].min())
        if distance <= maximum_distance:
            candidates.append((distance, name))
    candidates.sort()
    if not candidates:
        return 0, "no_nearby_established_owner"
    if (len(candidates) > 1 and candidates[1][0]
            < params.minimum_second_best_distance_ratio*candidates[0][0]):
        return 0, "ambiguous_nearby_owner"
    return candidates[0][1], "unique_nearby_established_owner"


def _remnant_reassignments(labels: np.ndarray, records: list[dict],
                           owner: int, protected: np.ndarray,
                           strong: pd.DataFrame,
                           params: Params) -> tuple[list[dict], str]:
    reassignments = []
    for record in records:
        t = int(record["frame"])
        parts, count = ndi.label(labels[t] == owner, STRUCTURE)
        for index in range(1, count + 1):
            mask = parts == index
            if np.any(mask & protected[t]):
                return [], "accepted_pixel_in_remnant"
            alternate, method = _alternate_remnant_owner(
                labels, strong, t, mask, owner, params)
            if alternate <= 0:
                return [], method
            reassignments.append({
                "frame": t, "mask": mask, "owner": alternate,
                "wrong_owner": owner, "method": method,
                "changed_pixels": int(mask.sum()),
            })
    return reassignments, "supported"


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray,
            params: Params = Params()) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or labels.shape != protected.shape
            or raw.shape[0] < len(labels) or raw.shape[1:] != labels.shape[1:]):
        raise ValueError("labels, raw and protected masks must align")
    result = labels.copy()
    occupied = np.zeros_like(labels, bool)
    strong = _strong_points(points, labels)
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    audits, applications = [], []
    applied_primaries: set[tuple[int, int, int]] = set()
    for proposal in discover(labels, points, params):
        key = (proposal["track_id"], proposal["owner"],
               proposal["wrong_owner"])
        if proposal.get("linked_short") and key not in applied_primaries:
            audit = {name: value for name, value in proposal.items()
                     if name not in ("early", "middle", "later")}
            audit.update({
                "outcome": "rejected", "reason": "primary_contact_not_applied",
                "untracked_remnant_components": 0,
                "untracked_remnant_pixels": 0,
                "reassigned_remnant_components": 0,
                "reassigned_remnant_pixels": 0,
                "changed_pixels": 0,
            })
            audits.append(audit)
            continue
        records, reason = _proposal_masks(labels, raw, protected, by_frame,
                                          proposal, params)
        if reason == "supported" and any(
                np.any(occupied[row["frame"]] & row["mask"])
                for row in records):
            reason = "overlaps_other_proposal"
        remnant_count, remnant_pixels = 0, 0
        reassignments = []
        if reason == "supported":
            supported, remnant_count, remnant_pixels = _separate_transient_remnants(
                labels, records, proposal["owner"], by_frame, params)
            if not supported:
                reason = "duplicate_identity_has_large_near_or_persistent_holder"
        if reason == "supported":
            reassignments, reason = _remnant_reassignments(
                labels, records, proposal["owner"], protected, strong, params)
        if reason == "supported":
            trial = result.copy()
            for row in records:
                trial[row["frame"]][row["mask"]] = proposal["owner"]
            for row in reassignments:
                trial[row["frame"]][row["mask"]] = row["owner"]
            if any(np.any((trial[row["frame"]] == proposal["owner"])
                          & ~row["mask"]) for row in records):
                reason = "restored_name_remains_on_other_component"
            if reason == "supported":
                result = trial
                if not proposal.get("linked_short"):
                    applied_primaries.add(key)
                for row in records:
                    occupied[row["frame"]] |= row["mask"]
                    applications.append({
                        "track_id": proposal["track_id"],
                        "owner": proposal["owner"],
                        "wrong_owner": proposal["wrong_owner"],
                        "frame": row["frame"] + 1,
                        "method": row["method"],
                        "changed_pixels": row["changed_pixels"],
                    })
                for row in reassignments:
                    occupied[row["frame"]] |= row["mask"]
                    applications.append({
                        "track_id": proposal["track_id"],
                        "owner": row["owner"],
                        "wrong_owner": row["wrong_owner"],
                        "frame": row["frame"] + 1,
                        "method": row["method"],
                        "changed_pixels": row["changed_pixels"],
                    })
        audit = {key: value for key, value in proposal.items()
                 if key not in ("early", "middle", "later")}
        audit.update({
            "outcome": "applied" if reason == "supported" else "rejected",
            "reason": reason,
            "untracked_remnant_components": remnant_count,
            "untracked_remnant_pixels": remnant_pixels,
            "reassigned_remnant_components": len(reassignments) if reason == "supported" else 0,
            "reassigned_remnant_pixels": sum(row["changed_pixels"] for row in reassignments) if reason == "supported" else 0,
            "changed_pixels": (sum(row["changed_pixels"] for row in records + reassignments)
                               if reason == "supported" else 0),
        })
        audits.append(audit)
    if (not np.array_equal(labels > 0, result > 0)
            or np.any((labels != result) & protected)
            or set(map(int, np.unique(result))) - set(map(int, np.unique(labels)))):
        raise AssertionError("bookended owner rule crossed parent boundary")
    return result, pd.DataFrame(audits), pd.DataFrame(applications)
