"""Restore a bookended soma name beside small untracked name remnants.

The eligibility test uses only physical points and accepted labels. It has
no movie, identity, track, coordinate, frame, region or feedback selector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import complete_owner_handoff_episodes_r10_a002 as partition
import owner_consensus


@dataclass(frozen=True)
class Params:
    minimum_bookend_frames: int = 8
    minimum_middle_frames: int = 8
    maximum_step_p95_radii: float = 0.8
    minimum_basin_area_ratio: float = 0.25
    maximum_basin_area_ratio: float = 4.0
    maximum_individual_remnant_area_ratio: float = 0.30
    maximum_total_remnant_area_ratio: float = 0.25
    minimum_remnant_separation_sum_radii: float = 3.0


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
    proposals = []
    for track_id, group in strong.groupby("track_id", sort=True):
        runs = _runs(group)
        for early, middle, later in zip(runs, runs[1:], runs[2:]):
            owner, wrong = int(early.owner.iloc[0]), int(middle.owner.iloc[0])
            if (owner <= 0 or wrong <= 0 or owner == wrong
                    or int(later.owner.iloc[0]) != owner
                    or len(early) < params.minimum_bookend_frames
                    or len(middle) < params.minimum_middle_frames
                    or len(later) < params.minimum_bookend_frames):
                continue
            path = pd.concat([early.tail(4), middle, later.head(4)])
            jumps = np.hypot(np.diff(path.x), np.diff(path.y))
            radii = np.maximum(path.radius_px.to_numpy(float)[:-1], 1.0)
            step_p95 = float(np.percentile(jumps / radii, 95))
            if step_p95 > params.maximum_step_p95_radii:
                continue
            proposals.append({
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
    return proposals


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
        if any(int(other.owner) == owner for other in others):
            return [], "established_name_has_separate_strong_holder"
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


def _small_untracked_remnants(labels: np.ndarray, records: list[dict],
                              owner: int, strong_by_frame: dict,
                              params: Params) -> tuple[bool, int, int]:
    """Check every original-name component that would become a duplicate."""
    component_count, pixel_count = 0, 0
    target_pixels = sum(int(record["changed_pixels"]) for record in records)
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
            if (area > params.maximum_individual_remnant_area_ratio*len(tx)
                    or distance < params.minimum_remnant_separation_sum_radii*scale):
                return False, component_count, pixel_count
            for point in strong_by_frame.get(t, pd.DataFrame()).itertuples(index=False):
                x, y = round(float(point.x)), round(float(point.y))
                if 0 <= x < remnant.shape[1] and 0 <= y < remnant.shape[0] and remnant[y, x]:
                    return False, component_count, pixel_count
            component_count += 1
            pixel_count += area
    if pixel_count > params.maximum_total_remnant_area_ratio*target_pixels:
        return False, component_count, pixel_count
    return True, component_count, pixel_count


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
    for proposal in discover(labels, points, params):
        records, reason = _proposal_masks(labels, raw, protected, by_frame,
                                          proposal, params)
        if reason == "supported" and any(
                np.any(occupied[row["frame"]] & row["mask"])
                for row in records):
            reason = "overlaps_other_proposal"
        remnant_count, remnant_pixels = 0, 0
        if reason == "supported":
            trial = result.copy()
            for row in records:
                trial[row["frame"]][row["mask"]] = proposal["owner"]
            duplicate_count = owner_consensus.count_new_duplicate_components(result, trial)
            if duplicate_count:
                supported, remnant_count, remnant_pixels = _small_untracked_remnants(
                    labels, records, proposal["owner"], by_frame, params)
                if not supported or duplicate_count > remnant_count:
                    reason = "duplicate_identity_has_substantial_or_tracked_holder"
            if reason == "supported":
                result = trial
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
        audit = {key: value for key, value in proposal.items()
                 if key not in ("early", "middle", "later")}
        audit.update({
            "outcome": "applied" if reason == "supported" else "rejected",
            "reason": reason,
            "untracked_remnant_components": remnant_count,
            "untracked_remnant_pixels": remnant_pixels,
            "changed_pixels": (sum(row["changed_pixels"] for row in records)
                               if reason == "supported" else 0),
        })
        audits.append(audit)
    if (not np.array_equal(labels > 0, result > 0)
            or np.any((labels != result) & protected)
            or set(map(int, np.unique(result))) - set(map(int, np.unique(labels)))):
        raise AssertionError("bookended owner rule crossed parent boundary")
    return result, pd.DataFrame(audits), pd.DataFrame(applications)
