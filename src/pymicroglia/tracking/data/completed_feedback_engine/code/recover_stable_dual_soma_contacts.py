"""Recover a stable second identity through sustained shared-outline contacts.

This is a post-R13 layer. It discovers contacts from physical tracks, current
labels and raw signal across the field; feedback is not a producer input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import complete_owner_handoff_episodes_r10_a002 as body
import owner_consensus
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_independent_owner_frames: int = 8
    minimum_reference_run_frames: int = 5
    minimum_host_owner_frames: int = 10
    minimum_owner_dominance: float = 2.0
    minimum_contact_run_frames: int = 3
    minimum_reference_area_ratio: float = 0.25
    maximum_reference_area_ratio: float = 4.0
    minimum_reference_area_percentile: float = 0.0


def _longest_consecutive(frames: list[int]) -> int:
    if not frames:
        return 0
    best = run = 1
    for before, after in zip(frames[:-1], frames[1:]):
        run = run + 1 if after == before + 1 else 1
        best = max(best, run)
    return best


def _references(scored: pd.DataFrame, params: Params) -> dict[int, dict]:
    """Use an independently observed run to identify a physical soma."""
    anchors = {}
    evidence = scored[(scored.state.astype(str) == "observed")
                      & scored.strong.astype(bool)
                      & scored.physically_visible.astype(bool)
                      & ~scored.in_encounter.astype(bool)
                      & scored.accepted_owner.astype(int).gt(0)]
    for track_id, group in evidence.groupby("track_id", sort=True):
        counts = group.accepted_owner.astype(int).value_counts()
        if counts.empty or int(counts.iloc[0]) < params.minimum_independent_owner_frames:
            continue
        second = int(counts.iloc[1]) if len(counts) > 1 else 0
        if second and int(counts.iloc[0]) < params.minimum_owner_dominance * second:
            continue
        owner = int(counts.index[0])
        rows = group[group.accepted_owner.astype(int).eq(owner)]
        frames = sorted(rows.frame.astype(int).tolist())
        if _longest_consecutive(frames) < params.minimum_reference_run_frames:
            continue
        anchors[int(track_id)] = {
            "owner": owner, "support_frames": int(counts.iloc[0]),
            "first_reference_frame": frames[0], "last_reference_frame": frames[-1],
        }
    return anchors


def _reference_area(labels: np.ndarray, group: pd.DataFrame, owner: int,
                    anchor: dict) -> float:
    areas = []
    for row in group.itertuples(index=False):
        if (int(row.accepted_owner) != owner or not bool(row.strong)
                or bool(row.in_encounter) or str(row.state) != "observed"):
            continue
        mask = body._at_component(labels[int(row.frame)], owner, row)
        if mask is not None:
            areas.append(int(mask.sum()))
    return float(np.median(areas)) if areas else 0.0


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray | None = None,
            params: Params = Params(),
            partition_params: body.Params = body.Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape:
        raise ValueError("raw and labels must align")
    if protected is None:
        protected = np.zeros_like(labels, bool)
    if protected.shape != labels.shape:
        raise ValueError("protected pixels must align")

    scored = physical.attach_owners(points, labels)
    for field in ("strong", "physically_visible", "in_encounter"):
        scored[field] = scored[field].astype(bool)
    anchors = _references(scored, params)
    reference_areas = {
        int(track_id): _reference_area(
            labels, scored[scored.track_id.astype(int).eq(int(track_id))],
            int(anchor["owner"]), anchor)
        for track_id, anchor in anchors.items()
    }
    area_population = np.array([area for area in reference_areas.values()
                                if area > 0], float)
    by_frame = {int(t): group for t, group in scored[
        scored.physically_visible & scored.strong
        & scored.state.astype(str).eq("observed")].groupby("frame")}
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    audits, applications = [], []

    for track_id, group in scored.groupby("track_id", sort=True):
        track_id = int(track_id)
        if track_id not in anchors:
            continue
        anchor = anchors[track_id]
        owner = int(anchor["owner"])
        area = reference_areas[track_id]
        if area <= 0:
            continue
        if (params.minimum_reference_area_percentile > 0
                and (not len(area_population) or float(np.mean(area_population <= area))
                     < params.minimum_reference_area_percentile)):
            continue
        proposals = {}
        refusals = {}
        for row in group.itertuples(index=False):
            t, host = int(row.frame), int(row.accepted_owner)
            if (host <= 0 or host == owner or not bool(row.strong)
                    or not bool(row.physically_visible)
                    or str(row.state) != "observed"):
                continue
            source = candidate[t]
            mask = body._at_component(source, host, row)
            if mask is None:
                refusals[t] = "host_component_missing"
                continue
            seats = [other for other in by_frame.get(t, pd.DataFrame()).itertuples(index=False)
                     if int(other.track_id) != track_id and body._inside(mask, other)
                     and body._distinct(row, other, partition_params)]
            # A contact is resolved only when exactly one independent resident
            # soma supplies the host name. Extra somata require a joint solver.
            residents = [other for other in seats
                         if int(other.track_id) in anchors
                         and int(anchors[int(other.track_id)]["owner"]) == host
                         and int(other.accepted_owner) == host
                         and int(anchors[int(other.track_id)]["support_frames"])
                         >= params.minimum_host_owner_frames]
            if len(seats) != 1 or len(residents) != 1:
                refusals[t] = "ambiguous_or_missing_resident_seat"
                continue
            if any(int(other.accepted_owner) == owner
                   and body._distinct(row, other, partition_params)
                   for other in by_frame.get(t, pd.DataFrame()).itertuples(index=False)):
                refusals[t] = "reference_owner_on_distinct_soma"
                continue
            selected = body._partition(mask, raw[t], row, residents[0], partition_params)
            if selected is None:
                refusals[t] = "raw_somata_not_separable"
                continue
            size = int(selected.sum())
            if not (params.minimum_reference_area_ratio * area <= size
                    <= params.maximum_reference_area_ratio * area):
                refusals[t] = "basin_area_outside_reference_range"
                continue
            if np.any(protected[t] & selected) or np.any(occupied[t] & selected):
                refusals[t] = "protected_or_prior_pixel"
                continue
            trial = source.copy()
            trial[selected] = owner
            if owner_consensus.count_new_duplicate_components(source[None], trial[None]):
                refusals[t] = "duplicate_soma"
                continue
            proposals[t] = (selected, host, size)

        # A lone marked frame is not enough evidence. Apply each connected run
        # as one episode, leaving unresolved frames explicit in the audit.
        applied = set()
        frames = sorted(proposals)
        run = []
        for t in [*frames, -2]:
            if run and t != run[-1] + 1:
                if len(run) >= params.minimum_contact_run_frames:
                    applied.update(run)
                run = []
            if t >= 0:
                run.append(t)
        for t in sorted(applied):
            selected, host, size = proposals[t]
            candidate[t][selected] = owner
            occupied[t][selected] = True
            applications.append({
                "track_id": track_id, "reference_owner": owner,
                "former_owner": host, "frame": t + 1,
                "changed_pixels": size, "reference_area_px": area,
                "independent_reference_frames": anchor["support_frames"],
            })
        if proposals or refusals:
            audits.append({
                "track_id": track_id, "reference_owner": owner,
                "independent_reference_frames": anchor["support_frames"],
                "reference_area_px": area, "eligible_contact_frames": len(proposals),
                "applied_frames": len(applied),
                "first_applied_frame": min(applied) + 1 if applied else None,
                "last_applied_frame": max(applied) + 1 if applied else None,
                "changed_pixels": sum(proposals[t][2] for t in applied),
                "refusals": "|".join(f"{t + 1}:{reason}" for t, reason in sorted(refusals.items())),
            })

    if (not np.array_equal(labels > 0, candidate > 0)
            or np.any((candidate != labels) & protected)
            or not np.array_equal(np.unique(labels), np.unique(candidate))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("stable dual-soma layer crossed an accepted boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
