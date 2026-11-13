"""Split an established soma from an outline with a persistent host seat.

The established multi-resident rule required the host soma to have an
independently named run. This sibling experiment also accepts a host track
that stays strong and consistently bears the host name through a long contact.
All producer discovery remains field-wide and independent of feedback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import complete_owner_handoff_episodes_r10_a002 as body
import owner_consensus
import recover_stable_dual_soma_contacts as dual
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_contact_run_frames: int = 3
    maximum_resident_seats: int = 3
    minimum_reference_area_ratio: float = 0.25
    maximum_reference_area_ratio: float = 4.0
    minimum_reference_area_percentile: float = 0.6
    minimum_persistent_host_frames: int = 20
    minimum_persistent_host_run: int = 12
    minimum_persistent_host_dominance: float = 3.0


def _persistent_host_owners(scored: pd.DataFrame, anchors: dict,
                            params: Params) -> dict[int, int]:
    """A durable physical track can establish which soma retains the host name."""
    eligible = scored[scored.physically_visible.astype(bool)
                      & scored.strong.astype(bool)
                      & scored.state.astype(str).eq("observed")]
    owners = {}
    for track_id, group in eligible.groupby("track_id"):
        track_id = int(track_id)
        counts = group[group.accepted_owner.astype(int).gt(0)] \
            .accepted_owner.astype(int).value_counts()
        if counts.empty:
            continue
        owner, support = int(counts.index[0]), int(counts.iloc[0])
        second = int(counts.iloc[1]) if len(counts) > 1 else 0
        if (support < params.minimum_persistent_host_frames
                or support < params.minimum_persistent_host_dominance * max(1, second)
                or (track_id in anchors and int(anchors[track_id]["owner"]) != owner)):
            continue
        frames = sorted(group[group.accepted_owner.astype(int).eq(owner)]
                        .frame.astype(int).unique())
        run, longest, previous = 0, 0, None
        for frame in frames:
            run = run + 1 if previous is not None and frame == previous + 1 else 1
            longest = max(longest, run)
            previous = frame
        if longest >= params.minimum_persistent_host_run:
            owners[track_id] = owner
    return owners


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray | None = None,
            params: Params = Params(),
            anchor_params: dual.Params = dual.Params(),
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
    anchors = dual._references(scored, anchor_params)
    persistent_hosts = _persistent_host_owners(scored, anchors, params)
    reference_areas = {
        int(track_id): dual._reference_area(
            labels, scored[scored.track_id.astype(int).eq(int(track_id))],
            int(anchor["owner"]), anchor)
        for track_id, anchor in anchors.items()
    }
    area_population = np.array([area for area in reference_areas.values()
                                if area > 0], float)
    visible = scored[scored.physically_visible & scored.strong
                     & scored.state.astype(str).eq("observed")]
    by_frame = {int(t): group for t, group in visible.groupby("frame")}
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
        if (not len(area_population) or float(np.mean(area_population <= area))
                < params.minimum_reference_area_percentile):
            continue
        proposals, refusals = {}, {}
        for row in group.itertuples(index=False):
            t, host = int(row.frame), int(row.accepted_owner)
            if (host <= 0 or host == owner or not bool(row.strong)
                    or not bool(row.physically_visible)
                    or str(row.state) != "observed"):
                continue
            source = candidate[t]
            mask = body._at_component(source, host, row)
            if mask is None:
                continue
            seats = [other for other in by_frame.get(t, pd.DataFrame()).itertuples(index=False)
                     if int(other.track_id) != track_id and body._inside(mask, other)
                     and body._distinct(row, other, partition_params)]
            if not 2 <= len(seats) <= params.maximum_resident_seats:
                continue
            if (any(int(other.accepted_owner) != host for other in seats)
                    or any(not body._distinct(a, b, partition_params)
                           for i, a in enumerate(seats) for b in seats[i + 1:])):
                refusals[t] = "resident_seats_not_independent_or_not_shared"
                continue
            residents = [other for other in seats
                         if ((int(other.track_id) in anchors
                              and int(anchors[int(other.track_id)]["owner"]) == host
                              and int(anchors[int(other.track_id)]["support_frames"])
                              >= anchor_params.minimum_host_owner_frames)
                             or persistent_hosts.get(int(other.track_id)) == host)]
            if not residents:
                refusals[t] = "host_name_has_no_independent_resident"
                continue
            if any(int(other.accepted_owner) == owner
                   and body._distinct(row, other, partition_params)
                   for other in by_frame.get(t, pd.DataFrame()).itertuples(index=False)):
                refusals[t] = "reference_owner_on_distinct_soma"
                continue
            selected = body._partition_many(mask, raw[t], row, seats, partition_params)
            if selected is None:
                refusals[t] = "multi_soma_raw_partition_failed"
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
            proposals[t] = (selected, host, size, len(seats))

        selected_frames = set()
        frames = sorted(proposals)
        run = []
        for t in [*frames, -2]:
            if run and t != run[-1] + 1:
                if len(run) >= params.minimum_contact_run_frames:
                    selected_frames.update(run)
                run = []
            if t >= 0:
                run.append(t)
        for t in sorted(selected_frames):
            selected, host, size, residents = proposals[t]
            candidate[t][selected] = owner
            occupied[t][selected] = True
            applications.append({
                "track_id": track_id, "reference_owner": owner,
                "former_owner": host, "frame": t + 1,
                "changed_pixels": size, "resident_seats": residents,
                "reference_area_px": area,
                "independent_reference_frames": anchor["support_frames"],
            })
        if proposals or refusals:
            audits.append({
                "track_id": track_id, "reference_owner": owner,
                "independent_reference_frames": anchor["support_frames"],
                "reference_area_px": area,
                "eligible_contact_frames": len(proposals),
                "applied_frames": len(selected_frames),
                "changed_pixels": sum(proposals[t][2] for t in selected_frames),
                "refusals": "|".join(f"{t + 1}:{reason}" for t, reason in sorted(refusals.items())),
            })

    if (not np.array_equal(labels > 0, candidate > 0)
            or np.any((candidate != labels) & protected)
            or not np.array_equal(np.unique(labels), np.unique(candidate))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("multi-resident layer crossed an accepted boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
