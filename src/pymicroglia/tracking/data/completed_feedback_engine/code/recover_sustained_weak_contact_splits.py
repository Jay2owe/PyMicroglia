"""Split a sustained two-soma contact through weak but visible point frames.

Discovery uses physical tracks, accepted labels and raw images across the
whole field. Feedback, movie IDs, coordinates and selected identities are
not producer inputs. Earlier accepted pixels and unsafe contact frames stay
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import complete_owner_handoff_episodes_r10_a002 as body
import owner_consensus
import recover_stable_dual_soma_contacts as dual
import recover_terminal_name_switch_contacts_lazy_areas as contact_rule
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_independent_owner_frames: int = 6
    minimum_reference_run_frames: int = 6
    minimum_host_owner_frames: int = 10
    minimum_structural_run_frames: int = 5
    minimum_eligible_run_frames: int = 5
    minimum_strong_partition_frames: int = 3
    minimum_weak_visible_frames: int = 1
    minimum_eligible_coverage: float = .8
    maximum_weak_to_strong_gap_frames: int = 1
    minimum_temporal_basin_jaccard: float = .6


def _runs(records: list[dict]) -> list[list[dict]]:
    runs = []
    for record in sorted(records, key=lambda item: item["frame"]):
        if (not runs or record["frame"] != runs[-1][-1]["frame"] + 1):
            runs.append([])
        runs[-1].append(record)
    return runs


def _near_trusted(frame: int, trusted: set[int], distance: int) -> bool:
    return any(abs(frame - other) <= distance for other in trusted)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def _temporal_contact(record: dict, before: dict, after: dict,
                      frame_labels: np.ndarray, frame_raw: np.ndarray,
                      protected: np.ndarray, reference_areas: dict[int, float],
                      params: Params) -> dict | None:
    """Use two trusted flanking basins as a broad marker in raw watershed."""
    if (record["trusted"] or record["contact"] is not None
            or before["frame"] != record["frame"]-1
            or after["frame"] != record["frame"]+1
            or not before["trusted"] or not after["trusted"]
            or before["contact"] is None or after["contact"] is None):
        return None
    mask = record["mask"]
    tracks, names, host = record["tracks"], record["names"], record["host"]
    if (np.any(mask & protected)
            or any(np.any(frame_labels[~mask] == name) for name in names)):
        return None
    target_idx = 1-names.index(host)
    host_idx = 1-target_idx
    current_points = record["points"]
    forecasts = []
    for neighbor in (before, after):
        if (neighbor["tracks"] != tracks or neighbor["names"] != names
                or neighbor["host"] != host):
            return None
        old = neighbor["points"][target_idx]
        now = current_points[target_idx]
        basin = neighbor["contact"]["basins"][target_idx]
        moved = ndi.shift(basin.astype(np.uint8),
                          shift=(float(now.y)-float(old.y),
                                 float(now.x)-float(old.x)),
                          order=0, mode="constant") > 0
        forecasts.append(moved & mask)
    core = ndi.binary_erosion(forecasts[0] & forecasts[1],
                              structure=body.STRUCTURE, iterations=1)
    target_seed = physical._marker_pixel(
        mask, float(current_points[target_idx].x),
        float(current_points[target_idx].y))
    host_seed = physical._marker_pixel(
        mask, float(current_points[host_idx].x),
        float(current_points[host_idx].y))
    if (target_seed is None or host_seed is None or not core[target_seed]
            or core[host_seed]):
        return None
    components, count = ndi.label(core, body.STRUCTURE)
    if count == 0:
        return None
    core = components == components[target_seed]
    markers = np.zeros(mask.shape, np.uint8)
    markers[core] = 1
    markers[host_seed] = 2
    assignment = watershed(
        -ndi.gaussian_filter(frame_raw.astype(np.float32), 1.0),
        markers=markers, mask=mask, connectivity=body.STRUCTURE)
    target_basin = assignment == 1
    host_basin = assignment == 2
    if (not np.array_equal(target_basin | host_basin, mask)
            or ndi.label(target_basin, body.STRUCTURE)[1] != 1
            or ndi.label(host_basin, body.STRUCTURE)[1] != 1
            or any(_jaccard(target_basin, forecast)
                   < params.minimum_temporal_basin_jaccard
                   for forecast in forecasts)):
        return None
    basins = [None, None]
    basins[target_idx] = target_basin
    basins[host_idx] = host_basin
    areas = [int(basin.sum()) for basin in basins]
    for area, track in zip(areas, tracks):
        reference = reference_areas[track]
        if not .25 <= area/reference <= 4.0:
            return None
    return {"host": host, "tracks": tracks, "names": names,
            "mask": mask, "basins": basins, "areas": areas,
            "temporal_support": True}


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params(),
            partition_params: body.Params = body.Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected pixels must align")
    scored = physical.attach_owners(points, labels)
    for field in ("strong", "physically_visible", "in_encounter"):
        scored[field] = scored[field].astype(bool)
    reference_params = dual.Params(
        minimum_independent_owner_frames=params.minimum_independent_owner_frames,
        minimum_reference_run_frames=params.minimum_reference_run_frames)
    anchors = dual._references(scored, reference_params)
    by_track = {int(track): group for track, group in scored.groupby("track_id")}
    reference_areas: dict[int, float] = {}
    visible_by_frame = {
        int(frame): list(group.itertuples(index=False))
        for frame, group in scored[scored.physically_visible].groupby("frame")}
    structural: dict[tuple[int, int, int], list[dict]] = {}
    for frame, visible in visible_by_frame.items():
        by_host: dict[int, list] = {}
        for point in visible:
            host = int(point.accepted_owner)
            if host > 0:
                by_host.setdefault(host, []).append(point)
        for host, host_points in by_host.items():
            if len(host_points) < 2:
                continue
            components, _ = ndi.label(labels[frame] == host, body.STRUCTURE)
            by_component: dict[int, list] = {}
            for point in host_points:
                component = physical._component_at(
                    components, float(point.x), float(point.y), search_radius=6)
                if component:
                    by_component.setdefault(int(component), []).append(point)
            for component, occupants in by_component.items():
                if len(occupants) != 2:
                    continue
                occupants = sorted(occupants, key=lambda point: int(point.track_id))
                tracks = tuple(int(point.track_id) for point in occupants)
                if (tracks[0] == tracks[1] or any(track not in anchors for track in tracks)):
                    continue
                names = tuple(int(anchors[track]["owner"]) for track in tracks)
                if names[0] == names[1] or host not in names:
                    continue
                host_track = tracks[names.index(host)]
                if int(anchors[host_track]["support_frames"]) < params.minimum_host_owner_frames:
                    continue
                target_track = tracks[1-names.index(host)]
                if int(anchors[target_track]["last_reference_frame"]) >= frame:
                    continue
                mask = components == component
                if (any(not body._inside(mask, point) for point in occupants)
                        or {int(point.track_id) for point in visible
                            if body._inside(mask, point)} != set(tracks)
                        or not body._distinct(occupants[0], occupants[1], partition_params)):
                    continue
                for track in tracks:
                    if track not in reference_areas:
                        anchor = anchors[track]
                        reference_areas[track] = dual._reference_area(
                            labels, by_track[track], int(anchor["owner"]), anchor)
                raw_partition = body._partition(
                    mask, raw[frame], occupants[0], occupants[1], partition_params)
                if raw_partition is None:
                    continue
                trusted = all(bool(point.strong) and str(point.state) == "observed"
                              for point in occupants)
                eligible = contact_rule._contact(
                    labels[frame], raw[frame], occupants, visible, anchors,
                    reference_areas, protected[frame], partition_params,
                    contact_rule.Params())
                key = (tracks[0], tracks[1], host)
                structural.setdefault(key, []).append({
                    "frame": frame, "tracks": tracks, "names": names,
                    "host": host, "trusted": trusted, "contact": eligible,
                    "mask": mask, "points": occupants})

    proposals: list[list[dict]] = []
    audits = []
    for key, records in structural.items():
        for run in _runs(records):
            if len(run) < params.minimum_structural_run_frames:
                continue
            trusted = {record["frame"] for record in run if record["trusted"]}
            if (len(trusted) < params.minimum_strong_partition_frames
                    or len(run)-len(trusted) < params.minimum_weak_visible_frames):
                continue
            for before, record, after in zip(run[:-2], run[1:-1], run[2:]):
                if record["contact"] is None:
                    frame = record["frame"]
                    record["contact"] = _temporal_contact(
                        record, before, after, labels[frame], raw[frame],
                        protected[frame], reference_areas, params)
            eligible_runs = _runs([record for record in run
                                   if record["contact"] is not None])
            for eligible_run in eligible_runs:
                if (len(eligible_run) < params.minimum_eligible_run_frames
                        or len(eligible_run) / len(run) < params.minimum_eligible_coverage):
                    continue
                if any(not record["trusted"] and (
                        not _near_trusted(record["frame"] - 1, trusted,
                                          params.maximum_weak_to_strong_gap_frames)
                        or not _near_trusted(record["frame"] + 1, trusted,
                                             params.maximum_weak_to_strong_gap_frames))
                       for record in eligible_run):
                    continue
                proposals.append(eligible_run)
                audits.append({"tracks": "|".join(map(str, key[:2])),
                               "host": key[2], "structural_first_frame": run[0]["frame"]+1,
                               "structural_last_frame": run[-1]["frame"]+1,
                               "structural_frames": len(run),
                               "eligible_first_frame": eligible_run[0]["frame"]+1,
                               "eligible_last_frame": eligible_run[-1]["frame"]+1,
                               "eligible_frames": len(eligible_run),
                               "trusted_frames": len(trusted)})

    # Concurrent proposals sharing a pixel are ambiguous: reject both.
    candidate = labels.copy()
    applications = []
    for index, run in enumerate(proposals):
        if any(record["frame"] == other["frame"]
               and np.any(record["mask"] & other["mask"])
               for other_index, other_run in enumerate(proposals)
               if other_index != index
               for record in run for other in other_run):
            continue
        for record in run:
            frame = record["frame"]
            contact = record["contact"]
            assert contact is not None
            for name, basin in zip(contact["names"], contact["basins"]):
                candidate[frame][basin] = name
            applications.append({"frame": frame+1, "host": contact["host"],
                                 "tracks": "|".join(map(str, contact["tracks"])),
                                 "names": "|".join(map(str, contact["names"])),
                                 "changed_pixels": int(np.count_nonzero(
                                     candidate[frame][record["mask"]]
                                     != labels[frame][record["mask"]])),
                                 "trusted_points": record["trusted"],
                                 "temporal_support": bool(contact.get("temporal_support", False))})
    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("sustained weak contact layer crossed accepted boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
