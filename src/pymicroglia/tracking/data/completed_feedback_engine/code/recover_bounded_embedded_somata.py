"""Give a bounded, independently visible embedded soma its own full path.

The discovery has no feedback, movie, cell-name, frame or region selector.
Existing names, foreground, accepted pixels and the unclaimed ledger are inputs
or protected controls; only one new name per qualified physical episode is
added. Acceptance scoring remains outside this producer.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import complete_owner_handoff_episodes_r10_a002 as body
import owner_consensus
import recover_stable_dual_soma_contacts as dual
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_contact_frames: int = 8
    minimum_host_reference_frames: int = 20
    minimum_strong_fraction: float = .8
    minimum_new_identity_frames: int = 10
    minimum_basin_area_ratio: float = .25
    maximum_basin_area_ratio: float = 4.0
    maximum_collapsed_valley_ratio: float = .3
    minimum_forecast_containment: float = .9
    minimum_extension_jaccard: float = .6
    minimum_extension_relative_contrast: float = .5


def _runs(records: list[dict]) -> list[list[dict]]:
    runs = []
    for row in sorted(records, key=lambda item: item["frame"]):
        if not runs or row["frame"] != runs[-1][-1]["frame"] + 1:
            runs.append([])
        runs[-1].append(row)
    return runs


def _shift_basin(basin: np.ndarray, old, new, mask: np.ndarray) -> np.ndarray:
    moved = ndi.shift(basin.astype(np.uint8),
                      shift=(float(new.y)-float(old.y),
                             float(new.x)-float(old.x)),
                      order=0, mode="constant") > 0
    return moved & mask


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    shared = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(shared/union) if union else 0.0


def _contrast(frame: np.ndarray, point) -> float:
    x, y = float(point.x), float(point.y)
    ymin, ymax = max(0, int(y)-11), min(frame.shape[0], int(y)+12)
    xmin, xmax = max(0, int(x)-11), min(frame.shape[1], int(x)+12)
    yy, xx = np.ogrid[ymin:ymax, xmin:xmax]
    radius = np.hypot(xx-x, yy-y)
    crop = frame[ymin:ymax, xmin:xmax].astype(np.float64)
    core = crop[radius <= 2.5]
    ring = crop[(radius >= 6) & (radius <= 10)]
    return float(np.mean(core)-np.median(ring)) if len(core) and len(ring) else 0.0


def _plausible(area: int, reference: float, params: Params) -> bool:
    return reference > 0 and (params.minimum_basin_area_ratio
                              <= area/reference <= params.maximum_basin_area_ratio)


def _broad_marker(record: dict, earlier: dict, later: dict,
                  raw: np.ndarray, reference: float,
                  params: Params) -> np.ndarray | None:
    if (not bool(record["target"].strong)
            or earlier["basin"] is None or later["basin"] is None):
        return None
    mask = record["mask"]
    target, resident = record["target"], record["resident"]
    valley = physical._valley_ratio(raw,
                                    (float(target.x), float(target.y)),
                                    (float(resident.x), float(resident.y)))
    if not np.isfinite(valley) or valley > params.maximum_collapsed_valley_ratio:
        return None
    forecasts = [_shift_basin(other["basin"], other["target"], target, mask)
                 for other in (earlier, later)]
    target_seed = physical._marker_pixel(mask, float(target.x), float(target.y))
    resident_seed = physical._marker_pixel(mask, float(resident.x), float(resident.y))
    if target_seed is None or resident_seed is None:
        return None
    core = forecasts[0] & forecasts[1]
    components, _ = ndi.label(core, body.STRUCTURE)
    component = int(components[target_seed])
    if component <= 0:
        return None
    core = components == component
    if core[resident_seed]:
        return None
    markers = np.zeros(mask.shape, np.uint8)
    markers[core] = 1
    markers[resident_seed] = 2
    assigned = watershed(-ndi.gaussian_filter(raw.astype(np.float32), 1.0),
                         markers=markers, mask=mask, connectivity=body.STRUCTURE)
    basin = assigned == 1
    residual = assigned == 2
    if (not np.array_equal(basin | residual, mask)
            or ndi.label(basin, body.STRUCTURE)[1] != 1
            or ndi.label(residual, body.STRUCTURE)[1] != 1
            or not _plausible(int(basin.sum()), reference, params)):
        return None
    if any(np.count_nonzero(forecast & basin) /
           max(1, np.count_nonzero(forecast)) < params.minimum_forecast_containment
           for forecast in forecasts):
        return None
    return basin


def _extend(direction: int, edge: dict, resident_track: int,
            labels: np.ndarray, raw: np.ndarray, protected: np.ndarray,
            resident_by_frame: dict[int, object], visible_by_frame: dict[int, list],
            reference_area: float, reference_contrast: float,
            params: Params) -> list[dict]:
    added = []
    old = edge
    frame = old["frame"] + direction
    while 0 <= frame < len(labels):
        resident = resident_by_frame.get(frame)
        if (resident is None or not bool(resident.physically_visible)
                or int(resident.accepted_owner) != int(old["host"])):
            break
        mask = body._at_component(labels[frame], old["host"], resident)
        if mask is None or not body._inside(mask, resident) or np.any(mask & protected[frame]):
            break
        predicted = SimpleNamespace(
            x=float(old["target"].x)+float(resident.x)-float(old["resident"].x),
            y=float(old["target"].y)+float(resident.y)-float(old["resident"].y),
            radius_px=float(old["target"].radius_px))
        if (not body._inside(mask, predicted)
                or any(body._inside(mask, point)
                       for point in visible_by_frame.get(frame, [])
                       if int(point.track_id) != resident_track)):
            break
        basin = body._partition(mask, raw[frame], predicted, resident, body.Params())
        if basin is None or not _plausible(int(basin.sum()), reference_area, params):
            break
        forecast = _shift_basin(old["basin"], old["target"], predicted, mask)
        if (_jaccard(forecast, basin) < params.minimum_extension_jaccard
                or _contrast(raw[frame], predicted)
                < params.minimum_extension_relative_contrast * reference_contrast):
            break
        current = {"frame": frame, "host": old["host"], "mask": mask,
                   "resident": resident, "target": predicted, "basin": basin,
                   "method": "raw_continuation"}
        added.append(current)
        old = current
        frame += direction
    return added


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected pixels must align")
    scored = physical.attach_owners(points, labels)
    for field in ("strong", "physically_visible", "in_encounter"):
        scored[field] = scored[field].astype(bool)
    anchors = dual._references(scored, dual.Params())
    visible = scored[scored.physically_visible]
    visible_by_frame = {int(frame): list(group.itertuples(index=False))
                        for frame, group in visible.groupby("frame")}
    by_track = {int(track): group for track, group in visible.groupby("track_id")}
    structural: dict[tuple[int, int, int], list[dict]] = {}
    for frame, rows in visible_by_frame.items():
        by_host: dict[int, list] = {}
        for point in rows:
            host = int(point.accepted_owner)
            if host > 0:
                by_host.setdefault(host, []).append(point)
        for host, host_rows in by_host.items():
            if len(host_rows) < 2:
                continue
            components, _ = ndi.label(labels[frame] == host, body.STRUCTURE)
            by_component: dict[int, list] = {}
            for point in host_rows:
                component = physical._component_at(
                    components, float(point.x), float(point.y), search_radius=6)
                if component:
                    by_component.setdefault(int(component), []).append(point)
            for component, pair in by_component.items():
                if len(pair) != 2:
                    continue
                mask = components == component
                if (not all(body._inside(mask, point) for point in pair)
                        or not body._distinct(pair[0], pair[1], body.Params())):
                    continue
                resident = [point for point in pair
                            if int(point.track_id) in anchors
                            and int(anchors[int(point.track_id)]["owner"]) == host
                            and int(anchors[int(point.track_id)]["support_frames"])
                            >= params.minimum_host_reference_frames]
                target = [point for point in pair
                          if int(point.track_id) not in anchors]
                if len(resident) != 1 or len(target) != 1:
                    continue
                key = (int(resident[0].track_id), int(target[0].track_id), host)
                structural.setdefault(key, []).append({
                    "frame": frame, "host": host, "mask": mask,
                    "resident": resident[0], "target": target[0]})

    proposals = []
    audit = []
    for (resident_track, target_track, host), records in structural.items():
        for run in _runs(records):
            first, last = run[0]["frame"], run[-1]["frame"]
            if (len(run) < params.minimum_contact_frames
                    or first == 0 or last == len(labels)-1
                    or sorted(by_track[target_track].frame.astype(int).tolist())
                    != [row["frame"] for row in run]
                    or any(np.any(row["mask"] & protected[row["frame"]]) for row in run)):
                continue
            strong = sum(bool(row["resident"].strong) and bool(row["target"].strong)
                         and str(row["resident"].state) == "observed"
                         and str(row["target"].state) == "observed" for row in run)
            if strong/len(run) < params.minimum_strong_fraction:
                continue
            for record in run:
                frame = record["frame"]
                record["basin"] = body._partition(
                    record["mask"], raw[frame], record["target"],
                    record["resident"], body.Params())
                record["method"] = "raw"
            areas = [int(record["basin"].sum()) for record in run
                     if record["basin"] is not None and bool(record["target"].strong)]
            if len(areas) < strong:
                continue
            reference_area = float(np.median(areas))
            if reference_area <= 0:
                continue
            for index, record in enumerate(run):
                basin = record["basin"]
                if basin is not None and _plausible(int(basin.sum()), reference_area, params):
                    continue
                record["basin"] = None
                if index == 0 or index == len(run)-1:
                    break
                earlier, later = run[index-1], run[index+1]
                if (earlier["basin"] is None or later["basin"] is None
                        or not _plausible(int(earlier["basin"].sum()), reference_area, params)
                        or not _plausible(int(later["basin"].sum()), reference_area, params)):
                    break
                recovered = _broad_marker(record, earlier, later,
                                          raw[record["frame"]], reference_area,
                                          params)
                if recovered is None:
                    break
                record["basin"] = recovered
                record["method"] = "two_sided_containment"
            if any(record["basin"] is None for record in run):
                continue
            contrasts = [_contrast(raw[record["frame"]], record["target"])
                         for record in run]
            reference_contrast = float(np.median(contrasts))
            if reference_contrast <= 0:
                continue
            resident_by_frame = {
                int(row.frame): row for row in scored[scored.track_id.astype(int)
                                                  .eq(resident_track)].itertuples(index=False)}
            before = _extend(-1, run[0], resident_track, labels, raw, protected,
                             resident_by_frame, visible_by_frame,
                             reference_area, reference_contrast, params)
            after = _extend(1, run[-1], resident_track, labels, raw, protected,
                            resident_by_frame, visible_by_frame,
                            reference_area, reference_contrast, params)
            path = [*reversed(before), *run, *after]
            if (len(path) < params.minimum_new_identity_frames
                    or not before and not after):
                continue
            proposals.append((resident_track, target_track, host, path))
            audit.append({
                "resident_track": resident_track, "target_track": target_track,
                "host": host, "structural_frames": len(run),
                "path_first_frame": path[0]["frame"]+1,
                "path_last_frame": path[-1]["frame"]+1,
                "path_frames": len(path), "before_extension_frames": len(before),
                "after_extension_frames": len(after),
                "temporal_recovered_frames": sum(
                    row["method"] == "two_sided_containment" for row in path),
                "reference_area_px": reference_area,
                "reference_contrast": reference_contrast,
            })

    candidate = labels.copy()
    applications = []
    next_name = int(labels.max())+1
    proposals.sort(key=lambda item: (item[3][0]["frame"], item[0], item[1]))
    for index, (resident_track, target_track, host, path) in enumerate(proposals):
        if any(row["frame"] == other["frame"]
               and np.any(row["mask"] & other["mask"])
               for other_index, (*_, other_path) in enumerate(proposals)
               if other_index != index for row in path for other in other_path):
            continue
        if next_name > np.iinfo(candidate.dtype).max:
            raise OverflowError("no label value available for a new soma identity")
        name = next_name
        next_name += 1
        for record in path:
            frame = record["frame"]
            basin = record["basin"]
            candidate[frame][basin] = name
            applications.append({
                "frame": frame+1, "new_name": name,
                "host": host, "resident_track": resident_track,
                "target_track": target_track,
                "changed_pixels": int(basin.sum()),
                "method": record["method"],
            })
    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not set(map(int, np.unique(labels))).issubset(
                set(map(int, np.unique(candidate))))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("embedded soma recovery crossed accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
