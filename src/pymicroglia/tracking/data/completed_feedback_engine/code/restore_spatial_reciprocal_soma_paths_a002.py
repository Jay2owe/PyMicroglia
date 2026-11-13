"""Restore rare spatially crossed cell names through supported soma paths.

The producer discovers geometry fieldwide. It never reads feedback or selects
recordings, identities, trackers, frames or coordinates by literal value.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import component_accounting
import recover_dormant_name_soma_basins as basin_rule
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_component_area: int = 25
    maximum_component_area: int = 1000
    maximum_pair_distance_px: float = 70.
    minimum_pair_separation_radii: float = 4.5
    minimum_crossed_distance_gain: float = 4.
    maximum_crossed_step_px: float = 30.
    maximum_crossed_area_log_cost: float = 1.3
    maximum_successor_distance_px: float = 22.
    maximum_successor_cost: float = 1.5
    maximum_successor_area_ratio: float = 6.
    maximum_missing_frames: int = 3
    minimum_raw_peak_retention: float = .35
    maximum_companion_valley_ratio: float = .75
    minimum_basin_reference_ratio: float = .2
    maximum_basin_reference_ratio: float = 2.5
    minimum_basin_source_fraction: float = .15
    minimum_path_changed_frames: int = 2
    minimum_event_changed_pixels: int = 250


class Frame:
    def __init__(self, labels: np.ndarray, points: list, yy: np.ndarray,
                 xx: np.ndarray):
        self.ids, self.owners, self.areas = component_accounting.label_value_components(labels)
        count = len(self.owners)
        if count:
            sx = np.bincount(self.ids.ravel(), weights=xx.ravel(), minlength=count+1)[1:]
            sy = np.bincount(self.ids.ravel(), weights=yy.ravel(), minlength=count+1)[1:]
            self.x, self.y = sx/self.areas, sy/self.areas
        else:
            self.x = self.y = np.empty(0, float)
        self.points: dict[int, list] = {}
        for point in points:
            y, x = round(float(point.y)), round(float(point.x))
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                cid = int(self.ids[y, x])
                if cid:
                    self.points.setdefault(cid, []).append(point)

    def mask(self, cid: int) -> np.ndarray:
        return self.ids == cid

    def center(self, cid: int) -> tuple[float, float]:
        return float(self.x[cid-1]), float(self.y[cid-1])

    def area(self, cid: int) -> int:
        return int(self.areas[cid-1])

    def owner(self, cid: int) -> int:
        return int(self.owners[cid-1])


def _discover(previous: Frame, current: Frame, params: Params) -> list[dict]:
    eligible = np.flatnonzero((current.areas >= params.minimum_component_area) &
                              (current.areas <= params.maximum_component_area))
    if len(eligible) < 2:
        return []
    tree = cKDTree(np.c_[current.x[eligible], current.y[eligible]])
    by_owner: dict[int, np.ndarray] = {}
    for name in np.unique(previous.owners):
        by_owner[int(name)] = np.flatnonzero(
            (previous.owners == name) &
            (previous.areas >= params.minimum_component_area) &
            (previous.areas <= params.maximum_component_area))
    result = []
    for left_index, right_index in tree.query_pairs(params.maximum_pair_distance_px):
        ai, bi = int(eligible[left_index]), int(eligible[right_index])
        name_a, name_b = int(current.owners[ai]), int(current.owners[bi])
        if name_a == name_b:
            continue
        separation = float(np.hypot(current.x[ai]-current.x[bi],
                                    current.y[ai]-current.y[bi]))
        radii = np.sqrt(current.areas[ai]/np.pi) + np.sqrt(current.areas[bi]/np.pi)
        if separation/radii < params.minimum_pair_separation_radii:
            continue
        old_as, old_bs = by_owner.get(name_a, []), by_owner.get(name_b, [])
        best = None
        center = np.array([(current.x[ai]+current.x[bi])/2,
                           (current.y[ai]+current.y[bi])/2])
        for pi in old_as:
            if np.hypot(previous.x[pi]-center[0], previous.y[pi]-center[1]) > 65:
                continue
            for qi in old_bs:
                if np.hypot(previous.x[qi]-center[0], previous.y[qi]-center[1]) > 65:
                    continue
                da = float(np.hypot(previous.x[pi]-current.x[bi],
                                    previous.y[pi]-current.y[bi]))
                db = float(np.hypot(previous.x[qi]-current.x[ai],
                                    previous.y[qi]-current.y[ai]))
                crossed = da+db
                same = float(np.hypot(previous.x[pi]-current.x[ai],
                                      previous.y[pi]-current.y[ai]) +
                             np.hypot(previous.x[qi]-current.x[bi],
                                      previous.y[qi]-current.y[bi]))
                area_cost = abs(float(np.log(previous.areas[pi]/current.areas[bi]))) + \
                    abs(float(np.log(previous.areas[qi]/current.areas[ai])))
                option = (crossed, da, db, same, area_cost, int(pi)+1, int(qi)+1)
                if best is None or option[0] < best[0]:
                    best = option
        if best is None:
            continue
        crossed, da, db, same, area_cost, prior_a, prior_b = best
        if (max(da, db) > params.maximum_crossed_step_px or
                same/max(crossed, 1.) < params.minimum_crossed_distance_gain or
                area_cost > params.maximum_crossed_area_log_cost):
            continue
        result.append({"current_a": ai+1, "current_b": bi+1,
                       "prior_a": prior_a, "prior_b": prior_b,
                       "name_a": name_a, "name_b": name_b,
                       "crossed_distance": crossed, "same_distance": same,
                       "pair_separation_radii": separation/radii,
                       "area_log_cost": area_cost})
    return result


def _peak(raw: np.ndarray, x: float, y: float) -> float:
    iy, ix = round(y), round(x)
    return float(raw[max(0, iy-2):min(raw.shape[0], iy+3),
                     max(0, ix-2):min(raw.shape[1], ix+3)].max())


def _raw_point(raw: np.ndarray, mask: np.ndarray, x: float, y: float,
               radius: float, others: list) -> SimpleNamespace | None:
    y0, y1 = max(0, int(y-radius-2)), min(raw.shape[0], int(y+radius+3))
    x0, x1 = max(0, int(x-radius-2)), min(raw.shape[1], int(x+radius+3))
    if y1 <= y0 or x1 <= x0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    allowed = mask[y0:y1, x0:x1] & ((xx-x)**2+(yy-y)**2 <= radius**2)
    for point in others:
        erode = max(3., .75*float(point.radius_px))
        allowed &= (xx-float(point.x))**2+(yy-float(point.y))**2 > erode**2
    if not allowed.any():
        return None
    original = ndi.gaussian_filter(raw[y0:y1, x0:x1].astype(np.float32), 1.)
    eligible = original.copy()
    eligible[~allowed] = -np.inf
    iy, ix = np.unravel_index(np.argmax(eligible), eligible.shape)
    if others and original[iy, ix] < ndi.maximum_filter(original, size=7)[iy, ix]:
        return None
    return SimpleNamespace(x=float(x0+ix), y=float(y0+iy), radius_px=max(2., radius/2),
                           track_id=-1)


def _trace(start: int, cid: int, prior: Frame, prior_cid: int,
           target: int, frames: list[Frame], raw: np.ndarray,
           protected: np.ndarray, params: Params) -> list[dict]:
    previous_x, previous_y = prior.center(prior_cid)
    reference_area = prior.area(prior_cid)
    reference_peak = _peak(raw[start-1], previous_x, previous_y)
    last_x, last_y = previous_x, previous_y
    last_track = -1
    excluded_tracks: set[int] = set()
    current_cid = cid
    missing = 0
    output = []
    radius = np.sqrt(reference_area/np.pi)
    for t in range(start, len(frames)):
        info = frames[t]
        if t != start:
            candidates = []
            for index in np.flatnonzero((info.areas >= 20) &
                    (info.areas <= params.maximum_successor_area_ratio*reference_area)):
                candidate_id = int(index)+1
                cx, cy = info.center(candidate_id)
                distance = float(np.hypot(cx-last_x, cy-last_y))
                if distance > max(params.maximum_successor_distance_px, 3*radius):
                    continue
                cost = distance/18. + .28*abs(float(np.log(info.areas[index]/reference_area)))
                same_track = any(int(p.track_id) == last_track
                                 for p in info.points.get(candidate_id, []))
                candidates.append((not same_track, cost, candidate_id))
            if not candidates:
                missing += 1
                if missing > params.maximum_missing_frames:
                    break
                continue
            candidates.sort()
            _, cost, current_cid = candidates[0]
            if cost > params.maximum_successor_cost:
                missing += 1
                if missing > params.maximum_missing_frames:
                    break
                continue
        mask = info.mask(current_cid)
        points = info.points.get(current_cid, [])
        same = [p for p in points if int(p.track_id) == last_track]
        seed = min(same, key=lambda p: np.hypot(float(p.x)-last_x, float(p.y)-last_y)) \
            if same else None
        if seed is None:
            distance_limit = (max(18., 3.*radius) if len(points) == 1
                              else max(8., 1.6*radius))
            nearby = [p for p in points
                      if int(p.track_id) not in excluded_tracks
                      and np.hypot(float(p.x)-last_x, float(p.y)-last_y)
                      <= distance_limit]
            seed = min(nearby, key=lambda p: np.hypot(float(p.x)-last_x,
                                                        float(p.y)-last_y)) if nearby else None
        if seed is None:
            seed = _raw_point(raw[t], mask, last_x, last_y,
                              max(12., 2.5*radius), points)
        if seed is None or _peak(raw[t], float(seed.x), float(seed.y)) < \
                params.minimum_raw_peak_retention*reference_peak:
            missing += 1
            if missing > params.maximum_missing_frames:
                break
            continue
        peers = []
        unsupported = False
        for other in points:
            if int(other.track_id) == int(seed.track_id):
                continue
            valley = physical._valley_ratio(raw[t], (float(seed.x), float(seed.y)),
                                            (float(other.x), float(other.y)))
            if valley > params.maximum_companion_valley_ratio:
                if int(seed.track_id) < 0:
                    unsupported = True
                    break
                combined = max(float(seed.radius_px)+float(other.radius_px), 1.)
                if np.hypot(float(seed.x)-float(other.x),
                            float(seed.y)-float(other.y)) > 1.25*combined:
                    unsupported = True
                    break
            else:
                peers.append(other)
        if unsupported:
            missing += 1
            if missing > params.maximum_missing_frames:
                break
            continue
        selected = basin_rule._basin(mask, raw[t], seed, peers,
                                     basin_rule.Params())
        area = int(selected.sum()) if selected is not None else 0
        if (not area or not params.minimum_basin_reference_ratio <= area/reference_area
                <= params.maximum_basin_reference_ratio
                or area < params.minimum_basin_source_fraction*int(mask.sum())
                or np.any(selected & protected[t])):
            missing += 1
            if missing > params.maximum_missing_frames:
                break
            continue
        output.append({"frame": t, "source": info.owner(current_cid),
                       "target": target, "mask": selected, "area": area,
                       "component_area": int(mask.sum()),
                       "companion_somata": len(peers),
                       "seed_track": int(seed.track_id)})
        excluded_tracks.update(int(other.track_id) for other in peers)
        last_x, last_y = float(seed.x), float(seed.y)
        last_track = int(seed.track_id) if int(seed.track_id) > 0 else last_track
        missing = 0
    return output


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("aligned complete labels, raw and accepted-pixel mask required")
    strong = points[(points.state.astype(str) == "observed") &
                    points.strong.astype(bool)].copy()
    by_frame = {int(t): list(g.itertuples(index=False)) for t, g in strong.groupby("frame")}
    yy, xx = np.indices(labels.shape[1:])
    frames = [Frame(labels[t], by_frame.get(t, []), yy, xx)
              for t in range(len(labels))]
    events = []
    for t in range(1, len(frames)):
        for event in _discover(frames[t-1], frames[t], params):
            event["frame"] = t
            events.append(event)
    candidate = labels.copy()
    audit, applications = [], []
    for index, event in enumerate(events, start=1):
        t = event["frame"]
        left = _trace(t, event["current_a"], frames[t-1], event["prior_b"],
                      event["name_b"], frames, raw, protected, params)
        right = _trace(t, event["current_b"], frames[t-1], event["prior_a"],
                       event["name_a"], frames, raw, protected, params)
        paths = [left, right]
        if any(len({r["frame"] for r in path if r["source"] != r["target"]}) <
               params.minimum_path_changed_frames for path in paths):
            audit.append({"event": index, "frame": t+1, "outcome": "short_supported_path",
                          "proposed_pixels": sum(r["area"] for path in paths for r in path),
                          **{k:v for k,v in event.items() if k != "frame"}})
            continue
        ordered = [(r["frame"], side, r) for side, path in enumerate(paths)
                   for r in path]
        ordered.sort(key=lambda row: (row[0], row[1]))
        staged = candidate.copy()
        blocked = set()
        applied_pixels = 0
        event_apps = []
        for frame, side, row in ordered:
            if side in blocked or row["source"] == row["target"]:
                continue
            selected = row["mask"]
            if np.any(staged[frame][selected] != row["source"]):
                blocked.add(side)
                continue
            moving_away = [other["mask"] for s, path in enumerate(paths)
                           if s != side and s not in blocked for other in path if other["frame"] == frame
                           and other["source"] == row["target"]
                           and other["target"] != row["target"]]
            if any(int(staged[frame, round(float(p.y)), round(float(p.x))]) == row["target"]
                   and not any(mask[round(float(p.y)), round(float(p.x))]
                               for mask in moving_away)
                   for p in by_frame.get(frame, [])):
                blocked.add(side)
                continue
            staged[frame][selected] = row["target"]
            applied_pixels += row["area"]
            event_apps.append({"event": index, "frame": frame+1,
                               "old_name": row["source"], "new_name": row["target"],
                               "changed_pixels": row["area"],
                               "source_area": row["component_area"],
                               "companion_somata": row["companion_somata"],
                               "seed_track": row["seed_track"]})
        if applied_pixels >= params.minimum_event_changed_pixels:
            candidate = staged
            applications.extend(event_apps)
        audit.append({"event": index, "frame": t+1,
                      "outcome": "applied" if applied_pixels >=
                      params.minimum_event_changed_pixels else "small_supported_path",
                      "proposed_pixels": sum(r["area"] for path in paths for r in path),
                      "changed_pixels": applied_pixels if applied_pixels >=
                      params.minimum_event_changed_pixels else 0,
                      "blocked_sides": len(blocked),
                      **{k:v for k,v in event.items() if k != "frame"}})
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("spatial exchange crossed accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
