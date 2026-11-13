"""Detect weak-name takeovers and mass-conserving dormant releases."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack  # noqa: E402


STRUCTURE = np.ones((3, 3), bool)
# detect keeps its own per-call component cache, so
# dormant_component_speed.make_detect hands it back unwrapped.
CACHES_OWN_COMPONENTS = True


def components(frame: np.ndarray, identity: int,
               minimum_area: int = 8) -> list[dict]:
    numbered, count = ndi.label(frame == identity, structure=STRUCTURE)
    result = []
    for number in range(1, count + 1):
        mask = numbered == number
        area = int(mask.sum())
        if area < minimum_area:
            continue
        centre = np.argwhere(mask).mean(axis=0)
        result.append({"component": number, "mask": mask, "area": area,
                       "y": float(centre[0]), "x": float(centre[1])})
    return result


class _CompactMask:
    """Read-only mask stored only over its nonzero rectangle."""

    def __init__(self, shape, region, values):
        self.shape = shape
        self.region = region
        self.values = np.array(values, dtype=bool, copy=True)
        self.values.flags.writeable = False

    def __array__(self, dtype=None, copy=None):
        result = np.zeros(self.shape, dtype=bool)
        result[self.region] = self.values
        return result if dtype is None else result.astype(dtype, copy=False)


def _frame_key(frame) -> tuple:
    return (frame.__array_interface__["data"][0], frame.shape,
            frame.strides, frame.dtype.str)


class _DetectFrames:
    """Components, label values and identity boxes for one detect call.

    The detector only reads labels, so each frame's answers serve every
    question asked of it during the call. One identity's components are
    labelled inside its box, which numbers them in the same raster order, and
    their masks are kept over their own box. The byte budget limits execution
    memory, never which components are found.

    The release enumeration, the prior-local search and persistence only use
    a component whose centroid lies within a radius of a point. A centroid lies
    inside its identity's box, so when that box is more than the radius plus
    one pixel from the point, or the identity is absent, no component can pass
    (far() is True) and its components need not be found.
    """
    budget = 32 * 1024 ** 2

    def __init__(self, labels: np.ndarray):
        # Holding labels keeps the pointer keys from being recycled.
        self.labels = labels
        self.allowed = {_frame_key(labels[t]) for t in range(len(labels))}
        self.cache: OrderedDict = OrderedDict()
        self.accounted = 0
        self.values: dict = {}
        self.boxes: dict = {}

    def components(self, frame: np.ndarray, identity: int,
                   minimum_area: int = 8) -> list[dict]:
        key = _frame_key(frame)
        if key not in self.allowed:
            return components(frame, identity, minimum_area)
        key = (key, int(identity))
        if key in self.cache:
            self.cache.move_to_end(key)
            items = self.cache[key][0]
        else:
            occupied = frame == identity
            ys = np.flatnonzero(occupied.any(axis=1))
            items = []
            if ys.size:
                xs = np.flatnonzero(occupied.any(axis=0))
                y0, y1 = int(ys[0]), int(ys[-1]) + 1
                x0, x1 = int(xs[0]), int(xs[-1]) + 1
                numbered, count = ndi.label(occupied[y0:y1, x0:x1],
                                            structure=STRUCTURE)
                for number, region in enumerate(ndi.find_objects(numbered), 1):
                    if region is None:
                        continue
                    local = numbered[region] == number
                    coords = np.argwhere(local)
                    oy, ox = y0 + region[0].start, x0 + region[1].start
                    coords += np.array([oy, ox])
                    centre = coords.mean(axis=0)
                    box = (slice(oy, y0 + region[0].stop),
                           slice(ox, x0 + region[1].stop))
                    items.append({"component": number,
                                  "mask": _CompactMask(frame.shape, box, local),
                                  "area": len(coords), "y": float(centre[0]),
                                  "x": float(centre[1])})
            items = tuple(items)
            cost = (sum(item["mask"].values.nbytes for item in items)
                    + 512 + len(items) * 1536)
            if cost <= self.budget:
                while self.cache and self.accounted + cost > self.budget:
                    _, (_, old_cost) = self.cache.popitem(last=False)
                    self.accounted -= old_cost
                self.cache[key] = (items, cost)
                self.accounted += cost
        return [dict(item) for item in items if item["area"] >= minimum_area]

    def frame_values(self, index: int) -> np.ndarray:
        if index not in self.values:
            self.values[index] = np.unique(self.labels[index])
        return self.values[index]

    def far(self, frames: np.ndarray, index: int, identity: int,
            point, radius) -> bool:
        """True only when no component of identity in frames[index] can lie
        within radius of point."""
        try:
            limit = float(radius)
            py, px = float(point[0]), float(point[1])
            identity = int(identity)
        except (TypeError, ValueError, IndexError):
            return False
        if (frames is not self.labels or not limit >= 0
                or not math.isfinite(limit) or not math.isfinite(py)
                or not math.isfinite(px) or identity < 1):
            return False
        found = self.boxes.get(index)
        if found is None:
            frame = np.asarray(self.labels[index])
            found = False
            if (frame.ndim == 2 and frame.dtype.kind == "u" and frame.size
                    and int(frame.max()) < 1 << 20):
                found = ndi.find_objects(frame)
            self.boxes[index] = found
        if found is False:
            return False
        if identity > len(found) or found[identity - 1] is None:
            return True
        rows, cols = found[identity - 1]
        dy = max(rows.start - py, py - (rows.stop - 1), 0.0)
        dx = max(cols.start - px, px - (cols.stop - 1), 0.0)
        return dy * dy + dx * dx > (limit + 1.0) ** 2


def largest(frame: np.ndarray, identity: int,
            minimum_area: int = 8, cache=None) -> dict | None:
    found = (components if cache is None else cache.components)(
        frame, identity, minimum_area)
    return max(found, key=lambda item: item["area"]) if found else None


def history(labels: np.ndarray, raw: np.ndarray, identity: int,
            source_frame: int, offset: int, window: int,
            minimum_area: int, cache: _DetectFrames | None = None
            ) -> tuple[list[int], list[float]]:
    areas, intensities = [], []
    first_source = offset + 1
    for source in range(max(first_source, source_frame - window), source_frame):
        index = source - offset - 1
        item = largest(labels[index], identity, minimum_area, cache=cache)
        if item is not None:
            areas.append(item["area"])
            intensities.append(float(raw[index][item["mask"]].mean()))
    return areas, intensities


def persistence(labels: np.ndarray, identity: int, source_frame: int,
                centre: np.ndarray, offset: int, minimum_area: int,
                frames: int = 3, radius: float = 35.0,
                cache: _DetectFrames | None = None) -> int:
    found = 1
    previous = centre
    for source in range(source_frame + 1,
                        min(source_frame + frames, offset + len(labels)) + 1):
        index = source - offset - 1
        if cache is not None and cache.far(labels, index, identity, previous, radius):
            break
        choices = (components if cache is None else cache.components)(
            labels[index], identity, minimum_area)
        if not choices:
            break
        item = min(choices, key=lambda part: float(np.linalg.norm(
            np.array([part["y"], part["x"]]) - previous)))
        current = np.array([item["y"], item["x"]])
        if float(np.linalg.norm(current - previous)) > radius:
            break
        found += 1
        previous = current
    return found


def detect(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
           actions: pd.DataFrame, offset: int, cfg: dict
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if actions.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "action" not in actions.columns:
        raise ValueError("nonempty action table has no action column")
    cache = _DetectFrames(labels)
    minimum_area = int(cfg.get("minimum_component_area_px", 8))
    history_window = int(cfg.get("history_window_frames", 6))
    first_source = offset + 1
    last_source = offset + len(labels)
    starts = actions[actions.action.astype(str) == "dormancy_started"].copy()
    starts = starts.sort_values("source_imagej_frame").drop_duplicates(
        ["prior_identity", "host_identity"], keep="first")
    events: list[dict] = []
    releases: list[dict] = []

    for row in starts.itertuples(index=False):
        start = int(row.source_imagej_frame)
        identity = int(row.prior_identity)
        host = int(row.host_identity)
        d_areas, d_raw = history(labels, raw, identity, start, offset,
                                 history_window, minimum_area, cache=cache)
        h_areas, _ = history(labels, raw, host, start, offset,
                             history_window, minimum_area, cache=cache)
        established = float(np.median(d_areas)) if d_areas else 0.0
        established_raw = float(np.median(d_raw)) if d_raw else 0.0
        large_floor = float(cfg.get("established_area_floor_px", 80))
        d_large = sum(area >= large_floor for area in d_areas)
        h_large = sum(area >= large_floor for area in h_areas)
        previous_host = float(h_areas[-1]) if h_areas else 0.0
        current_host = largest(labels[start - offset - 1], host, minimum_area,
                               cache=cache)
        current_host_area = float(current_host["area"]) if current_host else 0.0
        host_ratio = previous_host / max(current_host_area, 1.0)
        current_to_established = current_host_area / max(established, 1.0)
        weak_takeover = bool(
            d_large >= int(cfg.get("minimum_established_history_frames", 3))
            and established >= float(cfg.get(
                "minimum_takeover_established_area_px", 150))
            and h_large < int(cfg.get("minimum_host_history_frames", 3))
            and host_ratio <= float(cfg.get(
                "maximum_weak_host_previous_area_ratio", 0.2))
            and current_to_established <= float(cfg.get(
                "maximum_takeover_area_ratio", 1.8)))
        event_type = "weak_name_takeover" if weak_takeover else "true_dormancy"
        event_id = f"{event_type}_{identity}_{host}_{start}"
        events.append({
            "event_id": event_id, "event_type": event_type,
            "identity": identity, "host_identity": host,
            "start_source_imagej_frame": start,
            "established_area_px": established,
            "established_raw_mean": established_raw,
            "identity_large_history_frames": d_large,
            "host_large_history_frames": h_large,
            "previous_host_area_px": previous_host,
            "current_host_area_px": current_host_area,
            "previous_to_current_host_area_ratio": host_ratio,
        })
        if weak_takeover or established <= 0:
            continue

        # Enumerate permissively. The next stage applies every grid threshold.
        for source in range(start + 1, last_source + 1):
            prior_index = source - offset - 2
            index = source - offset - 1
            prior_host = largest(labels[prior_index], host, minimum_area,
                                 cache=cache)
            current_host = largest(labels[index], host, minimum_area,
                                   cache=cache)
            if prior_host is None:
                continue
            current_area = current_host["area"] if current_host else 0
            host_loss = max(0, prior_host["area"] - current_area)
            if host_loss / max(established, 1.0) < float(
                    cfg.get("enumeration_minimum_host_loss_ratio", 0.3)):
                continue
            host_centre = np.array([
                current_host["y"], current_host["x"]]) if current_host else np.array([
                    prior_host["y"], prior_host["x"]])
            for candidate_identity in sorted(
                    set(map(int, cache.frame_values(index))) - {0, host, identity}):
                if cache.far(labels, index, candidate_identity, host_centre, cfg.get(
                        "enumeration_maximum_radius_px", 60)):
                    continue
                for part in cache.components(labels[index], candidate_identity,
                                             minimum_area):
                    centre = np.array([part["y"], part["x"]])
                    distance = float(np.linalg.norm(centre - host_centre))
                    if distance > float(cfg.get(
                            "enumeration_maximum_radius_px", 60)):
                        continue
                    area_ratio = part["area"] / max(established, 1.0)
                    if not float(cfg.get("minimum_candidate_area_ratio", 0.25)) <= area_ratio <= float(cfg.get("maximum_candidate_area_ratio", 2.0)):
                        continue
                    prior_local = 0
                    for old_source in range(max(first_source, source - 10), source):
                        old_index = old_source - offset - 1
                        if cache.far(labels, old_index, candidate_identity, centre, 35):
                            continue
                        for old in cache.components(labels[old_index],
                                                    candidate_identity,
                                                    minimum_area):
                            old_centre = np.array([old["y"], old["x"]])
                            if float(np.linalg.norm(old_centre - centre)) <= 35:
                                prior_local = max(prior_local, old["area"])
                    red = float(np.mean(lag[prior_index][part["mask"]] >= float(
                        cfg.get("motion_threshold", 1.5))))
                    mass_match = 1.0 - abs(part["area"] - host_loss) / max(
                        part["area"], host_loss, 1)
                    releases.append({
                        "event_id": event_id, "identity": identity,
                        "host_identity": host,
                        "release_source_imagej_frame": source,
                        "candidate_identity": candidate_identity,
                        "candidate_component": int(part["component"]),
                        "candidate_area_px": int(part["area"]),
                        "candidate_y": float(part["y"]),
                        "candidate_x": float(part["x"]),
                        "host_loss_px": int(host_loss),
                        "host_loss_to_established_ratio": host_loss / max(established, 1.0),
                        "candidate_to_established_area_ratio": area_ratio,
                        "mass_match": mass_match,
                        "distance_from_host_px": distance,
                        "red_arrival_fraction": red,
                        "persistence_frames": persistence(
                            labels, candidate_identity, source, centre, offset,
                            minimum_area, cache=cache),
                        "prior_local_max_area_px": prior_local,
                        "prior_local_to_established_ratio": prior_local / max(established, 1.0),
                    })
    return pd.DataFrame(events), pd.DataFrame(releases)
