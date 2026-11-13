"""Build local birth-priority tracklets without future-territory confidence."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack  # noqa: E402

STRUCTURE = np.ones((3, 3), np.uint8)
LAYER_ORDER = {
    "large_static_isolated": 0,
    "large_static_near_smaller": 1,
    "large_mobile": 2,
    "small_or_dim": 3,
}


@dataclass
class Component:
    frame: int
    identity: int
    number: int
    pixels: np.ndarray
    area: int
    x: float
    y: float


@dataclass
class Tracklet:
    number: int
    identity: int
    nodes: list[Component] = field(default_factory=list)

    @property
    def birth(self) -> Component:
        return self.nodes[0]

    @property
    def last(self) -> Component:
        return self.nodes[-1]


def frame_components(frame: np.ndarray, frame_number: int,
                     minimum: int) -> list[Component]:
    result = []
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        parts, count = ndi.label(frame == identity, structure=STRUCTURE)
        areas = np.bincount(parts.ravel())
        for number in range(1, count + 1):
            if int(areas[number]) < minimum:
                continue
            mask = parts == number
            yy, xx = np.where(mask)
            result.append(Component(
                frame_number, identity, number,
                np.flatnonzero(mask.ravel()), int(areas[number]),
                float(xx.mean()), float(yy.mean())))
    return result


def component_mask(component: Component,
                   shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], bool)
    mask[component.pixels] = True
    return mask.reshape(shape)


def _overlap_intersect(first: Component, second: Component) -> int:
    return int(np.intersect1d(
        first.pixels, second.pixels, assume_unique=True).size)


# Each component's pixel list, checked once to be strictly increasing, kept
# while the component is (holding it keeps its id from being reused). The byte
# budget limits execution memory, never which pixels are counted.
_SORTED_PIXELS: OrderedDict = OrderedDict()
_SORTED_BUDGET = 32 * 1024 ** 2
_sorted_retained = 0


def _sorted_pixels(component: Component) -> np.ndarray | None:
    global _sorted_retained
    key = id(component)
    if key in _SORTED_PIXELS:
        _SORTED_PIXELS.move_to_end(key)
        return _SORTED_PIXELS[key][1]
    pixels = np.asarray(component.pixels)
    good = (pixels.ndim == 1 and pixels.dtype.kind in "iu"
            and bool(np.all(pixels[1:] > pixels[:-1])))
    value = pixels if good else None
    cost = pixels.nbytes + 512
    if cost <= _SORTED_BUDGET:
        while _SORTED_PIXELS and _sorted_retained + cost > _SORTED_BUDGET:
            _, (_, _, old_cost) = _SORTED_PIXELS.popitem(last=False)
            _sorted_retained -= old_cost
        _SORTED_PIXELS[key] = (component, value, cost)
        _sorted_retained += cost
    return value


def overlap(first: Component, second: Component) -> int:
    """Shared pixels of two components, counted by binary search in their
    sorted pixel lists, within the span the two lists share."""
    a, b = _sorted_pixels(first), _sorted_pixels(second)
    if a is None or b is None or a.dtype != b.dtype:
        return _overlap_intersect(first, second)
    if not len(a) or not len(b) or a[-1] < b[0] or b[-1] < a[0]:
        return 0
    if len(a) > len(b):
        a, b = b, a
    lo = np.searchsorted(a, b[0], side="left")
    hi = np.searchsorted(a, b[-1], side="right")
    a = a[lo:hi]
    positions = np.searchsorted(b, a)
    return int(np.count_nonzero(b[positions] == a))


def distance(first: Component, second: Component) -> float:
    return float(np.hypot(first.x - second.x, first.y - second.y))


def best_match(candidates: list[Component], reference: Component,
               radius: float, used: set[tuple[int, int]] | None = None,
               minimum_area_ratio: float = 0.2,
               maximum_area_ratio: float = 5.0) -> Component | None:
    used = used or set()
    choices = []
    for item in candidates:
        key = (item.identity, item.number)
        if key in used:
            continue
        shared = overlap(reference, item)
        separation = distance(reference, item)
        ratio = item.area / max(reference.area, 1)
        if shared == 0 and separation > radius:
            continue
        if ratio < minimum_area_ratio or ratio > maximum_area_ratio:
            continue
        choices.append(((shared, -separation,
                         -abs(np.log(max(ratio, 1e-12))), item.area), item))
    return max(choices, key=lambda value: value[0])[1] if choices else None


def build_tracklets(frames: list[list[Component]], cfg: dict) -> list[Tracklet]:
    """Same-identity tracklets, frame by frame.

    A tracklet keeps the identity it was made with, so each identity's
    tracklets, kept in creation order, are the active candidates in the same
    order; each frame's components grouped by identity keep their frame order.
    """
    tracklets: list[Tracklet] = []
    by_identity: dict[int, list[Tracklet]] = {}
    next_number = 1
    maximum_gap = int(cfg["maximum_tracklet_gap_frames"])
    radius = float(cfg["same_identity_tracklet_radius_px"])
    for frame_number, items in enumerate(frames):
        identities = sorted({item.identity for item in items})
        groups: dict[int, list[Component]] = {}
        for item in items:
            groups.setdefault(item.identity, []).append(item)
        for identity in identities:
            current = groups[identity]
            active = [tracklet for tracklet in by_identity.get(identity, ())
                      if frame_number - tracklet.last.frame - 1 <= maximum_gap]
            pairs = []
            for tracklet in active:
                for item in current:
                    shared = overlap(tracklet.last, item)
                    separation = distance(tracklet.last, item)
                    ratio = item.area / max(tracklet.last.area, 1)
                    if shared == 0 and separation > radius:
                        continue
                    pairs.append(((shared, -separation,
                                   -abs(np.log(max(ratio, 1e-12)))),
                                  tracklet, item))
            assigned_tracks: set[int] = set()
            assigned_nodes: set[int] = set()
            for _, tracklet, item in sorted(
                    pairs, key=lambda value: value[0], reverse=True):
                if tracklet.number in assigned_tracks or id(item) in assigned_nodes:
                    continue
                tracklet.nodes.append(item)
                assigned_tracks.add(tracklet.number)
                assigned_nodes.add(id(item))
            for item in current:
                if id(item) in assigned_nodes:
                    continue
                tracklet = Tracklet(next_number, identity, [item])
                tracklets.append(tracklet)
                by_identity.setdefault(tracklet.identity, []).append(tracklet)
                next_number += 1
    return tracklets


def approximate_edge_gap(first: Component, second: Component) -> float:
    return float(distance(first, second)
                 - np.sqrt(first.area / np.pi)
                 - np.sqrt(second.area / np.pi))


def classify_tracklet(tracklet: Tracklet, frames: list[list[Component]],
                      cfg: dict) -> dict:
    count = int(cfg["classification_observations"])
    early = tracklet.nodes[:count]
    birth_small = tracklet.birth.area <= int(cfg["small_birth_area_px"])
    median_area = float(np.median([node.area for node in early]))
    if birth_small or median_area < float(cfg["large_median_area_px"]):
        layer = "small_or_dim"
        p75_step = np.nan
        near_fraction = 0.0
    else:
        steps = [distance(first, second)
                 for first, second in zip(early, early[1:])]
        p75_step = float(np.percentile(steps, 75)) if steps else 0.0
        near = 0
        for node in early:
            smaller = [item for item in frames[node.frame]
                       if item is not node
                       and item.area <= int(cfg["small_birth_area_px"])]
            if smaller and min(approximate_edge_gap(node, item)
                               for item in smaller) <= float(
                                   cfg["near_smaller_edge_gap_px"]):
                near += 1
        near_fraction = near / max(len(early), 1)
        if p75_step > float(cfg["static_maximum_p75_step_px"]):
            layer = "large_mobile"
        elif near_fraction >= float(cfg["near_smaller_frame_fraction"]):
            layer = "large_static_near_smaller"
        else:
            layer = "large_static_isolated"
    return {
        "tracklet_id": tracklet.number,
        "canonical_identity": tracklet.identity,
        "birth_output_frame": tracklet.birth.frame + 1,
        "birth_area_px": tracklet.birth.area,
        "classification_observations_used": len(early),
        "early_median_area_px": median_area,
        "early_p75_step_px": p75_step,
        "early_near_smaller_fraction": near_fraction,
        "confidence_layer": layer,
        "priority_rank": LAYER_ORDER[layer],
        "observed_frames": len(tracklet.nodes),
    }


def hierarchy_tables(labels: np.ndarray, cfg: dict
                     ) -> tuple[pd.DataFrame, pd.DataFrame,
                                list[list[Component]], list[Tracklet]]:
    frames = [frame_components(labels[frame], frame,
                               int(cfg["minimum_component_area_px"]))
              for frame in range(len(labels))]
    tracklets = build_tracklets(frames, cfg)
    classes = [classify_tracklet(tracklet, frames, cfg)
               for tracklet in tracklets]
    class_table = pd.DataFrame(classes)
    class_by_id = {int(row.tracklet_id): row
                   for row in class_table.itertuples()}
    rows = []
    for tracklet in tracklets:
        classification = class_by_id[tracklet.number]
        for index, node in enumerate(tracklet.nodes):
            rows.append({
                "tracklet_id": tracklet.number,
                "node_index": index,
                "output_frame": node.frame + 1,
                "observed_identity": node.identity,
                "component_number": node.number,
                "area_px": node.area,
                "x": node.x,
                "y": node.y,
                "canonical_identity": int(
                    classification.canonical_identity),
                "birth_output_frame": int(
                    classification.birth_output_frame),
                "birth_area_px": int(classification.birth_area_px),
                "confidence_layer": str(classification.confidence_layer),
                "priority_rank": int(classification.priority_rank),
            })
    return pd.DataFrame(rows), class_table, frames, tracklets

