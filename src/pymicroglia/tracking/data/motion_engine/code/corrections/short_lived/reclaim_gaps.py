"""Globally assign unclaimed components to lower-cost persistent lineages."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import find_objects
from scipy.optimize import linear_sum_assignment

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
import sys
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402


class LabelMask(NDArrayOperatorsMixin):
    """A Boolean mask held as the equality test of one identity against a
    read-only labelled frame (the same class, by name, as PyMicroglia's
    frozen_io.LabelMask, so checks by type name treat both alike)."""
    __array_priority__ = 1000

    def __init__(self, labels, identity):
        self.labels = labels
        self.identity = int(identity)

    @property
    def shape(self):
        return self.labels.shape

    @property
    def dtype(self):
        return np.dtype(bool)

    @property
    def ndim(self):
        return self.labels.ndim

    @property
    def size(self):
        return self.labels.size

    def __len__(self):
        return len(self.labels)

    def __array__(self, dtype=None, copy=None):
        value = self.labels == self.identity
        return value if dtype is None else value.astype(dtype, copy=False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        converted = tuple(np.asarray(v) if isinstance(v, LabelMask) else v for v in inputs)
        if kwargs.get("out") is not None:
            kwargs["out"] = tuple(np.asarray(v) if isinstance(v, LabelMask) else v for v in kwargs["out"])
        return getattr(ufunc, method)(*converted, **kwargs)

    def __getitem__(self, key):
        return self.labels[key] == self.identity

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)


@dataclass
class Component:
    number: int
    mask: np.ndarray
    area: int
    y: float
    x: float


def components(mask: np.ndarray, minimum_area: int) -> list[Component]:
    numbered, count = ndi.label(mask, structure=np.ones((3, 3), bool))
    result: list[Component] = []
    for number in range(1, count + 1):
        item = numbered == number
        area = int(np.count_nonzero(item))
        if area < minimum_area:
            continue
        yy, xx = np.nonzero(item)
        result.append(Component(number, item, area, float(np.mean(yy)),
                                float(np.mean(xx))))
    return result


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    return float(np.mean(yy)), float(np.mean(xx))


def coverage(values: np.ndarray, mask: np.ndarray, predicate) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(predicate(values[mask])))


def motion_support(lag: np.ndarray, frame: int, candidate: np.ndarray,
                   previous: int, following: int, previous_mask: np.ndarray,
                   following_mask: np.ndarray, threshold: float) -> float:
    evidence: list[float] = []
    if previous == frame - 1 and frame > 0:
        transition = lag[frame - 1]
        blue = coverage(transition, previous_mask, lambda x: x <= -threshold)
        red = coverage(transition, candidate, lambda x: x >= threshold)
        evidence.append(min(blue, red))
    if following == frame + 1 and frame < len(lag):
        transition = lag[frame]
        blue = coverage(transition, candidate, lambda x: x <= -threshold)
        red = coverage(transition, following_mask, lambda x: x >= threshold)
        evidence.append(min(blue, red))
    return max(evidence, default=0.0)


def territory_support(candidate: np.ndarray, previous_mask: np.ndarray,
                      following_mask: np.ndarray, memory=None) -> float:
    if memory is None:
        history = ndi.binary_dilation(
            previous_mask | following_mask, structure=np.ones((3, 3), bool),
            iterations=3)
    else:
        history = memory.territory(previous_mask, following_mask)
    return float(np.count_nonzero(candidate & history) /
                 max(np.count_nonzero(candidate), 1))


def assignment_cost(candidate: np.ndarray, area: int,
                    predicted: tuple[float, float], radius: float,
                    reference_area: float, previous_mask: np.ndarray,
                    following_mask: np.ndarray, motion: float, cfg: dict,
                    memory=None) -> tuple[float, dict]:
    y, x = mask_centroid(candidate) if memory is None else memory.centroid(candidate)
    distance = float(np.hypot(y - predicted[0], x - predicted[1]))
    position = distance / max(radius, 1.0)
    area_term = abs(float(np.log2(max(area, 1) / max(reference_area, 1.0))))
    history = territory_support(candidate, previous_mask, following_mask, memory)
    terms = {
        "distance_px": distance,
        "position_cost": position,
        "area_cost": area_term,
        "territory_support": history,
        "motion_support": motion,
    }
    cost = (
        float(cfg["position_weight"]) * position
        + float(cfg["area_weight"]) * area_term
        + float(cfg["territory_weight"]) * (1.0 - history)
        + float(cfg["motion_weight"]) * (1.0 - motion))
    return cost, terms


def nearest_anchors(present: np.ndarray, frame: int
                    ) -> tuple[int, int] | None:
    before = present[present < frame]
    after = present[present > frame]
    if not len(before) or not len(after):
        return None
    return int(before[-1]), int(after[0])


def _frame_boxes(store: OrderedDict, key, frame):
    """find_objects of one integer label frame (False when it is not one), kept
    for the 64 latest frames."""
    boxes = store.get(key)
    if boxes is not None:
        store.move_to_end(key)
        return boxes
    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.dtype.kind not in "iu" or (
            frame.dtype.kind == "i" and frame.size and frame.min() < 0):
        boxes = False
    else:
        boxes = find_objects(frame)
    store[key] = boxes
    if len(store) > 64:
        store.popitem(last=False)
    return boxes


def _box_centroid(frame, identity: int, boxes) -> tuple[float, float] | None:
    """The centroid of `frame == identity` read inside its box, or None when it
    has no box."""
    if boxes is False or not 0 < identity <= len(boxes) or boxes[identity - 1] is None:
        return None
    y, x = boxes[identity - 1]
    yy, xx = np.nonzero(np.asarray(frame)[y, x] == identity)
    yy += y.start
    xx += x.start
    return float(np.mean(yy)), float(np.mean(xx))


# Speed (folded from persistence_profile_reuse): one reclaim call remembers what
# it has measured. Anchor masks are LabelMasks whose centroids are read inside
# the identity's find_objects box (the same coordinates in the same order); a
# donor's centroid is the one components() measured with the same formula; the
# current body's is read in its box before the frame's assignments are written;
# the distance to the current body is transformed once per identity and frame
# rather than once per donor; and each anchor pair's dilated history is kept
# while 16 MiB allow.
_TERRITORY_BYTES = 16 * 1024 ** 2


class _ReclaimMemory:
    """One reclaim call's anchor centroids, this frame's measured donors and
    bodies, frame boxes and anchor-pair histories."""

    def __init__(self):
        self.boxes = OrderedDict()
        self.anchors: dict[tuple[int, int], tuple[float, float]] = {}
        self.seeded: dict[int, tuple] = {}
        self.territories = OrderedDict()
        self.retained = 0

    def anchor_centroid(self, mask, frame: int, identity: int) -> tuple[float, float]:
        key = (frame, identity)
        if key not in self.anchors:
            value = None
            if type(mask) is LabelMask:
                value = _box_centroid(mask.labels, identity,
                                      _frame_boxes(self.boxes, ("labels", frame), mask.labels))
            if value is None:
                value = mask_centroid(mask)
            self.anchors[key] = value
        return self.anchors[key]

    def seed_donors(self, donors: list[Component]) -> None:
        # A new frame: earlier frames' bodies and donors are no longer read.
        self.seeded.clear()
        for donor in donors:
            self.seeded[id(donor.mask)] = (donor.mask, (donor.y, donor.x))

    def seed_current(self, mask: np.ndarray, result: np.ndarray, frame: int,
                     identity: int) -> None:
        value = _box_centroid(result[frame], identity,
                              _frame_boxes(self.boxes, ("result", frame), result[frame]))
        if value is not None:
            self.seeded[id(mask)] = (mask, value)

    def centroid(self, mask: np.ndarray) -> tuple[float, float]:
        entry = self.seeded.get(id(mask))
        if entry is not None and entry[0] is mask:
            return entry[1]
        return mask_centroid(mask)

    def territory(self, previous_mask, following_mask) -> np.ndarray:
        key = (id(previous_mask), id(following_mask))
        entry = self.territories.pop(key, None)
        if entry is not None:
            self.territories[key] = entry
            return entry[2]
        # Exact original morphology, with no border or radius changes.
        history = ndi.binary_dilation(
            previous_mask | following_mask, structure=np.ones((3, 3), bool),
            iterations=3)
        cost = history.nbytes + 512
        if cost <= _TERRITORY_BYTES:
            while self.territories and self.retained + cost > _TERRITORY_BYTES:
                _, (_, _, _, size) = self.territories.popitem(last=False)
                self.retained -= size
            self.territories[key] = (previous_mask, following_mask, history, cost)
            self.retained += cost
        return history


def reclaim(labels: np.ndarray, ownership: np.ndarray,
            unclaimed_ids: np.ndarray, profiles: pd.DataFrame,
            lag: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray,
                                                       pd.DataFrame]:
    result = ownership.copy()
    unclaimed = unclaimed_ids.copy()
    cfg = params["reclaim"]
    minimum_persistence = int(params["persistent_minimum_frames"])
    persistent = profiles.loc[
        profiles.frames_present.astype(int) >= minimum_persistence,
        "identity"].astype(int).tolist()
    areas_by_id = {identity: np.zeros(len(labels), int)
                   for identity in persistent}
    for frame, item in enumerate(labels):
        values, counts = np.unique(item, return_counts=True)
        for value, count in zip(map(int, values), map(int, counts)):
            if value in areas_by_id:
                areas_by_id[value][frame] = count
    present_by_id = {
        identity: np.flatnonzero(areas > 0)
        for identity, areas in areas_by_id.items()}
    mask_cache: dict[tuple[int, int], np.ndarray] = {}
    memory = _ReclaimMemory()

    def identity_mask(frame: int, identity: int) -> np.ndarray:
        key = (frame, identity)
        if key not in mask_cache:
            mask_cache[key] = LabelMask(labels[frame], identity)
        return mask_cache[key]
    maximum_gap = int(cfg["maximum_bracket_gap_frames"])
    minimum_donor = int(cfg["minimum_donor_area_px"])
    threshold = float(cfg["motion_threshold"])
    minimum_improvement = float(cfg["minimum_improvement"])
    actions: list[dict] = []
    for frame in range(len(result)):
        donors = components(unclaimed[frame] > 0, minimum_donor)
        memory.seed_donors(donors)
        if not donors:
            continue
        edges: list[dict] = []
        recipients: list[int] = []
        states: dict[int, dict] = {}
        for identity in persistent:
            areas = areas_by_id[identity]
            present = present_by_id[identity]
            anchors = nearest_anchors(present, frame)
            if anchors is None:
                continue
            previous, following = anchors
            gap = following - previous - 1
            if gap > maximum_gap:
                continue
            previous_mask = identity_mask(previous, identity)
            following_mask = identity_mask(following, identity)
            py, px = memory.anchor_centroid(previous_mask, previous, identity)
            fy, fx = memory.anchor_centroid(following_mask, following, identity)
            alpha = (frame - previous) / max(following - previous, 1)
            predicted = (py + alpha * (fy - py), px + alpha * (fx - px))
            speed = float(np.hypot(fy - py, fx - px) /
                          max(following - previous, 1))
            radius = min(float(cfg["maximum_search_radius_px"]),
                         float(cfg["base_search_radius_px"]) + speed * gap)
            local = areas[max(0, frame - 5):min(len(labels), frame + 6)]
            local = local[local > 0]
            if not len(local):
                continue
            reference_area = float(np.median(local))
            current_mask = result[frame] == identity
            current_distance = None
            memory.seed_current(current_mask, result, frame, identity)
            current_area = int(np.count_nonzero(current_mask))
            if current_area:
                current_motion = motion_support(
                    lag, frame, current_mask, previous, following,
                    previous_mask, following_mask, threshold)
                current_cost, current_terms = assignment_cost(
                    current_mask, current_area, predicted, radius,
                    reference_area, previous_mask, following_mask,
                    current_motion, cfg, memory)
            else:
                current_terms = {"distance_px": None, "position_cost": None,
                                 "area_cost": None, "territory_support": 0.0,
                                 "motion_support": 0.0}
                current_cost = 2.5 + float(cfg["gap_weight"]) * gap
            area_fraction = current_area / max(reference_area, 1.0)
            state = {
                "identity": identity, "previous": previous,
                "following": following, "gap": gap,
                "previous_mask": previous_mask,
                "following_mask": following_mask,
                "predicted": predicted, "radius": radius,
                "reference_area": reference_area,
                "current_mask": current_mask, "current_area": current_area,
                "current_cost": current_cost, "current_terms": current_terms,
                "area_fraction": area_fraction,
            }
            states[identity] = state
            for donor_index, donor in enumerate(donors):
                distance = float(np.hypot(donor.y - predicted[0],
                                          donor.x - predicted[1]))
                if distance > radius:
                    continue
                if current_area:
                    if current_distance is None:
                        current_distance = ndi.distance_transform_edt(~current_mask)
                    separation = float(np.min(current_distance[donor.mask]))
                else:
                    separation = float("inf")
                deficit = area_fraction < float(cfg["area_deficit_fraction"])
                if current_area and deficit and separation <= 3.0:
                    action = "augment"
                    candidate_mask = current_mask | donor.mask
                elif current_area:
                    action = "swap"
                    candidate_mask = donor.mask
                else:
                    action = "fill"
                    candidate_mask = donor.mask
                candidate_area = int(np.count_nonzero(candidate_mask))
                ratio = candidate_area / max(reference_area, 1.0)
                if ratio < 0.25 or ratio > 2.5:
                    continue
                candidate_motion = motion_support(
                    lag, frame, candidate_mask, previous, following,
                    previous_mask, following_mask, threshold)
                candidate_cost, terms = assignment_cost(
                    candidate_mask, candidate_area, predicted, radius,
                    reference_area, previous_mask, following_mask,
                    candidate_motion, cfg, memory)
                candidate_cost += float(cfg["gap_weight"]) * gap
                improvement = current_cost - candidate_cost
                if improvement < minimum_improvement:
                    continue
                edges.append({
                    "recipient": identity, "donor": donor_index,
                    "action": action, "improvement": improvement,
                    "candidate_cost": candidate_cost,
                    "candidate_area": candidate_area,
                    "donor_area": donor.area,
                    "separation_px": separation,
                    **terms,
                })
                if identity not in recipients:
                    recipients.append(identity)
        if not edges:
            continue
        donor_count, recipient_count = len(donors), len(recipients)
        size = donor_count + recipient_count
        matrix = np.zeros((size, size), float)
        edge_by_pair = {(item["donor"], recipients.index(item["recipient"])): item
                        for item in edges}
        for (donor_index, recipient_index), item in edge_by_pair.items():
            matrix[donor_index, recipient_index] = -item["improvement"]
        rows, cols = linear_sum_assignment(matrix)
        chosen = []
        for row, col in zip(rows, cols):
            item = edge_by_pair.get((int(row), int(col)))
            if item is not None and item["improvement"] >= minimum_improvement:
                chosen.append(item)
        for item in sorted(chosen, key=lambda x: -x["improvement"]):
            donor = donors[item["donor"]]
            state = states[item["recipient"]]
            released = 0
            if item["action"] == "swap":
                current = result[frame] == item["recipient"]
                released = int(np.count_nonzero(current))
                result[frame][current] = 0
                unclaimed[frame][current] = item["recipient"]
            result[frame][donor.mask] = item["recipient"]
            unclaimed[frame][donor.mask] = 0
            actions.append({
                "output_frame": frame + 1,
                "source_imagej_frame": frame + 1 + int(
                    params["source_frame_offset"]),
                "recipient_identity": item["recipient"],
                "action": item["action"],
                "donor_original_identities": ";".join(map(str, sorted(
                    set(map(int, np.unique(unclaimed_ids[frame][donor.mask])))
                    - {0}))),
                "donor_area_px": item["donor_area"],
                "released_current_area_px": released,
                "reference_area_px": state["reference_area"],
                "current_area_px": state["current_area"],
                "candidate_area_px": item["candidate_area"],
                "current_cost": state["current_cost"],
                "candidate_cost": item["candidate_cost"],
                "cost_improvement": item["improvement"],
                "distance_px": item["distance_px"],
                "search_radius_px": state["radius"],
                "position_cost": item["position_cost"],
                "area_cost": item["area_cost"],
                "territory_support": item["territory_support"],
                "motion_support": item["motion_support"],
                "bracket_gap_frames": state["gap"],
            })
    columns = [
        "output_frame", "source_imagej_frame", "recipient_identity", "action",
        "donor_original_identities", "donor_area_px",
        "released_current_area_px", "reference_area_px", "current_area_px",
        "candidate_area_px", "current_cost", "candidate_cost",
        "cost_improvement", "distance_px", "search_radius_px",
        "position_cost", "area_cost", "territory_support", "motion_support",
        "bracket_gap_frames",
    ]
    return result, unclaimed, pd.DataFrame(actions, columns=columns)

