"""Build local connected tracklets and audit young-challenger ownership changes."""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from project_paths import find_project_root


TUNING = Path(__file__).resolve().parents[1]
ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402
from merge_partition_speed import BoundedNdimage, slices as occupied_slices
from array_bounds import occupied_coordinates
ndi = BoundedNdimage()

STRUCTURE = np.ones((3, 3), np.uint8)


@dataclass
class Component:
    frame: int
    observed_identity: int
    local_component: int
    pixels: np.ndarray
    area: int
    x: float
    y: float
    tracklet: int = 0
    local_age: int = 1


@dataclass
class Tracklet:
    number: int
    observed_identity: int
    components: list[Component] = field(default_factory=list)

    @property
    def last(self) -> Component:
        return self.components[-1]


def frame_components(frame: np.ndarray, frame_index: int,
                     minimum_area: int = 1) -> list[Component]:
    """Return connected components without allowing one name to merge locations."""
    width = frame.shape[1]
    result = []
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        parts, count = ndi.label(frame == identity, structure=STRUCTURE)
        for component in range(1, count + 1):
            yy, xx = occupied_coordinates(parts == component)
            pixels = yy * width + xx
            if len(pixels) < minimum_area:
                continue
            result.append(Component(
                frame=frame_index, observed_identity=identity,
                local_component=component, pixels=pixels,
                area=int(len(pixels)), x=float(xx.mean()), y=float(yy.mean())))
    return result


def overlap_pixels(first: Component, second: Component) -> int:
    return int(np.intersect1d(first.pixels, second.pixels,
                              assume_unique=True).size)


def distance(first: Component, second: Component) -> float:
    return float(np.hypot(first.x - second.x, first.y - second.y))


def best_component(candidates: list[Component], reference: Component,
                   maximum_distance: float) -> Component | None:
    scored = []
    for candidate in candidates:
        overlap = overlap_pixels(candidate, reference)
        separation = distance(candidate, reference)
        if overlap == 0 and separation > maximum_distance:
            continue
        scored.append(((overlap, -separation), candidate))
    return max(scored, key=lambda row: row[0])[1] if scored else None


def build_tracklets(labels: np.ndarray, maximum_step_px: float = 15.0,
                    maximum_gap_frames: int = 2,
                    minimum_area: int = 1,
                    ) -> tuple[list[list[Component]], dict[int, Tracklet]]:
    """Link only spatially compatible components carrying the same observed name."""
    frames: list[list[Component]] = []
    tracklets: dict[int, Tracklet] = {}
    next_number = 1
    for frame_index, frame in enumerate(labels):
        current = frame_components(frame, frame_index, minimum_area)
        available = set(tracklets)
        for component in sorted(current, key=lambda item: -item.area):
            choices = []
            for number in list(available):
                track = tracklets[number]
                gap = frame_index - track.last.frame
                if (track.observed_identity != component.observed_identity
                        or gap < 1 or gap > maximum_gap_frames):
                    continue
                shared = overlap_pixels(track.last, component)
                separation = distance(track.last, component)
                if shared == 0 and separation > maximum_step_px * gap:
                    continue
                choices.append(((shared, -separation, -gap), track))
            if choices:
                track = max(choices, key=lambda row: row[0])[1]
                available.remove(track.number)
            else:
                track = Tracklet(next_number, component.observed_identity)
                tracklets[next_number] = track
                next_number += 1
            component.tracklet = track.number
            component.local_age = len(track.components) + 1
            track.components.append(component)
        frames.append(current)
    return frames, tracklets


def component_mask(component: Component, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], bool)
    mask[component.pixels] = True
    return mask.reshape(shape)


def detect_takeovers(labels: np.ndarray, lag: np.ndarray, cfg: dict,
                     source_offset: int,
                     ) -> tuple[pd.DataFrame, list[list[Component]],
                                dict[int, Tracklet]]:
    frames, tracklets = build_tracklets(
        labels, float(cfg["tracklet_maximum_step_px"]),
        int(cfg["tracklet_maximum_gap_frames"]),
        int(cfg["minimum_component_area_px"]))
    rows = []
    for t in range(len(labels) - 1):
        next_flat = labels[t + 1].ravel()
        next_by_identity: dict[int, list[Component]] = {}
        current_by_identity: dict[int, list[Component]] = {}
        for item in frames[t + 1]:
            next_by_identity.setdefault(item.observed_identity, []).append(item)
        for item in frames[t]:
            current_by_identity.setdefault(item.observed_identity, []).append(item)
        for owner in frames[t]:
            if (owner.area < int(cfg["minimum_owner_area_px"])
                    or owner.local_age < int(cfg["minimum_owner_age_frames"])):
                continue
            values, counts = np.unique(next_flat[owner.pixels], return_counts=True)
            alternatives = [(int(count), int(value))
                            for value, count in zip(values, counts)
                            if int(value) not in (0, owner.observed_identity)
                            and int(count) >= int(cfg["minimum_transfer_px"])]
            if not alternatives:
                continue
            transferred, challenger_identity = max(alternatives)
            destination = max(
                next_by_identity.get(challenger_identity, []),
                key=lambda item: overlap_pixels(owner, item), default=None)
            if destination is None:
                continue
            source_coverage = transferred / max(owner.area, 1)
            if source_coverage < float(cfg["minimum_owner_transfer_fraction"]):
                continue
            owner_next = best_component(
                next_by_identity.get(owner.observed_identity, []), owner,
                float(cfg["tracklet_maximum_step_px"]))
            owner_next_area = owner_next.area if owner_next is not None else 0
            owner_loss = 1.0 - owner_next_area / max(owner.area, 1)
            if owner_loss < float(cfg["minimum_owner_loss_fraction"]):
                continue
            challenger_previous = best_component(
                current_by_identity.get(challenger_identity, []), destination,
                float(cfg["challenger_search_px"]))
            challenger_previous_area = (challenger_previous.area
                                        if challenger_previous is not None else 0)
            growth = destination.area / max(challenger_previous_area, 1)
            if growth < float(cfg["minimum_challenger_growth_ratio"]):
                continue
            if destination.local_age > int(cfg["maximum_challenger_age_frames"]):
                continue
            transition = lag[t] if t < len(lag) else np.zeros(labels.shape[1:])
            owner_mask = component_mask(owner, labels.shape[1:])
            destination_mask = component_mask(destination, labels.shape[1:])
            threshold = float(cfg["motion_threshold"])
            blue = float(np.mean(transition[owner_mask] <= -threshold))
            red = float(np.mean(transition[destination_mask] >= threshold))
            red_destinations = []
            for possible in frames[t + 1]:
                if distance(owner, possible) > float(
                        cfg["maximum_motion_release_radius_px"]):
                    continue
                possible_mask = component_mask(possible, labels.shape[1:])
                coverage = float(np.mean(transition[possible_mask] >= threshold))
                if coverage >= float(cfg["minimum_red_destination_for_release"]):
                    red_destinations.append((coverage, possible))
            paired_release = bool(
                blue >= float(cfg["minimum_blue_departure_for_release"])
                and red_destinations)
            if paired_release:
                continue
            rows.append({
                "event_id": "",
                "source_output_frame": t + 1,
                "destination_output_frame": t + 2,
                "source_imagej_frame": t + 1 + source_offset,
                "destination_imagej_frame": t + 2 + source_offset,
                "owner_identity": owner.observed_identity,
                "challenger_identity": challenger_identity,
                "owner_tracklet": owner.tracklet,
                "challenger_tracklet": destination.tracklet,
                "owner_local_component": owner.local_component,
                "destination_local_component": destination.local_component,
                "owner_area_px": owner.area,
                "owner_next_area_px": owner_next_area,
                "challenger_previous_area_px": challenger_previous_area,
                "challenger_destination_area_px": destination.area,
                "transferred_px": transferred,
                "owner_transfer_fraction": source_coverage,
                "owner_loss_fraction": owner_loss,
                "challenger_growth_ratio": growth,
                "owner_local_age_frames": owner.local_age,
                "challenger_local_age_frames": destination.local_age,
                "owner_blue_departure_coverage": blue,
                "challenger_red_arrival_coverage": red,
                "paired_motion_release": False,
                "owner_x": owner.x, "owner_y": owner.y,
                "destination_x": destination.x, "destination_y": destination.y,
            })
    table = pd.DataFrame(rows)
    if len(table):
        table.insert(0, "rank", range(1, len(table) + 1))
        table["event_id"] = [f"T{number:04d}" for number in range(1, len(table) + 1)]
    return table, frames, tracklets

