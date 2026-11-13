"""Split a terminal two-soma contact when its host name switches to an alien name.

Only accepted masks, raw images, physical points and protected pixels enter
this producer. Neither feedback nor recording, track, identity or frame targets
are inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import complete_owner_handoff_episodes_r10_a002 as body
import owner_consensus
import recover_stable_dual_soma_contacts as dual
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    terminal_window_frames: int = 3
    minimum_basin_reference_area_ratio: float = .25
    maximum_basin_reference_area_ratio: float = 4.0


def _contact(frame: np.ndarray, raw: np.ndarray, points: list,
             all_visible_points: list,
             anchors: dict, reference_areas: dict, protected: np.ndarray,
             partition_params: body.Params, params: Params) -> dict | None:
    """Return a complete two-name partition for one shared outline."""
    if len(points) != 2:
        return None
    somata = list(points)
    tracks = [int(p.track_id) for p in somata]
    if (len(set(tracks)) != 2 or any(track not in anchors for track in tracks)
            or not body._distinct(somata[0], somata[1], partition_params)):
        return None
    names = [int(anchors[track]["owner"]) for track in tracks]
    if names[0] == names[1] or any(reference_areas.get(track, 0) <= 0
                                   for track in tracks):
        return None
    host = int(somata[0].accepted_owner)
    if host <= 0 or any(int(p.accepted_owner) != host for p in somata):
        return None
    mask = body._at_component(frame, host, somata[0])
    if (mask is None or any(not body._inside(mask, p) for p in somata)
            or np.any(mask & protected)
            or any(np.any(frame[~mask] == name) for name in names)):
        return None
    # A weak or interpolated third soma is still a separate physical cell.
    # Never assign its pixels to one of the two strong named residents.
    if {int(point.track_id) for point in all_visible_points
            if body._inside(mask, point)} != set(tracks):
        return None
    first = body._partition(mask, raw, somata[0], somata[1], partition_params)
    if first is None:
        return None
    basins = [first, mask & ~first]
    areas = [int(basin.sum()) for basin in basins]
    if any(not params.minimum_basin_reference_area_ratio
           <= area / reference_areas[track]
           <= params.maximum_basin_reference_area_ratio
           for area, track in zip(areas, tracks)):
        return None
    return {"host": host, "tracks": tracks, "names": names,
            "mask": mask, "basins": basins, "areas": areas}


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params(),
            partition_params: body.Params = body.Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected pixels must align")
    scored = physical.attach_owners(points, labels)
    for field in ("strong", "physically_visible", "in_encounter"):
        scored[field] = scored[field].astype(bool)
    anchors = dual._references(scored, dual.Params())
    by_track = {int(track): group for track, group in scored.groupby("track_id")}
    # Only terminal pairs can reach the area gate. Preserve the same
    # reference-area calculation while avoiding every unrelated track.
    reference_areas = {}
    terminal = scored[
        scored.frame.astype(int).ge(max(0, len(labels)-params.terminal_window_frames))
        & scored.physically_visible & scored.strong
        & scored.state.astype(str).eq("observed")]
    visible_by_frame = {
        int(frame): list(group.itertuples(index=False))
        for frame, group in scored[scored.physically_visible].groupby("frame")}
    contacts = {}
    for frame, rows in terminal.groupby("frame"):
        frame = int(frame)
        for host, group in rows.groupby("accepted_owner"):
            host = int(host)
            if host <= 0:
                continue
            # Group by connected host component, so distinct contacts cannot
            # become one proposal merely because they share a name.
            components, _ = ndi.label(labels[frame] == host,
                                      np.ones((3, 3), np.uint8))
            seats = {}
            for point in group.itertuples(index=False):
                y, x = int(round(float(point.y))), int(round(float(point.x)))
                if 0 <= y < components.shape[0] and 0 <= x < components.shape[1]:
                    component = int(components[y, x])
                    if component:
                        seats.setdefault(component, []).append(point)
            for component, occupants in seats.items():
                if len(occupants) != 2:
                    continue
                names = [int(anchors[int(p.track_id)]["owner"])
                         if int(p.track_id) in anchors else 0 for p in occupants]
                if 0 in names or names[0] == names[1]:
                    continue
                for point in occupants:
                    track = int(point.track_id)
                    if track not in reference_areas:
                        anchor = anchors[track]
                        reference_areas[track] = dual._reference_area(
                            labels, by_track[track], int(anchor["owner"]), anchor)
                contact = _contact(labels[frame], raw[frame], occupants,
                                   visible_by_frame.get(frame, []),
                                   anchors, reference_areas, protected[frame],
                                   partition_params, params)
                if contact is not None:
                    contacts[(frame, host, component)] = contact
    proposals = []
    for (later_frame, later_host, _), later in contacts.items():
        if later_host in later["names"]:
            continue
        for (earlier_frame, earlier_host, _), earlier in contacts.items():
            if (earlier_frame != later_frame-1
                    or earlier_host not in earlier["names"]
                    or set(earlier["tracks"]) != set(later["tracks"]) 
                    or set(earlier["names"]) != set(later["names"])):
                continue
            proposals.append((earlier_frame, earlier, later_frame, later))
    # Refuse overlapping evidence rather than choosing an arbitrary pair.
    candidate = labels.copy()
    applications = []
    selected = []
    for index, (t0, early, t1, late) in enumerate(proposals):
        if any((t0 == other[0] and np.any(early["mask"] & other[1]["mask"]))
               or (t1 == other[2] and np.any(late["mask"] & other[3]["mask"]))
               for other_index, other in enumerate(proposals)
               if other_index != index):
            continue
        if any((frame == prior_frame and np.any(contact["mask"] & prior_mask))
               for frame, contact in ((t0, early), (t1, late))
               for prior_frame, prior_mask in selected):
            continue
        for frame, contact in ((t0, early), (t1, late)):
            for name, basin in zip(contact["names"], contact["basins"]):
                candidate[frame][basin] = name
            selected.append((frame, contact["mask"]))
            applications.append({"frame": frame+1, "old_host": contact["host"],
                                 "track_a": contact["tracks"][0],
                                 "track_b": contact["tracks"][1],
                                 "name_a": contact["names"][0],
                                 "name_b": contact["names"][1],
                                 "pixels": int(contact["mask"].sum()),
                                 "basin_a_px": contact["areas"][0],
                                 "basin_b_px": contact["areas"][1]})
    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))
            or owner_consensus.count_new_duplicate_components(labels, candidate)):
        raise AssertionError("terminal name-switch layer crossed a protected boundary")
    audit = pd.DataFrame([{"earlier_frame":t0+1, "later_frame":t1+1,
                           "earlier_host":early["host"],"later_host":late["host"],
                           "tracks":"|".join(map(str, early["tracks"])),
                           "names":"|".join(map(str, early["names"]))}
                          for t0,early,t1,late in proposals])
    return candidate, audit, pd.DataFrame(applications)
