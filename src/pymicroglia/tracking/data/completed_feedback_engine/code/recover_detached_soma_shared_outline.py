"""Recover a dormant cell name through a long two-soma shared outline.

Only accepted labels, raw frames, soma tracks and original unclaimed masks
select proposals. The resident keeps its name; the later soma recovers its
unique earlier identity instead of receiving an invented new one.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

import complete_owner_handoff_episodes_r10_a002 as body
import recover_persistent_large_soma_satellites as partition
import recover_stable_dual_soma_contacts as dual
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_anchor_frames: int = 10
    minimum_visible_frames: int = 40
    minimum_soma_radius_px: float = 3.0
    minimum_consecutive_shared_frames: int = 30
    minimum_independently_detached_frames: int = 2
    minimum_raw_gap_contrast: float = 3.0
    minimum_prior_independent_frames: int = 5
    maximum_motion_fraction: float = .95
    minimum_match_margin: float = 1.5


def _longest_run(frames: list[int]) -> int:
    longest = current = 0
    previous = -10
    for frame in sorted(set(frames)):
        current = current + 1 if frame == previous + 1 else 1
        longest = max(longest, current)
        previous = frame
    return longest


def _strong(point) -> bool:
    return (bool(point.physically_visible) and bool(point.strong)
            and str(point.state) == "observed")


def _dormant_identity(labels: np.ndarray, raw: np.ndarray,
                      scored: pd.DataFrame, anchors: dict,
                      resident_name: int, onset,
                      params: Params) -> tuple[int | None, float]:
    """Find one vanished, previously independent soma within a motion budget."""
    start = int(onset.frame)
    resident = scored[(scored.accepted_owner.astype(int) == resident_name)
                      & scored.strong.astype(bool)
                      & scored.physically_visible.astype(bool)
                      & scored.state.eq("observed")]
    by_frame = {int(frame): list(group.itertuples(index=False))
                for frame, group in resident.groupby("frame")}
    matches = []
    for track, anchor in anchors.items():
        name = int(anchor["owner"])
        if (name <= 0 or name == resident_name
                or int(anchor["support_frames"]) < params.minimum_anchor_frames
                or np.any(labels[start:] == name)):
            continue
        source = scored[(scored.track_id.astype(int) == int(track))
                        & scored.accepted_owner.astype(int).eq(name)
                        & scored.strong.astype(bool)
                        & scored.physically_visible.astype(bool)
                        & scored.state.eq("observed")].sort_values("frame")
        if source.empty:
            continue
        last = source.iloc[-1]
        gap = start-int(last.frame)
        if gap <= 0 or gap > labels.shape[0]/3:
            continue
        prior_separate = 0
        for point in source.itertuples(index=False):
            for other in by_frame.get(int(point.frame), ()):
                if partition._raw_separate(raw[int(point.frame)],point,other):
                    prior_separate += 1
                    break
        if prior_separate < params.minimum_prior_independent_frames:
            continue
        distance = float(np.hypot(float(last.x)-float(onset.x),
                                  float(last.y)-float(onset.y)))
        budget = (float(last.radius_px)+float(onset.radius_px)
                  + 1.25*gap)
        fraction = distance/max(budget,1.)
        if fraction <= params.maximum_motion_fraction:
            matches.append((fraction, name, int(track)))
    if not matches:
        return None, float("inf")
    matches.sort()
    best = matches[0]
    if (len(matches) > 1 and
            matches[1][0] < params.minimum_match_margin*best[0]):
        return None, best[0]
    return best[1], best[0]


# Speed (folded from detached_outline_groups): discover reads the visible,
# frame-sorted rows only for tracks whose anchor has at least
# minimum_anchor_frames support frames, so only those tracks are selected
# (an integer track column selects them exactly with isin), then grouped,
# filtered and sorted with the same statements on the same rows in the same
# order. correct gives every track its visible rows in frame order: with
# integer track and frame columns and a Boolean visibility, when no track
# repeats a frame among its visible rows one stable sort by (track, frame)
# followed by the same groupby gives each track the same rows in the same
# order. Other tables keep the original statements.
def _anchored_groups(scored: pd.DataFrame, anchors: dict, minimum: int) -> dict:
    if (type(scored) is not pd.DataFrame or not scored.columns.is_unique
            or any(name not in scored.columns for name in ("track_id", "physically_visible", "frame"))
            or scored["track_id"].dtype.kind not in "iu" or scored["physically_visible"].dtype != bool):
        return {int(track): group[group.physically_visible].sort_values("frame")
                for track, group in scored.groupby("track_id")}
    needed = [int(track) for track, anchor in anchors.items()
              if int(anchor["support_frames"]) >= minimum]
    picked = scored[scored["track_id"].isin(needed)]
    return {int(track): group[group.physically_visible].sort_values("frame")
            for track, group in picked.groupby("track_id")}


def _visible_track_rows(scored: pd.DataFrame) -> dict:
    columns = scored.columns
    if (type(scored) is pd.DataFrame and columns.is_unique
            and all(name in columns for name in ("track_id", "physically_visible", "frame"))):
        track, visible, frame = scored["track_id"], scored["physically_visible"], scored["frame"]
        if (isinstance(track.dtype, np.dtype) and track.dtype.kind in "iu" and visible.dtype == np.bool_
                and isinstance(frame.dtype, np.dtype) and frame.dtype.kind in "iu"):
            tracks = track.to_numpy()
            shown = np.flatnonzero(visible.to_numpy())
            shown_tracks, frames = tracks[shown], frame.to_numpy()[shown]
            order = np.lexsort((frames, shown_tracks))
            shown_tracks, frames = shown_tracks[order], frames[order]
            if not np.any((shown_tracks[1:] == shown_tracks[:-1]) & (frames[1:] == frames[:-1])):
                rows = {int(key): list(group.itertuples(index=False))
                        for key, group in scored.take(shown[order]).groupby("track_id")}
                return {key: rows.get(key, []) for key in np.unique(tracks).tolist()}
    return {int(track): list(group[group.physically_visible]
                             .sort_values("frame").itertuples(index=False))
            for track, group in scored.groupby("track_id")}


def discover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
             scored: pd.DataFrame, params: Params = Params()) -> list[dict]:
    """Find persistent two-soma outlines with separate raw-object evidence."""
    anchors = dual._references(scored, dual.Params())
    groups = _anchored_groups(scored, anchors, params.minimum_anchor_frames)
    by_name: dict[int, list[int]] = {}
    for track, anchor in anchors.items():
        if int(anchor["support_frames"]) >= params.minimum_anchor_frames:
            by_name.setdefault(int(anchor["owner"]), []).append(int(track))
    candidates = []
    for name, tracks in by_name.items():
        for left_id, right_id in itertools.combinations(sorted(tracks), 2):
            left, right = groups[left_id], groups[right_id]
            if (len(left) < params.minimum_visible_frames
                    or len(right) < params.minimum_visible_frames
                    or min(float(left.radius_px.median()),
                           float(right.radius_px.median()))
                    < params.minimum_soma_radius_px):
                continue
            first = lambda group: int(group[group.apply(_strong, axis=1)].frame.min())
            # Earlier onset owns the established name; tie-break only orders
            # otherwise equal observations deterministically.
            left_key = (first(left), -int(anchors[left_id]["support_frames"]),
                        left_id)
            right_key = (first(right), -int(anchors[right_id]["support_frames"]),
                         right_id)
            earlier_id, later_id = ((left_id, right_id) if left_key < right_key
                                    else (right_id, left_id))
            earlier, later = groups[earlier_id], groups[later_id]
            if int(earlier.frame.min()) >= int(later.frame.min()):
                continue
            if not later.accepted_owner.astype(int).isin((0, name)).all():
                continue
            ea = {int(p.frame): p for p in earlier.itertuples(index=False)
                  if _strong(p)}
            lb = {int(p.frame): p for p in later.itertuples(index=False)
                  if _strong(p)}
            shared = []
            areas = []
            for frame in sorted(ea.keys() & lb.keys()):
                a, b = ea[frame], lb[frame]
                if (int(a.accepted_owner) != name
                        or int(b.accepted_owner) != name):
                    continue
                component = body._at_component(labels[frame], name, a)
                if (component is None or not body._inside(component, b)
                        or not partition._raw_separate(raw[frame], a, b)):
                    continue
                shared.append(frame)
                areas.append(int(component.sum()))
            run = _longest_run(shared)
            if run < params.minimum_consecutive_shared_frames:
                continue
            detached = []
            object_ids = set()
            for point in later.itertuples(index=False):
                if not _strong(point) or int(point.accepted_owner) != 0:
                    continue
                frame = int(point.frame)
                mask = partition._at_unclaimed_component(unclaimed[frame], point)
                if mask is None or not np.any(mask):
                    continue
                if (np.any(labels[frame][mask] != 0)
                        or partition._gap_contrast(raw[frame], point)
                        < params.minimum_raw_gap_contrast):
                    continue
                detached.append(frame)
                object_ids.add(int(unclaimed[frame, round(float(point.y)),
                                             round(float(point.x))]))
            if (len(detached) < params.minimum_independently_detached_frames
                    or len(object_ids) != 1):
                continue
            onset = next(point for point in later.itertuples(index=False)
                         if _strong(point))
            recovered_name, motion_fraction = _dormant_identity(
                labels, raw, scored, anchors, name, onset, params)
            if recovered_name is None:
                continue
            candidates.append({"earlier_track": earlier_id,
                               "later_track": later_id,
                               "shared_name": name,
                               "recovered_name": recovered_name,
                               "motion_fraction": motion_fraction,
                               "shared_raw_frames": len(shared),
                               "longest_shared_run": run,
                               "detached_frames": len(detached),
                               "later_visible_frames": len(later),
                               "median_shared_component_px": float(np.median(areas))})
    return sorted(candidates, key=lambda c: (-c["longest_shared_run"]
                                             * c["median_shared_component_px"],
                                             c["later_track"]))


def correct(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, protected: np.ndarray,
            params: Params = Params()
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if not (labels.shape == unclaimed.shape == raw.shape == protected.shape):
        raise ValueError("labels, raw, unclaimed and protected must align")
    scored = physical.attach_owners(points, labels)
    for field in ("physically_visible", "strong"):
        scored[field] = scored[field].astype(bool)
    by_frame = {int(frame): list(group.itertuples(index=False))
                for frame, group in scored.groupby("frame")}
    by_track = _visible_track_rows(scored)
    output = labels.copy()
    ledger = unclaimed.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    audit, applications = [], []
    path_params = partition.Params()
    for candidate in discover(labels, unclaimed, raw, scored, params):
        companion = {"satellite_track": candidate["later_track"]}
        proposals, reason = partition._propose_path(
            labels, unclaimed, raw, by_frame, by_track, protected,
            companion, path_params)
        if reason == "supported" and any(
                np.any(occupied[p["frame"]] & p["part"])
                for p in proposals):
            reason = "overlaps_previous_proposal"
        if reason == "supported" and len(proposals) != candidate["later_visible_frames"]:
            reason = "incomplete_later_soma_path"
        if reason != "supported":
            audit.append({**candidate, "outcome": "rejected", "reason": reason,
                          "resolved_frames": len(proposals)})
            continue
        new_name = int(candidate["recovered_name"])
        if any(np.any(labels[item["frame"]] == new_name)
               for item in proposals):
            audit.append({**candidate, "outcome": "rejected",
                          "reason": "recovered_name_has_simultaneous_holder",
                          "resolved_frames": len(proposals)})
            continue
        for item in proposals:
            frame, part = item["frame"], item["part"]
            output[frame][part] = new_name
            occupied[frame][part] = True
            if int(item["owner"]) == 0:
                ledger[frame][part] = 0
            applications.append({"earlier_track": candidate["earlier_track"],
                                 "later_track": candidate["later_track"],
                                 "frame": frame + 1,
                                 "old_name": int(item["owner"]),
                                 "new_name": new_name,
                                 "area_px": int(item["area_px"]),
                                 "method": item["method"]})
        audit.append({**candidate, "outcome": "applied",
                      "reason": "supported", "new_name": new_name,
                      "resolved_frames": len(proposals),
                      "changed_pixels": sum(item["area_px"]
                                            for item in proposals)})
    changed = output != labels
    gained = (output > 0) & (labels == 0)
    if (np.any(changed & protected)
            or np.any((labels > 0) & (output == 0))
            or not np.array_equal(ledger[~gained], unclaimed[~gained])
            or np.any(ledger[gained] != 0)
            or int(changed.sum()) != sum(row["area_px"] for row in applications)):
        raise AssertionError("detached soma rule crossed protected boundary")
    return output, ledger, pd.DataFrame(audit), pd.DataFrame(applications)
