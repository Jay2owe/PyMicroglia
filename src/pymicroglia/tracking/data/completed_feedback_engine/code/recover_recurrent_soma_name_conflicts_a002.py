"""Return a recurrent soma name and resolve independently anchored borrowers.

Discovery is fieldwide from approved masks, raw images and physical points.
Feedback, recording, identity, tracker, frame and coordinates do not select
an application. Accepted edits and foreground are immutable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import recover_dormant_name_soma_basins as prior
import separable_merge_recovery as physical
import resident_takeover as owner


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_path_movie_fraction: float = .15
    maximum_path_movie_fraction: float = .60
    minimum_strong_movie_fraction: float = .10
    minimum_later_return_movie_fraction: float = .02
    required_later_other_names: int = 2
    minimum_sole_reference_movie_fraction: float = .03
    maximum_step_p95_radii: float = 1.5
    maximum_single_step_radii: float = 2.5
    maximum_external_fragment_radii: float = 4.
    maximum_cross_soma_valley_ratio: float = .5
    minimum_raw_peak_retention: float = .4
    minimum_basin_area_ratio: float = .2
    maximum_basin_area_ratio: float = 1.75
    minimum_source_basin_fraction: float = .15
    minimum_changed_movie_fraction: float = .03
    minimum_changed_cell_equivalent_frames: float = 4.
    minimum_contact_frames: int = 1


def _peak(frame: np.ndarray, point) -> float:
    y, x = round(float(point.y)), round(float(point.x))
    return float(frame[max(0, y-2):min(frame.shape[0], y+3),
                       max(0, x-2):min(frame.shape[1], x+3)].max())


def _anchor(group: pd.DataFrame, strong: pd.DataFrame,
            movie_frames: int, params: Params) -> int | None:
    if (not math.ceil(movie_frames*params.minimum_path_movie_fraction)
            <= len(group) <= math.ceil(movie_frames*params.maximum_path_movie_fraction)
            or len(strong) < math.ceil(movie_frames*params.minimum_strong_movie_fraction)
            or not np.all(np.diff(group.frame.to_numpy(int)) == 1)):
        return None
    names = strong.owner.to_numpy(int)
    name = int(names[0])
    if name <= 0 or len(names) < 2 or int(names[1]) != name:
        return None
    first_other = np.flatnonzero(names != name)
    if not len(first_other):
        return None
    later = names[int(first_other[0]):]
    if (int(np.count_nonzero(later == name))
            < math.ceil(movie_frames*params.minimum_later_return_movie_fraction)
            or len(set(map(int, later)) - {0, name}) != params.required_later_other_names):
        return None
    xy = group[["y", "x"]].to_numpy(float)
    radius = np.maximum(group.radius_px.to_numpy(float), 1.)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(
        (radius[1:] + radius[:-1]) / 2, 1.)
    if (float(np.percentile(steps, 95)) > params.maximum_step_p95_radii
            or float(np.max(steps)) > params.maximum_single_step_radii):
        return None
    return name


def _external_anchor_is_local(labels: np.ndarray, t: int, name: int,
                              point, track: int, by_frame: dict[int, list],
                              params: Params) -> bool:
    parts, count = ndi.label(labels[t] == name, STRUCTURE)
    for cid in range(1, count+1):
        yy, xx = np.where(parts == cid)
        distance = float(np.min(np.hypot(yy-float(point.y), xx-float(point.x))))
        if distance > params.maximum_external_fragment_radii * max(float(point.radius_px), 1.):
            return False
        if any(int(other.track_id) != track
               and parts[round(float(other.y)), round(float(other.x))] == cid
               for other in by_frame.get(t, [])):
            return False
    return True


def _external_holder_repair(labels: np.ndarray, t: int, name: int,
                            point, track: int, strong: pd.DataFrame,
                            scored: pd.DataFrame,
                            reference_area: float,
                            by_frame: dict[int, list], protected: np.ndarray,
                            params: Params) -> list[tuple[np.ndarray, int, int]] | None:
    """Replace a borrowed name on a separate soma with its own recurrent name.

    Every distant occurrence must have exactly one independent strong soma.
    Its alternate name must recur on the same raw path, including an earlier
    observation, and must have no other strong holder after this transaction.
    """
    parts, count = ndi.label(labels[t] == name, STRUCTURE)
    repairs = []
    for cid in range(1, count+1):
        mask = parts == cid
        yy, xx = np.where(mask)
        distance = float(np.min(np.hypot(yy-float(point.y),
                                         xx-float(point.x))))
        holders = [other for other in by_frame.get(t, [])
                   if int(other.track_id) != track
                   and parts[round(float(other.y)), round(float(other.x))] == cid]
        if (distance <= params.maximum_external_fragment_radii
                * max(float(point.radius_px), 1.) and not holders):
            continue
        if not holders:
            # A small detached island can belong to a neighbouring strong
            # soma even when its centre falls outside the island itself.
            nearby = []
            for other in by_frame.get(t, []):
                if int(other.track_id) == track or int(other.owner) in (0, name):
                    continue
                distance_to_island = float(np.min(np.hypot(
                    yy-float(other.y), xx-float(other.x))))
                if distance_to_island <= float(other.radius_px):
                    nearby.append(other)
            if len(nearby) != 1 or len(yy) > reference_area:
                return None
            other = nearby[0]
            if np.any(protected[t] & mask):
                return None
            repairs.append((mask, int(other.owner), int(other.track_id)))
            continue
        if len(holders) != 1 or np.any(protected[t] & mask):
            return None
        holder = holders[0]
        lineage = strong[strong.track_id == int(holder.track_id)]
        alternatives = lineage[(lineage.owner > 0) & (lineage.owner != name)]
        if alternatives.empty:
            return None
        counts = alternatives.owner.value_counts()
        if (counts.iloc[0] < max(2, math.ceil(len(labels)*.02))
                or len(counts) > 1 and counts.iloc[0] == counts.iloc[1]):
            return None
        alternate = int(counts.index[0])
        if not bool(((alternatives.frame < t)
                     & alternatives.owner.eq(alternate)).any()):
            return None
        rival = [other for other in by_frame.get(t, [])
                 if int(other.track_id) not in (track, int(holder.track_id))
                 and int(other.owner) == alternate]
        if rival:
            gap = max(1, math.ceil(len(labels)*.03))
            for other in rival:
                histories = []
                for side in (-1, 1):
                    window = scored[
                        scored.frame.between(t-gap, t-1) if side < 0
                        else scored.frame.between(t+1, t+gap)]
                    left = window[(window.track_id == int(holder.track_id))
                                  & window.state.isin(["observed", "latent_visible"])]
                    right = window[(window.track_id == int(other.track_id))
                                   & window.state.isin(["observed", "latent_visible"])]
                    pair = left.merge(right, on="frame", suffixes=("_left", "_right"))
                    if pair.empty:
                        return None
                    distance = np.hypot(pair.x_left-pair.x_right,
                                        pair.y_left-pair.y_right)
                    separated = distance > np.maximum(
                        pair.radius_px_left, pair.radius_px_right)
                    coherent = [
                        bool(separated.iloc[i])
                        and owner._disk_owner(
                            labels[int(row.frame)], float(row.x_left),
                            float(row.y_left), float(row.radius_px_left))[0] == alternate
                        and owner._disk_owner(
                            labels[int(row.frame)], float(row.x_right),
                            float(row.y_right), float(row.radius_px_right))[0] == alternate
                        for i, row in enumerate(pair.itertuples(index=False))]
                    histories.append(any(coherent))
                if not all(histories):
                    return None
        repairs.append((mask, alternate, int(holder.track_id)))
    return repairs


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or labels.shape != raw.shape
            or labels.shape != protected.shape):
        raise ValueError("aligned complete labels, raw and protected pixels required")
    scored = points.copy()
    scored["strong"] = scored.strong.astype(bool)
    scored["owner"] = [prior._point_owner(labels, p)
                       for p in scored.itertuples(index=False)]
    strong = scored[(scored.state.astype(str) == "observed") & scored.strong]
    by_frame = {int(t): list(g.itertuples(index=False))
                for t, g in strong.groupby("frame")}
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    remaining = np.bincount(labels.ravel().astype(np.int64))
    audit, applications, borrowers = [], [], []
    for track, group in scored.groupby("track_id", sort=True):
        track = int(track)
        group = group.sort_values("frame")
        sole_strong = strong[strong.track_id == track].sort_values("frame")
        name = _anchor(group, sole_strong, len(labels), params)
        if name is None:
            continue
        refs = []
        first_other_frame = int(sole_strong[sole_strong.owner != name].frame.min())
        early_refs = late_refs = 0
        for p in sole_strong[sole_strong.owner == name].itertuples(index=False):
            t = int(p.frame)
            mask = prior._component(labels[t], p)
            if mask is None:
                continue
            if any(int(other.track_id) != track
                   and mask[round(float(other.y)), round(float(other.x))]
                   for other in by_frame.get(t, [])):
                continue
            refs.append((int(mask.sum()), _peak(raw[t], p)))
            if t < first_other_frame:
                early_refs += 1
            else:
                late_refs += 1
        if (len(refs) < math.ceil(len(labels)*params.minimum_sole_reference_movie_fraction)
                or early_refs < 2 or late_refs < 1):
            continue
        ref_area = float(np.median([r[0] for r in refs]))
        ref_peak = float(np.median([r[1] for r in refs]))
        proposals = []
        refusals = {}
        wrong_strong = 0
        for p in sole_strong.itertuples(index=False):
            t, old = int(p.frame), int(p.owner)
            if t < first_other_frame or old <= 0 or old == name:
                continue
            wrong_strong += 1
            if _external_anchor_is_local(labels, t, name, p, track,
                                         by_frame, params):
                holders = []
            else:
                holders = _external_holder_repair(
                    labels, t, name, p, track, strong, scored, ref_area, by_frame,
                    protected, params)
                if holders is None:
                    refusals[t] = "unresolved_independent_name_holder"
                    continue
            mask = prior._component(labels[t], p)
            if mask is None:
                refusals[t] = "source_component_missing"
                continue
            peers = [q for q in by_frame.get(t, []) if int(q.track_id) != track
                     and mask[round(float(q.y)), round(float(q.x))]]
            if any(not np.isfinite(v := physical._valley_ratio(
                   raw[t], (float(p.x), float(p.y)),
                   (float(q.x), float(q.y))))
                   or v > params.maximum_cross_soma_valley_ratio for q in peers):
                refusals[t] = "raw_somata_not_separable"
                continue
            if _peak(raw[t], p) < params.minimum_raw_peak_retention * ref_peak:
                refusals[t] = "raw_peak_not_retained"
                continue
            selected = prior._basin(mask, raw[t], p, peers, prior.Params())
            if selected is None:
                refusals[t] = "raw_basin_missing"
                continue
            area = int(selected.sum())
            if (not params.minimum_basin_area_ratio <= area/ref_area
                    <= params.maximum_basin_area_ratio
                    or area < params.minimum_source_basin_fraction * int(mask.sum())):
                refusals[t] = "implausible_raw_basin"
                continue
            if np.any(protected[t] & selected):
                refusals[t] = "accepted_pixel_in_basin"
                continue
            proposals.append((t, old, selected, area, int(mask.sum()),
                              len(peers), holders))
        total = sum(row[3] for row in proposals)
        contact_frames = sum(row[5] > 0 for row in proposals)
        reason = ""
        if (len(proposals) < math.ceil(len(labels)*params.minimum_changed_movie_fraction)
                or total < params.minimum_changed_cell_equivalent_frames*ref_area
                or contact_frames < params.minimum_contact_frames):
            reason = "insufficient_supported_path"
        elif any(np.any(candidate[t][selected] != old)
                 or np.any(occupied[t][selected])
                 or any(np.any(candidate[t][holder_mask] != name)
                        or np.any(occupied[t][holder_mask])
                        for holder_mask, _, _ in holders)
                 for t, old, selected, _, _, _, holders in proposals):
            reason = "overlaps_prior_proposal"
        else:
            taken = {}
            for _, old, _, area, _, _, _ in proposals:
                taken[old] = taken.get(old, 0) + area
            if any(remaining[old] <= amount for old, amount in taken.items()):
                reason = "would_remove_established_name"
        audit.append({"track_id": track, "anchor_name": name,
                      "first_frame": int(group.frame.min())+1,
                      "last_frame": int(group.frame.max())+1,
                      "wrong_strong_frames": wrong_strong,
                      "supported_frames": len(proposals),
                      "supported_contact_frames": contact_frames,
                      "changed_pixels": total if not reason else 0,
                      "refusals": "|".join(f"{t+1}:{why}" for t, why in sorted(refusals.items())),
                      "outcome": "applied" if not reason else reason})
        if reason:
            continue
        for t, old, selected, area, source_area, peers, holders in proposals:
            candidate[t][selected] = name
            occupied[t][selected] = True
            remaining[old] -= area
            remaining[name] += area
            applications.append({"track_id": track, "frame": t+1,
                                 "target_track_id": track,
                                 "old_name": old, "new_name": name,
                                 "changed_pixels": area,
                                 "source_area": source_area,
                                 "companion_somata": peers,
                                 "reference_area": ref_area,
                                 "role": "recurrent_target"})
            for holder_mask, alternate, holder_track in holders:
                holder_area = int(holder_mask.sum())
                candidate[t][holder_mask] = alternate
                occupied[t][holder_mask] = True
                remaining[name] -= holder_area
                remaining[alternate] += holder_area
                applications.append({"track_id": holder_track, "frame": t+1,
                                     "target_track_id": track,
                                     "old_name": name, "new_name": alternate,
                                     "changed_pixels": holder_area,
                                     "source_area": holder_area,
                                     "companion_somata": 0,
                                     "reference_area": ref_area,
                                     "role": "independent_name_borrower"})
                borrowers.append((track, holder_track, name, alternate,
                                  first_other_frame, int(group.frame.max()), ref_area))
    # Once a borrower's own identity is established, retain it even when the
    # main soma is already correctly named. Otherwise the borrower flashes
    # back on frames where only one side of the transaction needed changing.
    for target_track, holder_track, borrowed, alternate, first, last, ref_area in sorted(set(borrowers)):
        path = strong[(strong.track_id == holder_track)
                      & strong.frame.between(first, last)
                      & strong.owner.eq(borrowed)]
        for point in path.itertuples(index=False):
            t = int(point.frame)
            if prior._point_owner(candidate, point) != borrowed:
                continue
            if any(int(other.track_id) != holder_track
                   and prior._point_owner(candidate, other) == alternate
                   for other in by_frame.get(t, [])):
                continue
            mask = prior._component(labels[t], point)
            if mask is None:
                continue
            peers = [other for other in by_frame.get(t, [])
                     if int(other.track_id) != holder_track
                     and mask[round(float(other.y)), round(float(other.x))]]
            if any(not np.isfinite(value := physical._valley_ratio(
                    raw[t], (float(point.x), float(point.y)),
                    (float(other.x), float(other.y))))
                   or value > params.maximum_cross_soma_valley_ratio for other in peers):
                continue
            selected = prior._basin(mask, raw[t], point, peers, prior.Params())
            if (selected is None or not selected.any()
                    or np.any(protected[t] & selected)
                    or np.any(occupied[t] & selected)
                    or np.any(candidate[t][selected] != borrowed)):
                continue
            area = int(selected.sum())
            candidate[t][selected] = alternate
            occupied[t][selected] = True
            applications.append({"track_id": holder_track, "frame": t+1,
                                 "target_track_id": target_track,
                                 "old_name": borrowed, "new_name": alternate,
                                 "changed_pixels": area,
                                 "source_area": int(mask.sum()),
                                 "companion_somata": len(peers),
                                 "reference_area": ref_area,
                                 "role": "borrower_path_continuation"})
    for row in audit:
        row["target_changed_pixels"] = row["changed_pixels"]
        linked = [app for app in applications
                  if app["target_track_id"] == row["track_id"]]
        row["changed_pixels"] = sum(app["changed_pixels"] for app in linked)
        row["borrower_path_frames"] = sum(app["role"] != "recurrent_target" for app in linked)
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("recurrent-name layer crossed accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
