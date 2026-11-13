"""Repair repeated renaming of one soma when its old name gains a new holder.

Eligibility uses complete physical tracks and accepted labels only.  The rule
does not receive feedback, recording names, identities, coordinates or frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_visible_fraction: float = .90
    minimum_strong_fraction: float = .80
    maximum_first_visible_frame: int = 3
    maximum_last_visible_gap: int = 1
    minimum_anchor_frames: int = 20
    minimum_anchor_strong_fraction: float = .80
    minimum_later_labelled_frames: int = 30
    minimum_later_names: int = 3
    maximum_step_p95_radii_per_frame: float = 1.0
    maximum_step_radii_per_frame: float = 2.5
    minimum_holder_frames: int = 30
    minimum_holder_strong_fraction: float = .75
    maximum_holder_step_equivalent_radii: float = 3.0
    maximum_holder_track_head_start: int = 2
    minimum_holder_separation_radii: float = 4.0


def _runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        frame, owner = int(row.frame), int(row.accepted_owner)
        if (runs and runs[-1]["owner"] == owner
                and runs[-1]["end"] + 1 == frame):
            runs[-1]["end"] = frame
            runs[-1]["rows"].append(row)
        else:
            runs.append({"owner": owner, "start": frame, "end": frame,
                         "rows": [row]})
    return runs


def _component_at(frame: np.ndarray, name: int, row) -> np.ndarray | None:
    parts, _ = ndi.label(frame == int(name), STRUCTURE)
    y, x = round(float(row.y)), round(float(row.x))
    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and parts[y, x]:
        return parts == int(parts[y, x])
    yy, xx = np.where(parts > 0)
    if not len(yy):
        return None
    distance = np.hypot(yy-float(row.y), xx-float(row.x))
    best = int(np.argmin(distance))
    if float(distance[best]) > max(1.0, float(row.radius_px)):
        return None
    return parts == int(parts[yy[best], xx[best]])


def _component_records(labels: np.ndarray, name: int, first_frame: int,
                       strong_by_frame: dict[int, pd.DataFrame]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for frame in range(first_frame, len(labels)):
        parts, count = ndi.label(labels[frame] == int(name), STRUCTURE)
        records = []
        for component in range(1, count+1):
            mask = parts == component
            yy, xx = np.where(mask)
            strong_tracks = []
            strong_points = strong_by_frame.get(frame)
            if strong_points is not None:
                for point in strong_points.itertuples(index=False):
                    y, x = round(float(point.y)), round(float(point.x))
                    if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                            and mask[y, x]):
                        strong_tracks.append(int(point.track_id))
            records.append({
                "frame": frame,
                "mask": mask,
                "area": int(mask.sum()),
                "y": float(yy.mean()),
                "x": float(xx.mean()),
                "radius": float(np.sqrt(mask.sum()/np.pi)),
                "strong_tracks": tuple(sorted(set(strong_tracks))),
            })
        result[frame] = records
    return result


def _holder_chain(records: dict[int, list[dict]], switch: int, frames: int,
                  params: Params) -> tuple[list[dict], str]:
    chains = []
    for start in range(switch, frames):
        for seed in records.get(start, []):
            if not seed["strong_tracks"]:
                continue
            chain = [seed]
            for frame in range(start+1, frames):
                choices = []
                previous = chain[-1]
                for item in records.get(frame, []):
                    distance = float(np.hypot(item["x"]-previous["x"],
                                              item["y"]-previous["y"]))
                    scale = max((item["radius"]+previous["radius"])/2, 1.0)
                    choices.append((distance/scale, distance, item))
                if not choices:
                    break
                choices.sort(key=lambda value: (value[0], value[1], -value[2]["area"]))
                if choices[0][0] > params.maximum_holder_step_equivalent_radii:
                    break
                chain.append(choices[0][2])
            if chain[-1]["frame"] == frames-1:
                chains.append(chain)
    if not chains:
        return [], "no_anchor_holder_reaches_movie_end"
    chains.sort(key=lambda chain: (-len(chain), chain[0]["frame"],
                                   -sum(bool(item["strong_tracks"]) for item in chain)))
    best = chains[0]
    if len(best) < params.minimum_holder_frames:
        return [], "end_reaching_holder_too_short"
    if (sum(bool(item["strong_tracks"]) for item in best)
            < params.minimum_holder_strong_fraction*len(best)):
        return [], "holder_lacks_sustained_strong_soma"
    same_length = [chain for chain in chains[1:] if len(chain) == len(best)
                   and chain[0]["frame"] == best[0]["frame"]]
    if same_length:
        return [], "multiple_equal_end_reaching_holders"
    return best, "supported"


def correct(labels: np.ndarray, points: pd.DataFrame, protected: np.ndarray,
            params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != protected.shape:
        raise ValueError("labels and protected masks must be aligned movies")
    scored = physical.attach_owners(points, labels)
    for field in ("physically_visible", "strong"):
        scored[field] = scored[field].astype(bool)
    visible = scored[scored.physically_visible].copy()
    strong = scored[scored.state.astype(str).eq("observed")
                    & scored.strong.astype(bool)].copy()
    strong_by_frame = {int(frame): group for frame, group in strong.groupby("frame")}
    first_visible = visible.groupby("track_id").frame.min().astype(int).to_dict()
    candidate = labels.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    audits: list[dict] = []
    applications: list[dict] = []
    next_name = int(labels.max()) + 1

    for track, group in visible.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        if (len(group) < params.minimum_visible_fraction*len(labels)
                or int(group.frame.min()) > params.maximum_first_visible_frame
                or int(group.frame.max()) < len(labels)-1-params.maximum_last_visible_gap
                or int(group.strong.sum()) < params.minimum_strong_fraction*len(labels)):
            continue
        runs = _runs(group)
        anchors = [run for run in runs if run["owner"] > 0
                   and len(run["rows"]) >= params.minimum_anchor_frames
                   and sum(bool(row.strong) for row in run["rows"])
                   >= params.minimum_anchor_strong_fraction*len(run["rows"])]
        if not anchors:
            continue
        anchor = anchors[0]
        first_name = int(anchor["owner"])
        switch = int(anchor["end"])+1
        later = group[group.frame.astype(int).ge(switch)].copy()
        later_named = later[later.accepted_owner.astype(int).gt(0)]
        later_names = sorted(set(later_named.accepted_owner.astype(int))-{first_name})
        if (len(later_named) < params.minimum_later_labelled_frames
                or len(later_names) < params.minimum_later_names
                or later_named.accepted_owner.astype(int).eq(first_name).any()
                or int(later.frame.max()) < len(labels)-1-params.maximum_last_visible_gap):
            continue
        # Once the name anchor is established, assess the path that would be
        # changed.  Earlier ownerless movement can be cell entry or tracker
        # acquisition and is outside this identity correction.
        motion_group = group[group.frame.astype(int).ge(int(anchor["start"]))]
        positions = motion_group[["y", "x"]].to_numpy(dtype=float)
        radii = np.maximum(motion_group.radius_px.to_numpy(dtype=float), 1.0)
        gaps = np.maximum(np.diff(motion_group.frame.to_numpy(dtype=float)), 1.0)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        scaled = steps/np.maximum((radii[:-1]+radii[1:])/2, 1.0)/gaps
        if (len(scaled) and (np.percentile(scaled, 95)
                            > params.maximum_step_p95_radii_per_frame
                            or np.max(scaled) > params.maximum_step_radii_per_frame)):
            continue

        target_records = []
        reason = "supported"
        for row in later_named.itertuples(index=False):
            frame, owner = int(row.frame), int(row.accepted_owner)
            if not bool(row.strong) or str(row.state) != "observed":
                reason = "later_named_target_is_not_strong"
                break
            target = _component_at(labels[frame], owner, row)
            if target is None:
                reason = "target_soma_has_no_component"
                break
            other_strong = []
            for point in strong_by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
                if int(point.track_id) == int(track):
                    continue
                y, x = round(float(point.y)), round(float(point.x))
                if 0 <= y < target.shape[0] and 0 <= x < target.shape[1] and target[y, x]:
                    other_strong.append(int(point.track_id))
            if other_strong:
                reason = "target_component_contains_another_strong_soma"
                break
            if np.any(protected[frame] & target):
                reason = "target_crosses_prior_accepted_pixel"
                break
            target_records.append({"frame": frame, "mask": target,
                                   "old_name": owner, "area": int(target.sum()),
                                   "row": row})

        holder = []
        holder_tracks: set[int] = set()
        if reason == "supported":
            records = _component_records(labels, first_name, switch, strong_by_frame)
            holder, reason = _holder_chain(records, switch, len(labels), params)
        if reason == "supported":
            target_by_frame = {item["frame"]: item for item in target_records}
            for item in holder:
                frame = item["frame"]
                holder_tracks.update(item["strong_tracks"])
                if np.any(protected[frame] & item["mask"]):
                    reason = "holder_crosses_prior_accepted_pixel"
                    break
                target = target_by_frame.get(frame)
                if target is not None:
                    row = target["row"]
                    separation = float(np.hypot(item["x"]-float(row.x),
                                                item["y"]-float(row.y)))
                    scale = max(item["radius"]+float(row.radius_px), 1.0)
                    if separation < params.minimum_holder_separation_radii*scale:
                        reason = "holder_not_spatially_independent"
                        break
        if reason == "supported":
            holder_start = int(holder[0]["frame"])
            if not holder_tracks:
                reason = "holder_has_no_strong_track"
            elif any(int(first_visible[other]) < holder_start-params.maximum_holder_track_head_start
                     for other in holder_tracks):
                reason = "holder_track_predates_holder_path"
            else:
                for other in holder_tracks:
                    prior = visible[(visible.track_id.astype(int).eq(other))
                                    & visible.frame.astype(int).lt(holder_start)
                                    & visible.accepted_owner.astype(int).gt(0)]
                    if len(prior):
                        reason = "holder_track_has_earlier_positive_identity"
                        break
        if reason == "supported" and any(
                np.any(occupied[item["frame"]] & item["mask"])
                for item in target_records+holder):
            reason = "overlaps_another_proposal"

        audit = {
            "track_id": int(track),
            "anchor_name": first_name,
            "anchor_first_frame": int(anchor["start"])+1,
            "anchor_last_frame": int(anchor["end"])+1,
            "anchor_frames": len(anchor["rows"]),
            "later_labelled_frames": len(later_named),
            "later_names": "|".join(map(str, later_names)),
            "target_pixels": sum(item["area"] for item in target_records),
            "holder_first_frame": int(holder[0]["frame"])+1 if holder else 0,
            "holder_frames": len(holder),
            "holder_strong_frames": sum(bool(item["strong_tracks"]) for item in holder),
            "holder_tracks": "|".join(map(str, sorted(holder_tracks))),
            "holder_pixels": sum(item["area"] for item in holder),
            "step_p95_radii_per_frame": float(np.percentile(scaled, 95)) if len(scaled) else 0.0,
            "outcome": "applied" if reason == "supported" else "rejected",
            "reason": reason,
        }
        audits.append(audit)
        if reason != "supported":
            continue
        if next_name > np.iinfo(labels.dtype).max:
            raise OverflowError("no label value remains for the emergent holder")
        for item in target_records:
            frame, mask = item["frame"], item["mask"]
            candidate[frame][mask] = first_name
            occupied[frame][mask] = True
            applications.append({
                "kind": "restore_repeatedly_renamed_target",
                "track_id": int(track), "frame": frame+1,
                "old_name": item["old_name"], "new_name": first_name,
                "changed_pixels": item["area"],
            })
        for item in holder:
            frame, mask = item["frame"], item["mask"]
            candidate[frame][mask] = next_name
            occupied[frame][mask] = True
            applications.append({
                "kind": "name_emergent_holder_path",
                "track_id": int(track), "frame": frame+1,
                "old_name": first_name, "new_name": next_name,
                "changed_pixels": item["area"],
            })
        audits[-1]["new_holder_name"] = next_name
        next_name += 1

    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate > 0, labels > 0)):
        raise AssertionError("repeated-renaming correction crossed its protected boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
