"""Restore a short anchored soma name through later shared and split outlines.

Eligibility uses only physical tracks, accepted masks and raw signal. The
caller provides protected accepted pixels; feedback and recording names never
enter the decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

STRUCTURE = np.ones((3, 3), bool)


def _component(frame: np.ndarray, y: int, x: int) -> np.ndarray | None:
    name = int(frame[y, x])
    if name <= 0:
        return None
    parts, _ = ndi.label(frame == name, STRUCTURE)
    part = int(parts[y, x])
    return parts == part if part else None


def _split(source: np.ndarray, image: np.ndarray, target: tuple[int, int],
           other: tuple[int, int], markers_target: np.ndarray | None = None,
           sigma: float = .8) -> tuple[np.ndarray, np.ndarray]:
    markers = np.zeros(source.shape, np.uint8)
    if markers_target is None:
        markers[target] = 1
    else:
        markers[markers_target] = 1
    markers[other] = 2
    split = watershed(-ndi.gaussian_filter(image.astype(np.float32), sigma),
                      markers, mask=source, connectivity=STRUCTURE)
    return split == 1, split == 2


def _valley(image: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    signal = ndi.gaussian_filter(image.astype(np.float32), .8)
    count = max(2, int(np.ceil(np.hypot(a[0]-b[0], a[1]-b[1])*4)))
    ys = np.rint(np.linspace(a[0], b[0], count)).astype(int)
    xs = np.rint(np.linspace(a[1], b[1], count)).astype(int)
    return float(signal[ys, xs].min())/max(min(float(signal[a]), float(signal[b])), 1.)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b))/union if union else 0.


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != protected.shape or labels.ndim != 3:
        raise ValueError("Complete aligned labels, raw image and protection mask required")
    candidate = labels.copy()
    visible = points[points.state.isin(["observed", "latent_visible"])].copy()
    strong = points[(points.state == "observed") & (points.strong == True)].copy()
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    audit, applications = [], []
    used: set[tuple[int, int]] = set()
    for track, group in visible.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        frames = group.frame.to_numpy(dtype=int)
        if (len(frames) < 40 or not np.all(np.diff(frames) == 1)
                or group.strong.astype(bool).mean() < .9):
            continue
        ys = np.rint(group.y.to_numpy(dtype=float)).astype(int)
        xs = np.rint(group.x.to_numpy(dtype=float)).astype(int)
        if (np.any((ys < 0) | (ys >= labels.shape[1]))
                or np.any((xs < 0) | (xs >= labels.shape[2]))):
            continue
        names = labels[frames, ys, xs].astype(int)
        first = int(names[0])
        if first <= 0:
            continue
        differences = np.flatnonzero(names != first)
        if not len(differences):
            continue
        anchor = int(differences[0])
        if (anchor < 2 or anchor > 5 or len(names)-anchor < 30
                or np.any(names[anchor:] <= 0) or np.any(names[anchor:] == first)
                or len(set(names[anchor:])) < 2):
            continue
        positions = group[["y", "x"]].to_numpy(dtype=float)
        radii = np.maximum(group.radius_px.to_numpy(dtype=float), 1.)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        scaled = steps/np.maximum((radii[:-1]+radii[1:])/2, 1.)
        if np.percentile(scaled, 95) > 1. or np.max(scaled) > 2.5:
            continue
        reference_areas = []
        for i in range(anchor):
            source = _component(labels[int(frames[i])], int(ys[i]), int(xs[i]))
            if source is None:
                break
            seats = sum(source[round(float(pt.y)), round(float(pt.x))]
                        for pt in by_frame[int(frames[i])].itertuples(index=False))
            if seats != 1:
                break
            reference_areas.append(int(source.sum()))
        if len(reference_areas) != anchor:
            continue
        reference = float(np.median(reference_areas))
        records = []
        first_present = shared = 0
        independent_first_tracks: dict[int, set[int]] = {}
        reason = "eligible"
        for i in range(anchor, len(group)):
            t = int(frames[i])
            y, x = int(ys[i]), int(xs[i])
            source = _component(labels[t], y, x)
            if source is None:
                reason = "missing_target_source"
                break
            if (t, int(names[i])) in used:
                reason = "overlaps_another_proposal"
                break
            others = []
            for pt in by_frame[t].itertuples(index=False):
                py, px = round(float(pt.y)), round(float(pt.x))
                if int(pt.track_id) != int(track) and source[py, px]:
                    others.append(pt)
                if labels[t, py, px] == first and not source[py, px]:
                    independent_first_tracks.setdefault(int(pt.track_id), set()).add(t)
            if len(others) > 1:
                reason = "source_contains_more_than_two_strong_somata"
                break
            shared += len(others) == 1
            first_present += bool(np.any(labels[t] == first))
            records.append({"index": i, "frame": t, "source": source,
                            "target_point": (y, x),
                            "other": others[0] if others else None,
                            "target": None, "method": "whole_component",
                            "valley": None, "temporal_before": None,
                            "temporal_after": None})
        later = len(group)-anchor
        if reason == "eligible" and shared < .5*later:
            reason = "few_two_soma_shared_frames"
        if reason == "eligible" and first_present < .2*later:
            reason = "first_name_has_no_recurrent_separate_fragments"
        if reason == "eligible" and any(len(times) > 2 for times in independent_first_tracks.values()):
            reason = "first_name_has_sustained_independent_strong_holder"
        if reason == "eligible":
            for record in records:
                t, source, other = record["frame"], record["source"], record["other"]
                if other is None:
                    if not bool(group.iloc[record["index"]].strong):
                        reason = "weak_unshared_target"
                        break
                    record["target"] = source
                    continue
                shared_other = (round(float(other.y)), round(float(other.x)))
                if bool(group.iloc[record["index"]].strong):
                    valley = _valley(raw[t], record["target_point"], shared_other)
                    target, host = _split(source, raw[t], record["target_point"], shared_other)
                    if (valley > .85 or ndi.label(target, STRUCTURE)[1] != 1
                            or ndi.label(host, STRUCTURE)[1] != 1
                            or not np.array_equal(target | host, source)
                            or not .4 <= int(target.sum())/reference <= 2.5):
                        reason = "strong_raw_partition_not_plausible"
                        break
                    record.update({"target": target, "method": "strong_raw_soma",
                                   "valley": valley})
        if reason == "eligible":
            for j, record in enumerate(records):
                if record["target"] is not None:
                    continue
                if (j == 0 or j == len(records)-1
                        or records[j-1]["frame"] != record["frame"]-1
                        or records[j+1]["frame"] != record["frame"]+1
                        or records[j-1]["target"] is None
                        or records[j+1]["target"] is None
                        or records[j-1]["other"] is None
                        or records[j+1]["other"] is None
                        or not bool(group.iloc[records[j-1]["index"]].strong)
                        or not bool(group.iloc[records[j+1]["index"]].strong)
                        or any(int(names[neighbor["index"]]) != int(names[record["index"]])
                               for neighbor in (records[j-1], records[j+1]))):
                    reason = "weak_frame_not_flanked_by_trusted_same_host"
                    break
                host_track = int(record["other"].track_id)
                if any(int(records[k]["other"].track_id) != host_track
                       for k in (j-1, j+1)):
                    reason = "weak_frame_host_track_changes"
                    break
                forecasts = []
                for neighbor in (records[j-1], records[j+1]):
                    shift = (record["target_point"][0]-neighbor["target_point"][0],
                             record["target_point"][1]-neighbor["target_point"][1])
                    moved = ndi.shift(neighbor["target"].astype(np.uint8),
                                      shift=shift, order=0, mode="constant") > 0
                    forecasts.append(moved & record["source"])
                core = forecasts[0] & forecasts[1]
                other = record["other"]
                other_point = (round(float(other.y)), round(float(other.x)))
                if (not core[record["target_point"]] or core[other_point]
                        or ndi.label(core, STRUCTURE)[1] == 0):
                    reason = "weak_frame_no_separate_temporal_core"
                    break
                components, _ = ndi.label(core, STRUCTURE)
                core = components == components[record["target_point"]]
                target, host = _split(record["source"], raw[record["frame"]],
                                      record["target_point"], other_point,
                                      markers_target=core, sigma=1.0)
                overlaps = [_jaccard(target, forecast) for forecast in forecasts]
                if (ndi.label(target, STRUCTURE)[1] != 1
                        or ndi.label(host, STRUCTURE)[1] != 1
                        or not np.array_equal(target | host, record["source"])
                        or not .4 <= int(target.sum())/reference <= 2.5
                        or min(overlaps) < .6):
                    reason = "weak_temporal_partition_not_plausible"
                    break
                record.update({"target": target, "method": "temporal_raw_soma",
                               "temporal_before": overlaps[0],
                               "temporal_after": overlaps[1]})
        if reason == "eligible" and any(np.any(record["target"] & protected[record["frame"]])
                                        for record in records):
            reason = "would_change_earlier_accepted_pixels"
        audit.append({"track_id": int(track), "first_name": first,
                      "first_run_frames": anchor, "later_frames": later,
                      "shared_frames": shared, "first_name_fragment_frames": first_present,
                      "reference_area": reference,
                      "outcome": "applied" if reason == "eligible" else "rejected",
                      "reason": reason})
        if reason != "eligible":
            continue
        for record in records:
            t, target = record["frame"], record["target"]
            candidate[t][target] = first
            used.add((t, int(names[record["index"]])))
            applications.append({"track_id": int(track), "frame": t+1,
                                 "source_name": int(names[record["index"]]),
                                 "restored_name": first,
                                 "source_pixels": int(record["source"].sum()),
                                 "target_pixels": int(target.sum()),
                                 "changed_pixels": int(target.sum()),
                                 "method": record["method"],
                                 "valley_ratio": record["valley"],
                                 "temporal_jaccard_before": record["temporal_before"],
                                 "temporal_jaccard_after": record["temporal_after"]})
    changed = candidate != labels
    if (not np.array_equal(candidate > 0, labels > 0)
            or np.any(changed & protected)
            or set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels)))):
        raise AssertionError("short-anchor name recovery crossed accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
