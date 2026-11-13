"""Restore a corrected soma after its point tracker ends during a contact.

An approved correction supplies the earlier physical soma and name. A new
strong track supplies the later soma. Raw peaks bridge their short tracker
gap, and watershed separates the two bodies only where a deep intensity
valley supports it. The producer has no movie, name, track, frame, region or
feedback selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import recover_stable_soma_branch_handoffs as prior


@dataclass(frozen=True)
class Params:
    maximum_successor_lag: int = 6
    minimum_successor_strong_observations: int = 5
    maximum_successor_head_step_radii: float = 3.
    minimum_second_successor_distance_ratio: float = 1.5
    peak_search_radii: float = 1.75
    minimum_peak_relative_intensity: float = .4
    minimum_other_peak_separation_radii: float = 1.25
    maximum_source_projection_px: float = 2.
    maximum_bridge_skips: int = 1
    minimum_corrected_frames: int = 6
    minimum_corrected_pixels: int = 500
    basin: prior.Params = prior.Params()


def _near_component(frame: np.ndarray, x: float, y: float,
                    maximum_distance: float) -> tuple[int, np.ndarray, SimpleNamespace] | None:
    cy, cx = round(y), round(x)
    r = int(np.ceil(maximum_distance))
    options = []
    for yy in range(max(0, cy-r), min(frame.shape[0], cy+r+1)):
        for xx in range(max(0, cx-r), min(frame.shape[1], cx+r+1)):
            name = int(frame[yy, xx])
            if name <= 0:
                continue
            distance = float(np.hypot(yy-y, xx-x))
            if distance <= maximum_distance:
                options.append((distance, -float(xx==cx and yy==cy), name, yy, xx))
    if not options:
        return None
    _, _, name, yy, xx = min(options)
    parts, _ = ndi.label(frame == name, prior.STRUCTURE)
    mask = parts == parts[yy, xx]
    return name, mask, SimpleNamespace(x=xx, y=yy)


def _bridge_peak(frame: np.ndarray, predicted_x: float, predicted_y: float,
                 radius: float, reference_intensity: float,
                 other_points: list, params: Params) -> SimpleNamespace | None:
    smooth = ndi.gaussian_filter(frame.astype(np.float32), 1.)
    bound = params.peak_search_radii * radius
    cy, cx = round(predicted_y), round(predicted_x)
    r = int(np.ceil(bound))
    y0, y1 = max(0, cy-r), min(frame.shape[0], cy+r+1)
    x0, x1 = max(0, cx-r), min(frame.shape[1], cx+r+1)
    local = smooth[y0:y1, x0:x1]
    maxima = smooth == ndi.maximum_filter(smooth, size=5)
    choices = []
    for local_y, local_x in np.argwhere(maxima[y0:y1, x0:x1]):
        yy, xx = int(local_y+y0), int(local_x+x0)
        distance = float(np.hypot(xx-predicted_x, yy-predicted_y))
        if (distance > bound or
                float(smooth[yy, xx]) < params.minimum_peak_relative_intensity * reference_intensity or
                any(np.hypot(xx-float(p.x), yy-float(p.y)) <
                    params.minimum_other_peak_separation_radii * float(p.radius_px)
                    for p in other_points)):
            continue
        choices.append((distance, -float(local[local_y, local_x]), xx, yy))
    if not choices:
        return None
    _, _, x, y = min(choices)
    return SimpleNamespace(x=x, y=y)


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, prior_applications: pd.DataFrame,
            params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("labels, raw and protected pixels must align")
    candidate = labels.copy()
    audits, applications, proposals = [], [], []
    if prior_applications.empty:
        return candidate, pd.DataFrame(), pd.DataFrame()
    tracks = {int(track): group.sort_values("frame").set_index("frame")
              for track, group in points.groupby("track_id")}
    strong = points[(points.state.astype(str) == "observed") & points.strong.astype(bool)]
    by_frame = {int(t): list(group.itertuples(index=False))
                for t, group in strong.groupby("frame")}
    for (_, initial_name), episode in prior_applications.groupby(
            ["branch_track", "initial_name"], sort=False):
        if len(episode) < params.minimum_successor_strong_observations:
            continue
        branch_id = int(episode.branch_track.iloc[0])
        anchor_name = int(initial_name)
        end = int(episode.frame.max()) - 1
        branch = tracks.get(branch_id)
        if branch is None or end not in branch.index:
            continue
        anchor = branch.loc[end]
        if isinstance(anchor, pd.DataFrame):
            continue
        options = []
        for track_id, history in tracks.items():
            if track_id == branch_id or history.empty:
                continue
            start = int(history.index.min())
            if not (end < start <= end + params.maximum_successor_lag):
                continue
            head = history.loc[start]
            if (isinstance(head, pd.DataFrame) or str(head.state) != "observed"
                    or not bool(head.strong)):
                continue
            head_window = history.loc[start:min(start+params.maximum_successor_lag,
                                                len(labels)-1)]
            if int(((head_window.state == "observed") & head_window.strong).sum()) < params.minimum_successor_strong_observations:
                continue
            radius = max((float(anchor.radius_px)+float(head.radius_px))/2, 1.)
            step = np.hypot(float(head.x)-float(anchor.x),
                            float(head.y)-float(anchor.y))/radius
            if step <= params.maximum_successor_head_step_radii:
                options.append((step, track_id, start, head))
        options.sort(key=lambda row: row[0])
        if (not options or (len(options) > 1 and options[1][0] <
                            options[0][0]*params.minimum_second_successor_distance_ratio)):
            continue
        _, successor_id, head_frame, head = options[0]
        base = {"prior_branch_track": branch_id, "successor_track": successor_id,
                "initial_name": anchor_name, "first_frame": end+2}
        anchor_smooth = ndi.gaussian_filter(raw[end].astype(np.float32), 1.)
        head_smooth = ndi.gaussian_filter(raw[head_frame].astype(np.float32), 1.)
        reference = min(float(anchor_smooth[round(float(anchor.y)), round(float(anchor.x))]),
                        float(head_smooth[round(float(head.y)), round(float(head.x))]))
        selected = []
        bridge_skips = 0
        reason = "end_of_successor"
        for t in range(end+1, min(len(labels), int(tracks[successor_id].index.max())+1)):
            if np.any(labels[t] == anchor_name):
                reason = "original_name_has_another_holder"; break
            others = by_frame.get(t, [])
            if t < head_frame:
                w = (t-end)/(head_frame-end)
                px = (1-w)*float(anchor.x) + w*float(head.x)
                py = (1-w)*float(anchor.y) + w*float(head.y)
                radius = (1-w)*float(anchor.radius_px) + w*float(head.radius_px)
                soma = _bridge_peak(raw[t], px, py, radius, reference,
                                    others, params)
            else:
                if t not in tracks[successor_id].index:
                    reason = "successor_point_missing"; break
                soma = tracks[successor_id].loc[t]
                if (isinstance(soma, pd.DataFrame) or str(soma.state) != "observed"
                        or not bool(soma.strong)):
                    reason = "successor_not_strong"; break
            if soma is None:
                if t < head_frame and bridge_skips < params.maximum_bridge_skips:
                    bridge_skips += 1; continue
                reason = "raw_bridge_peak_missing"; break
            source = _near_component(labels[t], float(soma.x), float(soma.y),
                                     params.maximum_source_projection_px)
            if source is None:
                if t < head_frame and bridge_skips < params.maximum_bridge_skips:
                    bridge_skips += 1; continue
                reason = "source_outline_missing"; break
            owner, component, marker = source
            if owner == anchor_name:
                reason = "already_restored"; break
            peers = [p for p in others if int(p.track_id) != successor_id
                     and prior._inside(component, p)]
            basin = prior._basin(component, raw[t], [marker], peers, params.basin)
            if basin is None or np.any(basin & protected[t]):
                if t < head_frame and bridge_skips < params.maximum_bridge_skips:
                    bridge_skips += 1; continue
                reason = "raw_partition_or_boundary_unresolved"; break
            selected.append((t, owner, basin))
        total = sum(int(mask.sum()) for _, _, mask in selected)
        if len(selected) < params.minimum_corrected_frames or total < params.minimum_corrected_pixels:
            audits.append({**base, "outcome": "rejected", "reason": reason,
                           "eligible_frames": len(selected), "changed_pixels": 0})
            continue
        proposals.append((total, base, selected, reason))
    for total, base, selected, reason in sorted(proposals, reverse=True, key=lambda row: row[0]):
        if any(np.any(candidate[t][mask] != owner) for t, owner, mask in selected):
            audits.append({**base, "outcome": "rejected", "reason": "overlapping_proposal",
                           "eligible_frames": 0, "changed_pixels": 0})
            continue
        for t, owner, mask in selected:
            candidate[t][mask] = base["initial_name"]
            applications.append({**base, "frame": t+1, "source_name": owner,
                                 "changed_pixels": int(mask.sum())})
        audits.append({**base, "outcome": "applied", "reason": reason,
                       "eligible_frames": len(selected), "changed_pixels": total})
    if (np.any((candidate != labels) & protected) or
            not np.array_equal(candidate > 0, labels > 0) or
            not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("post-tracker raw rejoin crossed approved boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
