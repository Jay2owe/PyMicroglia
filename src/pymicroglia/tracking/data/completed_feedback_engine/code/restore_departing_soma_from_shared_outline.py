"""Keep a stable soma name through its last observed shared outline.

The candidate uses tracked soma positions, current labels and raw watershed
basins only. No movie, identity, track, frame, region or feedback selector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import recover_stable_soma_branch_handoffs as partition_rule


@dataclass(frozen=True)
class Params:
    prior_window_frames: int = 30
    minimum_prior_owner_frames: int = 26
    minimum_terminal_frames: int = 2
    maximum_terminal_frames: int = 3
    minimum_unobserved_tail_frames: int = 5
    maximum_successor_lag_frames: int = 3
    maximum_successor_step_radii: float = 2.5
    precontact_window_frames: int = 6
    minimum_exclusive_precontact_frames: int = 1
    minimum_changed_pixels: int = 100
    maximum_old_name_residual_pixels: int = 5
    maximum_step_radii: float = 2.5
    partition: partition_rule.Params = partition_rule.Params()


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("aligned labels, raw and protected arrays required")
    candidate = labels.copy()
    strong = points[(points.state.eq("observed")) & points.strong.astype(bool)].copy()
    yx = np.rint(strong[["y", "x"]].to_numpy(float)).astype(int)
    strong["owner"] = labels[strong.frame.to_numpy(int), yx[:, 0], yx[:, 1]]
    by_frame = {int(t): tuple(g.itertuples(index=False)) for t, g in strong.groupby("frame")}
    cache = {}
    proposals, audits, applications = [], [], []
    occupied = np.zeros_like(protected, bool)
    for track_id, group in strong.groupby("track_id"):
        group = group.sort_values("frame")
        if (len(group) < params.prior_window_frames+params.minimum_terminal_frames
                or int(group.frame.iloc[-1]) > len(labels)-1-params.minimum_unobserved_tail_frames):
            continue
        # The early part of the last window supplies the identity; the final
        # three observations are never used to choose it.
        history = group.iloc[-(params.prior_window_frames+params.maximum_terminal_frames):-params.maximum_terminal_frames]
        if len(history) < params.prior_window_frames:
            continue
        old_name = int(history.owner.mode().iloc[0])
        if old_name <= 0:
            continue
        k = len(group)-1
        while k >= 0 and int(group.owner.iloc[k]) > 0 and int(group.owner.iloc[k]) != old_name:
            k -= 1
        suffix = group.iloc[k+1:]
        if not params.minimum_terminal_frames <= len(suffix) <= params.maximum_terminal_frames:
            continue
        before = group.iloc[max(0, k-params.prior_window_frames+1):k+1]
        if (len(before) < params.prior_window_frames
                or int(before.owner.eq(old_name).sum()) < params.minimum_prior_owner_frames
                or int(before.frame.iloc[-1]) != int(suffix.frame.iloc[0])-1
                or not np.all(np.diff(suffix.frame.to_numpy(int)) == 1)):
            continue
        prior_two = group.iloc[k-1:k+1]
        if len(prior_two) != 2 or not prior_two.owner.eq(old_name).all():
            continue
        precontact = group.iloc[k-params.precontact_window_frames+1:k+1]
        if (len(precontact) != params.precontact_window_frames
                or not precontact.owner.eq(old_name).all()
                or not np.all(np.diff(precontact.frame.to_numpy(int)) == 1)):
            continue
        exclusive_count = 0
        for q in precontact.itertuples(index=False):
            t = int(q.frame)
            component = partition_rule._component(labels, t, old_name, q, cache)
            if component is None:
                break
            shared = any(
                    int(o.track_id) != int(track_id) and partition_rule._inside(component, o)
                    for o in by_frame.get(t, ()))
            exclusive_count += not shared
        if exclusive_count < params.minimum_exclusive_precontact_frames:
            continue
        last_observed = group.iloc[-1]
        last_component = partition_rule._component(
            labels, int(last_observed.frame), int(last_observed.owner),
            last_observed, cache)
        if last_component is None:
            continue
        co_residents = {int(o.track_id)
                        for o in by_frame.get(int(last_observed.frame), ())
                        if int(o.track_id) != int(track_id)
                        and partition_rule._inside(last_component, o)}
        # A later holder or a nearby newly tracked soma means the cell has a
        # possible continuation. Leave that physical path to a broader rule.
        if any(np.count_nonzero(labels[t] == old_name) > 0
               for t in range(int(last_observed.frame)+1, len(labels))):
            continue
        successor = False
        for lag in range(1, params.maximum_successor_lag_frames+1):
            t = int(last_observed.frame)+lag
            for o in by_frame.get(t, ()):
                if int(o.track_id) in co_residents:
                    continue
                step = np.hypot(float(o.x)-float(last_observed.x),
                                float(o.y)-float(last_observed.y))
                radius = max((float(o.radius_px)+float(last_observed.radius_px))/2, 1.)
                if step/radius <= params.maximum_successor_step_radii:
                    successor = True
                    break
            if successor:
                break
        if successor:
            continue
        frames = []
        reason = "supported"
        previous = prior_two.iloc[-1]
        for q in suffix.itertuples(index=False):
            t = int(q.frame)
            step = np.hypot(float(q.x)-float(previous.x), float(q.y)-float(previous.y))
            radius = max((float(q.radius_px)+float(previous.radius_px))/2, 1.)
            if step/radius > params.maximum_step_radii:
                reason = "soma_step_too_large"; break
            if int(np.count_nonzero(labels[t] == old_name)) > params.maximum_old_name_residual_pixels:
                reason = "earlier_name_held_elsewhere"; break
            source_name = int(q.owner)
            component = partition_rule._component(labels, t, source_name, q, cache)
            if component is None:
                reason = "target_component_missing"; break
            others = [o for o in by_frame.get(t, ())
                      if int(o.track_id) != int(track_id) and partition_rule._inside(component, o)]
            if not others:
                reason = "terminal_component_has_no_other_soma"; break
            selected = partition_rule._basin(component, raw[t], [q], others, params.partition)
            if (selected is None or np.any(protected[t] & selected)
                    or np.any(occupied[t] & selected)):
                reason = "raw_basin_or_protected_pixel_unresolved"; break
            frames.append((t, source_name, selected))
            previous = q
        total = sum(int(mask.sum()) for _, _, mask in frames)
        seed = {"track_id": int(track_id), "old_name": old_name,
                "first_frame": int(suffix.frame.iloc[0])+1,
                "last_frame": int(suffix.frame.iloc[-1])+1}
        if len(frames) != len(suffix) or total < params.minimum_changed_pixels:
            audits.append({**seed, "outcome": "rejected", "reason": reason,
                           "changed_pixels": 0})
            continue
        proposals.append((total, seed, frames))
    for total, seed, frames in sorted(proposals, reverse=True, key=lambda row: row[0]):
        if any(np.any(occupied[t] & mask) for t, _, mask in frames):
            audits.append({**seed, "outcome": "rejected", "reason": "overlapping_proposal",
                           "changed_pixels": 0})
            continue
        for t, source_name, mask in frames:
            candidate[t][mask] = seed["old_name"]
            occupied[t][mask] = True
            applications.append({**seed, "frame": t+1, "source_name": source_name,
                                 "changed_pixels": int(mask.sum())})
        audits.append({**seed, "outcome": "applied", "reason": "complete_terminal_raw_split",
                       "changed_pixels": total})
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("terminal split crossed accepted boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
