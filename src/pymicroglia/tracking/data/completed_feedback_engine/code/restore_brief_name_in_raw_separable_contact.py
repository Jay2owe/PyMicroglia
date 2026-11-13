"""Keep a soma's brief earlier name while it shares a neighbour's outline.

Eligible somata have their own last exclusive outline, then occupy another
long-lived soma's existing name for a complete strong observed run. The raw
image must separate every joined-frame soma basin. No recording, identity,
track, frame, region or feedback selector is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import recover_stable_soma_branch_handoffs as prior


@dataclass(frozen=True)
class Params:
    minimum_earlier_name_frames: int = 2
    minimum_contact_frames: int = 5
    minimum_contact_pixels: int = 500
    minimum_neighbour_stable_frames: int = 20
    maximum_step_radii: float = 2.5
    minimum_earlier_exclusive_area: int = 25
    maximum_residual_earlier_name_pixels: int = 5
    maximum_prior_reversal_pixels: int = 32
    maximum_prior_reversal_fraction: float = .2
    partition: prior.Params = prior.Params()


def _runs(group: pd.DataFrame) -> list[pd.DataFrame]:
    if group.empty:
        return []
    ordered = group.sort_values("frame")
    breaks = (ordered.frame.diff().fillna(1).ne(1) |
              ordered.owner.ne(ordered.owner.shift()))
    return [part for _, part in ordered.groupby(breaks.cumsum(), sort=False)]


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            original: np.ndarray, protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or labels.shape != raw.shape or
            labels.shape != original.shape or labels.shape != protected.shape or
            not np.array_equal(protected, labels != original)):
        raise ValueError("labels, raw, original and protected pixels must align")
    candidate = labels.copy()
    all_points = points.copy()
    coords = np.rint(all_points[["y", "x"]].to_numpy(float)).astype(int)
    if (np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= labels.shape[1])
            or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= labels.shape[2])):
        raise ValueError("soma point outside label movie")
    all_points["owner"] = labels[all_points.frame.to_numpy(int), coords[:, 0], coords[:, 1]]
    strong = all_points[(all_points.state.astype(str) == "observed") &
                        all_points.strong.astype(bool)].copy()
    by_frame = {int(t): list(part.itertuples(index=False))
                for t, part in strong.groupby("frame")}
    by_track = {int(track): group.sort_values("frame")
                for track, group in strong.groupby("track_id")}
    cache = {}
    proposals, audits, applications = [], [], []
    permitted_prior_reversal = np.zeros_like(protected, dtype=bool)
    for track_id, group in by_track.items():
        runs = _runs(group[group.owner.gt(0)])
        for earlier, contact in zip(runs, runs[1:]):
            if (len(earlier) < params.minimum_earlier_name_frames or
                    len(contact) < params.minimum_contact_frames or
                    int(contact.frame.iloc[0]) != int(earlier.frame.iloc[-1])+1):
                continue
            old_name, new_name = int(earlier.owner.iloc[0]), int(contact.owner.iloc[0])
            if old_name == new_name:
                continue
            last = earlier.iloc[-1]
            old_component = prior._component(labels, int(last.frame), old_name, last, cache)
            if (old_component is None or int(old_component.sum()) < params.minimum_earlier_exclusive_area
                    or any(int(q.track_id) != track_id and prior._inside(old_component, q)
                           for q in by_frame.get(int(last.frame), ()))):
                continue
            base = {"target_track": track_id, "earlier_name": old_name,
                    "contact_name": new_name, "first_frame": int(contact.frame.iloc[0])+1}
            frames = []
            reason = "end_of_contact_run"
            previous = last
            stable_holder = None
            for target in contact.itertuples(index=False):
                t = int(target.frame)
                if int(np.count_nonzero(labels[t] == old_name)) > params.maximum_residual_earlier_name_pixels:
                    reason = "earlier_name_has_other_holder"; break
                step = np.hypot(float(target.x)-float(previous.x),
                                float(target.y)-float(previous.y))
                radius = max((float(target.radius_px)+float(previous.radius_px))/2, 1.)
                if step/radius > params.maximum_step_radii:
                    reason = "soma_step_too_large"; break
                component = prior._component(labels, t, new_name, target, cache)
                if component is None:
                    reason = "contact_outline_missing"; break
                others = [q for q in by_frame.get(t, ())
                          if int(q.track_id) != track_id and prior._inside(component, q)]
                qualified = []
                if stable_holder is None:
                    for other in others:
                        history = by_track[int(other.track_id)]
                        before = history[(history.frame < int(contact.frame.iloc[0])) &
                                         history.owner.eq(new_name)]
                        recent = before.tail(params.minimum_neighbour_stable_frames)
                        if (len(recent) >= params.minimum_neighbour_stable_frames and
                                int(recent.frame.iloc[-1]) == int(contact.frame.iloc[0])-1 and
                                np.all(np.diff(recent.frame.to_numpy(int)) == 1)):
                            qualified.append(int(other.track_id))
                    if len(qualified) != 1:
                        reason = "no_unique_stable_neighbour"; break
                    stable_holder = qualified[0]
                elif stable_holder not in [int(q.track_id) for q in others]:
                    reason = "stable_neighbour_missing"; break
                selected = prior._basin(component, raw[t], [target], others,
                                        params.partition)
                if selected is None:
                    reason = "raw_split_or_approved_pixel_unresolved"; break
                reversal = selected & protected[t]
                reversed_pixels = int(reversal.sum())
                if (reversed_pixels > params.maximum_prior_reversal_pixels or
                        reversed_pixels > params.maximum_prior_reversal_fraction*int(selected.sum()) or
                        np.any(original[t][reversal] != old_name)):
                    reason = "prior_edit_not_supported_by_raw_soma"; break
                frames.append((t, selected, reversal))
                previous = target
            total = sum(int(mask.sum()) for _, mask, _ in frames)
            if (len(frames) < params.minimum_contact_frames or
                    total < params.minimum_contact_pixels or
                    len(frames) != len(contact)):
                audits.append({**base, "outcome": "rejected", "reason": reason,
                               "eligible_frames": len(frames), "changed_pixels": 0})
                continue
            proposals.append((total, base, frames, stable_holder))
    for total, base, frames, holder in sorted(proposals, reverse=True, key=lambda row: row[0]):
        if any(np.any(candidate[t][mask] != base["contact_name"])
               for t, mask, _ in frames):
            audits.append({**base, "outcome": "rejected", "reason": "overlapping_proposal",
                           "eligible_frames": 0, "changed_pixels": 0})
            continue
        for t, mask, reversal in frames:
            candidate[t][mask] = base["earlier_name"]
            permitted_prior_reversal[t] |= reversal
            applications.append({**base, "frame": t+1,
                                 "stable_neighbour_track": holder,
                                 "reversed_prior_pixels": int(reversal.sum()),
                                 "changed_pixels": int(mask.sum())})
        audits.append({**base, "outcome": "applied", "reason": "complete_raw_separable_contact",
                       "eligible_frames": len(frames), "changed_pixels": total,
                       "reversed_prior_pixels": sum(int(r.sum()) for _, _, r in frames),
                       "stable_neighbour_track": holder})
    if (np.any((candidate != labels) & protected & ~permitted_prior_reversal) or
            np.any(candidate[permitted_prior_reversal] != original[permitted_prior_reversal]) or
            not np.array_equal(candidate > 0, labels > 0) or
            not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("raw-separable contact restoration crossed declared boundary")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
