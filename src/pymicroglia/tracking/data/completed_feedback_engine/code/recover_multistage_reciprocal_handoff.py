"""Restore the first name of a complete physical soma after two handoffs.

Eligibility uses only accepted masks and observed strong soma tracks. The
separate holders of the first name receive the exclusive final wrong name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


STRUCTURE = np.ones((3, 3), bool)


def _runs(frames: np.ndarray, owners: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    for frame, owner in zip(frames, owners):
        frame, owner = int(frame), int(owner)
        if runs and runs[-1][0] == owner and runs[-1][2] + 1 == frame:
            runs[-1] = (owner, runs[-1][1], frame)
        else:
            runs.append((owner, frame, frame))
    return runs


def _point_component(frame: np.ndarray, name: int, y: int, x: int) -> np.ndarray:
    parts, _ = ndi.label(frame == name, STRUCTURE)
    part = int(parts[y, x])
    if not part:
        raise AssertionError("strong soma outside its accepted outline")
    return parts == part


def correct(labels: np.ndarray, points: pd.DataFrame
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate = labels.copy()
    strong = points[(points.state == "observed") & (points.strong == True)].copy()
    strong["frame"] = strong.frame.astype(int)
    by_frame = {int(frame): group for frame, group in strong.groupby("frame")}
    audits: list[dict] = []
    applications: list[dict] = []
    used: set[tuple[int, int]] = set()
    for track, group in strong.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        frames = group.frame.to_numpy(dtype=int)
        if (len(frames) != len(labels) or not np.array_equal(frames, np.arange(len(labels)))):
            continue
        ys = np.rint(group.y.to_numpy(dtype=float)).astype(int)
        xs = np.rint(group.x.to_numpy(dtype=float)).astype(int)
        if (np.any(ys < 0) or np.any(ys >= labels.shape[1]) or
                np.any(xs < 0) or np.any(xs >= labels.shape[2])):
            continue
        owners = labels[frames, ys, xs].astype(int)
        runs = _runs(frames, owners)
        if (len(runs) != 3 or len({run[0] for run in runs}) != 3 or
                any(run[0] == 0 for run in runs) or
                runs[0][2] - runs[0][1] + 1 < 10 or
                runs[1][2] - runs[1][1] + 1 < 10 or
                runs[2][2] - runs[2][1] + 1 < .5 * len(labels)):
            continue
        first, middle, final = [run[0] for run in runs]
        positions = group[["y", "x"]].to_numpy(dtype=float)
        radii = np.maximum(group.radius_px.to_numpy(dtype=float), 1.)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        scaled_steps = steps / np.maximum((radii[:-1] + radii[1:]) / 2, 1.)
        if np.percentile(scaled_steps, 95) > .8 or np.max(scaled_steps) > 2.5:
            continue
        proposed: list[tuple[int, np.ndarray, np.ndarray, int]] = []
        reason = "eligible"
        for t in range(runs[1][1], len(labels)):
            owner = middle if t <= runs[1][2] else final
            frame = labels[t]
            target = _point_component(frame, owner, int(ys[t]), int(xs[t]))
            others = by_frame[t]
            for pt in others.itertuples(index=False):
                if int(pt.track_id) == int(track):
                    continue
                yy, xx = round(float(pt.y)), round(float(pt.x))
                if 0 <= yy < target.shape[0] and 0 <= xx < target.shape[1] and target[yy, xx]:
                    reason = "target_outline_contains_another_strong_soma"
                    break
            if reason != "eligible":
                break
            holder = frame == first
            if t <= runs[1][2]:
                if np.any(holder):
                    reason = "first_name_occupied_during_middle_run"
                    break
                holder = np.zeros_like(target)
            elif int(np.count_nonzero(frame == final)) != int(target.sum()):
                reason = "final_name_has_another_holder"
                break
            if (t, first) in used or (t, owner) in used:
                reason = "overlaps_another_proposal"
                break
            proposed.append((t, target, holder, owner))
        audit = {"track_id": int(track), "first_name": int(first),
                 "middle_name": int(middle), "final_name": int(final),
                 "first_run_frames": runs[0][2]-runs[0][1]+1,
                 "middle_run_frames": runs[1][2]-runs[1][1]+1,
                 "final_run_frames": runs[2][2]-runs[2][1]+1,
                 "step_p95_radii": float(np.percentile(scaled_steps, 95)),
                 "outcome": "applied" if reason == "eligible" else "rejected",
                 "reason": reason}
        audits.append(audit)
        if reason != "eligible":
            continue
        for t, target, holder, owner in proposed:
            candidate[t][target] = first
            if np.any(holder):
                candidate[t][holder] = final
            used.add((t, first))
            used.add((t, owner))
            applications.append({"track_id": int(track), "frame": t+1,
                                 "target_old_name": int(owner),
                                 "target_new_name": int(first),
                                 "target_pixels": int(target.sum()),
                                 "holder_old_name": int(first),
                                 "holder_new_name": int(final),
                                 "holder_pixels": int(holder.sum()),
                                 "changed_pixels": int(target.sum()+holder.sum())})
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("multistage handoff changed foreground")
    if set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels))):
        raise AssertionError("multistage handoff introduced a name")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
