"""Exchange a stable late name handoff with its newly observed physical holder."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

STRUCTURE = np.ones((3, 3), bool)


def correct(labels: np.ndarray, points: pd.DataFrame
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3:
        raise ValueError("Expected a complete label movie")
    candidate = labels.copy()
    strong = points[(points.state == "observed") & (points.strong == True)].copy()
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    first_frame = strong.groupby("track_id").frame.min().to_dict()
    audits, applications = [], []
    used: set[tuple[int, int]] = set()
    n = len(labels)
    for track, group in strong.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        times = group.frame.to_numpy(dtype=int)
        if len(times) != n or not np.array_equal(times, np.arange(n)):
            continue
        ys = np.rint(group.y.to_numpy(dtype=float)).astype(int)
        xs = np.rint(group.x.to_numpy(dtype=float)).astype(int)
        if (np.any((ys < 0) | (ys >= labels.shape[1]))
                or np.any((xs < 0) | (xs >= labels.shape[2]))):
            continue
        owners = labels[times, ys, xs].astype(int)
        switches = np.flatnonzero(np.diff(owners))
        if len(switches) != 1 or owners[0] <= 0 or owners[-1] <= 0:
            continue
        switch = int(switches[0]) + 1
        if switch < .5*n or n-switch < .15*n:
            continue
        first, final = int(owners[0]), int(owners[-1])
        pos = group[["y", "x"]].to_numpy(dtype=float)
        radii = np.maximum(group.radius_px.to_numpy(dtype=float), 1.)
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        scaled = steps / np.maximum((radii[:-1]+radii[1:])/2, 1.)
        if np.percentile(scaled, 95) > .8 or np.max(scaled) > 2.5:
            continue
        proposals = []
        holder_tracks: set[int] = set()
        holder_strong_frames = 0
        holder_centroids = []
        target_pixels = extra_final_pixels = 0
        reason = "eligible"
        for t in range(switch, n):
            frame = labels[t]
            parts, _ = ndi.label(frame == final, STRUCTURE)
            part = int(parts[ys[t], xs[t]])
            if part == 0:
                reason = "target_point_outside_final_name"
                break
            target = parts == part
            holder = frame == first
            if not holder.any() or ndi.label(holder, STRUCTURE)[1] != 1:
                reason = "holder_missing_or_multiple_components"
                break
            if (t, first) in used or (t, final) in used:
                reason = "overlapping_accepted_proposal"
                break
            extra = int(np.count_nonzero(frame == final)) - int(target.sum())
            if extra > .35 * int(target.sum()):
                reason = "displaced_name_has_large_other_component"
                break
            target_pixels += int(target.sum())
            extra_final_pixels += extra
            strong_target = strong_holder = strong_other_final = 0
            for pt in by_frame[t].itertuples(index=False):
                py, px = round(float(pt.y)), round(float(pt.x))
                if target[py, px]:
                    strong_target += 1
                elif holder[py, px]:
                    strong_holder += 1
                    holder_tracks.add(int(pt.track_id))
                elif frame[py, px] == final:
                    strong_other_final += 1
            if strong_target != 1 or strong_holder > 1 or strong_other_final:
                reason = "multiple_strong_somata_or_displaced_holder"
                break
            holder_strong_frames += bool(strong_holder)
            holder_centroids.append(np.mean(np.argwhere(holder), axis=0))
            proposals.append((t, target, holder))
        if reason == "eligible" and extra_final_pixels > .05*target_pixels:
            reason = "too_many_displaced_name_fragments"
        if reason == "eligible" and holder_strong_frames < .8*(n-switch):
            reason = "holder_lacks_sustained_strong_soma"
        if reason == "eligible" and (not holder_tracks or any(
                int(first_frame[other]) < switch-2 for other in holder_tracks)):
            reason = "holder_is_an_older_independent_path"
        if reason == "eligible" and len(holder_centroids) > 1:
            jumps = np.linalg.norm(np.diff(holder_centroids, axis=0), axis=1)
            if np.max(jumps) > 5*np.median(radii[switch:]):
                reason = "holder_component_not_physically_continuous"
        audits.append({"track_id": int(track), "first_name": first,
                       "final_name": final, "switch_frame": switch+1,
                       "later_frames": n-switch, "holder_strong_frames": holder_strong_frames,
                       "holder_tracks": "|".join(map(str, sorted(holder_tracks))),
                       "extra_final_pixels": extra_final_pixels,
                       "outcome": "applied" if reason == "eligible" else "rejected",
                       "reason": reason})
        if reason != "eligible":
            continue
        for t, target, holder in proposals:
            candidate[t][target] = first
            candidate[t][holder] = final
            used.add((t, first))
            used.add((t, final))
            applications.append({"track_id": int(track), "frame": t+1,
                                 "target_old_name": final, "target_new_name": first,
                                 "target_pixels": int(target.sum()),
                                 "holder_old_name": first, "holder_new_name": final,
                                 "holder_pixels": int(holder.sum()),
                                 "changed_pixels": int(target.sum()+holder.sum())})
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("late handoff changed foreground")
    if set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels))):
        raise AssertionError("late handoff introduced a name")
    return candidate, pd.DataFrame(audits), pd.DataFrame(applications)
