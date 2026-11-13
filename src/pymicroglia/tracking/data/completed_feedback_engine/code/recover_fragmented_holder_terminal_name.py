"""Restore a long, single-soma terminal path with fragmented old-name holders."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

STRUCTURE = np.ones((3, 3), bool)


def correct(labels: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate = labels.copy()
    strong = points[(points.state == "observed") & points.strong.astype(bool)].copy()
    coords = np.rint(strong[["y", "x"]].to_numpy()).astype(int)
    strong["owner"] = labels[strong.frame.to_numpy(dtype=int), coords[:, 0], coords[:, 1]]
    by_frame = {int(t): frame_points for t, frame_points in strong.groupby("frame")}
    audit, applications = [], []
    for track, group in strong.groupby("track_id"):
        group = group.sort_values("frame")
        if len(group) < .9 * len(labels) or int(group.frame.min()) > 3 or int(group.frame.max()) < len(labels)-4:
            continue
        frames = group.frame.to_numpy(dtype=int)
        owners = group.owner.to_numpy(dtype=int)
        named = owners[owners > 0]
        if len(named) < .7 * len(labels) or named[0] == named[-1]:
            continue
        first, terminal = int(named[0]), int(named[-1])
        first_terminal = int(frames[np.flatnonzero(owners == terminal)[0]])
        last_first = int(frames[np.flatnonzero(owners == first)[-1]])
        earlier = owners[frames < first_terminal]
        later = owners[frames >= first_terminal]
        if (first_terminal <= last_first or last_first < .2 * len(labels)
                or len(later) < .25 * len(labels)
                or np.count_nonzero(earlier == first) < .2 * len(labels)
                or np.count_nonzero(later == terminal) < .25 * len(labels)
                or np.any(earlier[(earlier > 0)] != first)
                or np.any(later[(later > 0)] != terminal)):
            continue
        positions = group[["y", "x"]].to_numpy(dtype=float)
        radii = group.radius_px.to_numpy(dtype=float)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        steps /= np.maximum((radii[:-1] + radii[1:]) / 2, 1.)
        if np.percentile(steps, 95) > .8:
            continue
        records = []
        holder_counts: Counter[int] = Counter()
        first_present = 0
        reason = "eligible"
        for row in group.itertuples(index=False):
            t = int(row.frame)
            if t < first_terminal or int(row.owner) != terminal:
                continue
            y, x = round(float(row.y)), round(float(row.x))
            components, _ = ndi.label(labels[t] == terminal, STRUCTURE)
            if not components[y, x]:
                reason = "terminal_soma_has_no_outline"
                break
            target = components == components[y, x]
            other_somata = []
            for point in by_frame[t].itertuples(index=False):
                if int(point.track_id) == int(track):
                    continue
                py, px = round(float(point.y)), round(float(point.x))
                if target[py, px]:
                    other_somata.append(int(point.track_id))
                if int(labels[t, py, px]) == first:
                    holder_counts[int(point.track_id)] += 1
            first_present += bool(np.any(labels[t] == first))
            if other_somata or np.any(target & protected[t]):
                reason = "shared_soma_or_prior_accepted_pixel"
                break
            records.append((t, target, int(target.sum())))
        if reason == "eligible" and len(records) != int(np.count_nonzero(later == terminal)):
            reason = "incomplete_terminal_path"
        if reason == "eligible" and first_present < .5 * len(records):
            reason = "no_recurrent_first_name_holder"
        largest_holder = max(holder_counts.values(), default=0)
        if reason == "eligible" and largest_holder >= .5 * len(records):
            reason = "coherent_independent_first_name_holder"
        audit.append({"track_id": int(track), "first_name": first,
                      "terminal_name": terminal, "first_terminal_frame": first_terminal+1,
                      "later_labelled_frames": len(records), "first_name_holder_frames": first_present,
                      "largest_strong_holder_frames": largest_holder,
                      "target_pixels": sum(row[2] for row in records),
                      "outcome": "applied" if reason == "eligible" else "rejected",
                      "reason": reason})
        if reason != "eligible":
            continue
        for t, target, pixels in records:
            if np.any(candidate[t][target] != terminal):
                raise AssertionError("overlapping terminal corrections")
            candidate[t][target] = first
            applications.append({"track_id": int(track), "frame": t+1,
                                 "old_name": terminal, "restored_name": first,
                                 "changed_pixels": pixels})
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
