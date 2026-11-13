"""Fill supported interior gaps in recurring weak fragment paths.

This layer adds only same-name components between two selected path frames.
It has no recording, name, coordinate, frame or feedback selectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import label as value_components

import reunite_recurrent_weak_branch_paths as base


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, inventory: pd.DataFrame
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate, audit, applications = base.correct(labels, protected, inventory)
    _, selected = base.select_paths(inventory)
    if selected.empty:
        return candidate, audit, applications
    strong = points[points.state.astype(str).eq("observed")
                    & points.strong.astype(bool)]
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    added: list[dict] = []
    for (large_name, small_name, cluster), path in selected.groupby(
            ["large_name", "small_name", "cluster"], sort=True):
        ordered = path.sort_values("frame")
        observed = set(map(int, ordered.frame))
        for left, right in zip(ordered.iloc[:-1].itertuples(index=False),
                               ordered.iloc[1:].itertuples(index=False)):
            if right.frame - left.frame <= 1:
                continue
            if right.frame - left.frame > base.Params().maximum_temporal_link_frames:
                continue
            for frame in range(int(left.frame) + 1, int(right.frame)):
                if frame in observed:
                    continue
                t = frame - 1
                parts = value_components(labels[t], connectivity=2, background=0)
                large = labels[t] == int(large_name)
                if not large.any():
                    continue
                large_q95 = float(np.percentile(raw[t][large], 95))
                matches: list[tuple[float, int, np.ndarray, int, float]] = []
                for cid in np.unique(parts[labels[t] == int(small_name)]):
                    cid = int(cid)
                    if cid == 0:
                        continue
                    mask = parts == cid
                    area = int(mask.sum())
                    if not 10 <= area <= 150 or np.any(protected[t] & mask):
                        continue
                    yy, xx = np.where(mask)
                    x, y = float(xx.mean()), float(yy.mean())
                    radius = max(np.sqrt(area / np.pi), 1.0)
                    dl = float(np.hypot(x - float(left.x), y - float(left.y)))
                    dr = float(np.hypot(x - float(right.x), y - float(right.y)))
                    scale_l = max(radius + np.sqrt(float(left.small_area)/np.pi), 1.0)
                    scale_r = max(radius + np.sqrt(float(right.small_area)/np.pi), 1.0)
                    if (dl > 2.0 * scale_l * (frame - int(left.frame))
                            or dr > 2.0 * scale_r * (int(right.frame) - frame)):
                        continue
                    fraction = (frame - int(left.frame))/(int(right.frame)-int(left.frame))
                    predicted_x = float(left.x) + fraction*(float(right.x)-float(left.x))
                    predicted_y = float(left.y) + fraction*(float(right.y)-float(left.y))
                    predicted_distance = float(np.hypot(x-predicted_x, y-predicted_y))
                    if predicted_distance > max(8.0, 2.5*radius):
                        continue
                    ratio = float(np.percentile(raw[t][mask], 95))/max(large_q95, 1.0)
                    if ratio > base.Params().maximum_median_raw_q95_ratio:
                        continue
                    if any(
                        0 <= round(float(p.y)) < mask.shape[0]
                        and 0 <= round(float(p.x)) < mask.shape[1]
                        and mask[round(float(p.y)), round(float(p.x))]
                        for p in by_frame.get(t, points.iloc[0:0]).itertuples(index=False)
                    ):
                        continue
                    matches.append((predicted_distance, cid, mask, area, ratio))
                if len(matches) != 1:
                    continue
                _, cid, mask, area, ratio = matches[0]
                if np.any(candidate[t][mask] != int(small_name)):
                    continue
                candidate[t][mask] = int(large_name)
                added.append({
                    "frame": frame, "old_name": int(small_name),
                    "new_name": int(large_name), "cluster": int(cluster),
                    "changed_pixels": area, "raw_q95_ratio": ratio,
                    "temporal_absorption": np.nan, "strong_count": 0,
                    "evidence": "bookended_same_name_raw_weak",
                })
    if added:
        if len(applications):
            applications = applications.copy()
            applications["evidence"] = "recurrent_nearby_path"
            applications = pd.concat([applications, pd.DataFrame(added)], ignore_index=True)
        else:
            applications = pd.DataFrame(added)
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or int(np.count_nonzero(candidate != labels))
            != int(applications.changed_pixels.sum() if len(applications) else 0)):
        raise AssertionError("bookended extension crossed accepted boundary")
    return candidate, audit, applications
