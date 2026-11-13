"""Restore a returning soma's raw basin in a briefly misnamed mixed outline.

The rule reads approved masks, raw frames, soma tracks and protected pixels.
It contains no feedback, recording, identity, track, coordinate or frame target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as equal_value_components
from skimage.segmentation import watershed

import recover_large_bookended_owner_flickers as prior
import separable_merge_recovery as physical


@dataclass(frozen=True)
class Params:
    minimum_component_area: int = 150
    minimum_two_sided_overlap: float = .65
    minimum_returning_basin_fraction: float = .70
    minimum_other_basin_area: int = 8
    maximum_cross_soma_valley_ratio: float = .5
    minimum_same_soma_valley_ratio: float = .85
    watershed_sigma_px: float = 1.


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or raw.shape[:3] != labels.shape
            or protected.shape != labels.shape):
        raise ValueError("labels, raw and protected pixels must align")
    strong = points[(points.state.astype(str) == "observed")
                    & points.strong.astype(bool)]
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    by_track = {int(track): {int(p.frame): p for p in group.itertuples(index=False)}
                for track, group in points.groupby("track_id")}
    candidate = labels.copy()
    remaining = np.bincount(labels.ravel().astype(np.int64))
    audit, applications = [], []
    for t in range(1, len(labels)-1):
        frame = labels[t]
        components = equal_value_components(frame, connectivity=2, background=0)
        areas = np.bincount(components.ravel())
        residents: dict[int, list] = {}
        for point in by_frame.get(t, strong.iloc[0:0]).itertuples(index=False):
            y, x = round(float(point.y)), round(float(point.x))
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                cid = int(components[y, x])
                if cid:
                    residents.setdefault(cid, []).append(point)
        for cid in np.flatnonzero(areas >= params.minimum_component_area):
            cid = int(cid)
            if cid == 0 or len(residents.get(cid, [])) < 2:
                continue
            mask = components == cid
            old_name = int(frame[mask][0])
            before = np.bincount(labels[t-1][mask].astype(np.int64))
            after = np.bincount(labels[t+1][mask].astype(np.int64))
            threshold = params.minimum_two_sided_overlap * int(areas[cid])
            names = set(np.flatnonzero(before >= threshold)) & set(
                np.flatnonzero(after >= threshold))
            names.discard(0)
            names.discard(old_name)
            if len(names) != 1:
                continue
            new_name = int(next(iter(names)))
            source_area = int(areas[cid])
            base = {"frame": t+1, "component": cid, "old_name": old_name,
                    "new_name": new_name, "component_area": source_area,
                    "strong_residents": len(residents[cid])}
            reason = ""
            if np.any(protected[t] & mask):
                reason = "accepted_pixel_in_component"
            group = residents[cid]
            target = [point for point in group if prior._resident_continuity(
                by_track[int(point.track_id)], t, new_name, labels, raw,
                prior.Params())]
            target_ids = {int(point.track_id) for point in target}
            other = [point for point in group if int(point.track_id) not in target_ids]
            if not reason and (not target or not other):
                reason = "requires_returning_and_other_somata"
            if not reason:
                for point in target:
                    for peer in other:
                        valley = physical._valley_ratio(
                            raw[t], (float(point.x), float(point.y)),
                            (float(peer.x), float(peer.y)))
                        if not np.isfinite(valley) or valley > params.maximum_cross_soma_valley_ratio:
                            reason = "no_deep_cross_soma_valley"
                            break
                    if reason:
                        break
            if not reason:
                for i, point in enumerate(target):
                    for peer in target[i+1:]:
                        valley = physical._valley_ratio(
                            raw[t], (float(point.x), float(point.y)),
                            (float(peer.x), float(peer.y)))
                        if not np.isfinite(valley) or valley < params.minimum_same_soma_valley_ratio:
                            reason = "returning_somata_are_distinct"
                            break
                    if reason:
                        break
            if reason:
                audit.append({**base, "outcome": "rejected", "reason": reason,
                              "changed_pixels": 0})
                continue
            yy, xx = np.where(mask)
            box = np.s_[max(0, int(yy.min())-2):min(mask.shape[0], int(yy.max())+3),
                        max(0, int(xx.min())-2):min(mask.shape[1], int(xx.max())+3)]
            markers = np.zeros(mask[box].shape, np.uint16)
            seeds = set()
            for index, point in enumerate(group, 1):
                seed = physical._marker_pixel(mask[box],
                                              float(point.x)-box[1].start,
                                              float(point.y)-box[0].start)
                if seed is None or seed in seeds:
                    reason = "nonunique_raw_seed"
                    break
                seeds.add(seed)
                markers[seed] = index
            if reason:
                audit.append({**base, "outcome": "rejected", "reason": reason,
                              "changed_pixels": 0})
                continue
            partition = watershed(
                -ndi.gaussian_filter(raw[t][box].astype(np.float32),
                                     params.watershed_sigma_px),
                markers, mask=mask[box], connectivity=np.ones((3, 3), bool))
            target_markers = [index for index, point in enumerate(group, 1)
                              if int(point.track_id) in target_ids]
            target_basin = np.isin(partition, target_markers)
            changed_pixels = int(target_basin.sum())
            other_areas = [int((partition == index).sum())
                           for index, point in enumerate(group, 1)
                           if int(point.track_id) not in target_ids]
            if changed_pixels < params.minimum_returning_basin_fraction * source_area:
                reason = "returning_basin_too_small"
            elif not other_areas or min(other_areas) < params.minimum_other_basin_area:
                reason = "other_soma_basin_too_small"
            elif remaining[old_name] <= changed_pixels:
                reason = "would_remove_established_name"
            elif not np.all(candidate[t][box][target_basin] == old_name):
                reason = "overlapping_prior_proposal"
            if reason:
                audit.append({**base, "outcome": "rejected", "reason": reason,
                              "changed_pixels": 0})
                continue
            candidate[t][box][target_basin] = new_name
            remaining[old_name] -= changed_pixels
            application = {**base, "changed_pixels": changed_pixels,
                           "other_basin_pixels": source_area-changed_pixels,
                           "target_somata": len(target), "other_somata": len(other)}
            applications.append(application)
            audit.append({**application, "outcome": "applied", "reason": ""})
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("mixed soma basin correction crossed accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
