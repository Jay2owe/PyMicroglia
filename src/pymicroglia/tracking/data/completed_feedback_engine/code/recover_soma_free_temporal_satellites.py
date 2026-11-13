"""Rejoin a dim soma-free fragment using a large cell's adjacent-frame footprint.

This producer uses only current masks, raw intensities, physical soma points
and protected accepted pixels. It has no feedback or local selectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as equal_value_components


STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Params:
    minimum_fragment_area: int = 10
    maximum_fragment_area: int = 100
    minimum_host_area: int = 300
    maximum_gap_pixels: int = 2
    maximum_raw_q95_ratio: float = .6
    minimum_direct_temporal_overlap: float = .45
    minimum_expanded_temporal_overlap: float = .9
    temporal_expansion_pixels: int = 3


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray, params: Params = Params()
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.ndim != 3 or labels.shape != protected.shape
            or raw.shape[:3] != labels.shape):
        raise ValueError("labels, raw and protected pixels must align")
    candidate = labels.copy()
    strong = points[(points.state.astype(str) == "observed")
                    & points.strong.astype(bool)]
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    remaining = np.bincount(labels.ravel().astype(np.int64))
    audit = []
    applications = []
    for t, frame in enumerate(labels):
        comps = equal_value_components(frame, connectivity=2, background=0)
        areas = np.bincount(comps.ravel())
        boxes = ndi.find_objects(comps)
        occupied = set()
        for p in by_frame.get(t, strong.iloc[0:0]).itertuples(index=False):
            y, x = round(float(p.y)), round(float(p.x))
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                occupied.add(int(comps[y, x]))
        host_brightness: dict[int, float] = {}
        for cid in np.flatnonzero((areas >= params.minimum_fragment_area)
                                      & (areas <= params.maximum_fragment_area)):
            cid = int(cid)
            if cid == 0 or cid in occupied:
                continue
            box = boxes[cid-1]
            if box is None:
                continue
            margin = params.maximum_gap_pixels + 1
            y0 = max(0, box[0].start-margin)
            y1 = min(frame.shape[0], box[0].stop+margin)
            x0 = max(0, box[1].start-margin)
            x1 = min(frame.shape[1], box[1].stop+margin)
            region = np.s_[y0:y1, x0:x1]
            small = comps[region] == cid
            neighbors = set(map(int, np.unique(comps[region][
                ndi.binary_dilation(small, structure=STRUCTURE,
                                    iterations=params.maximum_gap_pixels)])))
            hosts = [hid for hid in neighbors if hid > 0 and hid != cid
                     and areas[hid] >= params.minimum_host_area]
            if len(hosts) != 1:
                continue
            host = int(hosts[0])
            host_box = boxes[host-1]
            host_name = int(frame[host_box][comps[host_box] == host][0])
            small_name = int(frame[region][small][0])
            if host_name == small_name or remaining[small_name] <= areas[cid]:
                continue
            if np.any(protected[t][region][small]):
                continue
            if host not in host_brightness:
                host_brightness[host] = float(np.percentile(
                    raw[t][host_box][comps[host_box] == host], 95))
            ratio = float(np.percentile(raw[t][region][small], 95)) / max(
                host_brightness[host], 1.)
            if ratio > params.maximum_raw_q95_ratio:
                continue
            direct = expanded = 0.
            for adjacent in (t-1, t+1):
                if not 0 <= adjacent < len(labels):
                    continue
                neighbor = labels[adjacent][region] == host_name
                direct = max(direct, float(np.mean(neighbor[small])))
                near = ndi.binary_dilation(neighbor, structure=STRUCTURE,
                    iterations=params.temporal_expansion_pixels)
                expanded = max(expanded, float(np.mean(near[small])))
            if (direct < params.minimum_direct_temporal_overlap
                    or expanded < params.minimum_expanded_temporal_overlap):
                continue
            # The same original connected outline is the atomic proposal.
            if not np.all(candidate[t][region][small] == small_name):
                raise AssertionError("overlapping soma-free satellite proposals")
            candidate[t][region][small] = host_name
            remaining[small_name] -= int(areas[cid])
            rows = {"frame": t+1, "component": cid,
                    "old_name": small_name, "new_name": host_name,
                    "changed_pixels": int(areas[cid]),
                    "host_area": int(areas[host]),
                    "raw_q95_ratio": ratio,
                    "adjacent_direct_overlap": direct,
                    "adjacent_expanded_overlap": expanded}
            audit.append({**rows, "outcome": "applied"})
            applications.append(rows)
    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError("soma-free satellite crossed an accepted boundary")
    return candidate, pd.DataFrame(audit), pd.DataFrame(applications)
