"""Remove smaller extremely bright optical sources from accepted outlines.

Eligibility uses only image intensity, temporal stability and connected
geometry. Feedback, recording names, identities and coordinates are absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as connected_labels

STRUCTURE = np.ones((3, 3), bool)
CORE_SIGNAL_FACTOR = 50.
CORE_TIME_FRACTION = .8
MIN_CORE_AREA = 20
MAX_CORE_AREA = 59
MIN_MEAN_SIGNAL_FACTOR = 100.
MAX_CORE_MEAN_CV = .10


def _eligible_cores(raw: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    background = np.array([np.median(frame[frame > 0]) if np.any(frame > 0) else 1.
                           for frame in raw], dtype=float)
    counts = np.zeros(raw.shape[1:], np.uint16)
    for t, frame in enumerate(raw):
        counts += frame >= CORE_SIGNAL_FACTOR * background[t]
    stable = counts >= int(np.ceil(CORE_TIME_FRACTION * len(raw)))
    components, count = ndi.label(stable, STRUCTURE)
    eligible = np.zeros(raw.shape[1:], bool)
    audit = []
    boxes = ndi.find_objects(components)
    for index in range(1, count+1):
        box = boxes[index-1]
        if box is None:
            continue
        local = components[box] == index
        area = int(local.sum())
        if not MIN_CORE_AREA <= area <= MAX_CORE_AREA:
            continue
        frame_mean = raw[:, box[0], box[1]][:, local].mean(axis=1)
        cv = float(frame_mean.std()/max(frame_mean.mean(), 1.))
        mean_signal_factor = float(np.mean(frame_mean/background))
        accepted = (cv <= MAX_CORE_MEAN_CV
                    and mean_signal_factor >= MIN_MEAN_SIGNAL_FACTOR)
        if accepted:
            eligible[box][local] = True
        audit.append({"core_index": index, "persistent_core_pixels": area,
                      "core_frame_mean_cv": cv,
                      "mean_signal_factor": mean_signal_factor,
                      "outcome": "eligible" if accepted else "outside_signal_stability_gate"})
    return eligible, pd.DataFrame(audit)


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if (labels.shape != raw.shape or labels.shape != protected.shape
            or labels.ndim != 3):
        raise ValueError("aligned complete label, raw and protection stacks required")
    candidate = labels.copy()
    core, core_audit = _eligible_cores(raw)
    strong = points[(points.state == "observed") & points.strong.astype(bool)]
    by_frame = {int(t): group for t, group in strong.groupby("frame")}
    applications = []
    if not core.any():
        return candidate, core_audit, pd.DataFrame(applications)
    for t, frame in enumerate(labels):
        component = connected_labels(frame, background=0, connectivity=2)
        selected = np.unique(component[core & (frame > 0)])
        selected = selected[selected > 0]
        if not len(selected):
            continue
        boxes = ndi.find_objects(component)
        frame_points = by_frame.get(t, strong.iloc[:0])
        for index in selected:
            box = boxes[int(index)-1]
            source = component[box] == index
            core_part = core[box] & source
            if not core_part.any():
                continue
            keep = np.zeros(source.shape, bool)
            point_count = 0
            for point in frame_points.itertuples(index=False):
                y, x = round(float(point.y)), round(float(point.x))
                if not (box[0].start <= y < box[0].stop
                        and box[1].start <= x < box[1].stop):
                    continue
                yy, xx = y-box[0].start, x-box[1].start
                if source[yy, xx]:
                    point_count += 1
                    if not core_part[yy, xx]:
                        keep[yy, xx] = True
            if keep.any():
                distance_to_core = ndi.distance_transform_edt(~core_part)
                distance_to_keep = ndi.distance_transform_edt(~keep)
                remove = source & (distance_to_core < distance_to_keep)
                method = "partition_around_other_somata"
            else:
                remove = source
                method = "entire_optical_outline"
            if not np.all(remove[core_part]):
                raise AssertionError("persistent source core was not removed")
            name = int(frame[box][source][0])
            if np.any(remove & protected[t][box]):
                applications.append({"frame": t+1, "source_name": name,
                                     "source_pixels": int(source.sum()),
                                     "core_pixels": int(core_part.sum()),
                                     "changed_pixels": 0,
                                     "other_soma_seeds": int(keep.sum()),
                                     "method": method,
                                     "outcome": "rejected_prior_accepted_pixel"})
                continue
            view = candidate[t][box]
            if np.any(view[remove] != name):
                raise AssertionError("artifact component changed before application")
            view[remove] = 0
            applications.append({"frame": t+1, "source_name": name,
                                 "source_pixels": int(source.sum()),
                                 "core_pixels": int(core_part.sum()),
                                 "changed_pixels": int(remove.sum()),
                                 "other_soma_seeds": int(keep.sum()),
                                 "method": method, "outcome": "applied"})
    return candidate, core_audit, pd.DataFrame(applications)
