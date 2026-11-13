"""Restore a one-frame lost name on a fast, raw-continuous soma path.

Discovery is fieldwide and independent of feedback, recording, identity,
track, frame and region. Every spatial and temporal scale is measured from
the input movie; accepted pixels and foreground are immutable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from restore_spatial_reciprocal_soma_paths_a002 import Frame


def _calibration(frames: list[Frame]) -> dict:
    areas = np.concatenate([f.areas for f in frames if len(f.areas)])
    typical_area = float(np.median(areas))
    radius = float(np.sqrt(typical_area/np.pi))
    minimum_area = typical_area/4
    steps, area_changes = [], []
    for previous, current in zip(frames, frames[1:]):
        a = {}
        b = {}
        for owner, area, x, y in zip(previous.owners, previous.areas,
                                     previous.x, previous.y):
            if area >= minimum_area:
                a.setdefault(int(owner), []).append((area,x,y))
        for owner, area, x, y in zip(current.owners, current.areas,
                                     current.x, current.y):
            if area >= minimum_area:
                b.setdefault(int(owner), []).append((area,x,y))
        for owner in a.keys() & b.keys():
            if len(a[owner]) != 1 or len(b[owner]) != 1:
                continue
            aa, ax, ay = a[owner][0]
            ba, bx, by = b[owner][0]
            steps.append(float(np.hypot(ax-bx, ay-by)))
            area_changes.append(abs(float(np.log(aa/ba))))
    if len(steps) < 20:
        raise AssertionError("recording lacks stable adjacent soma observations")
    return {"soma_radius": radius, "minimum_area": minimum_area,
            "step_q75": float(np.quantile(steps,.75)),
            "step_q995": float(np.quantile(steps,.995)),
            "area_q75": float(np.quantile(area_changes,.75))}


def _one(fr: Frame, owner: int, minimum_area: float):
    found = np.flatnonzero((fr.owners == owner) & (fr.areas >= minimum_area))
    return int(found[0])+1 if len(found) == 1 else None


def _closest(fr: Frame, owner: int, expected: np.ndarray):
    ids = np.flatnonzero(fr.owners == owner)
    if not len(ids):
        return None
    distances = np.hypot(fr.x[ids]-expected[0],fr.y[ids]-expected[1])
    return np.array(fr.center(int(ids[int(np.argmin(distances))])+1))


def _patch(raw: np.ndarray, t: int, xy: np.ndarray, radius: float):
    width = int(round(1.5*radius))
    x, y = np.rint(xy).astype(int)
    if (x-width < 0 or y-width < 0 or x+width >= raw.shape[2]
            or y+width >= raw.shape[1]):
        return None
    image = raw[t,y-width:y+width+1,x-width:x+width+1].astype(float)
    image = gaussian_filter(image,radius/10)
    spread = float(image.std())
    return (image-image.mean())/spread if spread > 0 else None


def correct(labels: np.ndarray, raw: np.ndarray, protected: np.ndarray
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.ndim != 3 or labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError("aligned complete labels, raw and protected masks required")
    yy, xx = np.indices(labels.shape[1:])
    frames = [Frame(labels[t], [], yy, xx) for t in range(len(labels))]
    scale = _calibration(frames)
    radius, minimum_area = scale["soma_radius"], scale["minimum_area"]
    candidate = labels.copy()
    audit, applications = [], []
    for t in range(2,len(frames)-2):
        before, middle, after = frames[t-1:t+2]
        missing = set(map(int,before.owners)) & set(map(int,after.owners)) - \
            set(map(int,middle.owners))
        for name in missing:
            old_id = _one(before,name,minimum_area)
            next_id = _one(after,name,minimum_area)
            if old_id is None or next_id is None:
                continue
            old = np.array(before.center(old_id))
            nxt = np.array(after.center(next_id))
            distance = float(np.linalg.norm(nxt-old))
            if distance > 2*scale["step_q995"]:
                continue
            prediction = (old+nxt)/2
            eligible = np.flatnonzero(middle.areas >= minimum_area)
            if not len(eligible):
                continue
            distances = np.hypot(middle.x[eligible]-prediction[0],
                                 middle.y[eligible]-prediction[1])
            cid = int(eligible[int(np.argmin(distances))])+1
            center = np.array(middle.center(cid))
            source = middle.owner(cid)
            area = middle.area(cid)
            d1, d2 = float(np.linalg.norm(center-old)), float(np.linalg.norm(nxt-center))
            area_log = abs(float(np.log(before.area(old_id)/area))) + \
                abs(float(np.log(area/after.area(next_id))))
            evidence = {"frame": t+1, "old_name": source, "new_name": name,
                        "component_id": cid, "area": area,
                        "midpoint_error_radii": float(distances.min()/radius),
                        "path_straightness": (d1+d2)/max(distance,radius),
                        "area_log_cost_scaled": area_log/max(scale["area_q75"],1e-12),
                        "step_q75": scale["step_q75"],
                        "step_q995": scale["step_q995"],
                        "soma_radius": radius,
                        "area_q75": scale["area_q75"]}
            reason = ""
            if max(d1,d2) > scale["step_q995"]:
                reason = "step_outside_recording_distribution"
            elif evidence["midpoint_error_radii"] > 1:
                reason = "off_interpolated_path"
            elif evidence["path_straightness"] > 1.25:
                reason = "path_not_direct"
            elif evidence["area_log_cost_scaled"] > 3:
                reason = "area_inconsistent"
            early = _closest(frames[t-2],name,old-(nxt-old)/2)
            late = _closest(frames[t+2],name,nxt+(nxt-old)/2)
            if not reason and (early is None or late is None):
                reason = "not_isolated_five_frame_path"
            if not reason:
                path = np.array([early,old,nxt,late])
                design = np.stack([np.ones(4),[-2.,-1.,1.,2.]],axis=1)
                fit = np.linalg.lstsq(design,path,rcond=None)[0]
                max_error = float(np.max(np.linalg.norm(path-design@fit,axis=1)))
                middle_error = float(np.linalg.norm(fit[0]-center))
                speed = float(np.linalg.norm(fit[1]))
                evidence.update(five_frame_max_residual_radii=max_error/radius,
                                five_frame_midpoint_error_radii=middle_error/radius,
                                speed_relative_to_recording_step=speed/max(scale["step_q75"],1e-12))
                if (max_error > 1.5*radius or middle_error > radius
                        or speed <= scale["step_q75"]):
                    reason = "five_frame_path_not_supported"
            if not reason:
                patches = [_patch(raw,index,xy,radius) for index,xy in
                           ((t-1,old),(t,center),(t+1,nxt))]
                if any(p is None for p in patches):
                    reason = "raw_patch_unavailable"
                else:
                    a,b,c = patches
                    ab,bc,ac = (float(np.mean(a*b)),float(np.mean(b*c)),
                                float(np.mean(a*c)))
                    evidence.update(raw_prev_mid=ab,raw_mid_next=bc,
                                    raw_prev_next=ac)
                    if min(ab,bc) <= ac:
                        reason = "raw_soma_not_bilaterally_continuous"
            selected = middle.mask(cid)
            if not reason and np.any(protected[t] & selected):
                reason = "accepted_pixel_protected"
            if not reason and np.any(candidate[t][selected] != source):
                reason = "competing_candidate"
            if not reason:
                candidate[t][selected] = name
                applications.append({"frame": t+1,"old_name": source,
                                     "new_name": name,"changed_pixels": area,
                                     "component_id": cid})
            audit.append({**evidence,"outcome": reason or "applied"})
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0,labels > 0)
            or not np.array_equal(np.unique(candidate),np.unique(labels))):
        raise AssertionError("moving-name layer crossed accepted boundary")
    return candidate,pd.DataFrame(audit),pd.DataFrame(applications)
