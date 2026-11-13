"""Expand undersized accepted short-dropout cores using their own raw soma.

The input application ledger comes from the general short-dropout producer.
Feedback, names of recordings, cell IDs, frames and regions are not inputs.
All previously accepted pixels are immutable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import recover_short_ownerless_soma_runs as core_rule
import separable_merge_recovery as physical

STRUCTURE = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class Params:
    maximum_current_area_fraction: float = .5
    minimum_expanded_area_fraction: float = .65
    maximum_expanded_area_fraction: float = 1.2
    minimum_peak_fraction: float = .08
    minimum_safe_fraction: float = .8
    minimum_reference_frames: int = 20


def _component(labels: np.ndarray, owner: int, point) -> np.ndarray | None:
    y, x = int(round(float(point.y))), int(round(float(point.x)))
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]) or labels[y,x] != owner:
        return None
    parts, _ = ndi.label(labels == owner, STRUCTURE)
    index = int(parts[y,x])
    return parts == index if index else None


def correct(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, prior_applications: pd.DataFrame,
            params: Params = Params()
            ) -> tuple[np.ndarray,np.ndarray,pd.DataFrame,pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != unclaimed.shape:
        raise ValueError("raw, labels and unclaimed must align")
    if prior_applications.empty:
        # Speed (folded from empty_dropout_return): the copies would equal the
        # inputs. The ownerless contact composition, the only caller, copies
        # before every later write, and its stage audit compares contents.
        return labels, unclaimed, pd.DataFrame(), pd.DataFrame()
    result, result_ledger = labels.copy(), unclaimed.copy()
    scored = physical.attach_owners(points, labels)
    for field in ("strong", "physically_visible", "in_encounter"):
        scored[field] = scored[field].astype(bool)
    by_key = scored.set_index(["track_id", "frame"])
    by_frame = {int(t): g for t,g in scored.groupby("frame")}
    audits, applications = [], []
    for track_id, group in prior_applications.groupby("track_id", sort=True):
        track_id = int(track_id)
        owners = group.reference_owner.astype(int).unique()
        if len(owners) != 1:
            continue
        owner = int(owners[0])
        first = int(group.frame.min())-1
        reference = scored[(scored.track_id.astype(int)==track_id)
                           & scored.frame.astype(int).lt(first)
                           & scored.strong & scored.physically_visible
                           & ~scored.in_encounter
                           & scored.state.astype(str).eq("observed")
                           & scored.accepted_owner.astype(int).eq(owner)]
        areas, low_ratios = [], []
        for point in reference.itertuples(index=False):
            t = int(point.frame)
            mask = _component(labels[t], owner, point)
            if mask is None:
                continue
            peak = core_rule._core(raw[t], point, .35)[2]
            if peak <= 0:
                continue
            areas.append(int(mask.sum()))
            low_ratios.append(float(np.quantile(raw[t][mask]/peak,.1)))
        if len(areas) < params.minimum_reference_frames:
            continue
        reference_area = float(np.median(areas))
        threshold = max(params.minimum_peak_fraction,float(np.median(low_ratios)))
        safe, refusals = [], []
        for app in group.itertuples(index=False):
            t = int(app.frame)-1
            try:
                point = by_key.loc[(track_id,t)]
            except KeyError:
                refusals.append(f"{t+1}:physical_point_missing")
                continue
            if isinstance(point,pd.DataFrame):
                refusals.append(f"{t+1}:ambiguous_physical_point")
                continue
            source = result[t]
            existing_component = _component(source,owner,point)
            if existing_component is None:
                refusals.append(f"{t+1}:accepted_core_not_at_point")
                continue
            old_area = int(existing_component.sum())
            if old_area >= params.maximum_current_area_fraction*reference_area:
                refusals.append(f"{t+1}:accepted_core_already_sized")
                continue
            core,region,peak,area = core_rule._core(raw[t],point,threshold)
            expanded_area = int(core.sum())
            if not (params.minimum_expanded_area_fraction*reference_area <= expanded_area
                    <= params.maximum_expanded_area_fraction*reference_area):
                refusals.append(f"{t+1}:raw_core_area_outside_reference_range")
                continue
            local_existing = existing_component[region]
            local_source = source[region]
            if (np.any(local_existing & ~core)
                    or np.any(core & (local_source>0) & ~local_existing)):
                refusals.append(f"{t+1}:accepted_outline_collision")
                continue
            addition = core & (local_source==0)
            if not np.any(addition) or not np.any(ndi.binary_dilation(local_existing,STRUCTURE)&addition):
                refusals.append(f"{t+1}:no_adjacent_new_pixels")
                continue
            other_inside = False
            for other in by_frame.get(t, scored.iloc[0:0]).itertuples(index=False):
                if int(other.track_id)==track_id or not bool(other.physically_visible):
                    continue
                y,x=int(round(float(other.y))),int(round(float(other.x)))
                if (region[0].start<=y<region[0].stop
                        and region[1].start<=x<region[1].stop
                        and core[y-region[0].start,x-region[1].start]):
                    other_inside = True
                    break
            if other_inside:
                refusals.append(f"{t+1}:other_physical_point_in_core")
                continue
            safe.append((t,region,addition,old_area,expanded_area))
        selected = safe if len(safe)/len(group)>=params.minimum_safe_fraction else []
        for t,region,addition,old_area,expanded_area in selected:
            pixels = int(addition.sum())
            result[t][region][addition] = owner
            result_ledger[t][region][addition] = 0
            applications.append({"track_id":track_id,"reference_owner":owner,
                                 "frame":t+1,"added_pixels":pixels,
                                 "old_area":old_area,"expanded_area":expanded_area,
                                 "reference_area":reference_area,
                                 "raw_peak_fraction":threshold})
        audits.append({"track_id":track_id,"reference_owner":owner,
                       "reference_frames":len(areas),"reference_area":reference_area,
                       "raw_peak_fraction":threshold,"prior_frames":len(group),
                       "safe_frames":len(safe),"applied_frames":len(selected),
                       "added_pixels":sum(int(v[2].sum()) for v in selected),
                       "refusals":"|".join(refusals)})
    changed = result != labels
    if (np.any(changed & (labels>0))
            or np.any((result_ledger != unclaimed)&~changed)
            or not np.array_equal(np.unique(labels),np.unique(result))):
        raise AssertionError("accepted pixels or identities changed")
    return result,result_ledger,pd.DataFrame(audits),pd.DataFrame(applications)
