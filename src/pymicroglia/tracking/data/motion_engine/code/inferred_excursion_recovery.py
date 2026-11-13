"""Field-wide repair of inference-only excursions between observed bookends.

Original detected foreground is immutable. Only pixels explicitly proved to
originate in the accepted gap-recovery stages may be withdrawn or relocated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import presence


DEFAULTS = dict(
    maximum_gap_frames=15, minimum_gap_frames=1,
    maximum_bookend_distance_radii=1.5, minimum_excursion_radii=1.5,
    minimum_observed_frames=3, maximum_area_quantile=.5,
    maximum_intensity_quantile=.5, search_radius_radii=3.,
    replacement_mode="replace_inferred", minimum_raw_area_ratio=.4,
    maximum_new_duplicate_components=0)


def parameters(overrides=None):
    supplied = overrides or {}
    unexpected = set(supplied) - set(DEFAULTS)
    if unexpected:
        raise ValueError("Unknown or targeted parameters: " + str(sorted(unexpected)))
    result = {**DEFAULTS, **supplied}
    if result["replacement_mode"] not in ("replace_inferred", "preserve_named"):
        raise ValueError("Invalid replacement mode")
    return result


def observations(labels, raw, inferred):
    rows = []
    for t, plane in enumerate(labels):
        for identity, slices in enumerate(ndi.find_objects(plane), 1):
            if slices is None:
                continue
            mask = plane[slices] == identity
            yy, xx = np.nonzero(mask)
            rows.append(dict(
                t=t, identity=identity, area=len(xx),
                x=float(xx.mean() + slices[1].start),
                y=float(yy.mean() + slices[0].start),
                mean=float(raw[t][slices][mask].mean()),
                inferred=bool(np.all(inferred[t][slices][mask])),
                components=int(ndi.label(mask, np.ones((3, 3)))[1])))
    return pd.DataFrame(rows, columns=[
        "t", "identity", "area", "x", "y", "mean", "inferred", "components"])


def discover(labels, raw, inferred, params):
    obs = observations(labels, raw, inferred)
    stats = obs.groupby("identity")[["area", "mean"]].median()
    area_cut = float(stats.area.quantile(params["maximum_area_quantile"]))
    mean_cut = float(stats["mean"].quantile(params["maximum_intensity_quantile"]))
    events, audit = [], []
    for identity, group in obs.groupby("identity"):
        group = group.sort_values("t").set_index("t", drop=False)
        flags = np.zeros(len(labels), bool)
        ticks = group.index.to_numpy()
        flags[ticks] = group.inferred.to_numpy() & (group.components.to_numpy() == 1)
        starts = np.flatnonzero(np.diff(np.r_[False, flags, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, flags, False].astype(int)) == -1) - 1
        for first, last in zip(starts, ends):
            item = dict(
                identity=int(identity), first=int(first), last=int(last),
                area_median=float(stats.loc[identity, "area"]),
                intensity_median=float(stats.loc[identity, "mean"]),
                accepted=False, reason="")
            reason = "eligible"
            if (stats.loc[identity, "area"] > area_cut
                    or stats.loc[identity, "mean"] > mean_cut):
                reason = "outside_small_dim_cohort"
            elif not (params["minimum_gap_frames"] <= last - first + 1
                      <= params["maximum_gap_frames"]):
                reason = "outside_gap_duration"
            elif first - 1 not in group.index or last + 1 not in group.index:
                reason = "unbracketed"
            elif int((~group.inferred).sum()) < params["minimum_observed_frames"]:
                reason = "insufficient_observed_history"
            if reason == "eligible":
                before, after = group.loc[first - 1], group.loc[last + 1]
                radius = float(np.sqrt(np.median([before.area, after.area]) / np.pi))
                start = before[["y", "x"]].to_numpy(float)
                end = after[["y", "x"]].to_numpy(float)
                bookend_distance = float(np.linalg.norm(end - start) / max(radius, 1))
                gap = group.loc[first:last]
                alpha = (gap.t.to_numpy() - first + 1) / (last - first + 2)
                expected = start[None, :] + alpha[:, None] * (end - start)[None, :]
                deviations = np.linalg.norm(
                    gap[["y", "x"]].to_numpy() - expected, axis=1) / max(radius, 1)
                item.update(radius=radius,
                            bookend_distance_radii=bookend_distance,
                            max_excursion_radii=float(deviations.max()))
                if bookend_distance > params["maximum_bookend_distance_radii"]:
                    reason = "moving_bookends"
                elif deviations.max() < params["minimum_excursion_radii"]:
                    reason = "no_large_inferred_excursion"
                else:
                    events.append((item, before, after, start, end, radius))
            item["reason"], item["accepted"] = reason, reason == "eligible"
            audit.append(item)
    events.sort(key=lambda event: (
        event[0]["first"], event[0]["last"], *event[3], *event[4]))
    return obs, events, pd.DataFrame(audit)


def component_excess(before, after):
    total = 0
    for owner in np.union1d(np.unique(before), np.unique(after)):
        if owner:
            total += max(0,
                ndi.label(after == owner, np.ones((3, 3)))[1]
                - ndi.label(before == owner, np.ones((3, 3)))[1])
    return int(total)


def produce(labels, raw, lag, inferred, overrides=None):
    params = parameters(overrides)
    if labels.shape != raw.shape or inferred.shape != labels.shape:
        raise ValueError("unaligned inputs")
    if inferred.dtype != bool:
        raise ValueError("inference provenance must be boolean")
    if (labels.ndim != 3 or lag.ndim != 3
            or lag.shape[1:] != labels.shape[1:]
            or len(lag) < max(1, len(labels) - 1)):
        raise ValueError("unaligned lag evidence")
    if np.any(inferred & (labels == 0)):
        raise ValueError("inference provenance must name existing pixels")
    obs, events, audit = discover(labels, raw, inferred, params)
    expected_area = obs.groupby("identity").area.median().to_dict()
    candidate, frames = labels.copy(), []
    claims = np.zeros(labels.shape, bool)
    base_params = {
        **presence.DEFAULTS, "carve_hosts": True, "allow_new_foreground": True,
        "minimum_recovered_area_ratio": params["minimum_raw_area_ratio"],
        "_background_level": presence._background_level(
            labels, raw, presence.DEFAULTS)}
    for number, (item, before, after, start, end, radius) in enumerate(events, 1):
        identity, first, last = item["identity"], item["first"], item["last"]
        proposed, details, failed = {}, [], False
        template = labels[int(before.t)] == identity
        for t in range(first, last + 1):
            initial = candidate[t]
            if np.any(claims[t] & (labels[t] == identity)):
                details.append(dict(episode=number, identity=identity, t=t,
                    reason="overlapping_prior_episode", applied=False))
                failed = True
                break
            work = initial.copy()
            removed = (work == identity) & inferred[t]
            work[removed] = 0
            centre = start + (t - first + 1) / (last - first + 2) * (end - start)
            descriptor = presence.Descriptor(
                identity=identity, t=t - 1, mask=template, position=centre,
                velocity=np.zeros(2), areas=[int(before.area), int(after.area)],
                means=[float(before["mean"]), float(after["mean"])])
            local_params = {
                **base_params, "search_px": params["search_radius_radii"] * radius,
                "maximum_search_px": params["search_radius_radii"] * radius}
            found = presence.likely_area(
                descriptor, t, raw, lag, work, expected_area, local_params,
                labels[t - 1] if t else None)
            detail = dict(episode=number, identity=identity, t=t,
                review_frame=t + 1, applied=False,
                inferred_removed=int(removed.sum()), reason="no_candidate_area")
            if found is None:
                details.append(detail); failed = True; continue
            (rr, cc), scores, free, _, context = found
            local = presence.sculpt(
                scores, free, int(round(descriptor.area)), local_params)
            if local is None:
                details.append(detail); failed = True; continue
            mask = np.zeros(labels.shape[1:], bool)
            mask[rr, cc] = local
            mask, _ = presence._settle_carves(mask, work, local_params)
            if not mask.any():
                details.append(detail); failed = True; continue
            okay, reason, measure = presence._recovery_reason(
                mask, descriptor, raw[t], mask & (work == 0), context, local_params)
            detail.update(reason=reason, **measure)
            if not okay:
                details.append(detail); failed = True; continue
            work[mask] = identity
            if params["replacement_mode"] == "preserve_named":
                work[removed & ~mask] = initial[removed & ~mask]
            observed_lost = (labels[t] > 0) & ~inferred[t] & (work == 0)
            lost_ids = set(np.unique(initial)) - set(np.unique(work))
            excess = component_excess(initial, work)
            detail.update(
                observed_pixels_removed=int(observed_lost.sum()),
                new_duplicate_components=excess,
                donor_ids_lost="|".join(map(str, sorted(lost_ids - {0}))),
                reassigned_pixels=int(np.count_nonzero(
                    (initial > 0) & (work > 0) & (work != initial))))
            if np.any(claims[t] & (work != initial)):
                detail["reason"], failed = "conflicting_prior_claim", True
            elif observed_lost.any() or lost_ids - {0}:
                detail["reason"], failed = "observed_foreground_or_identity_lost", True
            elif excess > params["maximum_new_duplicate_components"]:
                detail["reason"], failed = "new_duplicate_component", True
            proposed[t] = work
            details.append(detail)
        if not failed and len(proposed) == last - first + 1:
            for t, work in proposed.items():
                claims[t] |= candidate[t] != work
                candidate[t] = work
            for detail in details:
                detail["applied"] = True
        else:
            for detail in details:
                detail["episode_rolled_back"] = True
        frames.extend(details)
    frame_table = pd.DataFrame(frames)
    removed = (labels > 0) & (candidate == 0)
    if np.any(removed & ~inferred):
        raise AssertionError("pre-gap detected foreground was removed")
    summary = dict(
        targeting_mode="field_wide_discovery", events_discovered=len(events),
        episodes_applied=(int(frame_table.loc[
            frame_table.applied, "episode"].nunique()) if len(frame_table) else 0),
        changed_pixels=int(np.count_nonzero(candidate != labels)),
        changed_frames=int(np.any(candidate != labels, axis=(1, 2)).sum()),
        removed_inference_only_pixels=int(removed.sum()),
        removed_original_foreground_pixels=int((removed & ~inferred).sum()),
        added_pixels=int(np.count_nonzero((candidate > 0) & (labels == 0))))
    return candidate, audit, frame_table, summary


def update_ledger(labels, candidate, current_unclaimed,
                  pre_gap_unclaimed, inferred):
    arrays = [candidate, current_unclaimed, pre_gap_unclaimed, inferred]
    if any(array.shape != labels.shape for array in arrays):
        raise ValueError("unaligned ledger inputs")
    if np.any((labels > 0) & (current_unclaimed > 0)):
        raise ValueError("current ledger overlaps named foreground")
    result = current_unclaimed.copy()
    removed = (labels > 0) & (candidate == 0)
    if np.any(removed & ~inferred):
        raise ValueError("cannot remove pre-gap detected foreground")
    restore = removed & (pre_gap_unclaimed > 0) & (result == 0)
    result[restore] = pre_gap_unclaimed[restore]
    result[candidate > 0] = 0
    if np.any((current_unclaimed > 0) & (result == 0) & (candidate == 0)):
        raise AssertionError("existing unclaimed foreground was lost")
    return result

