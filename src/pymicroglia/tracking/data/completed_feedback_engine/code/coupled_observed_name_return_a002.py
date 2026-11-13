"""Return two bookended physical identities despite weak borrower detection.

The original raw track name is never used as identity authority. An observed
body must have the same isolated name on both sides of a short excursion.
Only transactions with an independently bookended, currently weak borrower
are new here. No feedback, movie, identity, frame or region is an input.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import resident_takeover as ownership
import complete_stationary_soma_path as soma
import recover_recurrent_soma_name_conflicts_a002 as evidence


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray):
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError('aligned labels, raw and protected pixels required')
    scored = ownership.attach_owners(points, labels)
    observed = scored[scored.state.eq('observed')]
    by_frame = {int(t): list(g.itertuples(index=False)) for t, g in observed.groupby('frame')}
    by_track = {int(tr): list(g.sort_values('frame').itertuples(index=False))
                for tr, g in observed.groupby('track_id')}
    gap = max(1, math.ceil(len(labels)*.05))
    bounds = evidence.Params()
    components = {}

    def component(point):
        key = int(point.frame), int(point.track_id)
        if key not in components:
            name = int(point.candidate_owner)
            components[key] = soma._component(labels[key[0]], name, point) if name else None
        return components[key]

    def peers(point, mask):
        return [p for p in by_frame.get(int(point.frame), [])
                if int(p.track_id) != int(point.track_id)
                and mask[round(float(p.y)), round(float(p.x))]]

    def bookends(point, excluded_name):
        t = int(point.frame)
        earlier, later = [], []
        for p in by_track[int(point.track_id)]:
            if not 0 < abs(int(p.frame)-t) <= gap:
                continue
            if int(p.candidate_owner) in (0, excluded_name):
                continue
            mask = component(p)
            if mask is None or peers(p, mask):
                continue
            (earlier if p.frame < t else later).append(p)
        if not earlier or not later:
            return None
        left, right = earlier[-1], later[0]
        if int(left.candidate_owner) != int(right.candidate_owner):
            return None
        return left, right

    overlap_calibration = {}

    def overlap_limit(track, name):
        key = int(track), int(name)
        if key not in overlap_calibration:
            samples = []
            previous = None
            for ref in by_track[int(track)]:
                refmask = component(ref)
                valid = (int(ref.candidate_owner) == name and refmask is not None
                         and not peers(ref, refmask))
                if valid and previous is not None and int(ref.frame)-int(previous.frame) == 1:
                    prior_mask = component(previous)
                    samples.append(np.count_nonzero(refmask & prior_mask)
                                   / min(int(refmask.sum()), int(prior_mask.sum())))
                previous = ref if valid else None
            overlap_calibration[key] = (
                float(np.quantile(samples, .05)) if len(samples) >= 2 else None,
                len(samples))
        return overlap_calibration[key]

    def supported(point, mask, ends):
        if peers(point, mask):
            return False
        areas = [int(component(p).sum()) for p in ends]
        peaks = [evidence._peak(raw[int(p.frame)], p) for p in ends]
        area_ratio = int(mask.sum()) / float(np.median(areas))
        if not bounds.minimum_basin_area_ratio <= area_ratio <= bounds.maximum_basin_area_ratio:
            return False
        if evidence._peak(raw[int(point.frame)], point) < bounds.minimum_raw_peak_retention*np.median(peaks):
            return False
        for ref in ends:
            radius = (float(ref.radius_px)+float(point.radius_px))/2
            step = math.hypot(float(ref.x-point.x), float(ref.y-point.y))
            if step > radius*abs(int(ref.frame)-int(point.frame)):
                return False
        # The body must still occupy its own outline footprint in both
        # directions. Overlap is relative to the smaller body, never pixels.
        threshold, sample_count = overlap_limit(point.track_id, ends[0].candidate_owner)
        if threshold is None:
            return False
        return all(np.count_nonzero(mask & component(p))
                   / min(int(mask.sum()), int(component(p).sum()))
                   >= threshold ** abs(int(p.frame)-int(point.frame))
                   for p in ends)

    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    audits, apps = [], []
    for track, group in by_track.items():
        for point in group:
            source = int(point.candidate_owner)
            if not point.strong or source <= 0:
                continue
            ends = bookends(point, source)
            if ends is None:
                continue
            wanted = int(ends[0].candidate_owner)
            t = int(point.frame)
            target = component(point)
            if target is None or not supported(point, target, ends):
                continue
            external, count = ndi.label(labels[t] == wanted, np.ones((3, 3), bool))
            if count == 0:
                continue
            patches = [(target, source, wanted, track, 'returning_target')]
            weak_holders = 0
            refusal = ''
            for number in range(1, count+1):
                mask = external == number
                holders = []
                for other in by_frame.get(t, []):
                    if int(other.track_id) == track:
                        continue
                    if int(other.candidate_owner) not in (0, wanted):
                        continue
                    # Weak observations may fall just outside the thin outline.
                    # Require that outline within the measured soma radius.
                    other_mask = soma._component(labels[t], wanted, other)
                    if other_mask is not None and np.array_equal(mask, other_mask):
                        holders.append(other)
                if len(holders) != 1:
                    refusal = 'external_name_has_no_unique_observed_body'
                    break
                holder = holders[0]
                references = bookends(holder, wanted)
                if references is None or not supported(holder, mask, references):
                    refusal = 'borrower_has_no_supported_own_name_on_both_sides'
                    break
                alternate = int(references[0].candidate_owner)
                # The alternative may be held by the main target only if it
                # too moves in this transaction. Any third holder blocks it.
                outside = (labels[t] == alternate) & ~target
                if outside.any():
                    refusal = 'borrower_return_name_still_occupied'
                    break
                weak_holders += not bool(holder.strong)
                patches.append((mask, wanted, alternate, int(holder.track_id), 'weak_observed_borrower'))
            if not refusal and not weak_holders:
                continue  # Existing strong-holder correction is protected.
            if not refusal and any(np.any(mask & (protected[t] | occupied[t]))
                                   for mask, _, _, _, _ in patches):
                refusal = 'accepted_or_other_candidate_pixels'
            row = dict(frame=t+1, track_id=track, source_name=source,
                       returning_name=wanted, weak_borrowers=weak_holders,
                       outcome=refusal or 'applied', changed_pixels=0)
            if not refusal:
                for mask, old, new, tr, role in patches:
                    candidate[t][mask] = new
                    occupied[t][mask] = True
                    area = int(mask.sum())
                    row['changed_pixels'] += area
                    apps.append(dict(frame=t+1, track_id=tr, target_track_id=track,
                                     old_name=old, new_name=new, changed_pixels=area, role=role,
                                     overlap_reference_pairs=overlap_limit(tr, new)[1],
                                     overlap_lower_reference_quantile=overlap_limit(tr, new)[0]))
            audits.append(row)
    if (np.any((candidate != labels) & protected)
            or not np.array_equal(candidate > 0, labels > 0)
            or not np.array_equal(np.unique(candidate), np.unique(labels))):
        raise AssertionError('coupled return crossed accepted output boundary')
    return candidate, pd.DataFrame(audits), pd.DataFrame(apps)
