"""Recover an established soma after unrelated use of its name has ended.

A short remote name reuse must not veto the original physical cell for the
rest of the recording. This extension handles the suffix where the earlier
name has no remaining external pixels. It does not infer a new name for the
ambiguous competing bodies. All discovery uses images and physical tracks;
no recording, identity, coordinate, frame or feedback selectors are accepted.
"""
from __future__ import annotations

import math
import recording_area_continuity
import numpy as np
import pandas as pd

import anchored_soma_owner_recovery_a002_snapshot as established
import resident_takeover as owner
import recover_recurrent_soma_name_conflicts_a002 as recurrence


AUDIT_COLUMNS = ['raw_track', 'earlier_identity', 'first_takeover_frame',
                 'last_external_frame', 'suffix_first_frame', 'path_last_frame',
                 'exclusive_anchor_frames', 'supported_suffix_frames',
                 'changed_pixels', 'decision', 'reason']
APP_COLUMNS = ['frame', 'raw_track', 'source_identity', 'restored_identity',
               'changed_pixels', 'role']


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != raw.shape or labels.shape != protected.shape:
        raise ValueError('aligned raw, labels and accepted-pixel protection required')
    n = len(labels)
    scored = owner.attach_owners(points, labels)
    area_calibration = recording_area_continuity.calibrate(labels, scored)
    area_limit = area_calibration["absolute_log_area_step_limit"]
    minimum_anchor = established._minimum(
        n, established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    proposals = owner.discover_proposals(scored, {
        'minimum_resident_run_frames': minimum_anchor,
        'minimum_takeover_run_frames': established._minimum(
            n, established.DEFAULTS['minimum_shared_contact_movie_fraction']),
        'maximum_adjoining_blank_frames': math.ceil(
            n*established.DEFAULTS['maximum_blank_movie_fraction']),
        'maximum_endpoint_step_radii': established.DEFAULTS['maximum_endpoint_step_radii'],
    })
    step_limit = established._recording_step_limit(scored)
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    audits, apps = [], []
    eligible = proposals[proposals.precohort_status.eq('eligible')]
    for proposal in eligible.itertuples(index=False):
        track, name = int(proposal.track_id), int(proposal.resident_owner)
        first = int(proposal.first_frame)
        group = scored[(scored.track_id == track)
                       & scored.physically_visible].sort_values('frame')
        strong = group[group.strong.astype(bool)]
        if strong.empty or int(strong.iloc[0].candidate_owner) != name:
            continue
        anchor = established._exclusive_earlier(
            scored, labels, track, name, first,
            math.ceil(n*established.DEFAULTS['maximum_blank_movie_fraction']))
        if anchor < minimum_anchor:
            continue
        path, linked = established._continue_track(
            scored, group[group.frame >= first], step_limit)
        if path.empty or not np.all(np.diff(path.frame.to_numpy(int)) == 1):
            continue
        # Inspect exact external pixels, not just named centres. A fragment
        # without a detected soma still counts as a competing use of a name.
        external_frames = []
        for point in path.itertuples(index=False):
            t = int(point.frame)
            target = (established.soma._component(labels[t], name, point)
                      if int(point.candidate_owner) == name else None)
            count = int(np.count_nonzero(labels[t] == name))
            if target is not None:
                count -= int(target.sum())
            if count > 0:
                external_frames.append(t)
        if not external_frames:
            continue  # Established recovery already owns conflict-free paths.
        last_external = max(external_frames)
        suffix = path[path.frame > last_external]
        if len(suffix) < anchor:
            continue
        row = dict(raw_track=track, earlier_identity=name,
                   first_takeover_frame=first+1, last_external_frame=last_external+1,
                   suffix_first_frame=last_external+2,
                   path_last_frame=int(path.frame.max())+1,
                   exclusive_anchor_frames=anchor, supported_suffix_frames=0,
                   changed_pixels=0, decision='rejected', reason='eligible')
        # A raw track identifier is not sufficient: motion must remain inside
        # the measured recording's normal adjacent-frame envelope.
        xy = path[['x', 'y']].to_numpy(float)
        radii = path.radius_px.to_numpy(float)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(
            (radii[:-1]+radii[1:])/2, np.finfo(float).eps)
        if len(steps) and float(np.max(steps)) > step_limit:
            row['reason'] = 'raw_path_step_outside_recording_envelope'
            audits.append(row)
            continue
        references = []
        for point in strong[strong.frame < first].itertuples(index=False):
            t = int(point.frame)
            if int(point.candidate_owner) != name:
                continue
            component = established.soma._component(labels[t], name, point)
            if component is not None and not established._strong_other(
                    scored, t, track, point, component):
                references.append((int(component.sum()), recurrence._peak(raw[t], point)))
        if len(references) < minimum_anchor:
            continue
        reference_area, reference_peak = np.median(references, axis=0)
        bounds = recurrence.Params()
        patches = []
        previous = None
        for before_point in group[group.frame < int(suffix.frame.min())].itertuples(index=False):
            before_name = int(before_point.candidate_owner)
            before_mask = established.soma._component(labels[int(before_point.frame)], before_name, before_point) if before_name > 0 else None
            previous = (int(before_point.frame), int(before_mask.sum())) if before_mask is not None else None
        row.update(area_reference_steps=area_calibration['samples'], area_step_limit=area_limit)
        for point in suffix.itertuples(index=False):
            t, source = int(point.frame), int(point.candidate_owner)
            if source == name:
                row['supported_suffix_frames'] += 1
                continue
            if source == 0:
                continue  # Existing missing outlines are reported, never filled.
            component = established.soma._component(labels[t], source, point)
            if component is None:
                row['reason'] = 'named_component_missing'
                break
            if established._strong_other(scored, t, int(point.track_id), point, component):
                row['reason'] = 'suffix_contains_shared_soma_outline'
                break
            area = int(component.sum())
            if area_limit is None:
                row['reason'] = 'no_recording_area_calibration'
                break
            if previous is not None and t == previous[0]+1:
                step = abs(float(np.log(area/previous[1])))
                if step > area_limit:
                    row['reason'] = 'area_step_outside_recording_envelope'
                    break
            previous = (t, area)
            if recurrence._peak(raw[t], point) < bounds.minimum_raw_peak_retention*reference_peak:
                row['reason'] = 'raw_soma_no_longer_supported'
                break
            if np.any(component & (protected[t] | occupied[t])):
                row['reason'] = 'accepted_or_other_candidate_pixels'
                break
            patches.append((t, component, source, int(point.track_id)))
            row['supported_suffix_frames'] += 1
        if row['reason'] == 'eligible' and patches:
            for t, component, source, raw_track in patches:
                candidate[t][component] = name
                occupied[t][component] = True
                apps.append(dict(frame=t+1, raw_track=raw_track,
                                 source_identity=source, restored_identity=name,
                                 changed_pixels=int(component.sum()),
                                 role='anchored_name_after_transient_reuse'))
            row['changed_pixels'] = sum(int(mask.sum()) for _, mask, _, _ in patches)
            row['decision'] = 'applied'
        audits.append(row)
    return (candidate, pd.DataFrame(audits, columns=AUDIT_COLUMNS + ["area_reference_steps", "area_step_limit"]),
            pd.DataFrame(apps, columns=APP_COLUMNS))
