"""Independent, contemporaneous soma occupancy for native identity assignment.

The detector's name is only a correspondence key between measurements. No
particular name, frame, location, recording or feedback mark is selected.
"""
from __future__ import annotations

import numpy as np


def measured_cores_by_frame(points):
    required = {'frame', 'track_id', 'x', 'y', 'strong', 'state'}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"Independent soma evidence lacks {sorted(missing)}")
    selected = points[points.strong.astype(bool) & points.state.eq('observed')]
    if 'physically_visible' in selected:
        selected = selected[selected.physically_visible.astype(bool)]
    elif 'visibility' in selected:
        selected = selected[selected.visibility.gt(0)]
    frames = {}
    for frame, rows in selected.groupby('frame', sort=False):
        # Duplicate detector rows must not create another biological witness.
        rows = rows.drop_duplicates(['track_id', 'x', 'y'])
        frames[int(frame)] = [(int(q.track_id), int(round(q.y)), int(round(q.x)))
                              for q in rows.itertuples(index=False)
                              if np.isfinite(q.x) and np.isfinite(q.y)]
    return frames


def _inside(mask, y, x):
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def retained_source_constraints(states, observations, target_t, frame_labels, cores,
                                raw_frame=None):
    """Disallow departure when the same independent core still occupies its source.

    A shared source outline is ambiguous. Genuine movement out of the old
    outline cannot trigger this constraint. All coordinate decisions are mask
    membership and component separation, with no added image-unit threshold.
    """
    current = cores.get(int(target_t), [])
    if not current:
        return {}
    current_by_track = {}
    for track, y, x in current:
        current_by_track.setdefault(track, []).append((y, x))
    forbidden = {}
    for state in states:
        earlier = [(track, y, x) for track, y, x in cores.get(int(state.t), [])
                   if _inside(state.mask, y, x)]
        if len(earlier) != 1:
            continue
        track, old_y, old_x = earlier[0]
        witnesses = current_by_track.get(track, [])
        if len(witnesses) != 1:
            continue
        y, x = witnesses[0]
        if not _inside(state.mask, y, x):
            continue
        occupied = int(frame_labels[y, x])
        if occupied <= 0:
            continue
        appearance = {}
        if raw_frame is not None:
            if state.bounds is None:
                retained_mean = float(raw_frame[state.mask].mean())
            else:
                y0,y1,x0,x1=state.bounds
                retained_mean=float(raw_frame[y0:y1,x0:x1][state.mask[y0:y1,x0:x1]].mean())
            retained_change=abs(float(np.log2((retained_mean+1.0)/(state.mean+1.0))))
            appearance=dict(retained_raw_mean=retained_mean,retained_mean_log2_change=retained_change)
        for observation in observations:
            if int(observation['local_id']) == occupied:
                continue
            a = state.bounds
            b = observation.get('bounds')
            if a is None or b is None:
                overlaps = np.any(state.mask & observation['mask'])
            else:
                y0, y1 = max(a[0], b[0]), min(a[1], b[1])
                x0, x1 = max(a[2], b[2]), min(a[3], b[3])
                overlaps = (y0 < y1 and x0 < x1 and np.any(
                    state.mask[y0:y1, x0:x1] & observation['mask'][y0:y1, x0:x1]))
            if overlaps:
                continue
            evidence=dict(appearance)
            if raw_frame is not None:
                destination_change=abs(float(np.log2((observation['mean']+1.0)/(state.mean+1.0))))
                # The retained point must still represent the prior cell body,
                # not just a dim remnant left after a genuine departure. Use the
                # same dimensionless mean-intensity change as the native cost.
                if retained_change>destination_change:continue
                evidence['destination_mean_log2_change']=destination_change
            forbidden[state.identity, int(observation['local_id'])] = dict(
                raw_track=track, raw_source_y=old_y, raw_source_x=old_x,
                raw_current_y=y, raw_current_x=x, occupied_local_id=occupied,**evidence)
    return forbidden
