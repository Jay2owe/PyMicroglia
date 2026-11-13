"""Exclude gap links which displace better uninterrupted core measurements.

Eligibility uses existing assignment costs and each detection's measured radius.
It does not introduce a match-cost cutoff or choose names, places or frames.
"""
import numpy as np


def measured_strong(observation):
    return bool(observation.get('measured_strong', observation['strong']))


def constrain_gap_links(cost, tracks, active, detections, frame, params, audit):
    if not params.get('preserve_measured_core_continuations', False) or not cost.size:
        return cost
    gaps = np.asarray([frame-int(tracks[index].observations[-1]['frame'])
                       for index in active])
    gap_rows = np.flatnonzero(gaps > 1)
    if not len(gap_rows):
        return cost
    blocked = np.zeros(cost.shape, dtype=bool)
    for row in np.flatnonzero(gaps == 1):
        history = tracks[active[int(row)]].observations
        if len(history) < 2:
            continue
        previous, last = history[-2:]
        if (int(last['frame'])-int(previous['frame']) != 1
                or not measured_strong(previous) or not measured_strong(last)):
            continue
        best = float(cost[row].min())
        choices = np.flatnonzero(cost[row] == best)
        if best >= 1e5 or len(choices) != 1:
            continue
        column = int(choices[0])
        detection = detections[column]
        if not measured_strong(detection):
            continue
        radius = max(float(last['radius_px']), float(detection['radius_px']))
        separation = np.hypot(float(detection['x'])-float(last['x']),
                              float(detection['y'])-float(last['y']))
        if radius <= 0 or separation > radius:
            continue
        # Evaluate every claim against the original costs. Row ordering cannot
        # manufacture a later claim, and equal-cost alternatives confer no claim.
        competitors = gap_rows[(cost[gap_rows, column] > best)
                               & (cost[gap_rows, column] < 1e5)]
        for competitor in competitors:
            blocked[competitor, column] = True
            if audit is not None:
                audit.append(dict(track_id=tracks[active[int(competitor)]].track_id,
                    frame=int(frame), x=float(detection['x']), y=float(detection['y']),
                    decision='gap_link_displaces_better_measured_core_continuation',
                    continuing_track_id=tracks[active[int(row)]].track_id,
                    gap=int(gaps[competitor]), gap_cost=float(cost[competitor,column]),
                    continuing_cost=best, displacement_over_radius=float(separation/radius)))
    if not blocked.any():
        return cost
    result = cost.copy()
    result[blocked] = 1e6  # Same forbidden-edge representation as the original linker.
    return result
