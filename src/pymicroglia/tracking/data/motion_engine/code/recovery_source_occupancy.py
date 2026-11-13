"""Prevent inferred departures from a soma that remains independently present.

Original gap-recovery hypotheses are fixed. A rejected claim never becomes
space for another claim, and no inferred position supplies a source witness.
"""
import numpy as np
import pandas as pd
from observed_soma_reservation import measured_cores_by_frame


def constrain_recovery_departures(observed, proposals, raw, points, params, *, lag=None):
    if observed.shape != proposals.shape or observed.shape != raw.shape:
        raise ValueError('Movie shapes differ')
    if np.any((observed > 0) & (observed != proposals)):
        raise ValueError('Expected non-carving recovery proposals')
    protect_destination = bool(params.get('protect_measured_destination', False))
    if protect_destination:
        if lag is None or lag.shape != (len(observed)-1, *observed.shape[1:]):
            raise ValueError('Destination evidence requires the original signed motion movie')
        from presence import DEFAULTS
        motion_threshold = float(params.get('motion_log2', DEFAULTS['motion_log2']))
    quantized = points.copy()
    quantized['x'] = np.floor(quantized.x + .5)
    quantized['y'] = np.floor(quantized.y + .5)
    cores = measured_cores_by_frame(quantized)
    by_track = {}
    for t, rows in cores.items():
        for track, y, x in rows:
            by_track.setdefault((t, track), []).append((y, x))
    def in_mask(mask, y, x):
        return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]
    measured = {}; inferred = {}
    for t, frame in enumerate(observed):
        for name in np.unique(frame):
            if name: measured.setdefault(int(name), []).append(t)
        for name in np.unique(proposals[t][(frame == 0) & (proposals[t] > 0)]):
            inferred.setdefault(int(name), set()).add(t)
    output = proposals.copy(); audit = []
    for name, claimed in inferred.items():
        actual = measured.get(name, [])
        for left, right in zip([-1] + actual, actual + [len(observed)]):
            frames = sorted(claimed.intersection(range(left + 1, right)))
            if not frames: continue
            rejected = set(); reasons = {}
            for anchor, direction in ((left, 1), (right, -1)):
                if anchor < 0 or anchor >= len(observed): continue
                source = observed[anchor] == name
                witnesses = [(track, y, x) for track, y, x in cores.get(anchor, [])
                             if in_mask(source, y, x)]
                if len(witnesses) != 1: continue
                track = witnesses[0][0]
                # Only an unbroken measured residence can contradict departure.
                # Once the measured soma moves or becomes uncertain, this anchor
                # cannot veto subsequent legitimate motion.
                end = right if direction == 1 else left
                for t in range(anchor + direction, end, direction):
                    current = by_track.get((t, track), [])
                    if len(current) != 1 or not in_mask(source, *current[0]): break
                    if t not in claimed: continue
                    hypothesis = (proposals[t] == name) & (observed[t] == 0)
                    if np.any(hypothesis & source): continue
                    if protect_destination:
                        # Another measured soma is an ownership ambiguity, not
                        # an unsupported speckle. Likewise preserve a destination
                        # with measured arrival (departure in reverse time).
                        if any(in_mask(hypothesis, y, x) for _, y, x in cores.get(t, [])):
                            continue
                        start, stop = sorted((anchor, t))
                        change = direction * np.nan_to_num(lag[start:stop, hypothesis], nan=0.0).sum(axis=0)
                        if np.any(change >= motion_threshold): continue
                    rejected.add(t)
                    reasons.setdefault(t, []).append((anchor + 1, track))
            for t in frames:
                mask = (proposals[t] == name) & (observed[t] == 0)
                pixels = int(mask.sum()) if t in rejected else 0
                if pixels: output[t][mask] = 0
                audit.append(dict(identity=name, frame=t+1, first_gap_frame=left+2,
                    last_gap_frame=right, decision='retained_source_departure_withdrawn'
                    if pixels else 'recovery_retained', withdrawn_pixels=pixels,
                    source_witnesses=str(reasons.get(t, []))))
    return output, pd.DataFrame(audit)
