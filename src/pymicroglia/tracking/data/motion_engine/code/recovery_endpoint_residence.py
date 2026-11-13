"""A measured soma remaining between both endpoints cannot also walk elsewhere.

Certify fixed gap proposals against independent measurements. Earlier inferred
outlines are not measurements. No feedback or recording-specific selectors enter
this operation, and rejecting a proposal cannot free space for another proposal.
"""
import numpy as np
import pandas as pd
from observed_soma_reservation import measured_cores_by_frame


def constrain_endpoint_residence(observed, proposals, raw, points, params=None):
    if observed.shape != proposals.shape or observed.shape != raw.shape:
        raise ValueError('Movie shapes differ')
    if np.any((observed > 0) & (observed != proposals)):
        raise ValueError('Expected additive, non-carving recovery proposals')
    quantized = points.copy()
    quantized['x'] = np.floor(quantized.x + .5)
    quantized['y'] = np.floor(quantized.y + .5)
    cores = measured_cores_by_frame(quantized)
    by_track = {}
    for t, entries in cores.items():
        for track, y, x in entries:
            by_track.setdefault((t, track), []).append((y, x))
    measured = {}; inferred = {}
    for t, frame in enumerate(observed):
        for name in np.unique(frame):
            if name: measured.setdefault(int(name), []).append(t)
        added = (frame == 0) & (proposals[t] > 0)
        for name in np.unique(proposals[t][added]):
            inferred.setdefault(int(name), set()).add(t)
    output = proposals.copy(); audit = []
    for name, frames in inferred.items():
        actual = measured.get(name, [])
        for left, right in zip(actual[:-1], actual[1:]):
            gap = sorted(frames.intersection(range(left + 1, right)))
            if not gap: continue
            sources = [observed[left] == name, observed[right] == name]
            witnesses = [[track for track, y, x in cores.get(anchor, [])
                          if 0 <= y < source.shape[0] and 0 <= x < source.shape[1] and source[y, x]]
                         for anchor, source in zip((left, right), sources)]
            if any(len(w) != 1 for w in witnesses) or witnesses[0] != witnesses[1]:
                continue
            track = witnesses[0][0]; common = sources[0] & sources[1]
            # The same independently measured soma must remain inside BOTH
            # endpoint outlines on every frame, including the endpoints.
            stationary = True
            for t in range(left, right + 1):
                pos = by_track.get((t, track), [])
                if len(pos) != 1 or not common[pos[0]]:
                    stationary = False; break
            if not stationary: continue
            templates = []
            for anchor, source in zip((left, right), sources):
                yy, xx = np.nonzero(source)
                ref = raw[anchor, yy, xx].astype(float)
                templates.append((yy, xx, ref, float(np.dot(ref, ref))))
            for t in gap:
                target = (proposals[t] == name) & (observed[t] == 0)
                if any(np.any(target & source) for source in sources): continue
                ty, tx = np.nonzero(target); scores = []; supported = True
                for yy, xx, ref, energy in templates:
                    shift = np.floor(np.array([ty.mean()-yy.mean(), tx.mean()-xx.mean()])+.5).astype(int)
                    dy, dx = yy+shift[0], xx+shift[1]
                    if energy <= 0 or min(dy.min(),dx.min()) < 0 or dy.max() >= raw.shape[1] or dx.max() >= raw.shape[2]:
                        supported = False; break
                    def similarity(vector):
                        return float(np.dot(ref,vector)/max(np.sqrt(energy*np.dot(vector,vector)),1.))
                    held = similarity(raw[t,yy,xx].astype(float))
                    moved = similarity(raw[t,dy,dx].astype(float))
                    scores.append((held,moved))
                    supported = supported and held > moved
                if not supported: continue
                output[t][target] = 0
                audit.append(dict(identity=name,frame=t+1,first_gap_frame=left+2,last_gap_frame=right,
                    raw_track=track,withdrawn_pixels=int(target.sum()),source_similarity_left=scores[0][0],
                    moved_similarity_left=scores[0][1],source_similarity_right=scores[1][0],moved_similarity_right=scores[1][1],
                    decision='continuous_measured_endpoint_residence_contradicts_inferred_departure'))
    return output, pd.DataFrame(audit)
