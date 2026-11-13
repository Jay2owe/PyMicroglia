"""A measured soma remaining between both endpoints cannot also walk elsewhere.

Certify fixed gap proposals against independent measurements. Earlier inferred
outlines are not measurements. No feedback or recording-specific selectors enter
this operation, and rejecting a proposal cannot free space for another proposal.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from observed_soma_reservation import measured_cores_by_frame


def constrain_envelope_inference(observed, proposals, raw, points, params=None):
    if observed.shape != proposals.shape or observed.shape != raw.shape:
        raise ValueError('Movie shapes differ')
    if np.any((observed > 0) & (observed != proposals)):
        raise ValueError('Expected additive, non-carving recovery proposals')
    quantized = points.copy()
    quantized['x'] = np.floor(quantized.x + .5)
    quantized['y'] = np.floor(quantized.y + .5)
    cores = measured_cores_by_frame(quantized)
    measured = {}; inferred = {}
    for t, frame in enumerate(observed):
        for name in np.unique(frame):
            if name: measured.setdefault(int(name), []).append(t)
        added = (frame == 0) & (proposals[t] > 0)
        for name in np.unique(proposals[t][added]):
            inferred.setdefault(int(name), set()).add(t)
    output = proposals.copy(); audit = []; measured_contact = {}
    for name, frames in inferred.items():
        actual = measured.get(name, [])
        for left, right in zip(actual[:-1], actual[1:]):
            gap = sorted(frames.intersection(range(left + 1, right)))
            if not gap: continue
            sources = [observed[left] == name, observed[right] == name]
            overlap = sources[0] & sources[1]
            if not overlap.any():continue
            common = sources[0] | sources[1]
            # Detector peak names can fragment or exchange within one small
            # outline. Certify spatial occupancy, not a tracker-name history.
            # Every measured endpoint peak must be in the observed endpoint envelope;
            # at least one independently measured peak must stay there on
            # every intervening frame. Endpoint outlines must overlap: their
            # union admits measured boundary jitter without a radius cutoff.
            # Raw template agreement at both anchors is also required.
            def inside(mask,y,x):
                return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y,x]
            eligible = True
            for anchor,source in zip((left,right),sources):
                witnesses = [(y,x) for _,y,x in cores.get(anchor,[]) if inside(source,y,x)]
                if not witnesses or not all(inside(common,y,x) for y,x in witnesses):
                    eligible=False;break
            if not eligible:continue
            if not all(any(inside(common,y,x) for _,y,x in cores.get(t,[]))
                       for t in range(left,right+1)):continue
            templates = []
            for anchor, source in zip((left, right), sources):
                yy, xx = np.nonzero(source)
                ref = raw[anchor, yy, xx].astype(float)
                templates.append((yy, xx, ref, float(np.dot(ref, ref))))
            for t in gap:
                target = (proposals[t] == name) & (observed[t] == 0)
                if any(np.any(target & source) for source in sources): continue
                # A disputed identity on a measured neighbour's boundary is
                # an ownership problem, not proof of empty foreground. Keep
                # that outline for a later ownership correction; only detached
                # inferred structures are eligible for withdrawal here.
                if t not in measured_contact:
                    measured_contact[t] = ndi.binary_dilation(observed[t]>0,np.ones((3,3),bool))
                if np.any(target & measured_contact[t]):continue
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
                    occupancy_evidence='continuous_measured_endpoint_envelope',withdrawn_pixels=int(target.sum()),source_similarity_left=scores[0][0],
                    moved_similarity_left=scores[0][1],source_similarity_right=scores[1][0],moved_similarity_right=scores[1][1],
                    decision='continuous_measured_endpoint_residence_contradicts_inferred_departure'))
    return output, pd.DataFrame(audit)
