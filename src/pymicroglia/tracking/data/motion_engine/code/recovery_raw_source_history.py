"""Reject detached guesses contradicted by continuously retained raw appearance.

Two adjacent measured observations calibrate background-subtracted appearance
and contrast. Every intervening raw frame must stay within their observed range;
both templates must prefer residence over the proposed movement. No feedback,
identity selectors, absolute intensity or length thresholds enter this operation.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from observed_soma_reservation import measured_cores_by_frame


def constrain_raw_source_history(observed, proposals, raw, points, params=None):
    if observed.shape != proposals.shape or observed.shape != raw.shape:
        raise ValueError('Movie shapes differ')
    if np.any((observed > 0) & (observed != proposals)):
        raise ValueError('Expected fixed additive proposals')
    points = points.copy()
    points['x'] = np.floor(points.x + .5)
    points['y'] = np.floor(points.y + .5)
    cores = measured_cores_by_frame(points)
    measured, inferred = {}, {}
    for t, frame in enumerate(observed):
        for name in np.unique(frame):
            if name: measured.setdefault(int(name), []).append(t)
        for name in np.unique(proposals[t][(frame == 0) & (proposals[t] > 0)]):
            inferred.setdefault(int(name), set()).add(t)
    output = proposals.copy()
    candidates, contacts, protected = {}, {}, {}

    def has_peak(mask, t):
        return any(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                   and mask[y, x] for _, y, x in cores.get(t, []))

    for name, frames in inferred.items():
        actual = measured.get(name, [])
        for index, (left, right) in enumerate(zip(actual[:-1], actual[1:])):
            if not frames.intersection(range(left + 1, right)): continue
            pairs = []
            if index > 0 and actual[index - 1] == left - 1:
                pairs.append((left - 1, left, range(left + 1, right), 'forward'))
            if index + 2 < len(actual) and actual[index + 2] == right + 1:
                pairs.append((right, right + 1, range(right - 1, left, -1), 'backward'))
            for first, second, scan, direction in pairs:
                masks = [observed[first] == name, observed[second] == name]
                if not np.any(masks[0] & masks[1]): continue
                envelope = masks[0] | masks[1]
                ring = ndi.binary_dilation(envelope, np.ones((3, 3), bool)) & ~envelope
                ry, rx = np.nonzero(ring)
                ey, ex = np.nonzero(envelope)
                if not len(ry): continue

                def residual(t, yy, xx, sy=0, sx=0):
                    return raw[t, yy + sy, xx + sx].astype(float) - float(np.median(raw[t, ry + sy, rx + sx]))

                def contrast(t):
                    inside = float(np.mean(raw[t, ey, ex]))
                    outside = float(np.median(raw[t, ry, rx]))
                    scale = np.hypot(inside, outside)
                    return (inside - outside) / scale if scale > 0 else 0.

                floor = min(contrast(first), contrast(second))
                if floor <= 0: continue
                templates = []
                for anchor, other, mask in [(first, second, masks[0]), (second, first, masks[1])]:
                    yy, xx = np.nonzero(mask)
                    ref = residual(anchor, yy, xx)
                    norm = float(np.linalg.norm(ref))
                    other_ref = residual(other, yy, xx)
                    den = norm * float(np.linalg.norm(other_ref))
                    minimum = float(np.dot(ref, other_ref) / den) if den > 0 else 0.
                    if minimum <= 0: break
                    templates.append((yy, xx, ref, norm, minimum))
                if len(templates) != 2: continue
                for t in scan:
                    current_contrast = contrast(t)
                    if current_contrast < floor: break
                    held = []
                    for yy, xx, ref, norm, minimum in templates:
                        vector = residual(t, yy, xx)
                        den = norm * float(np.linalg.norm(vector))
                        similarity = float(np.dot(ref, vector) / den) if den > 0 else 0.
                        if similarity < minimum: break
                        held.append(similarity)
                    if len(held) != 2: break
                    if t not in frames: continue
                    target = (proposals[t] == name) & (observed[t] == 0)
                    if np.any(target & envelope) or has_peak(target, t): continue
                    if any(np.any(target & (observed[f] > 0)) for f in (t - 1, t + 1) if 0 <= f < len(raw)): continue
                    if t not in contacts:
                        contacts[t] = ndi.binary_dilation(observed[t] > 0, np.ones((3, 3), bool))
                    if np.any(target & contacts[t]): continue
                    ty, tx = np.nonzero(target)
                    if not len(ty): continue
                    moved = []
                    for j, (yy, xx, ref, norm, minimum) in enumerate(templates):
                        sy, sx = np.floor([ty.mean() - yy.mean(), tx.mean() - xx.mean()] + np.array(.5)).astype(int)
                        if min(yy.min() + sy, ry.min() + sy, xx.min() + sx, rx.min() + sx) < 0: break
                        if max(yy.max() + sy, ry.max() + sy) >= raw.shape[1] or max(xx.max() + sx, rx.max() + sx) >= raw.shape[2]: break
                        vector = residual(t, yy, xx, sy, sx)
                        den = norm * float(np.linalg.norm(vector))
                        similarity = float(np.dot(ref, vector) / den) if den > 0 else 0.
                        if held[j] <= similarity: break
                        moved.append(similarity)
                    if len(moved) != 2: continue
                    span = range(first, t + 1) if direction == 'forward' else range(t, second + 1)
                    for f in span:
                        protected.setdefault(f, np.zeros(envelope.shape, bool))[:] |= envelope
                    candidates.setdefault((t, name), dict(identity=name, frame=t + 1,
                        first_gap_frame=left + 2, last_gap_frame=right,
                        first_measured_reference=first + 1, second_measured_reference=second + 1,
                        direction=direction, source_contrast=current_contrast, measured_contrast_floor=floor,
                        source_similarity_left=held[0], source_similarity_right=held[1],
                        measured_similarity_left=templates[0][-1], measured_similarity_right=templates[1][-1],
                        moved_similarity_left=moved[0], moved_similarity_right=moved[1],
                        withdrawn_pixels=int(target.sum()), decision='raw_measured_history_contradicts_detached_recovery'))
    audit = []
    for (t, name), record in candidates.items():
        target = (proposals[t] == name) & (observed[t] == 0)
        if t in protected and np.any(target & protected[t]):
            record['proposed_withdrawn_pixels'] = record['withdrawn_pixels']
            record['withdrawn_pixels'] = 0
            record['decision'] = 'preserved_raw_source_used_by_another_proof'
        else:
            output[t][target] = 0
        audit.append(record)
    return output, pd.DataFrame(audit)
