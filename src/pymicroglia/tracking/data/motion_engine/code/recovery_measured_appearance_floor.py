"""Keep inferred recovery from learning its appearance floor from its guesses."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from observed_soma_reservation import measured_cores_by_frame


def constrain_measured_appearance(observed, proposals, raw, points, params=None, *, proposal_history):
    if observed.shape != proposals.shape or observed.shape != raw.shape:
        raise ValueError('Movie shapes differ')
    if np.any((observed > 0) & (observed != proposals)):
        raise ValueError('Expected fixed additive recovery proposals')
    history = proposal_history[proposal_history.outcome.eq('recovered')]
    if history.duplicated(['identity', 'imagej_frame']).any():
        raise ValueError('Ambiguous recovery origin')
    points = points.copy()
    points['x'] = np.floor(points.x + .5); points['y'] = np.floor(points.y + .5)
    cores = measured_cores_by_frame(points)
    actual, contact, cache = {}, {}, {}
    for t, frame in enumerate(observed):
        for name in np.unique(frame):
            if name: actual.setdefault(int(name), []).append(t)
    output = proposals.copy(); audit = []
    def has_peak(mask, t):
        return any(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y,x] for _,y,x in cores.get(t, []))
    for q in history.itertuples(index=False):
        t = int(q.imagej_frame) - 1; name = int(q.identity); direction = str(q.direction)
        if direction not in ('forward','backward') or not 0 <= t < len(raw): raise ValueError('Invalid recovery origin')
        target = (proposals[t] == name) & (observed[t] == 0)
        if not target.any() or has_peak(target,t): continue
        frames = actual.get(name, [])
        before = [f for f in frames if f < t]; after = [f for f in frames if f > t]
        anchors = before[-2:] if direction == 'forward' else after[:2]
        if len(anchors) != 2: continue
        first, second = anchors
        if second != first + 1: continue
        key = name, first, second
        if key not in cache:
            masks = [observed[first] == name, observed[second] == name]
            envelope = masks[0] | masks[1]
            ring = ndi.binary_dilation(envelope, np.ones((3,3),bool)) & ~envelope
            refs = []; contrasts = []
            if np.any(masks[0] & masks[1]) and ring.any():
                backgrounds = [float(np.median(raw[f][ring])) for f in anchors]
                for f, background in zip(anchors, backgrounds):
                    inside = float(np.mean(raw[f][envelope])); scale = np.hypot(inside,background)
                    contrasts.append((inside-background)/scale if scale>0 else 0.)
                for j, mask in enumerate(masks):
                    yy,xx = np.nonzero(mask)
                    ref = raw[anchors[j],yy,xx].astype(float) - backgrounds[j]
                    other = raw[anchors[1-j],yy,xx].astype(float) - backgrounds[1-j]
                    norm = float(np.linalg.norm(ref)); den = norm * float(np.linalg.norm(other))
                    similarity = float(np.dot(ref,other)/den) if den>0 else 0.
                    refs.append((yy,xx,ref,norm,similarity))
            cache[key] = (envelope, ring, refs, min(contrasts) if contrasts else 0.)
        envelope, ring, refs, floor = cache[key]
        if floor<=0 or len(refs)!=2 or min(r[-1] for r in refs)<=0: continue
        if np.any(target & envelope): continue
        if any(np.any(target & (observed[f]>0)) for f in (t-1,t+1) if 0<=f<len(raw)): continue
        if t not in contact: contact[t]=ndi.binary_dilation(observed[t]>0,np.ones((3,3),bool))
        if np.any(target & contact[t]): continue
        target_ring = ndi.binary_dilation(target,np.ones((3,3),bool)) & ~target
        if not target_ring.any(): continue
        inside = float(np.mean(raw[t][target])); outside = float(np.median(raw[t][target_ring])); scale = np.hypot(inside,outside)
        contrast = (inside-outside)/scale if scale>0 else 0.
        if contrast >= floor: continue
        ty,tx = np.nonzero(target); ry,rx = np.nonzero(ring); scores=[]
        for yy,xx,ref,norm,minimum in refs:
            sy,sx = np.floor(np.array([ty.mean()-yy.mean(),tx.mean()-xx.mean()])+.5).astype(int)
            if min(yy.min()+sy,ry.min()+sy,xx.min()+sx,rx.min()+sx)<0: break
            if max(yy.max()+sy,ry.max()+sy)>=raw.shape[1] or max(xx.max()+sx,rx.max()+sx)>=raw.shape[2]: break
            vector = raw[t,yy+sy,xx+sx].astype(float)-float(np.median(raw[t,ry+sy,rx+sx]))
            den=norm*float(np.linalg.norm(vector)); similarity=float(np.dot(ref,vector)/den) if den>0 else 0.
            if similarity >= minimum: break
            scores.append(similarity)
        if len(scores)!=2: continue
        output[t][target]=0
        audit.append(dict(identity=name,frame=t+1,first_gap_frame=(before[-1]+2 if before else 1),last_gap_frame=(after[0] if after else len(raw)),direction=direction,
            first_measured_reference=first+1,second_measured_reference=second+1,
            measured_contrast_floor=floor,destination_contrast=contrast,
            measured_similarity_left=refs[0][-1],measured_similarity_right=refs[1][-1],
            destination_similarity_left=scores[0],destination_similarity_right=scores[1],
            withdrawn_pixels=int(target.sum()),decision='detached_recovery_below_measured_contrast_and_appearance'))
    return output,pd.DataFrame(audit)
