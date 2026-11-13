"""Require spatial raw support before accepting detached inferred foreground.

The historical scalar contrast test can select bright speckles from an empty
background. A detached proposed soma must additionally contain a connected
bright interior, using its own boundary median as the reference. Independently
measured soma footprints and measured paths remain protected.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def constrain_destination_soma(observed, proposals, raw, points, params=None, *, proposal_history, source_overlap_mode='any', measured_path_mode='any'):
    if source_overlap_mode not in ('any', 'measured_interior', 'raw_supported_interior'):
        raise ValueError('Unknown source overlap support mode')
    if measured_path_mode not in ('any', 'measured_interior', 'neighbour_interior'):
        raise ValueError('Unknown measured path support mode')
    if observed.shape != proposals.shape or observed.shape != raw.shape:
        raise ValueError('Movie shapes differ')
    if np.any((observed > 0) & (observed != proposals)):
        raise ValueError('Expected fixed additive recovery proposals')
    required = {'identity', 'imagej_frame', 'direction', 'outcome'}
    if not required.issubset(proposal_history.columns):
        raise ValueError('Recovery origin records are incomplete')
    history = proposal_history[proposal_history.outcome.eq('recovered')]
    if history.duplicated(['identity', 'imagej_frame']).any():
        raise ValueError('Ambiguous proposal origin')
    selected = points[points.strong.astype(bool) & points.state.eq('observed')]
    if 'physically_visible' in selected:
        selected = selected[selected.physically_visible.astype(bool)]
    elif 'visibility' in selected:
        selected = selected[selected.visibility.gt(0)]
    footprints = {}
    for q in selected.itertuples(index=False):
        t = int(q.frame); x = float(q.x); y = float(q.y)
        radius = float(getattr(q, 'radius_px', 0.))
        if not 0 <= t < len(raw) or not np.isfinite([x, y, radius]).all() or radius < 0:
            continue
        y0=max(0,int(np.floor(y-radius))); y1=min(raw.shape[1],int(np.ceil(y+radius))+1)
        x0=max(0,int(np.floor(x-radius))); x1=min(raw.shape[2],int(np.ceil(x+radius))+1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        coverage = footprints.setdefault(t, np.zeros(raw.shape[1:], bool))
        coverage[y0:y1, x0:x1] |= (yy-y)**2 + (xx-x)**2 <= radius**2
        # Preserve the independently measured centre even for a subpixel disk.
        iy, ix = int(np.floor(y+.5)), int(np.floor(x+.5))
        if 0 <= iy < raw.shape[1] and 0 <= ix < raw.shape[2]:
            coverage[iy, ix] = True
    actual = {}
    for t, frame in enumerate(observed):
        for name in np.unique(frame):
            if name:
                actual.setdefault(int(name), []).append(t)
    output = proposals.copy(); audit = []; contact = {}
    cross = ndi.generate_binary_structure(2, 1)
    measured_support = {}
    def support_at(t):
        if measured_path_mode == 'any':
            return observed[t] > 0
        if t not in measured_support:
            frame = observed[t]
            support = np.zeros(frame.shape, bool)
            # Bound each erosion to the measured object's extent. Zero padding
            # equals full-field erosion; unrelated identities never merge.
            for identity in np.unique(frame):
                if not identity:
                    continue
                yy, xx = np.nonzero(frame == identity)
                roi = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
                body = frame[roi] == identity
                interior = ndi.binary_erosion(body, cross)
                support[roi] |= interior if interior.any() else body
            measured_support[t] = support
        return measured_support[t]
    for q in history.itertuples(index=False):
        t = int(q.imagej_frame)-1; name = int(q.identity); direction = str(q.direction)
        if direction not in ('forward', 'backward') or not 0 <= t < len(raw):
            raise ValueError('Invalid recovery origin')
        target = (proposals[t] == name) & (observed[t] == 0)
        if not target.any():
            continue
        seen = actual.get(name, [])
        before = [f for f in seen if f < t]; after = [f for f in seen if f > t]
        anchor = (before[-1] if before else None) if direction == 'forward' else (after[0] if after else None)
        if anchor is None:
            continue
        source = observed[anchor] == name
        if np.any(target & source):
            if source_overlap_mode == 'any':
                continue
            sy, sx = np.nonzero(source)
            source_roi = np.s_[sy.min():sy.max()+1, sx.min():sx.max()+1]
            measured_interior = ndi.binary_erosion(source[source_roi], cross)
            # An unresolved thin source cannot establish an interior boundary.
            if not measured_interior.any():
                continue
            if np.any(target[source_roi] & measured_interior):
                if source_overlap_mode == 'measured_interior':
                    continue
                # A copied outline is not independent signal evidence. Reuse
                # the destination's boundary-relative spatial test at its
                # measured source. Keep the entire geometric interior when
                # any coherent source soma is present; do not demand exact
                # overlap with a fluctuating bright core.
                source_body = source[source_roi]
                source_patch = raw[anchor][source_roi]
                boundary = source_body & ~measured_interior
                level = np.median(source_patch[boundary])
                if ndi.binary_erosion(source_body & (source_patch > level), cross).any():
                    continue
        if any(f in footprints and np.any(target & footprints[f]) for f in (t-1, t, t+1)):
            continue
        if any(np.any(target & (support_at(f) if measured_path_mode == 'measured_interior' else observed[f] > 0)) for f in (t-1, t+1) if 0 <= f < len(raw)):
            continue
        if t not in contact:
            contact[t] = ndi.binary_dilation(support_at(t), np.ones((3, 3), bool))
        if np.any(target & contact[t]):
            continue
        yy, xx = np.nonzero(target)
        roi = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
        small = target[roi]; patch = raw[t][roi]
        interior = ndi.binary_erosion(small, cross)
        boundary = small & ~interior
        level = float(np.median(patch[boundary]))
        coherent = ndi.binary_erosion(small & (patch > level), cross)
        if coherent.any():
            continue
        output[t][target] = 0
        audit.append(dict(identity=name, frame=t+1,
            first_gap_frame=(before[-1]+2 if before else 1), last_gap_frame=(after[0] if after else len(raw)),
            direction=direction, source_anchor_frame=anchor+1,
            destination_boundary_median=level, destination_geometric_interior=int(interior.sum()),
            destination_raw_coherent_interior=int(coherent.sum()),
            measured_soma_footprint_contact=False, adjacent_measured_path_overlap=False,
            measured_neighbour_contact=False, withdrawn_pixels=int(target.sum()),
            decision='detached_recovery_without_spatial_raw_soma_support'))
    return output, pd.DataFrame(audit)
