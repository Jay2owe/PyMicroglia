"""Measured, raw-supported coexistence of two proposed gap endpoints.

No identity is selected or inferred here. The same evidence can distinguish a
gap in assigned labels from physical travel. Callers decide how to use it.
"""
from __future__ import annotations

import numpy as np
from inferred_raw_soma_evidence import has_raw_soma


class CoexistingSomaGapEvidence:
    def __init__(self, raw, points):
        self.raw = raw
        self.by_frame = {}
        self.by_track = {}
        rows = points[points.strong.astype(bool) & points.state.eq('observed')]
        if 'physically_visible' in rows:
            rows = rows[rows.physically_visible.astype(bool)]
        elif 'visibility' in rows:
            rows = rows[rows.visibility.gt(0)]
        for q in rows.itertuples(index=False):
            if not np.isfinite(q.x) or not np.isfinite(q.y):
                continue
            frame, track = int(q.frame), int(q.track_id)
            if not 0 <= frame < len(raw):
                raise ValueError('Raw points must use the raw movie frame system')
            position = (
                int(np.floor(q.y + .5)), int(np.floor(q.x + .5)))
            existing = self.by_track.setdefault(track, {}).get(frame)
            if existing is not None and existing != position:
                raise ValueError('One measured track has conflicting positions')
            self.by_track[track][frame] = position
            self.by_frame.setdefault(frame, {})[track] = position
        self.endpoint_cache = {}
        self.shape_cache = {}

    def _endpoint(self, mask, frame):
        ys, xs = np.nonzero(mask)
        if not len(ys):
            return None
        witnesses = [(track, position) for track, position in
                     self.by_frame.get(frame, {}).items()
                     if 0 <= position[0] < mask.shape[0]
                     and 0 <= position[1] < mask.shape[1]
                     and mask[position]]
        if len(witnesses) != 1:
            return None
        track, position = witnesses[0]
        # Relative mask geometry makes this independent of field coordinates.
        rel_y, rel_x = ys-position[0], xs-position[1]
        key = (frame, track, rel_y.tobytes(), rel_x.tobytes())
        if key not in self.endpoint_cache:
            self.endpoint_cache[key] = (track, rel_y, rel_x, key)
        return self.endpoint_cache[key]

    def _translated(self, endpoint, frame):
        track, rel_y, rel_x, _ = endpoint
        centre = self.by_track[track].get(frame)
        if centre is None:
            return None
        ys, xs = rel_y+centre[0], rel_x+centre[1]
        if (ys.min()<0 or xs.min()<0 or ys.max()>=self.raw.shape[1]
                or xs.max()>=self.raw.shape[2]):
            return None
        y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        mask = np.zeros((y1-y0, x1-x0), bool)
        mask[ys-y0, xs-x0] = True
        return ys, xs, mask, self.raw[frame, y0:y1, x0:x1]

    def _shape_supported(self, endpoint, frame, translated):
        key = (endpoint[3], frame)
        if key not in self.shape_cache:
            self.shape_cache[key] = has_raw_soma(translated[2], translated[3])
        return self.shape_cache[key]

    def measure(self, source_mask, source_frame, target_mask, target_frame):
        source_frame, target_frame = int(source_frame), int(target_frame)
        result = dict(coexisting_raw_somas=False, frames_checked=0,
                      reason='adjacent_assignment')
        if abs(source_frame-target_frame) <= 1:
            return result
        source = self._endpoint(source_mask, source_frame)
        target = self._endpoint(target_mask, target_frame)
        if source is None or target is None:
            result['reason'] = 'endpoint_has_no_unique_measured_core'
            return result
        result.update(source_core=source[0], target_core=target[0])
        if source[0] == target[0]:
            result['reason'] = 'same_measured_core'
            return result
        lo, hi = sorted((source_frame, target_frame))
        for frame in range(lo, hi+1):
            left, right = self._translated(source, frame), self._translated(target, frame)
            if left is None or right is None:
                result['reason'] = 'missing_measurement_or_outside_field'
                return result
            # Exact translated-mask overlap; no radius/distance threshold.
            li = left[0]*self.raw.shape[2]+left[1]
            ri = right[0]*self.raw.shape[2]+right[1]
            if np.intersect1d(li, ri, assume_unique=True).size:
                result['reason'] = 'endpoint_territories_overlap'
                return result
            if (not self._shape_supported(source, frame, left)
                    or not self._shape_supported(target, frame, right)):
                result['reason'] = 'independent_raw_soma_shape_not_supported'
                return result
            result['frames_checked'] += 1
        result.update(coexisting_raw_somas=True, reason='both_raw_somas_coexist_throughout_gap')
        return result
