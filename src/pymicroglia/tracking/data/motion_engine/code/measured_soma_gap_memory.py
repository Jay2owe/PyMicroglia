"""Do not let an unmeasured fragment redefine a soma's gap-search source.

The size comparison uses that body's own observed core radii. There is no
recording, biological identity, detector identity, frame or region selector.
Installation is scoped to one tracking run; the established tracker is restored
afterwards. Normal adjacent-frame links use the original features unchanged.
"""
from contextlib import contextmanager
from dataclasses import replace
import inspect

import numpy as np


class MeasuredSomaGapMemory:
    def __init__(self, points_by_frame, direction, events):
        self.points = points_by_frame
        self.direction = int(direction)
        self.events = events
        self.sources = {}
        self.cache = {}
        self.observation_areas = {}

    def core_area(self, mask, frame):
        points = self.points.get(int(frame))
        if points is None or not len(points):
            return None
        # Exact containment is deliberate: nearby peaks do not prove that a
        # partial outline has independently measured its own soma.
        y = np.floor(points[:, 0] + .5).astype(int)
        x = np.floor(points[:, 1] + .5).astype(int)
        valid = (y >= 0) & (y < mask.shape[0]) & (x >= 0) & (x < mask.shape[1])
        points, y, x = points[valid], y[valid], x[valid]
        radii = points[mask[y, x], 2]
        if not len(radii):
            return None
        # A shared outline may contain more than one measured core. Only a
        # fragment smaller than even the smallest core is classified here.
        return float(np.pi * np.min(radii) ** 2)

    def source(self, state):
        cached = self.cache.get(state.identity)
        if cached is not None and cached[0] is state:
            return cached[1]
        area = self.core_area(state.mask, state.t)
        previous = self.sources.get(state.identity)
        if area is not None:
            previous = (state, area)
            self.sources[state.identity] = previous
            answer = None
        elif previous is not None and state.area < previous[1]:
            answer = previous
        elif previous is not None:
            # A dim but soma-sized silhouette remains an ordinary observation.
            self.sources[state.identity] = (state, previous[1])
            answer = None
        else:
            answer = None
        self.cache[state.identity] = (state, answer)
        return answer

    def features(self, original, state, observation, target_t, lag, direction, params):
        reference = self.source(state)
        ordinary = original(state, observation, target_t, lag, direction, params)
        if (reference is None or abs(int(target_t) - int(state.t)) <= 1
                or ordinary.get('independent_source_soma_retained', False)):
            return ordinary
        key = (int(target_t), int(observation['local_id']))
        if key not in self.observation_areas:
            self.observation_areas[key] = self.core_area(observation['mask'], target_t)
        if self.observation_areas[key] is None:
            return ordinary
        measured, minimum_area = reference
        # The identity and its protected reservation membership are unchanged.
        # A snapshot supplies the position, velocity, morphology and actual time
        # of the preceding body that could contain its observed soma.
        effective = replace(measured, identity=state.identity)
        result = dict(original(effective, observation, target_t, lag, direction, params))
        measurement_gap = result['gap_steps']
        result.update(gap_steps=ordinary['gap_steps'],
            soma_measurement_gap_steps=measurement_gap,
            soma_source_frame=int(measured.t),
            soma_source_area=int(measured.area),
            unmeasured_fragment_area=int(state.area),
            measured_core_area_lower_bound=float(minimum_area),
            ordinary_fragment_link_cost=float(ordinary['cost']),
            ordinary_fragment_link_allowed=bool(ordinary['allowed']),
            gap_source_evidence='preceding_soma_sized_body')
        self.events.append(dict(direction=self.direction, identity=int(state.identity),
            source_assignment_frame=int(state.t), source_measurement_frame=int(measured.t),
            target_frame=int(target_t), target_local_id=int(observation['local_id']),
            fragment_area=int(state.area), measured_core_area_lower_bound=minimum_area,
            ordinary_allowed=bool(ordinary['allowed']), allowed=bool(result['allowed']),
            ordinary_cost=float(ordinary['cost']), cost=float(result['cost'])))
        return result


def observed_core_geometry(points):
    """Accept native-frame measured positions, without detector-name memory."""
    p = points[points.state.eq('observed') & points.strong.astype(bool)].copy()
    if 'physically_visible' in p:
        p = p[p.physically_visible.astype(bool)]
    if p.empty:
        return {}
    values = p[['frame', 'y', 'x', 'radius_px']].to_numpy(float)
    p = p[np.isfinite(values).all(axis=1) & (values[:, 3] > 0)]
    return {int(t): g[['y', 'x', 'radius_px']].to_numpy(float)
            for t, g in p.groupby('frame')}


@contextmanager
def measured_soma_gap_memory(points, events=None):
    """Use the same general source measurement in both tracking directions.

Point frames must already align with the source movie. This context is for a
single tracking process; it is not a thread-safe global configuration switch.
"""
    import tracking
    if events is None:
        events = []
    geometry = observed_core_geometry(points)
    original_direction = tracking.track_direction
    original_features = tracking.link_features
    signature = inspect.signature(original_direction)

    def direction_wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        params = dict(bound.arguments['params'])
        params['_measured_gap_memory'] = MeasuredSomaGapMemory(
            geometry, bound.arguments['direction'], events)
        bound.arguments['params'] = params
        return original_direction(*bound.args, **bound.kwargs)

    def feature_wrapper(state, observation, target_t, lag, direction, params):
        memory = params.get('_measured_gap_memory')
        if memory is None:
            return original_features(state, observation, target_t, lag, direction, params)
        return memory.features(original_features, state, observation, target_t,
                               lag, direction, params)

    tracking.track_direction = direction_wrapper
    tracking.link_features = feature_wrapper
    try:
        yield events
    finally:
        tracking.track_direction = original_direction
        tracking.link_features = original_features
