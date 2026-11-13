"""Do not let an unmeasured fragment redefine a soma's gap-search source.

The size comparison uses that body's own observed core radii. There is no
recording, biological identity, detector identity, frame or region selector.
Installation is scoped to one tracking run; the established tracker is restored
afterwards. Whole-body observations and supported rapid-motion links retain their original features. Consecutive sub-soma fragments use the last measured whole source.
"""
from contextlib import contextmanager
from dataclasses import replace
import inspect

import numpy as np


def trace_measured_core_path(geometry, initial, start, end):
    """Follow unique closest measured cores in core-radius and elapsed-time units.

    Missing observations keep the last position. Ties stop the path rather
    than selecting by table order. Detector and identity names are unused.
    """
    current = np.asarray(initial, float)
    last = int(start)
    history = [(last, current)]
    if start == end:
        return history
    direction = 1 if end > start else -1
    for t in range(start + direction, end + direction, direction):
        other = geometry.get(t, np.empty((0, 3)))
        if not len(other):
            continue
        cost = (((other[:, :2] - current[:2]) ** 2).sum(axis=1)
                / (other[:, 2] + current[2]) ** 2 / abs(t - last))
        minimum = cost.min()
        if minimum > 1 and not np.isclose(minimum, 1., rtol=16*np.finfo(float).eps, atol=0.):
            continue
        contenders = np.flatnonzero(np.isclose(cost, minimum,
            rtol=16*np.finfo(float).eps, atol=0.))
        if len(contenders) != 1:
            break
        current = other[contenders[0]]
        last = t
        history.append((last, current))
    return history


class MeasuredSomaGapMemory:
    def __init__(self, points_by_frame, direction, events):
        self.points = points_by_frame
        self.direction = int(direction)
        self.events = events
        self.sources = {}
        self.cache = {}
        self.observation_areas = {}
        self.measured_path_cache = {}

    def contained_cores(self, mask, frame):
        points = self.points.get(int(frame), np.empty((0, 3)))
        y = np.floor(points[:, 0] + .5).astype(int)
        x = np.floor(points[:, 1] + .5).astype(int)
        valid = (y >= 0) & (y < mask.shape[0]) & (x >= 0) & (x < mask.shape[1])
        points, y, x = points[valid], y[valid], x[valid]
        return points[mask[y, x]]

    def core_scaled_gap_supported(self, source, observation, target_t):
        old = self.contained_cores(source.mask, source.t)
        new = self.contained_cores(observation['mask'], target_t)
        if not len(old) or not len(new):
            return False
        # The original tracker expands a fixed pixel search radius with the
        # square root of elapsed frames. A restored source needs support at
        # the scale of its measured soma instead: its source/target core disks
        # must overlap after the same elapsed-time uncertainty expansion.
        # There is no fitted pixel distance or named-cell exception.
        distance_squared = ((old[:, None, :2] - new[None, :, :2]) ** 2).sum(axis=2)
        uncertainty_squared = ((old[:, None, 2] + new[None, :, 2]) ** 2
                               * abs(int(target_t) - int(source.t)))
        if not np.any(distance_squared <= uncertainty_squared):
            return False
        # A stale source cannot override the measured history of the target.
        # Trace from the proposed target back to the retained source time.
        # Source-to-target tracing alone can drift onto an unrelated body.
        for core in new:
            key = (int(target_t), int(source.t), tuple(core))
            if key not in self.measured_path_cache:
                self.measured_path_cache[key] = trace_measured_core_path(
                    self.points, core, int(target_t), int(source.t))[-1]
            end_t, end_core = self.measured_path_cache[key]
            if end_t == source.t and np.any(np.all(old == end_core, axis=1)):
                return True
        return False

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
        undersized_measured_body = (area is not None and state.area < area
            and not np.isclose(float(state.area), area,
                               rtol=16 * np.finfo(float).eps, atol=0.0))
        if area is not None and not undersized_measured_body:
            previous = (state, area)
            self.sources[state.identity] = previous
            answer = None
        elif undersized_measured_body:
            # A measured centre inside a partial mask does not establish that
            # the mask contains its measured soma. Keep the earlier whole body;
            # without one, do not create a source-memory override.
            answer = previous
        elif (previous is not None and state.area < previous[1]
              and not np.isclose(float(state.area), previous[1],
                                 rtol=16 * np.finfo(float).eps, atol=0.0)):
            # A core area reconstructed from its radius can exceed the exact
            # equal pixel count by floating-point roundoff. That is not shrinkage.
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
        if (reference is None
                or ordinary.get('independent_source_soma_retained', False)
                or ordinary.get('fast_motion_handoff', False)
                or ordinary.get('reversal_aware_handoff', False)):
            return ordinary
        # A fragment can corrupt the source even without a missed frame.
        # Consecutive detections do not make its displaced centroid a soma
        # measurement. The existing measured-core correspondence still has
        # to reconnect the proposed body to the retained physical source.
        # Established rapid-motion evidence is preserved above.
        key = (int(target_t), int(observation['local_id']))
        if key not in self.observation_areas:
            self.observation_areas[key] = self.core_area(observation['mask'], target_t)
        if self.observation_areas[key] is None:
            return ordinary
        measured, minimum_area = reference
        core_supported = self.core_scaled_gap_supported(measured, observation, target_t)
        # The identity and its protected reservation membership are unchanged.
        # A snapshot supplies the position, velocity, morphology and actual time
        # of the preceding body that could contain its observed soma.
        effective = replace(measured, identity=state.identity)
        result = dict(original(effective, observation, target_t, lag, direction, params))
        result['allowed'] = bool(result['allowed'] and core_supported)
        result['measured_core_gap_supported'] = core_supported
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
            ordinary_cost=float(ordinary['cost']), cost=float(result['cost']),
            measured_core_gap_supported=core_supported))
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
