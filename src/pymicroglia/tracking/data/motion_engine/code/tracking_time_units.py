"""Explicit physical-time translation for track gaps. Spatial units remain pixels; observation counts remain counts."""
from copy import deepcopy
from fractions import Fraction
import math

def interval(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError('A positive, finite acquisition interval in minutes is required')
    return Fraction(str(value))

def calibrate(params, reference_interval_min):
    dt = interval(reference_interval_min)
    original = deepcopy(params)
    required = ('maximum_track_gap_frames', 'maximum_reconnection_gap_frames', 'base_link_distance_px', 'additional_link_px_per_gap', 'reconnection_base_distance_px', 'reconnection_distance_px_per_gap')
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError('Explicit reference settings required: ' + ', '.join(missing))
    return dict(schema='motion.tracking-time-units.v1', reference_interval_min=str(dt), reference_parameters=original, maximum_track_elapsed_min=str(dt * int(params['maximum_track_gap_frames'])), maximum_reconnection_elapsed_min=str(dt * (int(params['maximum_reconnection_gap_frames']) + 1)), link_distance_intercept_px=str(Fraction(str(params['base_link_distance_px'])) - Fraction(str(params['additional_link_px_per_gap']))), link_distance_growth_px_per_min=str(Fraction(str(params['additional_link_px_per_gap'])) / dt), reconnection_distance_intercept_px=str(Fraction(str(params['reconnection_base_distance_px']))), reconnection_distance_growth_px_per_min=str(Fraction(str(params['reconnection_distance_px_per_gap'])) / dt))

def resolve(spec, frame_interval_min):
    if spec['schema'] != 'motion.tracking-time-units.v1':
        raise ValueError('Unsupported time-unit schema')
    dt = interval(frame_interval_min)
    params = deepcopy(spec['reference_parameters'])
    if dt == Fraction(spec['reference_interval_min']):
        calibrated = calibrate(params, Fraction(spec['reference_interval_min']))
        physical_fields = ('maximum_track_elapsed_min', 'maximum_reconnection_elapsed_min',
                           'link_distance_intercept_px', 'link_distance_growth_px_per_min',
                           'reconnection_distance_intercept_px', 'reconnection_distance_growth_px_per_min')
        if all(Fraction(str(spec[key])) == Fraction(str(calibrated[key])) for key in physical_fields):
            return params
    params['maximum_track_gap_frames'] = int(Fraction(spec['maximum_track_elapsed_min']) // dt)
    params['maximum_reconnection_gap_frames'] = int(Fraction(spec['maximum_reconnection_elapsed_min']) // dt) - 1
    growth = Fraction(spec['link_distance_growth_px_per_min']) * dt
    params['additional_link_px_per_gap'] = float(growth)
    params['base_link_distance_px'] = float(Fraction(spec['link_distance_intercept_px']) + growth)
    params['reconnection_base_distance_px'] = float(Fraction(spec['reconnection_distance_intercept_px']))
    params['reconnection_distance_px_per_gap'] = float(Fraction(spec['reconnection_distance_growth_px_per_min']) * dt)
    return params
