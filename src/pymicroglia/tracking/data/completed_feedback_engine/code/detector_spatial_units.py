"""Explicit physical lengths for the four configurable detector scales.

This changes the ruler only. It does not make the detector invariant to image
resampling: pixel-grid extrema, core shapes and internal radius bounds remain.
No acquisition scale is assumed, and no recording-specific value lives here.
"""
from copy import deepcopy
from fractions import Fraction
import math

FIELDS = ('core_sigma_px', 'background_sigma_px',
          'minimum_peak_distance_px', 'core_window_radius_px')
SCHEMA = 'motion.detector-spatial-units.v1'


def nonnegative_length(value):
    if isinstance(value, bool):
        raise ValueError('A finite nonnegative length is required')
    try:
        result = Fraction(str(value))
        valid = result >= 0 and math.isfinite(float(result))
    except (ValueError, ZeroDivisionError, TypeError, OverflowError) as error:
        raise ValueError('A finite nonnegative length is required') from error
    if not valid:
        raise ValueError('A finite nonnegative length is required')
    return result


def positive_length(value):
    result = nonnegative_length(value)
    if result == 0:
        raise ValueError('Pixel calibration must be positive and finite')
    return result


def resolve_config(config):
    """Optional detector units; other tracking scales remain explicitly pixels."""
    owner = config.values.get('accepted_postprocessing', {}).get('field_wide_owner_consensus', {})
    spec = owner.get('detector_spatial_units')
    if spec is None:
        return config
    values = deepcopy(config.values)
    target = values['accepted_postprocessing']['field_wide_owner_consensus']
    target['raw_detection'] = resolve(target['raw_detection'], spec, values['pixel_size_um'])
    return type(config)(values, config.path)


def calibrate(parameters, pixel_size_um, source):
    """Freeze explicit pixel settings as lengths, with their calibration source."""
    pixel_size = positive_length(pixel_size_um)
    if not isinstance(source, str) or not source.strip():
        raise ValueError('A calibration source is required')
    reference = {key: deepcopy(parameters[key]) for key in FIELDS}
    lengths = {key: str(nonnegative_length(reference[key]) * pixel_size)
               for key in FIELDS}
    return dict(schema=SCHEMA, reference_pixel_size_um=str(pixel_size),
                source=source, reference_parameters=reference, lengths_um=lengths)


def resolve(parameters, specification, pixel_size_um):
    """Resolve lengths on a square-pixel grid; preserve native numeric types.

    The integer measurement-window radius uses the existing truncation rule.
    A zero or subpixel radius retains the original one-pixel measurement window.
    """
    if specification['schema'] != SCHEMA:
        raise ValueError('Unsupported spatial-unit schema')
    pixel_size = positive_length(pixel_size_um)
    reference_size = positive_length(specification['reference_pixel_size_um'])
    result = deepcopy(parameters)
    for key in FIELDS:
        length = nonnegative_length(specification['lengths_um'][key])
        reference = specification['reference_parameters'][key]
        if pixel_size == reference_size and length == nonnegative_length(reference) * reference_size:
            result[key] = deepcopy(reference)
        elif key == 'core_window_radius_px':
            result[key] = int(length // pixel_size)
        else:
            result[key] = float(length / pixel_size)
    return result
