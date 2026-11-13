"""Movement allowances in micrometres and minutes, with native exactness.

Only the declared gap/distance settings are translated. Observation counts,
appearance windows, contact radii and ownership thresholds are not converted.
"""
from copy import deepcopy
from fractions import Fraction
from detector_spatial_units import positive_length

SCHEMA = 'motion.tracking-physical-units.v1'
FIELDS = ('maximum_track_gap_frames', 'maximum_reconnection_gap_frames',
          'base_link_distance_px', 'additional_link_px_per_gap',
          'maximum_link_distance_px', 'motion_link_extension_px',
          'reconnection_base_distance_px', 'reconnection_distance_px_per_gap',
          'maximum_reconnection_distance_px')


def number(value):
    if isinstance(value, bool):
        raise ValueError('A finite numeric setting is required')
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError) as error:
        raise ValueError('A finite numeric setting is required') from error


def calibrate(parameters, pixel_size_um, frame_interval_min, source):
    px=positive_length(pixel_size_um);dt=positive_length(frame_interval_min)
    if not isinstance(source,str) or not source.strip():
        raise ValueError('A calibration source is required')
    ref={key:deepcopy(parameters[key]) for key in FIELDS}
    vals={key:number(value) for key,value in ref.items()}
    if any(value < (-1 if key == 'maximum_reconnection_gap_frames' else 0)
           for key, value in vals.items()):
        raise ValueError('Reference settings must be nonnegative, except -1 disables reconnection')
    if any(vals[key].denominator!=1 for key in FIELDS[:2]):
        raise ValueError('Reference frame gaps must be integers')
    physical=dict(
        maximum_track_elapsed_min=dt*vals['maximum_track_gap_frames'],
        maximum_reconnection_elapsed_min=dt*(vals['maximum_reconnection_gap_frames']+1),
        link_distance_intercept_um=px*(vals['base_link_distance_px']-vals['additional_link_px_per_gap']),
        link_distance_growth_um_per_min=px*vals['additional_link_px_per_gap']/dt,
        maximum_link_distance_um=px*vals['maximum_link_distance_px'],
        motion_link_extension_um=px*vals['motion_link_extension_px'],
        reconnection_distance_intercept_um=px*vals['reconnection_base_distance_px'],
        reconnection_distance_growth_um_per_min=px*vals['reconnection_distance_px_per_gap']/dt,
        maximum_reconnection_distance_um=px*vals['maximum_reconnection_distance_px'])
    return dict(schema=SCHEMA,source=source,reference_pixel_size_um=str(px),
                reference_interval_min=str(dt),reference_parameters=ref,
                physical={key:str(value) for key,value in physical.items()})


def resolve(parameters, specification, pixel_size_um, frame_interval_min):
    if specification['schema']!=SCHEMA:
        raise ValueError('Unsupported tracking physical-unit schema')
    px=positive_length(pixel_size_um);dt=positive_length(frame_interval_min)
    physical={key:number(value) for key,value in specification['physical'].items()}
    if any(value<0 for key,value in physical.items() if key!='link_distance_intercept_um'):
        raise ValueError('Durations, growth rates and caps must be nonnegative')
    resolved=dict(
        maximum_track_gap_frames=int(physical['maximum_track_elapsed_min']//dt),
        maximum_reconnection_gap_frames=int(physical['maximum_reconnection_elapsed_min']//dt)-1,
        base_link_distance_px=(physical['link_distance_intercept_um']+dt*physical['link_distance_growth_um_per_min'])/px,
        additional_link_px_per_gap=dt*physical['link_distance_growth_um_per_min']/px,
        maximum_link_distance_px=physical['maximum_link_distance_um']/px,
        motion_link_extension_px=physical['motion_link_extension_um']/px,
        reconnection_base_distance_px=physical['reconnection_distance_intercept_um']/px,
        reconnection_distance_px_per_gap=dt*physical['reconnection_distance_growth_um_per_min']/px,
        maximum_reconnection_distance_px=physical['maximum_reconnection_distance_um']/px)
    if resolved['base_link_distance_px']<0:
        raise ValueError('The declared intercept gives a negative link allowance at this sampling interval')
    result=deepcopy(parameters)
    for key,value in resolved.items():
        reference=specification['reference_parameters'][key]
        result[key]=(deepcopy(reference) if value==number(reference) else
                     int(value) if key in FIELDS[:2] else float(value))
    return result


def resolve_config(config):
    owner=config.values.get('accepted_postprocessing',{}).get('field_wide_owner_consensus',{})
    spec=owner.get('tracking_physical_units')
    if spec is None:
        return config
    if owner.get('time_units') is not None:
        raise ValueError('Choose tracking_physical_units or time_units; both would define the same gap settings')
    values=deepcopy(config.values)
    target=values['accepted_postprocessing']['field_wide_owner_consensus']
    target['latent_tracking']=resolve(target['latent_tracking'],spec,values['pixel_size_um'],values['frame_interval_min'])
    return type(config)(values,config.path)
