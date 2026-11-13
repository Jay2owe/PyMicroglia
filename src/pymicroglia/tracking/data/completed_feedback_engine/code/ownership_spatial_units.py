"""Optional physical calibration for the ownership operation's existing spatial bounds.

This changes the ruler, not the bound-selection policy. Default ownership
behaviour retains the established 2..6 pixel sampling interval and 2 pixel seed floor. Bounds are not inferred
from individual recordings, identities or regions.
"""
from copy import deepcopy
from detector_spatial_units import positive_length,nonnegative_length

SCHEMA='motion.ownership-spatial-units.v1'
DEFAULTS={'minimum_owner_sample_radius_px':2.0,'maximum_owner_sample_radius_px':6.0,'minimum_component_seed_radius_px':2.0}


def calibrate(parameters,pixel_size_um,source):
    size=positive_length(pixel_size_um)
    if not isinstance(source,str) or not source.strip():
        raise ValueError('A calibration source is required')
    values={key:deepcopy(parameters.get(key,value)) for key,value in DEFAULTS.items()}
    if nonnegative_length(values['minimum_owner_sample_radius_px'])>nonnegative_length(values['maximum_owner_sample_radius_px']):
        raise ValueError('Ownership-radius bounds must be ordered')
    return dict(schema=SCHEMA,source=source,reference_pixel_size_um=str(size),
        reference_parameters=values,originally_present=[key for key in DEFAULTS if key in parameters],
        lengths_um={key:str(nonnegative_length(value)*size) for key,value in values.items()})


def resolve(parameters,specification,pixel_size_um):
    if specification['schema']!=SCHEMA:raise ValueError('Unsupported ownership-radius schema')
    size=positive_length(pixel_size_um);reference=positive_length(specification['reference_pixel_size_um'])
    result=deepcopy(parameters)
    lengths={key:nonnegative_length(specification['lengths_um'][key]) for key in DEFAULTS}
    if lengths['minimum_owner_sample_radius_px']>lengths['maximum_owner_sample_radius_px']:
        raise ValueError('Ownership-radius bounds must be ordered')
    for key,length in lengths.items():
        value=specification['reference_parameters'][key]
        if size==reference and length==nonnegative_length(value)*reference:
            if key in specification['originally_present']:result[key]=deepcopy(value)
            else:result.pop(key,None)
        else:result[key]=float(length/size)
    return result


def resolve_config(config):
    owner=config.values.get('accepted_postprocessing',{}).get('field_wide_owner_consensus',{})
    specification=owner.get('ownership_spatial_units')
    if specification is None:return config
    values=deepcopy(config.values);target=values['accepted_postprocessing']['field_wide_owner_consensus']
    target['owner_consensus']=resolve(target['owner_consensus'],specification,values['pixel_size_um'])
    return type(config)(values,config.path)


def bounds(parameters=None):
    values={} if parameters is None else parameters
    result=tuple(float(nonnegative_length(values.get(key,default))) for key,default in DEFAULTS.items())
    if result[0]>result[1]:raise ValueError("Ownership-radius bounds must be ordered")
    return result
