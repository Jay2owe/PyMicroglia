"""Optional physical lengths for the existing encounter geometry floors."""
from copy import deepcopy
from detector_spatial_units import positive_length,nonnegative_length
SCHEMA='motion.contact-spatial-units.v1'
DEFAULTS={'minimum_encounter_radius_px':10.0,'minimum_separable_distance_px':4.0}

def bounds(parameters):
    return tuple(float(nonnegative_length(parameters.get(key,value))) for key,value in DEFAULTS.items())

def calibrate(parameters,pixel_size_um,source):
    size=positive_length(pixel_size_um)
    if not isinstance(source,str) or not source.strip():raise ValueError('A calibration source is required')
    values={key:deepcopy(parameters.get(key,value)) for key,value in DEFAULTS.items()}
    return dict(schema=SCHEMA,source=source,reference_pixel_size_um=str(size),reference_parameters=values,
                originally_present=[key for key in DEFAULTS if key in parameters],lengths_um={key:str(nonnegative_length(value)*size) for key,value in values.items()})

def resolve(parameters,specification,pixel_size_um):
    if specification['schema']!=SCHEMA:raise ValueError('Unsupported contact-unit schema')
    size=positive_length(pixel_size_um);reference=positive_length(specification['reference_pixel_size_um']);result=deepcopy(parameters)
    for key in DEFAULTS:
        length=nonnegative_length(specification['lengths_um'][key]);value=specification['reference_parameters'][key]
        if size==reference and length==nonnegative_length(value)*reference:
            if key in specification['originally_present']:result[key]=deepcopy(value)
            else:result.pop(key,None)
        else:result[key]=float(length/size)
    return result

def resolve_config(config):
    owner=config.values.get('accepted_postprocessing',{}).get('field_wide_owner_consensus',{})
    specification=owner.get('contact_spatial_units')
    if specification is None:return config
    values=deepcopy(config.values);target=values['accepted_postprocessing']['field_wide_owner_consensus']
    target['latent_tracking']=resolve(target['latent_tracking'],specification,values['pixel_size_um'])
    return type(config)(values,config.path)
