"""Optional elapsed-time representation of two existing ownership bounds.

Support counts remain counts. A source-owner run spans first to last sample;
an adjoining gap counts missing frames between observed endpoints.
"""
from copy import deepcopy
from detector_spatial_units import positive_length
from tracking_physical_units import number

SCHEMA='motion.ownership-time-units.v1'
DEFAULTS={'maximum_source_run_frames':12,'maximum_adjoining_gap_frames':6}


def calibrate(parameters,frame_interval_min,source):
    interval=positive_length(frame_interval_min)
    if not isinstance(source,str) or not source.strip():raise ValueError('A calibration source is required')
    reference={key:deepcopy(parameters.get(key,value)) for key,value in DEFAULTS.items()}
    values={key:number(value) for key,value in reference.items()}
    if any(value.denominator!=1 for value in values.values()):raise ValueError('Ownership frame limits must be integers')
    if values['maximum_source_run_frames']<1 or values['maximum_adjoining_gap_frames']<-1:
        raise ValueError('Source runs require at least one frame; adjoining gap must be at least -1')
    return dict(schema=SCHEMA,source=source,reference_interval_min=str(interval),reference_parameters=reference,
        originally_present=[key for key in DEFAULTS if key in parameters],
        durations_min=dict(maximum_source_span=str((values['maximum_source_run_frames']-1)*interval),
                           maximum_adjoining_elapsed=str((values['maximum_adjoining_gap_frames']+1)*interval)))


def resolve(parameters,specification,frame_interval_min):
    if specification['schema']!=SCHEMA:raise ValueError('Unsupported ownership time schema')
    interval=positive_length(frame_interval_min);reference_interval=positive_length(specification['reference_interval_min'])
    source=number(specification['durations_min']['maximum_source_span'])
    adjoining=number(specification['durations_min']['maximum_adjoining_elapsed'])
    if source<0 or adjoining<0:raise ValueError('Elapsed durations must be nonnegative')
    resolved={'maximum_source_run_frames':int(source//interval)+1,'maximum_adjoining_gap_frames':int(adjoining//interval)-1}
    reference=specification['reference_parameters'];result=deepcopy(parameters)
    expected=calibrate(reference,reference_interval,specification['source'])['durations_min']
    native=(interval==reference_interval and source==number(expected['maximum_source_span']) and adjoining==number(expected['maximum_adjoining_elapsed']))
    for key,value in resolved.items():
        if native:
            if key in specification['originally_present']:result[key]=deepcopy(reference[key])
            else:result.pop(key,None)
        else:result[key]=value
    return result


def resolve_config(config):
    owner=config.values.get('accepted_postprocessing',{}).get('field_wide_owner_consensus',{})
    spec=owner.get('ownership_time_units')
    if spec is None:return config
    values=deepcopy(config.values);target=values['accepted_postprocessing']['field_wide_owner_consensus']
    target['owner_consensus']=resolve(target['owner_consensus'],spec,values['frame_interval_min'])
    return type(config)(values,config.path)
