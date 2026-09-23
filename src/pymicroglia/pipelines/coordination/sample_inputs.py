"""Recording and biological-unit bookkeeping over complete saved pair effects."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines._screening import _json_value


META=['source_run','movie','sample','sample_confirmed','condition']
DEFINITION=['question','evidence_level','reference','target','reference_measurement_id','target_measurement_id',
    'representation','adjustment','statistic','distance_effect','summary','distance_unit','effect_unit','estimate_unit',
    'evidence_kind','model_id','reference_state_id','target_state_id','coordination_question','coordination_measure','window_hours','step_hours','is_hypothesis']


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value)


def clock_profile(frame,max_gap,hour_column='hours'):
    """Recorded cadence is a compatibility key; missing stretches stay separate."""
    if hour_column=='elapsed_hours':
        values=frame.loc[frame.valid_transition_opportunity,hour_column].to_numpy(float)
        return {'intervals_hours':sorted(set(np.round(values,9).tolist())),'unsupported_intervals':int((~frame.valid_transition_opportunity).sum())}
    if 'frame_index' in frame:frame=frame.sort_values('frame_index')
    if 'within_range' in frame:frame=frame.loc[frame.within_range]
    hours=frame[hour_column].to_numpy(float);delta=np.diff(hours)
    good=np.isfinite(delta)&(delta>0)&(delta<=max_gap)
    if 'frame_index' in frame:good&=np.diff(frame.frame_index.to_numpy(float))==1
    return {'intervals_hours':sorted(set(np.round(delta[good],9).tolist())),'unsupported_intervals':int((~good).sum())}


def member_records(resolved,prepared,evidence,state_source=None):
    """Select saved scalar meanings, never significance-selected observations."""
    request=resolved.request;rows=[];definitions={};profiles={}
    for endpoint,frame in prepared['traces'].groupby('endpoint_id',sort=False):profiles[endpoint]=clock_profile(frame,request.support['max_gap_hours'])
    state_profiles={}
    if state_source is not None and state_source['status']=='accepted':
        for kind,frame,column in [('state_cooccupancy',state_source['assignments'],'hours'),('switch_process',state_source['time']['steps'],'elapsed_hours')]:
            for key,group in frame.groupby(['source_run','movie','identity'],sort=False):
                state_profiles[(*key,kind)]=clock_profile(group,request.support['max_gap_hours'],column)
    for raw in evidence['effects'].to_dict('records'):
        row=_json_value(raw);question=row['question'];kind='saved_association';value=None
        if question=='characteristics':
            kind='spatial_pattern' if row['evidence_level']=='recording' else 'pair_characteristic_difference'
            value=row.get('descriptive_effect') if row['evidence_level']=='recording' else row.get('effect')
        elif question in {'simultaneous','proximity','states'}:value=row.get('descriptive_effect',row.get('effect'))
        elif question=='delay':
            # Full common-window search values are not mixed with variable
            # original per-lag overlap peaks. Neither selection uses a p-value.
            if finite(row.get('search_statistic')):
                kind='complete_common_window_maximum_absolute_lag_association';value=row['search_statistic']
            else:
                kind='descriptive_overlap_maximum_absolute_lag_association'
                peaks=row.get('descriptive_peak_effects') or []
                value=max((abs(item) for item in peaks if finite(item)),default=None)
        elif question=='rhythm':kind='native_rhythm_timing';value=None
        timing=[]
        for role in ['reference','target']:
            if question=='states':profile=state_profiles.get((row['source_run'],row['movie'],row[role+'_identity'],row['evidence_kind']),{'intervals_hours':[],'unsupported_intervals':None})
            else:profile=profiles.get(row.get(role+'_endpoint_id'),{'intervals_hours':[],'unsupported_intervals':None})
            timing.append(profile)
        definition={key:row.get(key) for key in DEFINITION}
        definition.update(value_kind=kind,reference_cadence_hours=timing[0]['intervals_hours'],target_cadence_hours=timing[1]['intervals_hours'],
            geometry_definition=request.geometry['definition'],neighbourhood=request.geometry['neighbourhood'],
            declared_time_range_hours=request.time_range_hours,increment=request.increment,
            delay_grid={key:request.questions['delay'].get(key) for key in ['range_hours','resolution_hours']} if question=='delay' else None)
        if question=='states' and state_source is not None:
            definition['state_time_settings']=state_source.get('time_provenance',{}).get('settings')
        question_id=content_id(definition);definitions[question_id]={'question_id':question_id,**definition}
        geometry=row.get('geometry_population_eligible')
        eligible=finite(value) and geometry is not False and row['status'] not in {'insufficient','outside_geometry_population','ineligible','unresolvable','unaccepted_model'}
        reason='Finite saved effect in the complete declared population' if eligible else 'Rhythm timing requires its native comparable-role summary' if question=='rhythm' else row['reason']
        baseline=row.get('is_hypothesis') is False
        unused=question!='states' or row.get('model_unused_by_both_endpoints') is True
        # Native no-test delay profiles can have different observation support
        # at different lags; they remain a separately labelled description.
        formal=eligible and not baseline and unused and kind!='descriptive_overlap_maximum_absolute_lag_association'
        rows.append({**{key:row.get(key) for key in META},'question_id':question_id,'effect_id':row['effect_id'],'pair_id':row.get('pair_id'),
            'state_pair_id':row.get('state_pair_id'),'source_result_id':row['source_result_id'],'source_scientific_id':row['source_scientific_id'],
            'question':question,'value_kind':kind,'value':float(value) if eligible else None,'saved_value':value,
            'eligible':eligible,'formal_eligible':formal,'status':'available' if eligible else 'unavailable','reason':reason,
            'inference_exclusion':None if formal else 'Original-level display baseline' if baseline else 'Sample contributed to defining the accepted state model' if not unused else 'Lag maxima use different observation populations across the requested grid' if kind=='descriptive_overlap_maximum_absolute_lag_association' else reason,
            'reference_identity':row.get('reference_identity'),'target_identity':row.get('target_identity'),
            'original_status':row['status'],'original_decision':row['decision'],'original_pair_supported':row['supported'],
            'reference_clock':timing[0],'target_clock':timing[1],
            'paired_observations':row.get('paired_observations'),'overlap_span_hours':row.get('overlap_span_hours'),
            'joint_support_hours':row.get('joint_support_hours'),'geometry_population_eligible':geometry,
            'native_timing_evidence':row.get('native_timing_evidence')})
    return _table(rows,META+['question_id','effect_id','pair_id','state_pair_id','value','eligible','formal_eligible','status','reason']),_table(list(definitions.values()),['question_id',*DEFINITION,'value_kind'])


def aggregate(resolved,prepared,members,definitions,method):
    """Equal recording contributions within a biological unit, after pair reduction."""
    if method not in {'mean','median'}:raise ValueError('Sample aggregation must be mean or median')
    reduce=lambda values:float(np.mean(values) if method=='mean' else np.median(values)) if len(values) else None
    cells=prepared['cells'];inventory=[];recordings=[];summaries=[]
    source=resolved.inputs.source_run
    for assignment in resolved.inputs.samples:
        cell_rows=cells.loc[cells.movie.eq(assignment.movie)]
        unit_id=content_id({'source_run':source,'level':'biological_sample' if assignment.confirmed else 'recording',
            'sample':assignment.sample if assignment.confirmed else assignment.movie})
        inventory.append({'source_run':source,'movie':assignment.movie,'sample':assignment.sample,'sample_confirmed':assignment.confirmed,
            'condition':resolved.request.conditions.get(assignment.movie),'unit_id':unit_id,'cells':len(cell_rows),
            'cell_identities':cell_rows.identity.tolist(),'requested_pairs':int(prepared['inventory'].movie.eq(assignment.movie).sum()),
            'unit_level':'biological_sample' if assignment.confirmed else 'unconfirmed_recording'})
    records=_table(inventory,META+['unit_id','cells','cell_identities','requested_pairs','unit_level'])
    if records.movie.duplicated().any():raise ValueError('Repeated recording in sample inventory')
    for definition in definitions.to_dict('records'):
        qid=definition['question_id']
        for record in inventory:
            group=members.loc[members.question_id.eq(qid)&members.movie.eq(record['movie'])]
            usable=group.loc[group.eligible]
            permitted=bool(len(usable) and usable.formal_eligible.all())
            identities={int(value) for key in ['reference_identity','target_identity'] for value in usable.get(key,pd.Series(dtype=float)).dropna()}
            recordings.append({**record,'question_id':qid,'aggregation':method,'value':reduce(usable.value.tolist()),
                'formal_eligible':permitted,'status':'available' if len(usable) else 'unavailable',
                'reason':'All finite compatible effects; pair significance was not used' if len(usable) else 'No compatible finite original effects for this recorded question',
                'effects_requested':len(group),'effects_eligible':len(usable),'contributing_cells':len(identities),'contributing_cell_identities':sorted(identities),
                'member_effect_ids':group.effect_id.tolist(),'eligible_effect_ids':usable.effect_id.tolist(),
                'excluded_effects':group.loc[~group.eligible,['effect_id','reason']].to_dict('records'),
                'inference_exclusions':sorted(set(usable.loc[~usable.formal_eligible,'inference_exclusion'].dropna()))})
    recording_table=_table(recordings,META+['unit_id','question_id','aggregation','value','formal_eligible','status','reason','effects_requested','effects_eligible'])
    units=[]
    for unit_id,population in records.groupby('unit_id',sort=True):
        first=population.iloc[0];conditions=set(population.condition.dropna())
        if len(conditions)>1 or conditions and population.condition.isna().any():raise ValueError('A biological sample has inconsistent condition assignments')
        unit={'source_run':source,'unit_id':unit_id,'sample':first['sample'],'sample_confirmed':bool(first.sample_confirmed),
            'condition':next(iter(conditions)) if conditions else None,'unit_level':first.unit_level,'movies':population.movie.tolist(),
            'recordings':len(population),'cells':int(population.cells.sum()),'requested_pairs':int(population.requested_pairs.sum())}
        units.append(unit)
        for definition in definitions.to_dict('records'):
            group=recording_table.loc[recording_table.unit_id.eq(unit_id)&recording_table.question_id.eq(definition['question_id'])]
            usable=group.loc[group.status.eq('available')]
            permitted=bool(unit['sample_confirmed'] and len(usable) and usable.formal_eligible.all())
            summaries.append({**unit,'question_id':definition['question_id'],'aggregation':method,'value':reduce(usable.value.tolist()),
                'formal_eligible':permitted,'status':'available' if len(usable) else 'unavailable',
                'reason':'One aggregate for this biological sample across its compatible recordings' if unit['sample_confirmed'] and len(usable) else 'Unconfirmed recording; descriptive only' if len(usable) else 'No compatible finite recording effects',
                'recordings_eligible':len(usable),'effects_requested':int(group.effects_requested.sum()),'effects_eligible':int(group.effects_eligible.sum()),
                'contributing_movies':usable.movie.tolist(),'member_effect_ids':sum(group.member_effect_ids.tolist(),[]),
                'eligible_effect_ids':sum(usable.eligible_effect_ids.tolist(),[]),'excluded_movies':group.loc[group.status.ne('available'),['movie','reason']].to_dict('records'),
                'inference_exclusions':(['Unconfirmed biological sample'] if not unit['sample_confirmed'] else [])+sorted({value for values in usable.inference_exclusions for value in values})})
    return {'recording_inventory':records,'unit_inventory':_table(units,['source_run','unit_id','sample','sample_confirmed','condition','unit_level','movies','recordings','cells','requested_pairs']),
        'recording_summaries':recording_table,'unit_summaries':_table(summaries,['source_run','unit_id','question_id','sample','sample_confirmed','condition','value','formal_eligible','status','reason'])}
