"""Complete original change inventories and equal-recording biological units."""
import numpy as np
import pandas as pd
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.intervention.windows import finite
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines._screening import _json_value


def unit_id(source,movie,sample):
    return content_id({'source_run':source,'sample':sample['sample']} if sample['confirmed'] else {'source_run':source,'unconfirmed_recording':movie})


def clock_key(prepared,movie,anchor):
    """Compare native frame windows only with an established common clock basis.

This describes original clock observations; it does not extend or fill them.
Clock-key rounding removes sub-microsecond floating-point identity noise only.
"""
    frame=prepared['traces'].loc[prepared['traces'].movie.eq(movie)]
    if 'frame_index' not in frame:return None
    frame=frame[['frame_index','hours']].dropna().drop_duplicates().sort_values('frame_index')
    if len(frame)<2 or frame.frame_index.duplicated().any():return None
    indices=frame.frame_index.to_numpy(float);hours=frame.hours.to_numpy(float);delta=np.diff(hours)
    if not np.isfinite(indices).all() or not np.isfinite(hours).all() or not (indices>=0).all() or not (indices%1==0).all() or not (np.diff(indices)==1).all() or not (delta>0).all():return None
    if not np.allclose(delta,delta[0],rtol=1e-7,atol=1e-9):return None
    return {'first_original_frame_index':int(indices[0]),'cadence_hours':round(float(delta[0]),9),
        'anchor_from_first_original_frame_hours':round(float(anchor-hours[0]),9),'clock_key_rounding_hours':1e-9}


def window_key(window,clock):
    if window['coordinate']=='frames':
        return {'coordinate':'frames','start':window['start'],'end':window['end'],'original_clock':clock}
    low=window['start'] if window['coordinate']=='relative_hours' else window['start_hours']-window['anchor_hours']
    high=window['end'] if window['coordinate']=='relative_hours' else window['end_hours']-window['anchor_hours']
    return {'coordinate':'relative_hours','start':round(float(low),9),'end':round(float(high),9)}


def aggregate(prepared,evidence,resolved,settings):
    source=resolved.inputs.source_run;units={};recordings=[];membership={}
    for movie,recording in resolved.recordings.as_dict().items():
        sample=recording['sample'];uid=unit_id(source,movie,sample);membership[movie]=uid
        cells=[cell.as_dict() for cell in resolved.inputs.cells if cell.movie==movie]
        row={'source_run':source,'movie':movie,'unit_id':uid,'sample':sample['sample'],'sample_confirmed':sample['confirmed'],
            'condition':recording['condition'],'anchor_kind':recording['anchor']['kind'],'cells':cells,'cell_count':len(cells)}
        recordings.append(row)
        if uid not in units:units[uid]={key:row[key] for key in ['source_run','unit_id','sample','sample_confirmed','condition','anchor_kind']};units[uid].update(movies=[],cells=[])
        units[uid]['movies'].append(movie);units[uid]['cells'].extend(cells)
    windows={row['window_id']:row for row in _json_value(prepared['windows'].to_dict('records'))}
    clocks={movie:clock_key(prepared,movie,recording['anchor']['hours']) for movie,recording in resolved.recordings.as_dict().items()}
    definitions={};changes=[]
    for row in _json_value(evidence['effects'].to_dict('records')):
        baseline,target=windows[row['baseline_window_id']],windows[row['target_window_id']]
        comparable=all(window['coordinate']!='frames' or clocks[row['movie']] is not None for window in [baseline,target])
        quantities=sorted({spec['quantity'] for spec in settings['comparisons'] if all(spec[key]==row[key] for key in ['measurement','baseline','target_window'])})
        for quantity in quantities:
            definition={'measurement':row['measurement'],'measurement_id':row['measurement_id'],'summary_operation':row['summary_operation'],
                'unit':'fraction' if quantity=='relative_change' else row['unit'],'quantity':quantity,'baseline':row['baseline'],'target_window':row['target_window'],
                'baseline_definition':window_key(baseline,clocks[row['movie']]),'target_definition':window_key(target,clocks[row['movie']]),
                'model':resolved.request.evidence.as_dict() if quantity=='estimate' else None,
                'relative_settings':resolved.request.relative_effects.get(row['measurement']) if quantity=='relative_change' else None,
                'window_comparability':'established_original_clock' if comparable else 'unknown_frame_clock'}
            if not comparable:definition['unestablished_recording']=row['movie']
            did=content_id(definition);definitions[did]={'definition_id':did,**definition}
            value=row.get(quantity);eligible=bool(row['eligible'] and finite(value) and comparable)
            changes.append({**{key:row[key] for key in ['source_run','movie','identity','measurement','measurement_id','effect_id','comparison_id','baseline_window_id','target_window_id']},
                'definition_id':did,'unit_id':membership[row['movie']],'value':float(value) if finite(value) else None,'eligible':eligible,
                'reason':'Available original change with declared window support' if eligible else 'Insufficient original window support' if not row['eligible'] else 'Unestablished original frame-clock comparability' if not comparable else 'Requested change quantity unavailable',
                'within_cell_outcome':row['outcome'],'within_cell_supported':row['response_supported'],'population':'All original cells; no within-cell significance filter'})
    reducer=np.mean if settings['aggregation']=='mean' else np.median;recording_summaries=[];unit_summaries=[]
    for did,definition in definitions.items():
        for recording in recordings:
            local=[row for row in changes if row['definition_id']==did and row['movie']==recording['movie']]
            if not local:continue
            available=[row for row in local if row['eligible']]
            recording_summaries.append({'definition_id':did,**{key:recording[key] for key in ['source_run','movie','unit_id','sample','sample_confirmed','condition','anchor_kind']},
                'value':float(reducer([row['value'] for row in available])) if available else None,
                'requested_cells':len(local),'available_cells':len(available),'all_effect_ids':[row['effect_id'] for row in local],
                'eligible_effect_ids':[row['effect_id'] for row in available],'aggregation':settings['aggregation'],'status':'available' if available else 'no_eligible_cell_changes'})
        # Every original unit remains in each declared definition, including units
        # without matching windows; absence cannot become a zero-valued sample.
        for uid,unit in units.items():
            local=[row for row in recording_summaries if row['definition_id']==did and row['unit_id']==uid];available=[row for row in local if finite(row['value'])]
            value=float(reducer([row['value'] for row in available])) if available else None
            unit_summaries.append({'definition_id':did,**unit,'value':value,'eligible':value is not None,'formal_eligible':value is not None and unit['sample_confirmed'],
                'requested_recordings':len(unit['movies']),'matching_recordings':len(local),'available_recordings':len(available),
                'requested_cells':sum(row['requested_cells'] for row in local),'available_cells':sum(row['available_cells'] for row in available),
                'all_effect_ids':[eid for row in local for eid in row['all_effect_ids']],
                'eligible_effect_ids':[eid for row in available for eid in row['eligible_effect_ids']],
                'aggregation':settings['aggregation'],'weighting':'Equal available recording summaries within each biological sample',
                'status':'available_confirmed_sample' if value is not None and unit['sample_confirmed'] else 'descriptive_unconfirmed_recording' if value is not None else 'no_eligible_matching_changes'})
    return {'unit_inventory':_table(list(units.values()),['source_run','unit_id','sample','sample_confirmed','condition','anchor_kind','movies','cells']),
        'recording_inventory':_table(recordings,['source_run','movie','unit_id','cells']),
        'cell_changes':_table(changes,['definition_id','unit_id','effect_id','value','eligible']),
        'recording_summaries':_table(recording_summaries,['definition_id','unit_id','movie','value','all_effect_ids','eligible_effect_ids']),
        'unit_summaries':_table(unit_summaries,['definition_id','unit_id','sample','condition','value','eligible','formal_eligible']),
        'definitions':_table(list(definitions.values()),['definition_id','measurement','baseline','target_window','quantity'])}
