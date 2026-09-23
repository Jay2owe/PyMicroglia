"""Original observations and complete event/window inventories before inference."""
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
from pathlib import Path
import json
import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import observations, _table
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, read_verified_tables, write_table

KEYS=['source_run','movie','identity','measurement','measurement_id']
TABLES=['cells','measurements','traces','recordings','windows','window_members','intervals','comparisons']


def identity(context):
    read_verified_tables(context.table_paths,context.request.inputs.table_hashes)
    request=context.request
    return content_id({'inputs':request.inputs.as_dict(),'measurements':[m.as_dict() for m in request.measurements],
        'recordings':request.recordings.as_dict(),'support':request.request.support.as_dict(),
        'libraries':{'numpy':np.__version__,'pandas':pd.__version__},
        'code':{name:file_hash(source_file(name)) for name in ['intervention_windows.py','intervention_options.py','coordination_inputs.py','relationship_inputs.py','contracts.py','screening.py']}})


def finite(value):
    return isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value)


def clock(resolved,tables,movie):
    """Use the original recording frame table when supplied; never extrapolate."""
    if 'frame_summary' in resolved.inputs.table_hashes:
        sources=[tables['frame_summary']];kind='original_recording_frame_table'
    else:sources=[tables[m.table] for m in resolved.measurements if 'window' not in m.grain];kind='observed_measurement_extent'
    parts=[source.loc[source.stem.eq(movie),[key for key in ['frame_index','hours'] if key in source]] for source in sources if 'hours' in source]
    frame=pd.concat(parts,ignore_index=True).drop_duplicates() if parts else pd.DataFrame(columns=['hours'])
    valid=True
    if 'frame_index' in frame:
        valid=bool(frame.frame_index.map(lambda value:finite(value) and value>=0 and float(value).is_integer()).all())
        valid=valid and not frame.duplicated('frame_index').any()
        frame=frame.sort_values('frame_index',kind='stable')
        times=pd.to_numeric(frame.hours,errors='coerce').to_numpy(float);finite_times=times[np.isfinite(times)]
        valid=valid and bool(np.all(np.diff(finite_times)>0))
    else:finite_times=pd.to_numeric(frame.hours,errors='coerce').to_numpy(float);finite_times=finite_times[np.isfinite(finite_times)]
    return frame,{'source_run':resolved.inputs.source_run,'movie':movie,'clock_source':kind,'clock_valid':valid,
        'recorded_start_hours':float(np.min(finite_times)) if len(finite_times) else None,
        'recorded_end_hours':float(np.max(finite_times)) if len(finite_times) else None,
        'recorded_timepoints':len(frame),'extent_meaning':'Observed original endpoints only; no extension beyond the first or final recorded observation'}


def held(frame,window):
    key='frame_index' if window['coordinate']=='frames' else 'hours'
    if key not in frame:return pd.Series(False,index=frame.index)
    low=window['start'] if key=='frame_index' else window['start_hours'];high=window['end'] if key=='frame_index' else window['end_hours']
    return frame[key].ge(low)&frame[key].lt(high)


def support_intervals(trace,window,window_id,max_gap):
    """Both endpoints must be original members of this window; retain refusals."""
    selected=held(trace,window);rows=[];previous=None;segment=0
    for row in trace.to_dict('records'):
        if previous is not None and selected.loc[previous['_index']] and selected.loc[row['_index']]:
            dt=row['hours']-previous['hours'];valid=all(previous.get(key) is True and row.get(key) is True for key in ['raw_valid','clock_valid'])
            reason='valid_adjacent_observations'
            if not valid:reason='invalid_value_or_clock'
            elif not finite(dt) or dt<=0 or dt>max_gap:valid=False;reason='clock_or_elapsed_gap'
            elif row['sequence_index']!=previous['sequence_index']+1:valid=False;reason='missing_original_sequence'
            elif finite(row.get('frame_index')) and finite(previous.get('frame_index')) and row['frame_index']!=previous['frame_index']+1:valid=False;reason='missing_original_frame'
            if not valid:segment+=1
            record={'window_id':window_id,'start_observation':previous['observation_id'],'end_observation':row['observation_id'],
                'start_hours':previous['hours'],'end_hours':row['hours'],'start_relative_hours':previous['relative_hours'],'end_relative_hours':row['relative_hours'],
                'elapsed_hours':float(dt) if finite(dt) and dt>0 else None,'observed_interval_hours':float(dt) if valid else 0.,'valid':bool(valid),'reason':reason,'segment':segment}
            rows.append({**record,'interval_id':content_id(record)})
        else:segment+=1
        previous=row
    return rows


def assess(row,support):
    reasons=[]
    if not row['clock_valid']:reasons.append('invalid_clock')
    if not finite(row['summary_value']):reasons.append('missing_window_value')
    if row['valid_observations'] is None:reasons.append('unavailable_observation_count')
    elif row['valid_observations']<support['min_observations']:reasons.append('insufficient_observations')
    if row['valid_fraction'] is None:reasons.append('unavailable_valid_fraction')
    elif row['valid_fraction']<support['min_valid_fraction']:reasons.append('low_valid_fraction')
    if row['coverage_fraction'] is None:reasons.append('unavailable_temporal_coverage')
    elif row['coverage_fraction']<support['min_coverage']:reasons.append('low_temporal_coverage')
    truncated=row['recording_coverage_fraction'] is not None and row['recording_coverage_fraction']<1.-1e-12
    if row['recording_coverage_fraction'] is not None and row['recording_coverage_fraction']<support['min_coverage']:reasons.append('insufficient_recording_coverage')
    return {**row,'recording_truncated':truncated,'eligible':not reasons,'status':('usable_partial_window' if truncated else 'usable') if not reasons else reasons[0],'reasons':reasons,
        'reason':'Recorded summary with declared original observation support' if not reasons else '; '.join(reasons)}


def prepare(resolved,tables):
    support=resolved.request.support;records={};clocks={}
    for movie in resolved.recordings:
        clocks[movie],records[movie]=clock(resolved,tables,movie)
    traces=[];measurements=[];windows=[];members=[];intervals=[];comparisons=[]
    reducers={'mean':np.mean,'median':np.median,'min':np.min,'max':np.max}
    for cell in resolved.inputs.cells:
        recording=resolved.recordings[cell.movie];anchor=recording['anchor'];sample=recording['sample'];original_clock=records[cell.movie]
        for measured in resolved.measurements:
            base={**cell.as_dict(),'measurement':measured.column,'measurement_id':measured.record_id,'table':measured.table,
                'summary_operation':measured.summary,'unit':measured.unit,'anchor_hours':anchor['hours'],'anchor_kind':anchor['kind'],
                'anchor_label':anchor['label'],'sample':sample['sample'],'sample_confirmed':sample['confirmed'],'condition':recording['condition']}
            is_summary='window' in measured.grain;source=tables[measured.table]
            if not is_summary:
                trace,metadata,_=observations(cell,measured,source,None);trace['relative_hours']=trace.hours-anchor['hours']
                trace['_index']=trace.index
                traces.extend(_json_value(trace.drop(columns=['_index']).to_dict('records')))
                measurements.append({**base,**metadata,'input_kind':'original_trace'})
            else:
                trace=pd.DataFrame();source=source.loc[source.stem.eq(cell.movie)&source.identity.eq(cell.identity)]
                metadata={'clock_valid':True};measurements.append({**base,'input_kind':'saved_window_summary','source_rows':len(source),
                    'reason':'Original saved summaries; no raw observations are reconstructed'})
            local={}
            for window in recording['windows']:
                window_id=content_id({'cell':cell.as_dict(),'measurement':measured.record_id,'anchor':anchor,'window':window})
                row={**base,**window,'window_id':window_id,'window':window['name'],'input_kind':'saved_window_summary' if is_summary else 'original_trace',
                    'clock_valid':metadata['clock_valid'] and original_clock['clock_valid'],'summary_value':None,'observations':None,'valid_observations':None,
                    'valid_fraction':None,'observed_interval_hours':None,'coverage_fraction':None,'recording_coverage_fraction':None,
                    'recorded_start_hours':original_clock['recorded_start_hours'],'recorded_end_hours':original_clock['recorded_end_hours'],
                    'clock_source':original_clock['clock_source'],'first_observed_hours':None,'last_observed_hours':None,'gap_count':None,
                    'boundary_rule':'start inclusive, end exclusive','coverage_meaning':'Adjacent original valid observations within this window; no interpolation, endpoint extension or reassigned boundary observation'}
                captured=clocks[cell.movie].loc[held(clocks[cell.movie],window)]
                row['captured_timepoints']=len(captured)
                captured_hours=pd.to_numeric(captured.hours,errors='coerce') if 'hours' in captured else pd.Series(dtype=float)
                captured_hours=captured_hours.loc[np.isfinite(captured_hours)]
                row['first_captured_hours']=float(captured_hours.min()) if len(captured_hours) else None
                row['last_captured_hours']=float(captured_hours.max()) if len(captured_hours) else None
                if window['coordinate']=='frames':
                    row['requested_frames']=int(window['end']-window['start']);row['requested_hours']=None
                    row['recording_coverage_fraction']=len(captured)/row['requested_frames'] if original_clock['recorded_timepoints'] else None
                else:
                    width=window['end_hours']-window['start_hours'];row['requested_hours']=width;row['requested_frames']=None
                    if finite(original_clock['recorded_start_hours']) and finite(original_clock['recorded_end_hours']):
                        span=max(0.,min(window['end_hours'],original_clock['recorded_end_hours'])-max(window['start_hours'],original_clock['recorded_start_hours']))
                        row['recording_coverage_fraction']=span/width
                if is_summary:
                    saved=source.loc[source.window.eq(window['name'])]
                    if len(saved):
                        original=saved.iloc[0].to_dict();value=original[measured.column];row['summary_value']=float(value) if finite(value) else None
                        row['source_summary_id']=content_id({'cell':cell.as_dict(),'measurement':measured.record_id,'window':window['name'],'input_hash':resolved.inputs.table_hashes[measured.table]})
                        for key in ['observations','valid_observations','observed_interval_hours']:
                            value=original.get(key)
                            if value is not None and not pd.isna(value):
                                if not finite(value) or value<0 or (key!='observed_interval_hours' and not float(value).is_integer()):raise ValueError('Invalid recorded window support: '+key)
                                row[key]=float(value) if key=='observed_interval_hours' else int(value)
                        if row['observations'] is not None and row['valid_observations'] is not None:
                            if row['valid_observations']>row['observations']:raise ValueError('Saved valid observation count exceeds all observations')
                            row['valid_fraction']=row['valid_observations']/row['observations'] if row['observations'] else None
                else:
                    selected=trace.loc[held(trace,window)];valid=selected.loc[selected.raw_valid&selected.clock_valid]
                    row.update(observations=len(selected),valid_observations=len(valid),valid_fraction=len(valid)/len(selected) if len(selected) else None,
                        first_observed_hours=float(valid.hours.min()) if len(valid) else None,last_observed_hours=float(valid.hours.max()) if len(valid) else None,
                        summary_value=float(reducers[measured.summary](valid.raw_value.to_numpy(float))) if len(valid) else None)
                    members.extend({**base,'window_id':window_id,'window':window['name'],'observation_id':item['observation_id'],
                        'hours':item['hours'],'relative_hours':item['relative_hours'],'frame_index':item.get('frame_index'),
                        'value_valid':bool(item['raw_valid'] and item['clock_valid'])} for item in selected.to_dict('records'))
                    elapsed=support_intervals(trace,window,window_id,support['max_gap_hours']);intervals.extend({**base,**item} for item in elapsed)
                    row['observed_interval_hours']=sum(item['observed_interval_hours'] for item in elapsed);row['gap_count']=sum(not item['valid'] for item in elapsed)
                if row['requested_hours'] is not None and row['observed_interval_hours'] is not None:
                    if row['observed_interval_hours']>row['requested_hours']+1e-10:raise ValueError('Recorded observation exposure exceeds requested window duration')
                    row['coverage_fraction']=row['observed_interval_hours']/row['requested_hours']
                elif row['requested_frames'] is not None and row['valid_observations'] is not None:
                    row['coverage_fraction']=row['valid_observations']/row['requested_frames']
                    row['coverage_meaning']='Valid original frame observations / requested frames; elapsed hours retained separately without converting frame bounds'
                row=assess(row,support);windows.append(row);local[window['name']]=row
            for window in recording['windows']:
                if window['baseline'] is None:continue
                baseline,target=local[window['baseline']],local[window['name']]
                definition={**base,'baseline_window_id':baseline['window_id'],'target_window_id':target['window_id']}
                comparisons.append({**definition,'comparison_id':content_id(definition),'baseline':window['baseline'],'target_window':window['name'],
                    'baseline_value':baseline['summary_value'],'target_value':target['summary_value'],'baseline_status':baseline['status'],'target_status':target['status'],
                    'eligible':baseline['eligible'] and target['eligible'],'status':'usable' if baseline['eligible'] and target['eligible'] else 'insufficient_window_support',
                    'reason':'Original window summaries and observation support; no response test yet','evidence_performed':False})
    return {'cells':_table([cell.as_dict() for cell in resolved.inputs.cells],['source_run','movie','identity']),
        'measurements':_table(measurements,KEYS+['input_kind','summary_operation']),
        'traces':_table(traces,KEYS+['observation_id','hours','relative_hours','raw_value','raw_valid','clock_valid','sequence_index','frame_index']),
        'recordings':_table(list(records.values()),['source_run','movie','clock_source','clock_valid','recorded_start_hours','recorded_end_hours']),
        'windows':_table(windows,KEYS+['window_id','window','summary_value','eligible','status']),
        'window_members':_table(members,KEYS+['window_id','window','observation_id','hours','relative_hours','value_valid']),
        'intervals':_table(intervals,KEYS+['window_id','interval_id','start_hours','end_hours','observed_interval_hours','valid','reason']),
        'comparisons':_table(comparisons,KEYS+['comparison_id','baseline_window_id','target_window_id','baseline_value','target_value','eligible','status'])}


def produce(context):
    tables=read_verified_tables(context.table_paths,context.request.inputs.table_hashes);outputs=prepare(context.request,tables)
    context.output.mkdir(parents=True);refs=[]
    for name,frame in outputs.items():
        path=context.output/(name+'.json');path = write_table(path,frame);refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    path=context.output/'provenance.json';_write_json(path,{'schema_version':1,'scientific_id':context.scientific_id,
        'resolved_request':context.request.as_dict(),'design_id':context.saved('intervention-design').outcome.scientific_id,
        'scientific_tests_performed':False,'windows_retimed':False,'missingness':'Every requested original cell/measurement/window is retained; unknown support is not zero',
        'summary_authority':'Public NumPy mean, median, min or max over valid original window observations; saved summaries retain their original operation',
        'libraries':{'numpy':np.__version__,'pandas':pd.__version__},
        'time_support':'Only adjacent valid observations both owned by one window supply elapsed support; gaps and final intervals are not filled or extended'})
    refs.append(ArtifactRef('provenance',path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed','Saved complete original-clock and event-relative windows, coverage, summaries and comparison membership',tuple(refs))


def read_windows(saved):
    provenance=read_document(saved.artifact('provenance'))
    if provenance.get('schema_version')!=1 or provenance.get('scientific_id')!=saved.outcome.scientific_id:raise ValueError('Intervention window source identity/schema mismatch')
    data={name:read_table(saved.artifact(name)) for name in TABLES}
    if data['windows'].window_id.duplicated().any() or data['comparisons'].comparison_id.duplicated().any():raise ValueError('Repeated original window or comparison identity')
    lookup=data['windows'].set_index('window_id').to_dict('index')
    for row in data['comparisons'].to_dict('records'):
        for key in ['baseline_window_id','target_window_id']:
            target=lookup.get(row[key])
            if target is None or any(row[name]!=target[name] for name in KEYS):raise ValueError('Window comparison lost its exact original measured cell')
    data['provenance']=provenance
    return data
