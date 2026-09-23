"""Independent window rhythms, detection transitions and native direct contrasts."""
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
from pathlib import Path
import json
import pymicroglia.workbench as circadian
from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines.intervention.windows import KEYS, finite, read_windows
from pymicroglia.pipelines.intervention.evidence import MISSING_POLICY
import pymicroglia.pipelines.intervention.rhythm_windows as windows
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, producer_identity, read_table, write_table

TABLES=['window_results','window_traces','window_details','window_families',
        'comparisons','direct_comparisons','direct_families','native_details']


def policy(block,measurements):
    """Validate declarations only; numerical validation belongs to Workbench."""
    if not block.get('enabled',False):return {'enabled':False}
    chosen=block.get('measurements',list(measurements))
    if not isinstance(chosen,list) or not chosen or any(not isinstance(item,str) for item in chosen) or len(set(chosen))!=len(chosen) or set(chosen)-set(measurements):
        raise ValueError('Rhythm measurements must be distinct selected intervention measurements')
    scope=block.get('correction_scope','all')
    if scope not in {'all','measurement','movie_measurement'}:raise ValueError('Unknown rhythm correction scope')
    comparisons=block.get('comparisons')
    if comparisons is not None:
        if not isinstance(comparisons,list) or not comparisons:raise ValueError('Declare a nonempty rhythm comparison list or omit it for all original comparisons')
        used=set()
        for spec in comparisons:
            if not isinstance(spec,dict) or set(spec)-{'measurement','baseline','target_window','component_period_band'}:raise ValueError('Unknown rhythm comparison declaration')
            if spec.get('measurement') not in chosen or any(not isinstance(spec.get(key),str) or not spec[key].strip() for key in ['baseline','target_window']):raise ValueError('Each rhythm comparison requires a selected measurement and original baseline/target_window names')
            band=spec.get('component_period_band')
            if band is not None and (not isinstance(band,list) or len(band)!=2 or not all(finite(x) for x in band) or not 0<band[0]<band[1]):raise ValueError('Component matching requires an ordered positive period band in hours')
            sid=content_id(spec)
            if sid in used:raise ValueError('Repeated rhythm comparison declaration')
            used.add(sid)
    direct=block.get('direct_change',{'method':'none'})
    allowed={'method','properties','confidence','independent_window_errors','residual_model_justification',
             'max_relative_period_error','phase_reference','period_equivalence_fraction','max_phase_extrapolation_cycles'}
    if not isinstance(direct,dict) or set(direct)-allowed or direct.get('method') not in {'none','native_fit_covariance'}:raise ValueError('Direct rhythm change method must be none or native_fit_covariance')
    if direct['method']=='none':
        if set(direct)!={'method'}:raise ValueError('Disabled direct rhythm comparison has no analysis settings')
    else:
        direct={'properties':['period','amplitude'],'confidence':.95,'max_relative_period_error':.1,**direct}
        props=direct['properties']
        if not isinstance(props,list) or not props or any(not isinstance(x,str) for x in props) or len(set(props))!=len(props) or set(props)-{'period','amplitude','phase'}:raise ValueError('Choose distinct period, amplitude or phase component properties')
        for key,low,high in [('confidence',.5,1.),('max_relative_period_error',0.,.5)]:
            if not finite(direct[key]) or not low<direct[key]<high:raise ValueError(key+' is outside its supported range')
        for key in ['independent_window_errors','residual_model_justification']:
            if not isinstance(direct.get(key),str) or not direct[key].strip():raise ValueError('Direct rhythm comparison requires '+key)
        if 'phase' in props:
            if direct.get('phase_reference')!='recording_anchor':raise ValueError('Phase comparison requires phase_reference recording_anchor, the actual declared event/control time')
            if not finite(direct.get('period_equivalence_fraction')) or not 0<direct['period_equivalence_fraction']<.5:raise ValueError('Phase requires a declared practical period_equivalence_fraction below .5')
            if not finite(direct.get('max_phase_extrapolation_cycles')) or not 0<=direct['max_phase_extrapolation_cycles']<=.5:raise ValueError('Declare allowed phase extrapolation between zero and half a fitted cycle')
        elif any(key in direct for key in ['phase_reference','period_equivalence_fraction','max_phase_extrapolation_cycles']):raise ValueError('Phase-specific settings require the phase property')
    return {'enabled':True,'measurements':chosen,'comparisons':comparisons,'direct_change':direct,'correction_scope':scope}


def identity(context):
    chosen=policy(context.request.request.rhythms.as_dict(),[m.column for m in context.request.measurements])
    return content_id({'windows':context.saved('aligned-windows').outcome.scientific_id,'policy':chosen,
        'recipes':windows.recipes(context.request) if chosen['enabled'] else None,
        'native_environment':producer_identity() if chosen['enabled'] else None,
        'code':{name:file_hash(source_file(name)) for name in ['intervention_rhythms.py','intervention_rhythm_windows.py']}})


def requested_comparisons(prepared,chosen):
    original=_json_value(prepared['comparisons'].to_dict('records'));result=[];found=set()
    for comparison in original:
        if comparison['measurement'] not in chosen['measurements']:continue
        specs=chosen['comparisons']
        if specs is None:specs=[{'measurement':comparison['measurement'],'baseline':comparison['baseline'],'target_window':comparison['target_window']}]
        for spec in specs:
            if any(comparison[key]!=spec[key] for key in ['measurement','baseline','target_window']):continue
            found.add(content_id(spec));result.append((comparison,spec))
    if chosen['comparisons'] is not None and {content_id(s) for s in chosen['comparisons']}-found:raise ValueError('A requested rhythm comparison is absent from the original declared window design')
    return result


def native_record(row,detail,traces,band):
    result={'series_id':content_id({key:row[key] for key in KEYS}), 'window_id':row['window_id'],
        'time_origin':'2000-01-01','analysis_settings':detail['applied_recipe'],
        'hours':[item['hours'] for item in traces],'values':[item['filtered_value'] for item in traces],
        'frame_index':[item['frame_index'] for item in traces],
        'detected':row['detected'],'period_supported':row['period_supported'],
        'estimate':detail['estimate_result'],'significance':detail['significance_result'],'component_period_band':band}
    if all(value is None for value in result['frame_index']):result.pop('frame_index')
    return result


def adjust(rows,scope):
    families=[];groups={}
    for row in rows:
        row.update(q_value=None,family_id=None,change_supported=False)
        label='all' if scope=='all' else row['measurement'] if scope=='measurement' else row['movie']+' / '+row['measurement']
        groups.setdefault((label,row['alpha'],row['correction']),[]).append(row)
    for (label,alpha,correction),members in sorted(groups.items()):
        members.sort(key=lambda row:row['direct_id']);ids=[row['direct_id'] for row in members]
        fid=content_id({'level':'within_cell_rhythm_parameter_change','scope':label,'members':ids,'alpha':alpha,'correction':correction})
        valid=[finite(row['p_value']) for row in members]
        adjusted=circadian.adjust_pvalues([row['p_value'] if use else 1. for row,use in zip(members,valid)],correction) if any(valid) else [None]*len(members)
        for row,use,q in zip(members,valid,adjusted):
            supported=bool(use and q<alpha)
            row.update(q_value=float(q) if use else None,family_id=fid,change_supported=supported,
                outcome='supported_parameter_change' if supported else 'no_detected_parameter_change' if use else 'untestable',
                family_requested=len(members),family_tested=sum(valid))
        families.append({'family_id':fid,'evidence_level':'within_cell_rhythm_parameter_change','scope':label,'requested_scope':scope,
            'members':ids,'requested':len(members),'tested':sum(valid),'alpha':alpha,'correction':correction,
            'missing_probability_policy':MISSING_POLICY})
    return families


def analyse(prepared,resolved,chosen=None):
    chosen=chosen or policy(resolved.request.rhythms.as_dict(),[m.column for m in resolved.measurements])
    data=windows.analyse(prepared,resolved);by_id={row['window_id']:row for row in _json_value(data['window_results'].to_dict('records'))}
    details={row['window_id']:row for row in _json_value(data['window_details'].to_dict('records'))};originals={};comparisons=[];direct=[];native_details=[]
    for item in _json_value(data['window_traces'].to_dict('records')):originals.setdefault(item['window_id'],[]).append(item)
    for comparison,spec in requested_comparisons(prepared,chosen):
        baseline,target=[by_id[comparison[key]] for key in ['baseline_window_id','target_window_id']]
        if any(any(row[key]!=comparison[key] for key in KEYS) for row in [baseline,target]):raise ValueError('Rhythm comparison lost its original measured cell')
        rid=content_id({'comparison':comparison['comparison_id'],'component_period_band':spec.get('component_period_band')})
        transition=baseline['detection_outcome']+' to '+target['detection_outcome']
        common={**{key:comparison[key] for key in [*KEYS,'comparison_id','baseline_window_id','target_window_id','baseline','target_window','unit']},
            'rhythm_comparison_id':rid,'component_period_band':spec.get('component_period_band')}
        comparisons.append({**common,'detection_transition':transition,'baseline_detected':baseline['detected'],'target_detected':target['detected'],
            'baseline_period_supported':baseline['period_supported'],'target_period_supported':target['period_supported'],
            'baseline_window_status':baseline['window_status'],'target_window_status':target['window_status'],
            'transition_meaning':'Separate window test outcomes; a change of significance does not establish rhythm gain, loss or a direct parameter change'})
        config=dict(chosen['direct_change'])
        if config['method']=='none':continue
        if config.pop('phase_reference',None)=='recording_anchor':config['phase_reference_hours']=baseline['anchor_hours']
        config['alpha']=baseline['alpha']
        if config['alpha']>=.5:raise ValueError('Native direct rhythm comparison requires alpha below .5')
        records=[native_record(row,details[row['window_id']],originals.get(row['window_id'],[]),spec.get('component_period_band')) for row in [baseline,target]]
        refusal=None
        if not baseline['original_window_eligible'] or not target['original_window_eligible']:refusal='Insufficient original before/after window support'
        filtering=details[baseline['window_id']]['applied_recipe']['filtering']
        if filtering and filtering.get('method','none')!='none':refusal='Native fitted-parameter covariance does not propagate uncertainty/dependence from a separate trace filter; original window fits remain available'
        if refusal:
            native={'comparisons':[{'property':prop,'status':'untestable','effect':None,'p_value':None,'interval':None,
                'reason':refusal,'unit':'hours' if prop=='period' else 'cycles' if prop=='phase' else 'measurement units'} for prop in config['properties']]}
        else:native=circadian.rhythm_window_comparison(*records,settings=config)
        native_details.append({'rhythm_comparison_id':rid,'native_result':native,'original_windows':[row['window_id'] for row in records],
            'native_input_identity':content_id(records),'applied_settings':config})
        for item in native['comparisons']:
            interval=item.get('interval') or [None,None]
            direct.append({**common,**item,'direct_id':content_id({'rhythm_comparison_id':rid,'property':item['property'],'method':config}),
                'measurement_unit':comparison['unit'],'method':config['method'],'evidence_level':'within_cell_rhythm_parameter_change',
                'interval_low':interval[0],'interval_high':interval[1],'alpha':baseline['alpha'],'correction':baseline['correction'],
                'workbench_version':circadian.WORKBENCH_VERSION,'phase_reference_hours':config.get('phase_reference_hours'),
                'meaning':'Model-conditional fitted component contrast, not general rhythm strength, gain/loss or causality'})
    families=adjust(direct,chosen['correction_scope'])
    data.update(comparisons=_table(comparisons,KEYS+['rhythm_comparison_id','comparison_id','detection_transition']),
        direct_comparisons=_table(direct,KEYS+['rhythm_comparison_id','direct_id','property','effect','p_value','q_value','change_supported','outcome']),
        direct_families=_table(families,['family_id','evidence_level','members','requested','tested']),
        native_details=_table(native_details,['rhythm_comparison_id','native_result','native_input_identity','applied_settings']))
    return data


def produce(context):
    chosen=policy(context.request.request.rhythms.as_dict(),[m.column for m in context.request.measurements])
    if not chosen['enabled']:return StepResult(context.step.name,context.scientific_id,'skipped-empty','Independent window rhythm analysis was explicitly disabled')
    source=context.saved('aligned-windows');data=analyse(read_windows(source),context.request,chosen)
    context.output.mkdir(parents=True);refs=[]
    for name,frame in data.items():
        path=context.output/(name+'.json');path = write_table(path,frame);refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    rule=Settings({'criterion':'Corrected direct component-parameter comparison within the complete original declared family',
        'evidence_level':'within_cell_rhythm_parameter_change','significance_transition_is_not_direct_evidence':True})
    selected=[row for row in _json_value(data['direct_comparisons'].to_dict('records')) if row['change_supported']]
    cells={content_id({key:row[key] for key in ['source_run','movie','identity']}):{key:row[key] for key in ['source_run','movie','identity']} for row in selected}
    selections=[SelectionRecord('rhythm-change-cells',context.scientific_id,rule,tuple(Settings(cells[key]) for key in sorted(cells))),
        SelectionRecord('rhythm-change-comparisons',context.scientific_id,rule,tuple(Settings({key:row[key] for key in [*KEYS,'direct_id','rhythm_comparison_id','property']}) for row in selected))]
    path=context.output/'provenance.json';_write_json(path,{'schema_version':1,'scientific_id':context.scientific_id,'windows_id':source.outcome.scientific_id,
        'resolved_request':context.request.as_dict(),'applied_policy':chosen,'native_environment':producer_identity(),
        'missing_probability_policy':MISSING_POLICY,'numeric_time_origin':'2000-01-01 (documented native numeric-trace epoch)',
        'selection_scope':'All original selected cell/measurement/windows before report selection; window rhythmicity and direct change have separate correction families',
        'preprocessing_scope':'Each original window independently; no cross-intervention filter or whole-recording fit',
        'selections':[row.as_dict() for row in selections]})
    refs.append(ArtifactRef('provenance',path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed','Saved independent window rhythms, detection transitions and native direct component comparisons',tuple(refs),tuple(selections))


def read_rhythms(saved,expected_windows_id=None):
    provenance=read_document(saved.artifact('provenance'))
    if provenance.get('schema_version')!=1 or provenance.get('scientific_id')!=saved.outcome.scientific_id:raise ValueError('Intervention rhythm identity/schema mismatch')
    if expected_windows_id is not None and provenance['windows_id']!=expected_windows_id:raise ValueError('Rhythm results use different original windows')
    data={name:read_table(saved.artifact(name)) for name in TABLES}
    for name,key in [('window_results','window_id'),('window_details','window_id'),('comparisons','rhythm_comparison_id'),('direct_comparisons','direct_id')]:
        if data[name][key].duplicated().any():raise ValueError('Repeated original rhythm result: '+name)
    if set(data['window_details'].window_id)!=set(data['window_results'].window_id) or set(data['window_traces'].window_id)-set(data['window_results'].window_id):raise ValueError('Rhythm result lost its original window membership')
    if data['window_traces'].duplicated(['window_id','observation_id']).any():raise ValueError('Repeated original rhythm window observation')
    for detail in data['window_details'].to_dict('records'):
        actual=data['window_traces'].loc[data['window_traces'].window_id.eq(detail['window_id']),'observation_id'].tolist()
        if len(actual)!=len(detail['source_observation_ids']) or set(actual)!=set(detail['source_observation_ids']):raise ValueError('Rhythm window observations differ from the saved native input membership')
    if set(data['direct_comparisons'].rhythm_comparison_id)-set(data['comparisons'].rhythm_comparison_id):raise ValueError('Direct comparison lost its original window comparison')
    data['provenance']=provenance;return data
