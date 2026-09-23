"""Independent original-window Workbench fits and complete detection families."""
import numpy as np
import pandas as pd
import pymicroglia.workbench as circadian
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines.intervention.windows import finite
from pymicroglia.pipelines.intervention.evidence import MISSING_POLICY
from pymicroglia.pipelines._screening import _json_value


def recipes(resolved):
    native=resolved.rhythm_analysis.as_dict();chosen=resolved.request.rhythms.get('measurements',[m.column for m in resolved.measurements])
    result={}
    for metric in chosen:
        recipe=native['measurement_recipes'].get(metric)
        result[metric]={'options':recipe['analysis_options'] if recipe else native['analysis_options'],
            'params':recipe['rhythm_params'] if recipe else native['rhythm_params'],'filtering':recipe['filtering'] if recipe else None,
            'applied_recipe':recipe,'profile_provenance':native['profile_provenance']}
    return result


def analyse(prepared,resolved):
    declarations=recipes(resolved);source=prepared['traces'];members=prepared['window_members'];rows=[];traces=[];details=[]
    source_rows={row['observation_id']:row for row in _json_value(source.to_dict('records'))}
    for window in _json_value(prepared['windows'].to_dict('records')):
        if window['measurement'] not in declarations:continue
        recipe=declarations[window['measurement']];options=recipe['options'];params=recipe['params'];wid=window['window_id']
        ids=members.loc[members.window_id.eq(wid),'observation_id'].tolist()
        if len(set(ids))!=len(ids) or set(ids)-set(source_rows):raise ValueError('Window rhythm analysis lost original observation membership')
        observations=sorted((source_rows[oid] for oid in ids),key=lambda row:row['sequence_index'])
        if any(any(row[key]!=window[key] for key in ['source_run','movie','identity','measurement_id']) for row in observations):raise ValueError('Rhythm window contains another measured cell')
        hours=np.array([row['hours'] if finite(row['hours']) else np.nan for row in observations],float)
        values=np.array([row['raw_value'] if row['raw_valid'] and finite(row['raw_value']) else np.nan for row in observations],float)
        filtered=values.copy();processing=None;refusal=None
        if window['input_kind']!='original_trace':refusal='original_window_trace_unavailable'
        elif not window['clock_valid']:refusal='invalid_original_window_clock'
        elif not window['eligible']:refusal='insufficient_original_window_support: '+window['reason']
        elif not len(observations):refusal='no_original_window_observations'
        if refusal is None and recipe['filtering'] is not None:
            processing=circadian.filter_rhythm_trace(hours.tolist(),values.tolist(),recipe['filtering']);filtered=np.asarray(processing['values'],float)
            if filtered.shape!=values.shape:raise ValueError('Native rhythm preprocessing changed the original window observation count')
        evaluated={}
        if refusal is None:
            long=pd.DataFrame({'window_trace':0,'hours':hours,'value':filtered})
            native=circadian.estimate_grouped_rhythms(long,group_columns=['window_trace'],value_column='value',params=params,
                method=options['fit_method'],significance_method=options['significance_method'],detrend=options['detrend'],
                detrend_window_hours=options['detrend_window_hours'],min_observations=options['min_observations'],min_cycles=options['min_cycles'],
                correction='none',capture_details=True)
            if len(native)!=1:raise ValueError('Native window rhythm evaluation changed its single original series')
            evaluated=_json_value(native.iloc[0].to_dict())
        estimate=evaluated.pop('estimate_result',{}) or {};significance=evaluated.pop('significance_result',{}) or {}
        for field in ['window_trace','metric','q_value','significant','rhythm_status','family_tests']:evaluated.pop(field,None)
        base={key:window.get(key) for key in ['source_run','movie','identity','measurement','measurement_id','window_id','window','unit','anchor_hours','anchor_kind','anchor_label','sample','sample_confirmed','condition','coordinate','start','end','start_hours','end_hours']}
        p=evaluated.get('p_value');valid=finite(p) and 0<=p<=1 and evaluated.get('test_status')=='ok'
        period_supported=bool(evaluated.get('estimate_status')=='ok' and finite(evaluated.get('period_hours')) and not evaluated.get('period_underdetermined',True) and not evaluated.get('period_at_search_edge',True))
        rows.append({**base,**evaluated,'window_status':refusal or ('insufficient_cycles_or_unresolved_period' if not period_supported else 'supported_period_estimate'),
            'original_window_status':window['status'],'original_window_eligible':window['eligible'],'original_observations':len(observations),
            'p_value':float(p) if valid else None,'q_value':None,'detected':False,'period_supported':period_supported,'family_id':None,
            'test_status':evaluated.get('test_status','not_tested'),'reason':refusal or evaluated.get('reason',''),
            'method':options['fit_method'],'significance_method':options['significance_method'],'workbench_version':circadian.WORKBENCH_VERSION,
            'min_cycles':options['min_cycles'],'min_observations':options['min_observations'],'alpha':options['rhythmic_alpha'],
            'correction':options['multiple_testing'],'recipe_id':content_id(recipe),'observed_interval_hours':window['observed_interval_hours'],
            'coverage_fraction':window['coverage_fraction'],'gap_count':window['gap_count'],'recording_truncated':window['recording_truncated']})
        details.append({'window_id':wid,'estimate_result':estimate,'significance_result':significance,'filtering_result':processing,
            'applied_recipe':recipe,'native_series':estimate.get('native_series',{}),'native_result':estimate.get('native_result',{}),
            'processed_trace':estimate.get('display_processed_trace') or significance.get('display_processed_trace'),
            'processed_trace_provenance':estimate.get('display_processing_run_record') or significance.get('display_processing_run_record'),
            'source_observation_ids':ids,'scientific_window_rule':'Only this original window was filtered and fitted; no whole-recording or cross-intervention preparation'})
        for i,observation in enumerate(observations):
            traces.append({**observation,'window_id':wid,'window':window['window'],'filtered_value':float(filtered[i]) if np.isfinite(filtered[i]) else None,
                'filtering_applied':recipe['filtering'] is not None and processing is not None,'recipe_id':content_id(recipe)})
    families=[];groups={};scope=resolved.request.rhythms.get('correction_scope','all')
    for row in rows:
        label='all' if scope=='all' else row['measurement'] if scope=='measurement' else row['movie']+' / '+row['measurement']
        groups.setdefault((label,row['alpha'],row['correction']),[]).append(row)
    for (label,alpha,correction),group in sorted(groups.items()):
        group.sort(key=lambda row:row['window_id']);ids=[row['window_id'] for row in group]
        fid=content_id({'evidence_level':'individual_window_rhythmicity','scope':label,'members':ids,'alpha':alpha,'correction':correction})
        tested=[finite(row['p_value']) for row in group];q=circadian.adjust_pvalues([row['p_value'] if valid else 1. for row,valid in zip(group,tested)],correction) if any(tested) else [None]*len(group)
        for row,valid,adjusted in zip(group,tested,q):
            row.update(q_value=float(adjusted) if valid else None,family_id=fid,detected=bool(valid and adjusted<alpha),
                detection_outcome='untestable' if not valid else 'significant' if adjusted<alpha else 'not-significant')
        families.append({'family_id':fid,'evidence_level':'individual_window_rhythmicity','scope':label,'requested_scope':scope,'alpha':alpha,'correction':correction,
            'members':ids,'requested':len(group),'tested':sum(tested),'missing_probability_policy':MISSING_POLICY,
            'family_definition':'Original selected measurement/windows; differing explicitly applied alpha/correction recipes retain separate declared families'})
    return {'window_results':_table(rows,['window_id','measurement','method','significance_method','p_value','q_value','detected','period_supported','window_status']),
        'window_traces':_table(traces,['window_id','observation_id','hours','relative_hours','raw_value','filtered_value']),
        'window_details':_table(details,['window_id','estimate_result','significance_result','applied_recipe','source_observation_ids']),
        'window_families':_table(families,['family_id','evidence_level','members','requested','tested'])}
