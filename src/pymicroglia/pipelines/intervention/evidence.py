"""Complete original-unit changes and separately declared within-cell evidence."""
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
from pathlib import Path
import json
import numpy as np

import pymicroglia.workbench as circadian
from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines.intervention.windows import KEYS, finite, read_windows
import pymicroglia.pipelines.intervention.statistics as native
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table

TABLES=['effects','families','cell_decisions','native_details']
MISSING_POLICY='Every declared comparison remains in its original family. An unavailable probability occupies a conservative p=1 correction slot internally and is saved as p/q unavailable, never a tested null.'


relative_settings=native.relative_settings


def identity(context):
    request=context.request.request
    return content_id({'windows':context.saved('aligned-windows').outcome.scientific_id,
        'evidence':request.evidence.as_dict(),'inference':request.inference.as_dict(),
        'relative_effects':request.relative_effects.as_dict(),'meaningful_effects':request.meaningful_effects.as_dict(),
        'libraries':native.version() if request.evidence['method']!='none' else {'numpy':np.__version__},
        'code':{name:file_hash(source_file(name)) for name in ['intervention_evidence.py','intervention_statistics.py','intervention_windows.py','contracts.py','screening.py']},
        'correction_gateway':file_hash(Path(circadian.__file__)),'workbench':circadian.WORKBENCH_VERSION})


def relative(change,baseline,setting):
    if setting is None:return None,'not_requested'
    if change is None or not finite(baseline):return None,'unavailable_original_values'
    if abs(baseline)<setting['minimum_absolute_baseline']:return None,'baseline_too_close_to_zero'
    if setting['denominator']=='baseline' and baseline<=0:return None,'nonpositive_ratio_baseline'
    denominator=abs(baseline) if setting['denominator']=='absolute_baseline' else baseline
    value=change/denominator
    return (float(value),'descriptive_fractional_change') if finite(value) else (None,'nonfinite_relative_result')


def adjust(rows,inference):
    groups={};families=[]
    for row in rows:
        row.update(q_value=None,family_id=None,family_requested=0,family_tested=0,statistically_supported=False,response_supported=False)
        if row['formal_hypothesis']:
            key=row['measurement'] if inference['correction_scope']=='measurement' else 'all'
            groups.setdefault(key,[]).append(row)
    for key,members in sorted(groups.items()):
        members.sort(key=lambda item:item['effect_id']);ids=[item['effect_id'] for item in members]
        family_id=content_id({'level':'within_cell_time_series','scope':key,'members':ids,'inference':inference})
        valid=[finite(item['p_value']) for item in members]
        adjusted=circadian.adjust_pvalues([item['p_value'] if use else 1. for item,use in zip(members,valid)],inference['multiple_testing']) if any(valid) else [None]*len(members)
        for row,use,q in zip(members,valid,adjusted):
            row.update(q_value=float(q) if use else None,family_id=family_id,family_requested=len(members),family_tested=sum(valid),
                statistically_supported=bool(use and q<=inference['alpha']))
        families.append({'family_id':family_id,'evidence_level':'within_cell_time_series','scope':key,'requested':len(members),'tested':sum(valid),
            'members':ids,'alpha':inference['alpha'],'correction':inference['multiple_testing'],'missing_probability_policy':MISSING_POLICY})
    for row in rows:
        value=row['estimate'];threshold=row['meaningful_effect_threshold']
        row['model_estimate_exceeds_threshold']=bool(finite(value) and abs(value)>=threshold) if threshold is not None else None
        qualifies=threshold is None or row['model_estimate_exceeds_threshold']
        row['response_supported']=bool(row['statistically_supported'] and qualifies and finite(value) and value!=0)
        if row['response_supported']:outcome='increase' if value>0 else 'decrease'
        elif row['statistically_supported'] and not qualifies:outcome='detected_below_meaningful_threshold'
        elif finite(row['p_value']):outcome='no_detected_change'
        else:outcome='inconclusive'
        row.update(outcome=outcome,response_direction='increase' if finite(value) and value>0 else 'decrease' if finite(value) and value<0 else 'zero' if finite(value) else None,
            uncertainty_meaning='Pointwise model interval; multiple-testing correction applies to probabilities, not these intervals',
            no_detected_change_meaning='No detected change does not establish absence, equivalence or recovery')
    return rows,families


def analyse(prepared,resolved):
    request=resolved.request;options=native.settings(request.evidence.as_dict());inference=request.inference.as_dict()
    windows={row['window_id']:row for row in _json_value(prepared['windows'].to_dict('records'))}
    observations={row['observation_id']:row for row in _json_value(prepared['traces'].to_dict('records'))}
    members={key:[] for key in windows}
    for row in _json_value(prepared['window_members'].to_dict('records')):
        original=observations.get(row['observation_id'])
        if original is None or any(original[key]!=row[key] for key in KEYS):raise ValueError('Window member lost its original measured observation')
        members[row['window_id']].append(original)
    for key in members:members[key].sort(key=lambda item:item['sequence_index'])
    rows=[];details=[]
    for comparison in _json_value(prepared['comparisons'].to_dict('records')):
        baseline,target=windows[comparison['baseline_window_id']],windows[comparison['target_window_id']]
        for window in [baseline,target]:
            if any(window[key]!=comparison[key] for key in KEYS):raise ValueError('Comparison changed original cell or measurement')
        before,after=comparison['baseline_value'],comparison['target_value']
        change=float(after-before) if finite(before) and finite(after) and finite(after-before) else None
        specification=request.relative_effects.get(comparison['measurement']);ratio,ratio_status=relative(change,before,specification)
        threshold=request.meaningful_effects.get(comparison['measurement'])
        definition={'comparison_id':comparison['comparison_id'],'method':options,'summary':comparison['summary_operation']}
        row={**comparison,'effect_id':content_id(definition),'evidence_level':'within_cell_time_series','method':options['method'],
            'absolute_change':change,'absolute_change_direction':'increase' if change is not None and change>0 else 'decrease' if change is not None and change<0 else 'zero' if change==0 else None,
            'relative_change':ratio,'relative_status':ratio_status,'relative_settings':specification,'relative_unit':'fraction',
            'meaningful_effect_threshold':threshold,'absolute_change_exceeds_threshold':abs(change)>=threshold if change is not None and threshold is not None else None,
            'formal_hypothesis':options['method']!='none','alpha':inference.get('alpha'),'correction':inference.get('multiple_testing'),
            'estimate':None,'interval_low':None,'interval_high':None,'p_value':None,'estimand':options.get('estimand'),
            'baseline_coverage':baseline['coverage_fraction'],'target_coverage':target['coverage_fraction'],
            'baseline_observations':baseline['observations'],'target_observations':target['observations'],
            'status':'untestable','reason':'Insufficient original window support for the requested comparison',
            'absolute_change_meaning':'Chosen follow-up summary minus chosen baseline summary, in original measurement units; distinct from a conditional-model contrast'}
        if comparison['eligible'] and options['method']!='none':
            result=native.compare(members[baseline['window_id']],members[target['window_id']],operation=comparison['summary_operation'],options=options,alpha=inference['alpha'])
            row.update({key:result[key] for key in ['status','reason','estimate','interval_low','interval_high','p_value','estimand']})
            details.append({'effect_id':row['effect_id'],'comparison_id':comparison['comparison_id'],'native_result':result})
        elif options['method']=='none':row.update(status='descriptive',reason='Within-cell significance was explicitly disabled; available original-unit changes remain descriptive')
        rows.append(row)
    rows,families=adjust(rows,inference);cell_rows=[]
    for cell in resolved.inputs.cells:
        local=[row for row in rows if all(row[key]==value for key,value in cell.as_dict().items())]
        selected=[row for row in local if row['response_supported']]
        cell_rows.append({**cell.as_dict(),'all_effect_ids':[row['effect_id'] for row in local],'responding_effect_ids':[row['effect_id'] for row in selected],
            'increased_effect_ids':[row['effect_id'] for row in selected if row['outcome']=='increase'],
            'decreased_effect_ids':[row['effect_id'] for row in selected if row['outcome']=='decrease'],
            'outcome':'responding' if selected else 'no_detected_response' if any(finite(row['p_value']) for row in local) else 'inconclusive',
            'requested_comparisons':len(local),'tested_comparisons':sum(finite(row['p_value']) for row in local)})
    return {'effects':_table(rows,KEYS+['effect_id','comparison_id','absolute_change','estimate','p_value','q_value','outcome','response_supported']),
        'families':_table(families,['family_id','evidence_level','scope','members','requested','tested']),
        'cell_decisions':_table(cell_rows,['source_run','movie','identity','all_effect_ids','responding_effect_ids','outcome']),
        'native_details':_table(details,['effect_id','comparison_id','native_result'])}


def produce(context):
    source=context.saved('aligned-windows');prepared=read_windows(source);data=analyse(prepared,context.request)
    context.output.mkdir(parents=True);refs=[]
    for name,frame in data.items():
        path=context.output/(name+'.json');path = write_table(path,frame);refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    rule=Settings({'full_declared_population':True,'correction':context.request.request.inference.as_dict(),
        'meaningful_effects':context.request.request.meaningful_effects.as_dict(),'criterion':'Corrected supported conditional-model contrast and any declared original-unit magnitude threshold',
        'inference_level':'Within-cell temporal observations, not independent biological replicates'})
    effects=_json_value(data['effects'].to_dict('records'));cells=_json_value(data['cell_decisions'].to_dict('records'))
    selections=[SelectionRecord('responding-cells',context.scientific_id,rule,tuple(Settings(row) for row in cells if row['responding_effect_ids']))]
    for name,direction in [('responding-comparisons',None),('increased-comparisons','increase'),('decreased-comparisons','decrease')]:
        selections.append(SelectionRecord(name,context.scientific_id,rule,tuple(Settings({key:row[key] for key in [*KEYS,'effect_id','comparison_id','baseline_window_id','target_window_id','outcome']})
            for row in effects if row['response_supported'] and (direction is None or row['outcome']==direction))))
    provenance={'schema_version':1,'scientific_id':context.scientific_id,'windows_id':source.outcome.scientific_id,
        'source_run':context.request.inputs.source_run,'resolved_request':context.request.as_dict(),
        'libraries':native.version() if context.request.request.evidence['method']!='none' else {'numpy':np.__version__},
        'workbench_version':circadian.WORKBENCH_VERSION,'missing_probability_policy':MISSING_POLICY,
        'references':native.REFERENCES,'selection_scope':'Complete requested population before any display limit',
        'selections':[selection.as_dict() for selection in selections]}
    path=context.output/'provenance.json';_write_json(path,provenance);refs.append(ArtifactRef('provenance',path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed','Saved all original-unit changes, temporal-model evidence, full families and immutable response selections',tuple(refs),tuple(selections))


def read_evidence(saved,expected_windows_id=None):
    provenance=read_document(saved.artifact('provenance'))
    if provenance.get('schema_version')!=1 or provenance.get('scientific_id')!=saved.outcome.scientific_id:raise ValueError('Intervention evidence identity/schema mismatch')
    if expected_windows_id is not None and provenance['windows_id']!=expected_windows_id:raise ValueError('Intervention evidence uses different original windows')
    data={name:read_table(saved.artifact(name)) for name in TABLES};effects=data['effects']
    if effects.effect_id.duplicated().any() or effects.comparison_id.duplicated().any():raise ValueError('Repeated original intervention effect/comparison')
    data['provenance']=provenance;return data
