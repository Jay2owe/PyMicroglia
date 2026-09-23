"""Explicit treatment/control changes at the original biological-sample unit."""
from pymicroglia._results import read_document
from pathlib import Path
from importlib.metadata import version
import json
import numpy as np

import pymicroglia.workbench as circadian
import pymicroglia.measure.sample_contrasts as sample_contrasts
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines.intervention.windows import finite, read_windows
from pymicroglia.pipelines.intervention.evidence import read_evidence, MISSING_POLICY
import pymicroglia.pipelines.intervention.control_statistics as native
import pymicroglia.pipelines.intervention.control_inputs as inputs
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table

TABLES=['unit_inventory','recording_inventory','cell_changes','recording_summaries','unit_summaries','definitions',
        'comparisons','comparison_members','matches','families','native_details']


def policy(resolved):
    return native.control_policy(resolved.request.controls.as_dict(),[m.column for m in resolved.measurements],resolved.request.inference.as_dict())


def identity(context):
    if not context.request.request.controls['enabled']:return content_id({'enabled':False,'code':file_hash(__file__)})
    return content_id({'windows':context.saved('aligned-windows').outcome.scientific_id,'evidence':context.saved('response-evidence').outcome.scientific_id,
        'settings':policy(context.request),'inference':context.request.request.inference.as_dict(),'matching':[row.as_dict() for row in context.request.request.matching],
        'code':{path.name:file_hash(path) for path in [Path(__file__),Path(native.__file__),Path(inputs.__file__),Path(sample_contrasts.__file__),Path(circadian.__file__)]},
        'libraries':{name:version(name) for name in ['numpy','pandas','scipy']},'workbench':circadian.WORKBENCH_VERSION})


def compare_units(data,resolved,settings):
    units=_json_value(data['unit_inventory'].to_dict('records'));summaries=_json_value(data['unit_summaries'].to_dict('records'))
    definitions=_json_value(data['definitions'].to_dict('records'));matching=[row.as_dict() for row in resolved.request.matching]
    original_samples={row['sample']:row for row in units if row['sample_confirmed']}
    results=[];members=[];matches=[];details=[]
    for spec in settings['comparisons']:
        selected=[row for row in units if row['condition'] in {spec['reference_condition'],spec['target_condition']}]
        dids=[row['definition_id'] for row in definitions if all(row[key]==spec[key] for key in ['measurement','baseline','target_window','quantity'])
            and any(item['definition_id']==row['definition_id'] and item['condition'] in {spec['reference_condition'],spec['target_condition']} and item['matching_recordings']>0 for item in summaries)]
        dids=sorted(set(dids));cid=content_id({'specification':spec['comparison_spec_id'],'definitions':dids,'units':sorted(row['unit_id'] for row in selected)})
        refusal=[]
        if not dids:refusal.append('No original change has the requested measurement and baseline/follow-up definition')
        elif len(dids)!=1:refusal.append('The requested groups have incompatible original window, clock or change definitions; strata remain separate')
        did=dids[0] if len(dids)==1 else None
        local={row['unit_id']:row for row in summaries if row['definition_id']==did} if did else {}
        groups={role:[row for row in selected if row['condition']==spec[role+'_condition']] for role in ['reference','target']}
        if any(row['anchor_kind']!='control' for row in groups['reference']) or any(row['anchor_kind']!='intervention' for row in groups['target']):
            refusal.append('The declared conditions do not consistently identify control reference samples and intervention target samples')
        eligible={role:{row['sample']:local[row['unit_id']] for row in group if row['sample_confirmed'] and row['unit_id'] in local and local[row['unit_id']]['formal_eligible']} for role,group in groups.items()}
        all_samples={role:{row['sample'] for row in group if row['sample_confirmed']} for role,group in groups.items()}
        relevant=[row for row in matching if row['reference_sample'] in all_samples['reference'] or row['target_sample'] in all_samples['target']]
        complete=[]
        for match in relevant:
            original_a=original_samples.get(match['reference_sample']);original_b=original_samples.get(match['target_sample'])
            if original_a is None or original_b is None:raise ValueError('Declared matching lost an original confirmed biological sample')
            correct=match['reference_sample'] in all_samples['reference'] and match['target_sample'] in all_samples['target']
            a=eligible['reference'].get(match['reference_sample']);b=eligible['target'].get(match['target_sample']);included=bool(correct and a is not None and b is not None)
            saved={'comparison_id':cid,**match,'complete':included,'included':included and spec['design']=='matched','reference_unit_id':original_a['unit_id'],'target_unit_id':original_b['unit_id'],
                'status':'complete_original_match' if included else 'outside_declared_condition_contrast' if not correct else 'missing_eligible_member',
                'reference_value':local.get(original_a['unit_id'],{}).get('value'),'target_value':local.get(original_b['unit_id'],{}).get('value')}
            matches.append(saved)
            if included:complete.append((match,a,b))
        if spec['design']=='matched':
            if not any(row['reference_sample'] in all_samples['reference'] and row['target_sample'] in all_samples['target'] for row in relevant):refusal.append('No original biological-sample matching was declared for this condition contrast')
            pair_ids=[row[0]['match_id'] for row in complete]
            if len(set(pair_ids))!=len(pair_ids):raise ValueError('Repeated original biological-sample match')
            chosen={'reference':[row[1] for row in complete],'target':[row[2] for row in complete]}
        else:
            if any(row['reference_sample'] in all_samples['reference'] and row['target_sample'] in all_samples['target'] for row in relevant):refusal.append('Declared matching between these groups cannot be ignored by an independent-sample test')
            chosen={role:sorted(group.values(),key=lambda row:row['unit_id']) for role,group in eligible.items()}
        if set(row['unit_id'] for row in chosen['reference'])&set(row['unit_id'] for row in chosen['target']):raise ValueError('One biological sample occurs in both sides of a contrast')
        if not chosen['reference'] or not chosen['target']:refusal.append('No confirmed eligible biological units on at least one side of the comparison')
        evidence={'method':'none'} if refusal else spec['evidence']
        result=native.compare([row['value'] for row in chosen['reference']],[row['value'] for row in chosen['target']],evidence,matched=spec['design']=='matched')
        if refusal:result.update(status='untestable',reason='; '.join(dict.fromkeys(refusal)))
        formal=spec['evidence']['method']!='none'
        details.append({'comparison_id':cid,'native_result':result,'actual_matching':[row[0] for row in complete],
            'reference_unit_ids':[row['unit_id'] for row in chosen['reference']],'target_unit_ids':[row['unit_id'] for row in chosen['target']],
            'requested_evidence':spec['evidence'],'preanalysis_refusals':refusal})
        for role,group in groups.items():
            included={row['unit_id'] for row in chosen[role]}
            for unit in group:
                row=local.get(unit['unit_id']);used=unit['unit_id'] in included
                members.append({'comparison_id':cid,'definition_id':did,**unit,'role':role,'value':row['value'] if row else None,
                    'included':used,'formal_included':used and formal and not refusal,
                    'reason':'Complete original biological-unit aggregate' if used else 'Unconfirmed biological sample' if not unit['sample_confirmed'] else 'Missing or incompatible original change' if row is None or not row['formal_eligible'] else 'No complete original match for this comparison',
                    'all_effect_ids':row['all_effect_ids'] if row else [],'eligible_effect_ids':row['eligible_effect_ids'] if row else [],
                    'candidate_definition_ids':[item['definition_id'] for item in summaries if item['unit_id']==unit['unit_id'] and item['definition_id'] in dids and item['matching_recordings']>0]})
        definition=next((row for row in definitions if row['definition_id']==did),{})
        interval=result['effect_interval']
        results.append({'comparison_id':cid,'name':spec['name'],'comparison_spec_id':spec['comparison_spec_id'],'source_run':resolved.inputs.source_run,
            'definition_id':did,'definition_ids':dids,**{key:spec[key] for key in ['measurement','baseline','target_window','quantity','design','reference_condition','target_condition']},
            'unit':definition.get('unit'),'evidence_level':'biological_sample','method':spec['evidence']['method'],'formal_hypothesis':formal,
            'aggregation':settings['aggregation'],'effect':result['effect'],'interval_low':interval[0] if interval else None,'interval_high':interval[1] if interval else None,
            'interval_status':result['interval_status'],'p_value':result['p_value'],'status':result['status'],'reason':result['reason'],
            'requested_reference_units':len(groups['reference']),'requested_target_units':len(groups['target']),
            'reference_samples':len(chosen['reference']),'target_samples':len(chosen['target']),'complete_matches':len(complete) if spec['design']=='matched' else None,
            'reference_cells':sum(row['available_cells'] for row in chosen['reference']),'target_cells':sum(row['available_cells'] for row in chosen['target']),
            'reference_recordings':sum(row['available_recordings'] for row in chosen['reference']),'target_recordings':sum(row['available_recordings'] for row in chosen['target']),
            'effect_meaning':result['effect_scope'],'interval_meaning':result['interval_scope'],'population':'All eligible original cell changes; no responder-only selection',
            'causal_interpretation':'Observed treatment/control change association; causality requires an appropriate experimental design'})
    return results,members,matches,details


def correct(rows,inference):
    groups={};families=[]
    for row in rows:
        row.update(q_value=None,family_id=None,supported=False,alpha=inference.get('alpha'),correction=inference.get('multiple_testing'))
        if row['formal_hypothesis']:groups.setdefault(row['measurement'] if inference['correction_scope']=='measurement' else 'all',[]).append(row)
    for scope,members in sorted(groups.items()):
        members.sort(key=lambda row:row['comparison_id']);ids=[row['comparison_id'] for row in members]
        fid=content_id({'evidence_level':'biological_sample','scope':scope,'members':ids,'inference':inference})
        valid=[finite(row['p_value']) for row in members];adjusted=circadian.adjust_pvalues([row['p_value'] if use else 1. for row,use in zip(members,valid)],inference['multiple_testing']) if any(valid) else [None]*len(members)
        for row,use,q in zip(members,valid,adjusted):row.update(q_value=float(q) if use else None,family_id=fid,supported=bool(use and q<=inference['alpha']))
        families.append({'family_id':fid,'evidence_level':'biological_sample','scope':scope,'members':ids,'requested':len(members),'tested':sum(valid),
            'alpha':inference['alpha'],'correction':inference['multiple_testing'],'missing_probability_policy':MISSING_POLICY})
    for row in rows:row['outcome']='supported_difference' if row['supported'] else 'no_detected_difference' if finite(row['p_value']) else row['status']
    return rows,families


def produce(context):
    settings=policy(context.request)
    if not settings['enabled']:return StepResult(context.step.name,context.scientific_id,'skipped-empty','Treatment/control sample comparisons were explicitly disabled')
    windows=context.saved('aligned-windows');source=context.saved('response-evidence')
    prepared=read_windows(windows);evidence=read_evidence(source,windows.outcome.scientific_id)
    data=inputs.aggregate(prepared,evidence,context.request,settings);rows,members,matches,details=compare_units(data,context.request,settings)
    rows,families=correct(rows,context.request.request.inference.as_dict())
    data.update(comparisons=_table(rows,['comparison_id','definition_id','measurement','effect','p_value','q_value','outcome']),
        comparison_members=_table(members,['comparison_id','unit_id','role','value','included','formal_included']),
        matches=_table(matches,['comparison_id','match_id','reference_sample','target_sample','included','status']),
        families=_table(families,['family_id','evidence_level','members','requested','tested']),native_details=_table(details,['comparison_id','native_result']))
    context.output.mkdir(parents=True);refs=[]
    for name,frame in data.items():
        path=context.output/(name+'.json');path = write_table(path,frame);refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    path=context.output/'provenance.json';_write_json(path,{'schema_version':1,'scientific_id':context.scientific_id,'windows_id':windows.outcome.scientific_id,
        'evidence_id':source.outcome.scientific_id,'resolved_request':context.request.as_dict(),'applied_settings':settings,
        'libraries':{name:version(name) for name in ['numpy','pandas','scipy']},'workbench_version':circadian.WORKBENCH_VERSION,
        'population':'All eligible original changes; significance of individual cells never defines the comparison population',
        'experimental_unit':'Original confirmed biological sample; equal available recording summaries within a sample; exact declared matching only',
        'missing_probability_policy':MISSING_POLICY,'references':sample_contrasts.REFERENCES})
    refs.append(ArtifactRef('provenance',path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed','Saved complete biological units, original matching, control-change contrasts and separate families',tuple(refs))


def read_controls(saved,expected_evidence_id=None):
    provenance=read_document(saved.artifact('provenance'))
    if provenance.get('schema_version')!=1 or provenance.get('scientific_id')!=saved.outcome.scientific_id:raise ValueError('Control-comparison identity/schema mismatch')
    if expected_evidence_id is not None and provenance['evidence_id']!=expected_evidence_id:raise ValueError('Control comparisons use different original change evidence')
    data={name:read_table(saved.artifact(name)) for name in TABLES}
    if data['unit_inventory'].unit_id.duplicated().any() or data['comparisons'].comparison_id.duplicated().any():raise ValueError('Repeated original biological unit or control comparison')
    data['provenance']=provenance;return data
