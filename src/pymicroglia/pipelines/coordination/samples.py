"""Complete compatible recording effects, one aggregate per biological sample."""
from __future__ import annotations
from pymicroglia._results import read_document
from pymicroglia._sources import source_file

import json
from pathlib import Path

import pymicroglia.measure.sample_contrasts as native
from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.coordination.evidence import read_evidence
from pymicroglia.pipelines.coordination.inputs import _table, read_inputs
from pymicroglia.pipelines.coordination.sample_inputs import aggregate, member_records
from pymicroglia.pipelines.coordination.state_inputs import load_states
from pymicroglia.pipelines._runner import Unavailable
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


def policy(request):
    q=request.sample_summary
    if not q['enabled']:return {'enabled':False}
    try:settings=native.policy(q['evidence'])
    except native.UnsupportedContrast as error:raise Unavailable(str(error)) from error
    seen=set()
    for a,b in q['contrasts']:
        key=frozenset((a,b))
        if key in seen:raise ValueError('Sample condition contrasts repeat an unordered pair')
        seen.add(key)
    if settings['method']!='none' and not q['contrasts']:raise ValueError('Formal sample comparisons require explicit condition contrasts')
    return {'enabled':True,'aggregation':q['aggregation'],'evidence':settings,'contrasts':q['contrasts'],'timing':q.get('timing',{})}


def implementation(rhythm=False):
    paths=[Path(__file__),Path(native.__file__),*(source_file(name) for name in
        ['coordination_sample_inputs.py','coordination_sample_timing.py','coordination_state_inputs.py','coordination_evidence.py']),source_file('circadian.py')]
    result={'code':{path.name:file_hash(path) for path in paths}}
    if rhythm:
        import pymicroglia.workbench as circadian
        result['workbench']=circadian.rhythm_environment()
    return result


def identity(context):
    if not context.request.request.sample_summary['enabled']:return content_id({'enabled':False,'code':file_hash(__file__)})
    settings=policy(context.request.request)
    return content_id({'pair_inputs_id':context.saved('pair-inputs').outcome.scientific_id,
        'evidence_id':context.saved('coordination-evidence').outcome.scientific_id,'settings':settings,
        'inference':context.request.request.inference,'implementation':implementation(context.request.request.questions['rhythm']['enabled'])})


def comparisons(units,definitions,settings,inference):
    """Independent-unit contrasts retain all requested hypotheses and exclusions."""
    import pymicroglia.workbench as circadian
    rows=[];details=[];buckets={}
    for definition in definitions.to_dict('records'):
        if definition['value_kind']=='native_rhythm_timing':continue
        group=units.loc[units.question_id.eq(definition['question_id'])]
        hypothesis=definition.get('is_hypothesis') is not False and definition['value_kind']!='descriptive_overlap_maximum_absolute_lag_association'
        formal=hypothesis and settings['evidence']['method']!='none'
        for reference,target in settings['contrasts']:
            relevant=group.loc[group.condition.isin([reference,target])]
            complete=relevant.loc[relevant.value.notna()]
            usable=complete.loc[complete.formal_eligible] if formal else complete.loc[complete.sample_confirmed]
            a,b=[usable.loc[usable.condition.eq(name)] for name in [reference,target]]
            if usable.unit_id.duplicated().any() or set(a.unit_id)&set(b.unit_id):raise ValueError('A biological sample enters an independent contrast more than once')
            chosen=settings['evidence'] if formal else {'method':'none'}
            result=native.compare(a.value.tolist(),b.value.tolist(),chosen)
            cid=content_id({'question_id':definition['question_id'],'reference_condition':reference,'target_condition':target})
            descriptive=[complete.loc[complete.condition.eq(name),'value'] for name in [reference,target]]
            key={'comparison_id':cid,'question_id':definition['question_id'],'question':definition['question'],'value_kind':definition['value_kind'],
                'reference_condition':reference,'target_condition':target,'method':chosen['method'],'formal_hypothesis':formal,'aggregation':settings['aggregation']}
            used=set(usable.unit_id)
            row={**key,**{name:result[name] for name in ['effect','p_value','effect_interval','interval_status','status','reason','reference_samples','target_samples']},
                'descriptive_effect_all_recorded_units':float(descriptive[1].mean()-descriptive[0].mean()) if all(len(values) for values in descriptive) else None,
                'descriptive_scope':'All available unit summaries, including explicitly unconfirmed recordings and model-development populations; not the formal biological-sample estimate',
                'eligible_units':sorted(used),'excluded_units':relevant.loc[~relevant.unit_id.isin(used),['unit_id','sample_confirmed','status','reason','inference_exclusions']].to_dict('records'),
                'q_value':None,'supported':False,'family_id':None,'family_requested':0,'family_tested':0,'alpha':None,'correction':None}
            rows.append(row);details.append({**key,'evidence':result,'reference_unit_ids':a.unit_id.tolist(),'target_unit_ids':b.unit_id.tolist()})
            if formal:buckets.setdefault(definition['question'] if inference['correction_scope']=='question' else 'all',[]).append(row)
    families=[]
    for question,members in sorted(buckets.items()):
        members.sort(key=lambda row:row['comparison_id']);ids=[row['comparison_id'] for row in members]
        alpha=inference['alpha'];method=inference['multiple_testing'];valid=[row['p_value'] is not None for row in members]
        family=content_id({'evidence_level':'biological_sample','question':question,'members':ids,'alpha':alpha,'correction':method})
        adjusted=circadian.adjust_pvalues([row['p_value'] if ok else 1. for row,ok in zip(members,valid)],method) if any(valid) else [None]*len(members)
        for row,ok,q in zip(members,valid,adjusted):
            row.update(q_value=float(q) if ok else None,supported=bool(ok and q<=alpha),family_id=family,
                family_requested=len(members),family_tested=sum(valid),alpha=alpha,correction=method)
            if ok:row.update(status='detected_difference' if row['supported'] else 'no_detected_difference',
                reason='Corrected independent biological-sample contrast under the declared common-distribution null' if row['supported'] else 'The corrected sample test did not detect a difference; absence is not established')
        families.append({'family_id':family,'evidence_level':'biological_sample','question':question,'requested':len(members),'tested':sum(valid),
            'unavailable':len(members)-sum(valid),'members':ids,'alpha':alpha,'correction':method,
            'missing_probability_policy':'Unavailable requested comparisons retain a probability-one slot only during correction; saved p/q remain null'})
    return _table(rows,['comparison_id','question_id','question','reference_condition','target_condition','effect','p_value','q_value','supported','status','reason']),\
        _table(families,['family_id','evidence_level','question','requested','tested','unavailable','members','alpha','correction']),details


def produce(context):
    request=context.request.request;settings=policy(request)
    if not settings['enabled']:return StepResult(context.step.name,context.scientific_id,'skipped-empty','Biological-sample summaries were explicitly disabled')
    prepared=read_inputs(context.saved('pair-inputs'));evidence=read_evidence(context.saved('coordination-evidence'),context.saved('pair-inputs').outcome.scientific_id)
    source=load_states(context.request) if request.questions['states']['enabled'] and evidence['effects'].question.eq('states').any() else None
    members,definitions=member_records(context.request,prepared,evidence,source)
    outputs=aggregate(context.request,prepared,members,definitions,settings['aggregation']);outputs.update(effect_members=members,definitions=definitions)
    results,families,details=comparisons(outputs['unit_summaries'],definitions,settings,request.inference)
    outputs.update(comparisons=results,families=families)
    from pymicroglia.pipelines.coordination.sample_timing import summarise
    timing,timing_details,timing_policy=summarise(context.request,evidence,settings['timing'])
    outputs.update(timing)
    context.output.mkdir(parents=True);refs=[]
    for name,frame in outputs.items():
        path=context.output/(name+'.json');path = write_table(path,frame)
        refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    selections=(SelectionRecord('supported-sample-comparisons',context.scientific_id,Settings({'families':families.family_id.tolist(),
        'scope':'Only independent biological-sample contrasts; pair support and report selection were not used'}),
        tuple(Settings({'comparison_id':row['comparison_id'],'question_id':row['question_id']}) for row in results.to_dict('records') if row['supported'])),)
    provenance={'schema_version':1,'scientific_id':context.scientific_id,'pair_inputs_id':context.saved('pair-inputs').outcome.scientific_id,
        'evidence_id':context.saved('coordination-evidence').outcome.scientific_id,'settings':settings,'inference':request.inference,
        'implementation':implementation(request.questions['rhythm']['enabled']),'timing_policy':timing_policy,
        'scalar_weighting':settings['aggregation']+' over complete compatible finite effects within each recording, then '+settings['aggregation']+' over available recording summaries within each biological sample',
        'counts':'Cell counts are recording-qualified observations; several recordings of the same sample remain one biological experimental unit',
        'selection':'Complete original effects including non-significant and untestable results; scientific support and display limits do not define the population',
        'formal_scope':'One finite predeclared aggregate per confirmed independent sample; original model-development populations and display baselines remain descriptive',
        'uncertainty':'Ordinary contrasts concern observed sample summaries, with no latent coupling/measurement-error propagation. Optional percentile intervals are not simultaneous family intervals.',
        'timing':'Native compatible distinct-role timing summaries are separate from ordinary linear sample contrasts; no local circular calculation or direct condition-phase test',
        'original_pair_tests_repeated':False,'state_models_fitted':False,'sample_selections':[item.as_dict() for item in selections]}
    for name,document in [('provenance',provenance),('comparison_details',details),('timing_details',timing_details)]:
        path=context.output/(name+'.json');_write_json(path,document);refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed','Saved complete recording and biological-sample summaries, independent sample contrasts and native timing limits',tuple(refs),selections)


def read_samples(saved,expected_evidence=None):
    provenance=read_document(saved.artifact('provenance'))
    if provenance.get('schema_version')!=1 or provenance.get('scientific_id')!=saved.outcome.scientific_id:raise ValueError('Coordination sample schema/identity mismatch')
    if expected_evidence is not None and provenance['evidence_id']!=expected_evidence:raise ValueError('Sample summaries belong to another complete evidence result')
    tables={ref.name:read_table(saved.artifact(ref.name)) for ref in saved.outcome.artifacts if ref.name not in {'provenance','comparison_details','timing_details'}}
    if tables['unit_inventory'].unit_id.duplicated().any() or tables['unit_summaries'].duplicated(['unit_id','question_id']).any():raise ValueError('Sample output repeats a biological unit/question')
    if [item.as_dict() for item in saved.outcome.selections]!=provenance['sample_selections']:raise ValueError('Sample selections differ from their frozen provenance')
    tables['provenance']=provenance
    return tables
