"""Public Workbench summaries for compatible, distinct ordered endpoint roles."""
from __future__ import annotations

from types import SimpleNamespace

from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines.coordination.sources import load_source
from pymicroglia.pipelines.rhythm.agreement import independence
from pymicroglia.pipelines._screening import _json_value, read_screen
from pymicroglia.pipelines.rhythm.timing_samples import _in_stratum, validate_options


def pair_names(native):
    """The supplied native record is a measured cell pair, not one cell."""
    return {key.replace('cells','pair_records').replace('cell','pair'):
        value.replace('cells','pair records').replace('cell','pair') if isinstance(value,str) else value for key,value in native.items()}


def summarise(resolved,evidence,declared):
    import pymicroglia.workbench as circadian
    empty={'timing_populations':_table([],['timing_question_id','reference','target','scope','stratum','status','reason']),
        'timing_units':_table([],['timing_question_id','unit','sample','sample_confirmed','status','reason']),
        'timing_members':_table([],['timing_question_id','effect_id','pair_id','unit','summary_eligible','reason'])}
    if not resolved.request.questions['rhythm']['enabled']:return empty,[],{'enabled':False}
    settings=validate_options(dict(declared));frame=evidence['effects'].loc[evidence['effects'].question.eq('rhythm')]
    requested_pairs=sorted({(pair['reference'],pair['target']) for pair in resolved.request.measurement_pairs})
    conditions=sorted(set(resolved.request.conditions.values())|{name for pair in resolved.request.sample_summary['contrasts'] for name in pair})
    scopes=[('all',None),*(('condition:'+condition,condition) for condition in conditions)]
    count=len(requested_pairs)*len(settings['period_strata'])*len(scopes)
    options=settings['summary_options'];confidence=options.get('confidence',.95)
    if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not 0<confidence<1:raise ValueError('Timing summary confidence must be a finite fraction between zero and one')
    effective=1-(1-confidence)/max(1,count) if settings['interval_correction']=='bonferroni' else confidence
    options={**options,'confidence':effective};assignments={item.movie:item for item in resolved.inputs.samples}
    populations=[];units=[];members=[];details=[];screen=None
    for reference,target in requested_pairs:
        group=frame.loc[frame.reference.eq(reference)&frame.target.eq(target)]
        independent=False;reasons=[];dependencies=[]
        if reference!=target:
            if screen is None:
                binding=evidence['provenance']['branch_provenance']['rhythm-coordination']['source_binding']
                source,_=load_source({key:binding[key] for key in ['execution_record','scientific_id']},recipe='rhythm-discovery',step='rhythm-screen')
                screen=read_screen(source.root,expected_id=source.outcome.scientific_id)
            independent,reasons,dependencies=independence(screen,SimpleNamespace(reference=reference,target=target),assignments)
        for scope,condition in scopes:
            subset=group if condition is None else group.loc[group.condition.eq(condition)]
            for stratum in settings['period_strata']:
                key={'reference':reference,'target':target,'scope':scope,'condition':condition,'stratum':stratum['name'],'period_stratum_hours':stratum['period_hours']}
                key['timing_question_id']=content_id(key);records=[];annotations={};unit_info={}
                for row in _json_value(subset.to_dict('records')):
                    assignment=assignments[row['movie']]
                    unit=content_id({'source_run':resolved.inputs.source_run,'level':'biological_sample' if assignment.confirmed else 'recording',
                        'sample':assignment.sample if assignment.confirmed else assignment.movie})
                    unit_info[unit]={'sample':assignment.sample,'sample_confirmed':assignment.confirmed,'unit_level':'biological_sample' if assignment.confirmed else 'unconfirmed_recording'}
                    timing=dict(row['native_timing_evidence']);inside=_in_stratum(timing,stratum['period_hours'])
                    if not inside:timing.update(status='ineligible',reason='Original native period uncertainty is not wholly inside the declared half-open stratum')
                    records.append({'id':row['effect_id'],'unit':unit,'reference':reference,'target':target,'timing':timing})
                    annotations[row['effect_id']]={'effect_id':row['effect_id'],'pair_id':row['pair_id'],'source_run':row['source_run'],'movie':row['movie'],
                        'reference_identity':row['reference_identity'],'target_identity':row['target_identity'],'unit':unit,'in_declared_stratum':inside}
                if reference==target:
                    reason='Identical measurements on numerically ordered cells do not define comparable biological endpoint roles for a pooled signed phase offset; original pair timing remains available'
                    populations.append({**key,'status':'incomparable_endpoint_roles','reason':reason,'requested_pair_records':len(records),'requested_units':len(unit_info),
                        'mean_offset_hours':None,'resultant':None,'independent_units':False,'sampling_region':None,'repetition_status':'unresolved'})
                    units.extend({**key,**info,'unit':unit,'status':'incomparable_endpoint_roles','reason':reason} for unit,info in unit_info.items())
                    members.extend({**key,**annotations[row['id']],'summary_eligible':False,'within_unit_comparable':False,'across_unit_comparable':False,'reason':reason} for row in records)
                    details.append({**key,'native_called':False,'reason':reason});continue
                result=circadian.rhythm_timing_summary(records,independent_units=independent,settings=options)
                populations.append({**key,**pair_names(result['population']),'independence_reasons':reasons,
                    'weighting':'Equal measured-pair vectors within a sample, then equal biological-sample vectors; separate from ordinary scalar recording aggregation'})
                units.extend({**key,**unit_info[row['unit']],**pair_names(row)} for row in result['units'])
                members.extend({**key,**annotations[row['id']],**row} for row in result['members'])
                details.append({**key,'native_called':True,'result':result,'decision_dependencies':dependencies,'native_record_kind':'original measured cell pair'})
    return {'timing_populations':_table(populations,list(empty['timing_populations'].columns)),
        'timing_units':_table(units,list(empty['timing_units'].columns)),'timing_members':_table(members,list(empty['timing_members'].columns))},details,{
            'enabled':True,'settings':settings,'effective_summary_options':options,'interval_families':count,'workbench_version':circadian.WORKBENCH_VERSION,
            'native_scope':'Conditional comparable distinct ordered measurement roles; no arbitrary renaming of identical measurements into biological roles',
            'weighting':'Equal original pair records within each sample and equal sample vectors across samples; pair records sharing cells are not independent replicates',
            'contrast_scope':'No direct condition-phase contrast or ordinary linear test on angles/concentration; original and per-condition native uncertainty remain separate',
            'no_shared_tissue_clock':True,'source_pair_timing_recomputed':False}
