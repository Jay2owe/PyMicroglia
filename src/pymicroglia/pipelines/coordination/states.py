"""Accepted-state co-occupancy and observed switching between measured cells."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path

import numpy as np
import pandas as pd

import pymicroglia.measure.relationship_statistics as native
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import _table, read_inputs
from pymicroglia.pipelines.coordination.state_inputs import cell_pairs, joint_exposure, load_states, switch_events, transition_support
from pymicroglia.pipelines.coordination.simultaneous import temporal_refusal
from pymicroglia.pipelines.relationships.inputs import match_observations, paired_support
from pymicroglia.pipelines.relationships.options import _number
from pymicroglia.pipelines.rhythm.discovery import _known_keys, _object
from pymicroglia.pipelines._runner import Unavailable
from pymicroglia.pipelines._screening import _write_json, file_hash, write_table


KEYS = ['source_run','movie','reference_identity','target_identity','state_pair_id','model_id','sample','sample_confirmed','condition']


def options(question, support, definitions=None):
    if question['statistic']!='phi' or question['evidence']['method'] not in {'none','truncated_time_shift'}:
        raise Unavailable('Accepted-state coordination supports phi (binary-indicator Pearson correlation) with optional conservative temporal independence evidence')
    native.validate_question({'enabled':True,'statistic':'pearson','evidence':question['evidence']})
    declared=_object(question['settings'],'states.settings')
    _known_keys(declared,{'occupancy','switching','state_pairs','coincidence_tolerance_hours','min_joint_hours','min_observations'},'states.settings')
    result={'occupancy':True,'switching':True,'state_pairs':'all','coincidence_tolerance_hours':0.,
        'min_joint_hours':support['min_span_hours'],'min_observations':support['min_observations'],**declared}
    if any(not isinstance(result[key],bool) for key in ['occupancy','switching']):raise ValueError('State occupancy/switching choices must be boolean')
    if not result['occupancy'] and not result['switching']:raise ValueError('Enable state occupancy or switching comparison')
    for key in ['coincidence_tolerance_hours','min_joint_hours']:
        result[key]=_number(result[key],'states.settings.'+key,minimum=0)
    result['min_observations']=_number(result['min_observations'],'states.settings.min_observations',minimum=2,integer=True)
    pairs=result['state_pairs']
    if not (isinstance(pairs,str) and pairs in {'all','matching'}):
        if not isinstance(pairs,list) or not pairs:raise ValueError('state_pairs requires all, matching or explicit accepted-state-ID pairs')
        if any(not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(not isinstance(v,str) for v in pair) for pair in pairs):
            raise ValueError('Explicit state pairs require two complete accepted state IDs')
        pairs=list(dict.fromkeys(tuple(pair) for pair in pairs));result['state_pairs']=[list(pair) for pair in pairs]
    if definitions is not None:
        states=definitions.state_id.tolist()
        if pairs=='all':selected=[(a,b) for a in states for b in states]
        elif pairs=='matching':selected=[(a,a) for a in states]
        else:
            selected=[tuple(pair) for pair in pairs]
            if any(a not in states or b not in states for a,b in selected):raise ValueError('Requested categorical label belongs to another model or is not an accepted state')
        result['resolved_state_pairs']=selected
    return result


def implementation_version():
    return {'code':{path.name:file_hash(path) for path in [Path(__file__),Path(native.__file__),
        source_file('coordination_state_inputs.py'),source_file('coordination_sources.py'),
        source_file('coordination_simultaneous.py'),source_file('relationship_inputs.py')]},
        'libraries':{name:library_version(name) for name in ['numpy','pandas','scipy']}}


def identity(context):
    q=context.request.request.questions['states']
    if not q['enabled']:return content_id({'enabled':False,'code':file_hash(__file__)})
    options(q,context.request.request.support);source=load_states(context.request)
    return content_id({'pair_inputs':context.saved('pair-inputs').outcome.scientific_id,
        'source':{key:{'scientific_id':value['scientific_id'],'artifacts':value['artifacts']} for key,value in source['bindings'].items()},
        'model_id':source['model_id'],'question':{key:value for key,value in q.items() if key!='source'},'implementation':implementation_version()})


def indicator_series(assignments, state_id, clock_valid):
    frame=assignments.sort_values('frame_index').copy()
    frame['sequence_index']=np.arange(len(frame));frame['clock_valid']=clock_valid
    good=frame.status.eq('assigned') & frame.within_range & frame.time_status.eq('recorded')
    frame['processed_value']=frame.state_id.eq(state_id).astype(float).where(good,np.nan)
    frame['processed_valid']=good & clock_valid
    return frame


def switching_series(steps, assignments, clock_valid):
    frame=steps.sort_values('source_hours').copy();lookup=assignments.set_index('observation_id').frame_index.to_dict()
    frame['observation_id']=frame.step_id;frame['frame_index']=frame.target_observation_id.map(lookup)
    frame=frame.sort_values('frame_index');frame['sequence_index']=np.arange(len(frame))
    frame['hours']=(frame.source_hours+frame.target_hours)/2.;frame['clock_valid']=clock_valid
    frame['within_range']=np.isfinite(frame.hours.to_numpy(dtype=float,na_value=np.nan))
    frame['processed_value']=frame.counted_switch.astype(float).where(frame.valid_transition_opportunity,np.nan)
    frame['processed_valid']=frame.valid_transition_opportunity & clock_valid
    return frame


def model_unused(frame, inventory):
    if frame.empty:return False,'This endpoint has no saved assignment observations'
    if not frame.role.eq('assignment_only').all():return False,'This endpoint was used for learning or state-model support; only descriptive coordination is reported'
    confirmed=frame.loc[frame.sample_confirmed,'sample'].dropna().unique()
    used=inventory.loc[~inventory.role.eq('assignment_only') & inventory.sample_confirmed,'sample'].dropna()
    if any(sample in set(used) for sample in confirmed):return False,'Another recording of this biological sample was used for model learning/support'
    return True,'Endpoint and any confirmed biological sample were reserved for applying the frozen accepted model'


def indicator_evidence(pair, left, right, question, settings, support, joint_hours, unused, kind, state_pair=None):
    matched,diagnostic=match_observations(left,right,0.,support);coverage=paired_support(matched,diagnostic,
        {**support.as_dict(),'min_observations':settings['min_observations'],'min_span_hours':settings['min_joint_hours']})
    effect,reason=native.coefficient(matched.reference_value,matched.target_value,'pearson')
    row={**pair,'question':'states','evidence_level':'pair','evidence_kind':kind,'statistic':'phi','is_hypothesis':True,
        'reference_state_id':state_pair[0] if state_pair else None,'target_state_id':state_pair[1] if state_pair else None,
        **coverage,'effect':effect,'descriptive_effect':effect,'p_value':None,
        'status':'descriptive' if effect is not None else 'unresolvable','reason':reason,
        'method':question['evidence']['method'],'applied_method':'none','evidence_settings':question['evidence'],
        'confidence_low':None,'confidence_high':None,'uncertainty_status':'not_provided_by_method',
        'joint_support_hours':joint_hours,'model_unused_by_both_endpoints':all(item[0] for item in unused),
        'model_population_reasons':[item[1] for item in unused],
        'effect_population':'Matched original binary membership observations' if kind=='state_cooccupancy' else 'Matched original eligible transition-interval indicators; distinct from tolerance-based event matching'}
    if coverage['status']!='eligible' or joint_hours<settings['min_joint_hours']:
        row.update(status='insufficient',reason='Insufficient original joint observations or physical state/transition exposure')
    if not pair['geometry_population_eligible']:row.update(status='outside_geometry_population',reason='Outside the declared scientific geometry population')
    if row['status']=='descriptive' and question['evidence']['method']!='none':
        if not row['model_unused_by_both_endpoints']:
            row.update(status='descriptive_model_used',reason='State definition and these endpoints are not an unused model-application population')
        else:
            refusal=temporal_refusal(left,right,support)
            if refusal:row.update(status='untestable',reason=refusal)
            else:
                result=native.same_time_evidence(left,right,question={'enabled':True,'statistic':'pearson','evidence':question['evidence']},
                    support={**support.as_dict(),'min_observations':settings['min_observations'],'min_span_hours':settings['min_joint_hours']})
                for key in ['effect','p_value','status','reason','tested_observations','tested_start_hours','tested_end_hours','minimum_attainable_p','effect_population']:
                    if key in result:row[key]=result[key]
                row.update(applied_method=question['evidence']['method'],native_details=result)
    return row,matched


def event_coincidence(left, right, tolerance):
    def trace(frame):
        return frame.assign(hours=frame.event_hours,observation_id=frame.event_id,processed_value=1.,
            processed_valid=frame.eligible_for_coincidence,within_range=True)
    settings={'matching':'nearest_unique','matching_tolerance_hours':tolerance}
    matched,diagnostic=match_observations(trace(left),trace(right),0.,settings)
    rows=[{'reference_event_id':row['reference_observation'],'target_event_id':row['target_observation'],
        'reference_midpoint_hours':row['reference_hours'],'target_midpoint_hours':row['target_hours'],
        'target_minus_reference_hours':row['matching_error_hours']} for row in matched.to_dict('records')]
    a,b=int(left.eligible_for_coincidence.sum()),int(right.eligible_for_coincidence.sum())
    return rows,{'reference_events':len(left),'target_events':len(right),'reference_eligible_events':a,'target_eligible_events':b,
        'matched_events':len(rows),'ambiguous_reference_events':diagnostic['ambiguous_observations'],
        'reference_matched_fraction':len(rows)/a if a else None,'target_matched_fraction':len(rows)/b if b else None,
        'coincidence_tolerance_hours':tolerance,'matching_rule':'nearest_unique; ties and target collisions excluded; each original step at most once',
        'coincidence_p_value':None,'coincidence_meaning':'Descriptive interval-midpoint coincidence under the declared tolerance; true switch times remain bounded, not known exactly'}


def analyse(resolved, prepared, source, scientific_id):
    q=resolved.request.questions['states'];settings=options(q,resolved.request.support,source.get('definitions'))
    pairs,memberships=cell_pairs(prepared['inventory'],source['model_id'],source['bindings']['support']['scientific_id'])
    summaries=[];evidence=[];exposures=[];occupancy=[];events=[];coincidences=[];observations=[];transition_intervals=[]
    if source['status']=='accepted':
        key=['source_run','movie','identity'];groups={tuple(k):frame for k,frame in source['assignments'].groupby(key,sort=False)}
        time_groups={name:{tuple(k):frame for k,frame in source['time'][name].groupby(key,sort=False)} for name in ['exposures','steps','cell_statistics']}
    for pair in pairs:
        if source['status']!='accepted':
            summaries.append({**pair,'status':'unaccepted_model','reason':source['reason'],'joint_observed_hours':None,'joint_assigned_hours':None})
            continue
        locals_=[];unused=[];clocks=[]
        for role in ['reference','target']:
            cell=(pair['source_run'],pair['movie'],pair[role+'_identity'])
            assignments=groups.get(cell,source['assignments'].iloc[:0]);statistics=time_groups['cell_statistics'].get(cell,source['time']['cell_statistics'].iloc[:0])
            clock_valid=bool(len(statistics) and not statistics.status.eq('invalid_clock_order').any())
            locals_.append({'assignments':assignments,'exposures':time_groups['exposures'].get(cell,source['time']['exposures'].iloc[:0]),
                'steps':time_groups['steps'].get(cell,source['time']['steps'].iloc[:0])})
            unused.append(model_unused(assignments,source['inventory']));clocks.append(clock_valid)
        left,right=locals_;joint=joint_exposure(left['exposures'],right['exposures'])
        total=sum(row['duration_hours'] for row in joint);observed=sum(row['duration_hours'] for row in joint if row['jointly_observed'])
        assigned=sum(row['duration_hours'] for row in joint if row['jointly_assigned']);same=sum(row['duration_hours'] for row in joint if row['same_state'] is True)
        summary={**pair,'status':'descriptive' if observed>0 else 'insufficient','reason':'Original joint state exposure' if observed>0 else 'No jointly observed physical state exposure',
            'joint_reference_hours':total,'joint_observed_hours':observed,'joint_assigned_hours':assigned,
            'joint_unknown_hours':observed-assigned,'joint_unobserved_hours':total-observed,'same_state_hours':same,
            'same_state_fraction_observed':same/observed if observed else None,'same_state_fraction_assigned':same/assigned if assigned else None,
            'agreement_p_value':None,'agreement_meaning':'Duration-weighted agreement of the accepted saved states; per-state indicator tests are separate hypotheses',
            'model_accepted':True,'model_unused_by_both_endpoints':all(value[0] for value in unused)}
        exposures.extend({**pair,**row} for row in joint)
        for a in source['definitions'].state_id:
            for b in source['definitions'].state_id:
                duration=sum(row['duration_hours'] for row in joint if row['jointly_assigned'] and row['reference_state_id']==a and row['target_state_id']==b)
                occupancy.append({**pair,'reference_state_id':a,'target_state_id':b,'joint_state_hours':duration,
                    'joint_observed_hours':observed,'joint_assigned_hours':assigned,
                    'fraction_observed':duration/observed if observed else None,'fraction_assigned':duration/assigned if assigned else None})
        def save_evidence(a,b,hours,kind,state_pair=None):
            row,matched=indicator_evidence(pair,a,b,q,settings,resolved.request.support,hours,unused,kind,state_pair)
            result_id=content_id({'scientific_id':scientific_id,'state_pair_id':pair['state_pair_id'],'kind':kind,'states':state_pair})
            evidence.append({**row,'result_id':result_id})
            observations.extend({**pair,'result_id':result_id,'evidence_kind':kind,**item} for item in matched.to_dict('records'))
        if settings['occupancy']:
            for a,b in settings['resolved_state_pairs']:
                save_evidence(indicator_series(left['assignments'],a,clocks[0]),indicator_series(right['assignments'],b,clocks[1]),assigned,'state_cooccupancy',(a,b))
        if settings['switching']:
            support=transition_support(left['steps'],right['steps']);transition_intervals.extend({**pair,**row} for row in support)
            first,second=switch_events(left['steps'],support),switch_events(right['steps'],support)
            matched,event_summary=event_coincidence(first,second,settings['coincidence_tolerance_hours']);summary.update(event_summary)
            for role,frame in [('reference',first),('target',second)]:events.extend({**pair,'endpoint_role':role,**row} for row in frame.to_dict('records'))
            coincidences.extend({**pair,**row} for row in matched)
            save_evidence(switching_series(left['steps'],left['assignments'],clocks[0]),switching_series(right['steps'],right['assignments'],clocks[1]),
                sum(row['duration_hours'] for row in support),'switch_process')
        summaries.append(summary)
    return {'pair_summary':_table(summaries,KEYS+['status','reason','joint_observed_hours','joint_assigned_hours']),
        'pair_effects':_table(evidence,KEYS+['result_id','question','evidence_level','evidence_kind','effect','p_value','status','reason']),
        'pair_membership':_table(memberships,[*prepared['inventory'].columns,'state_pair_id','model_id']),
        'joint_exposure':_table(exposures,KEYS+['start_hours','end_hours','duration_hours','jointly_observed','jointly_assigned','support']),
        'joint_occupancy':_table(occupancy,KEYS+['reference_state_id','target_state_id','joint_state_hours','fraction_observed','fraction_assigned']),
        'switch_events':_table(events,KEYS+['endpoint_role','event_id','event_hours','earliest_hours','latest_hours','eligible_for_coincidence']),
        'switch_matches':_table(coincidences,KEYS+['reference_event_id','target_event_id','reference_midpoint_hours','target_midpoint_hours']),
        'indicator_observations':_table(observations,KEYS+['result_id','evidence_kind','reference_observation','target_observation','reference_hours','target_hours']),
        'joint_transition_support':_table(transition_intervals,KEYS+['start_hours','end_hours','duration_hours'])},settings


def produce(context):
    q=context.request.request.questions['states']
    if not q['enabled']:return StepResult(context.step.name,context.scientific_id,'skipped-empty','Optional state comparison was disabled; no state source was opened')
    options(q,context.request.request.support);source=load_states(context.request);prepared=read_inputs(context.saved('pair-inputs'))
    outputs,settings=analyse(context.request,prepared,source,context.scientific_id)
    context.output.mkdir(parents=True);refs=[]
    for name,frame in outputs.items():
        path=context.output/(name+'.json');path = write_table(path,frame)
        refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    path=context.output/'provenance.json'
    _write_json(path,{'schema_version':1,'scientific_id':context.scientific_id,'pair_inputs_id':context.saved('pair-inputs').outcome.scientific_id,
        'source_bindings':source['bindings'],'model_id':source['model_id'],'model_status':source['status'],'model_decision':source['decision'],
        'settings':settings,'evidence':q['evidence'],'implementation':implementation_version(),'native_reference':native.REFERENCE,
        'state_model_fitted':False,'assignments_changed':False,'source_time_statistics_recomputed':False,
        'state_transformation':source.get('assignment_provenance',{}).get('transformation'),
        'source_time_provenance':source.get('time_provenance'),
        'time_agreement':'Descriptive original duration-weighted joint state occupancy, with separate observed and assigned denominators',
        'indicator_evidence':'Per-state binary membership and exact eligible-interval switch-process phi/time-shift tests; these are not an overall agreement or tolerance-match probability',
        'model_population':'Inferential outcomes require unused model-application endpoints, excluding any known biological sample used for learning/support',
        'event_time':'Unique saved original transition steps; midpoint comparison retains original event-time bounds and unresolved within-interval timing',
        'families':'Unique oriented cell/model state hypotheses; repeated measurement pair memberships do not create repeated tests. Stage 09 corrects all declared hypotheses.'})
    refs.append(ArtifactRef('provenance',path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed',
        'Saved accepted-state joint exposure, unique switch comparisons and separate persistence-aware indicator evidence' if source['status']=='accepted' else 'Saved explicit unaccepted-model diagnostics; no state assignment or comparison was performed',tuple(refs))
