"""Original measured response/recovery episodes with sampling and censoring bounds."""
from pymicroglia._results import read_document
from pathlib import Path
from importlib.metadata import version
import json
import numpy as np
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines.intervention.windows import finite, read_windows
from pymicroglia.pipelines.intervention.evidence import read_evidence
import pymicroglia.pipelines.intervention.timing_rules as rules
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table

TABLES=['timing','criteria','episodes','episode_members','support_intervals']


def policy(resolved):
    return rules.policy(resolved.request.timing.as_dict(),[m.column for m in resolved.measurements])


def identity(context):
    if not context.request.request.timing['enabled']:return content_id({'enabled':False,'code':file_hash(__file__)})
    return content_id({'windows':context.saved('aligned-windows').outcome.scientific_id,'evidence':context.saved('response-evidence').outcome.scientific_id,
        'settings':policy(context.request),'code':{path.name:file_hash(path) for path in [Path(__file__),Path(rules.__file__)]},
        'libraries':{name:version(name) for name in ['numpy','pandas','scipy']}})


def criterion(value,baseline,threshold,direction):
    if not finite(value):return False
    delta=value-baseline
    return delta>=threshold if direction=='increase' else delta<=-threshold if direction=='decrease' else abs(delta)<=threshold


def scan(prepared,evidence,resolved,settings):
    windows={row['window_id']:row for row in _json_value(prepared['windows'].to_dict('records'))}
    records={row['movie']:row for row in _json_value(prepared['recordings'].to_dict('records'))}
    traces=prepared['traces'];membership=prepared['window_members'];max_gap=resolved.request.support['max_gap_hours']
    summaries=[];criteria=[];episodes=[];members=[];intervals=[]
    for effect in _json_value(evidence['effects'].to_dict('records')):
        baseline,target=windows[effect['baseline_window_id']],windows[effect['target_window_id']]
        row={key:effect[key] for key in ['source_run','movie','identity','measurement','measurement_id','effect_id','comparison_id','baseline_window_id','target_window_id',
            'baseline','target_window','summary_operation','anchor_hours','anchor_kind','anchor_label','sample','sample_confirmed','condition']}
        tid=content_id({'comparison':effect['comparison_id'],'settings':settings});row.update(timing_id=tid,method=settings['method'],
            within_cell_outcome=effect['outcome'],within_cell_supported=effect['response_supported'],original_unit=effect['unit'],
            baseline_value=baseline['summary_value'],baseline_status=baseline['status'],baseline_reference='saved_window_summary',
            original_change=effect['absolute_change'],target_window_eligible=target['eligible'],
            response_observed=False,recovery_observed=False,response_delay_hours=None,response_lower_hours=None,response_upper_hours=None,
            recovery_delay_hours=None,recovery_lower_hours=None,recovery_upper_hours=None,response_episode_id=None,recovery_episode_id=None,
            response_timing_status='unavailable',recovery_status='unavailable',response_label=None,recovery_censored=False,
            observed_endpoint_hours=None,observed_endpoint_relative_hours=None,expected_endpoint_hours=target.get('last_captured_hours'),
            clock_source=records[effect['movie']]['clock_source'],observation_status='unavailable',gap_count=0,
            interval_meaning=rules.BOUND_MEANING,probability=None,
            timing_scope='First qualifying observed criterion episode within the requested follow-up after the declared anchor; earlier or unobserved events are not excluded',
            recovery_meaning='Observed sustained return within the declared baseline tolerance; not a statistical equivalence or absence-of-effect conclusion')
        threshold=settings['thresholds'].get(effect['measurement'])
        refusal=None
        if threshold is None:refusal='timing_not_requested_for_measurement'
        elif baseline['input_kind']!='original_trace' or target['input_kind']!='original_trace':refusal='original_timing_observations_unavailable'
        elif not baseline['eligible'] or not finite(baseline['summary_value']):refusal='insufficient_original_baseline'
        elif baseline['coordinate']=='frames' and (not finite(baseline['last_observed_hours']) or baseline['last_observed_hours']>=effect['anchor_hours']):refusal='baseline_not_established_before_anchor'
        elif baseline['coordinate']!='frames' and baseline['end_hours']>effect['anchor_hours']:refusal='baseline_not_established_before_anchor'
        if threshold is not None:
            if effect['unit'] and threshold['unit']!=effect['unit']:raise ValueError('Timing threshold unit differs from the original measurement unit')
            row.update(unit=threshold['unit'],unit_status='recorded_source_unit' if effect['unit'] else 'user_declared_unit_for_original_unlabelled_values',
                response_threshold=threshold['change'],requested_direction=threshold['direction'],response_persistence_hours=settings['persistence_hours'])
        if refusal:row.update(status=refusal,reason=refusal);summaries.append(row);continue
        full=traces.loc[traces.movie.eq(effect['movie'])&traces.identity.eq(effect['identity'])&traces.measurement_id.eq(effect['measurement_id'])].sort_values('sequence_index',kind='stable')
        full_rows=_json_value(full.to_dict('records'));source={item['observation_id']:item for item in full_rows}
        ids=membership.loc[membership.window_id.eq(target['window_id']),'observation_id'].tolist()
        if len(set(ids))!=len(ids) or set(ids)-set(source):raise ValueError('Timing source lost original follow-up observation membership')
        chosen=set(ids);all_rows=[item for item in full_rows if item['observation_id'] in chosen]
        observed=[item for item in all_rows if finite(item['relative_hours']) and item['relative_hours']>=0]
        valid=[item for item in observed if item['raw_valid'] and item['clock_valid'] and finite(item['raw_value'])]
        response_directions=['increase','decrease'] if threshold['direction']=='either' else [threshold['direction']]
        recovery=settings['recovery'];tolerance=recovery['tolerances'][effect['measurement']]['value'] if recovery['enabled'] else None
        row.update(recovery_tolerance=tolerance,recovery_persistence_hours=recovery.get('persistence_hours'),
            recovery_status='not_requested' if not recovery['enabled'] else 'no_qualifying_response',original_target_observations=len(all_rows),valid_post_anchor_observations=len(valid))
        for item in all_rows:
            known=bool(item['raw_valid'] and item['clock_valid'] and finite(item['raw_value']));after=finite(item['relative_hours']) and item['relative_hours']>=0
            criteria.append({'timing_id':tid,**{key:item.get(key) for key in ['source_run','movie','identity','measurement','measurement_id','observation_id','hours','relative_hours','frame_index','sequence_index','raw_value']},
                'valid':known,'after_anchor':after,'baseline_value':baseline['summary_value'],'change':item['raw_value']-baseline['summary_value'] if known else None,
                'increase':criterion(item['raw_value'],baseline['summary_value'],threshold['change'],'increase') if known else None,
                'decrease':criterion(item['raw_value'],baseline['summary_value'],threshold['change'],'decrease') if known else None,
                'within_recovery_tolerance':criterion(item['raw_value'],baseline['summary_value'],tolerance,'return') if known and tolerance is not None else None})
        local=[]
        definitions=[('response',direction,threshold['change'],settings['persistence_hours']) for direction in response_directions]
        if recovery['enabled']:definitions.append(('recovery','return',tolerance,recovery['persistence_hours']))
        previous_by_id={item['observation_id']:full_rows[i-1] if i else None for i,item in enumerate(full_rows)}
        for kind,direction,change,persistence in definitions:
            marks=[bool(item['raw_valid'] and item['clock_valid'] and criterion(item['raw_value'],baseline['summary_value'],change,direction)) for item in observed]
            found=rules.episodes(observed,marks,persistence,max_gap,kind=kind,direction=direction)
            for episode in found:
                if episode['first_index']==0:
                    first=observed[0];previous=previous_by_id[first['observation_id']]
                    if previous is not None:
                        episode['previous_observation_id']=previous['observation_id']
                        if rules.adjacent(previous,first,max_gap) and not criterion(previous['raw_value'],baseline['summary_value'],change,direction):
                            episode.update(lower_hours=previous['hours'],lower_relative_hours=previous['relative_hours'],bound_status='adjacent_observations')
                episode.update(timing_id=tid,comparison_id=effect['comparison_id']);episode['episode_id']=content_id(episode);local.append(episode)
        for i,item in enumerate(observed[1:],1):
            previous=observed[i-1];adjacent=rules.adjacent(previous,item,max_gap)
            intervals.append({'timing_id':tid,'start_observation_id':previous['observation_id'],'end_observation_id':item['observation_id'],
                'start_hours':previous['hours'],'end_hours':item['hours'],'valid':adjacent,
                'observed_interval_hours':item['hours']-previous['hours'] if adjacent else 0.,'reason':'original_adjacent_valid_observations' if adjacent else 'original_value_clock_or_sequence_gap'})
        row['gap_count']=sum(not item['valid'] for item in intervals if item['timing_id']==tid)
        if valid:
            last=valid[-1];row.update(observed_endpoint_hours=last['hours'],observed_endpoint_relative_hours=last['relative_hours'])
            expected=target.get('last_captured_hours')
            if not finite(expected):row['observation_status']='original_clock_endpoint_unavailable'
            elif last['hours']<expected-1e-9:row['observation_status']='lost_observation_before_original_clock_endpoint'
            else:row['observation_status']='reached_original_clock_endpoint'
        else:row['observation_status']='no_valid_post_anchor_observation'
        candidates=sorted((item for item in local if item['kind']=='response' and item['qualified']),key=lambda item:item['start_hours'])
        response=candidates[0] if candidates else None
        if response is not None:
            row.update(response_observed=True,response_episode_id=response['episode_id'],response_direction=response['direction'],
                response_delay_hours=response['start_relative_hours'],response_lower_hours=response['lower_relative_hours'],response_upper_hours=response['upper_relative_hours'],
                response_timing_status=response['bound_status'],status='observed_response',reason='Declared change criterion persisted over adjacent original observations')
            immediate=settings.get('immediate_hours')
            if immediate is not None:
                if response['lower_relative_hours'] is None:row['response_label']='timing_unresolved'
                elif response['upper_relative_hours']<=immediate:row['response_label']='observed_within_declared_immediate_window'
                elif response['lower_relative_hours']>=immediate:row['response_label']='observed_after_declared_immediate_window'
                else:row['response_label']='immediate_boundary_unresolved'
            if recovery['enabled']:
                recovered=sorted((item for item in local if item['kind']=='recovery' and item['qualified'] and item['start_hours']>response['qualified_hours']),key=lambda item:item['start_hours'])
                if recovered:
                    first=recovered[0];row.update(recovery_observed=True,recovery_episode_id=first['episode_id'],recovery_delay_hours=first['start_relative_hours'],
                        recovery_lower_hours=first['lower_relative_hours'],recovery_upper_hours=first['upper_relative_hours'],recovery_status='observed_sustained_return',
                        status='observed_response_and_return')
                else:
                    last=valid[-1];after=[item for item in intervals if item['timing_id']==tid and item['end_hours']>=response['start_hours']]
                    returning=criterion(last['raw_value'],baseline['summary_value'],tolerance,'return')
                    continuing=criterion(last['raw_value'],baseline['summary_value'],threshold['change'],response['direction'])
                    if row['observation_status']=='lost_observation_before_original_clock_endpoint':state='recovery_unobserved_after_lost_observation'
                    elif any(not item['valid'] for item in after):state='recovery_unresolved_with_observation_gaps'
                    elif returning:state='return_observed_without_required_persistence'
                    elif continuing:
                        state='response_observed_at_recording_end' if finite(records[effect['movie']]['recorded_end_hours']) and abs(last['hours']-records[effect['movie']]['recorded_end_hours'])<1e-9 else 'response_observed_at_followup_end'
                    else:state='recovery_criterion_not_observed_by_endpoint'
                    row.update(recovery_status=state,recovery_censored=True)
        else:
            row.update(status='response_persistence_unresolved' if any(item['kind']=='response' for item in local) else 'no_observed_qualifying_response',
                reason='No declared persistent response episode was observed; this does not establish biological absence',response_timing_status='not_observed')
        episodes.extend(local)
        for episode in local:
            members.extend({'timing_id':tid,'episode_id':episode['episode_id'],'observation_id':oid,'position':i} for i,oid in enumerate(episode['observation_ids']))
        summaries.append(row)
    return {'timing':_table(summaries,['timing_id','effect_id','comparison_id','measurement','status','response_observed','recovery_observed']),
        'criteria':_table(criteria,['timing_id','observation_id','hours','raw_value','valid']),
        'episodes':_table(episodes,['timing_id','episode_id','kind','direction','qualified','start_hours','end_hours','observation_ids']),
        'episode_members':_table(members,['timing_id','episode_id','observation_id','position']),
        'support_intervals':_table(intervals,['timing_id','start_observation_id','end_observation_id','valid','observed_interval_hours'])}


def produce(context):
    settings=policy(context.request)
    if not settings['enabled']:return StepResult(context.step.name,context.scientific_id,'skipped-empty','Observed response timing and recovery were explicitly disabled')
    windows=context.saved('aligned-windows');evidence=context.saved('response-evidence')
    data=scan(read_windows(windows),read_evidence(evidence,windows.outcome.scientific_id),context.request,settings)
    context.output.mkdir(parents=True);refs=[]
    for name,frame in data.items():
        path=context.output/(name+'.json');path = write_table(path,frame);refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    path=context.output/'provenance.json';_write_json(path,{'schema_version':1,'scientific_id':context.scientific_id,'windows_id':windows.outcome.scientific_id,
        'evidence_id':evidence.outcome.scientific_id,'resolved_request':context.request.as_dict(),'applied_settings':settings,
        'libraries':{name:version(name) for name in ['numpy','pandas','scipy']},'references':rules.REFERENCES,'bound_meaning':rules.BOUND_MEANING,
        'scientific_meaning':'Observed user-declared measurement episodes, separate from stage-03 statistical response evidence; no per-frame tests, latent change-point inference or fitted recovery model',
        'persistence':'First to last adjacent original qualifying observations; no interpolation, holds across gaps or final-frame extension',
        'sampling_limit':'Between-observation excursions or earlier episodes outside the requested follow-up remain unobserved; brackets are not probabilistic uncertainty'})
    refs.append(ArtifactRef('provenance',path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed','Saved original observed threshold episodes, sampling bounds, actual recovery criteria and incomplete observation outcomes',tuple(refs))


def read_timing(saved,expected_evidence_id=None):
    provenance=read_document(saved.artifact('provenance'))
    if provenance.get('schema_version')!=1 or provenance.get('scientific_id')!=saved.outcome.scientific_id:raise ValueError('Timing identity/schema mismatch')
    if expected_evidence_id is not None and provenance['evidence_id']!=expected_evidence_id:raise ValueError('Timing uses different original response evidence')
    data={name:read_table(saved.artifact(name)) for name in TABLES}
    if data['timing'].timing_id.duplicated().any():raise ValueError('Repeated original timing comparison')
    if set(data['episodes'].timing_id)-set(data['timing'].timing_id) or set(data['episode_members'].episode_id)-set(data['episodes'].episode_id):raise ValueError('Timing episode membership lost its original result')
    data['provenance']=provenance;return data
