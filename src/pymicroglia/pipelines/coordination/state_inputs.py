"""Verified accepted-state sources, joint physical exposure and unique switches."""
from __future__ import annotations
from pymicroglia._results import read_document

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines.behaviour.assignments import read_assignments
from pymicroglia.pipelines.behaviour.durations import read_statistics
from pymicroglia.pipelines.behaviour.validation import read_support
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines.coordination.sources import load_source
from pymicroglia.pipelines._screening import file_hash, read_table


def related(binding, step, expected_id=None):
    path = Path(binding['execution_record'])
    if file_hash(path) != binding['execution_record_sha256']: raise ValueError('Pinned state execution record changed')
    record = read_document(path)
    selected = [row for row in record['steps'] if row['step'] == step]
    if len(selected) != 1: raise ValueError('Selected state execution lacks one exact '+step+' result')
    scientific_id = selected[0]['result']['scientific_id']
    if expected_id is not None and scientific_id != expected_id: raise ValueError('Related state artifact belongs to another scientific input')
    return load_source({'execution_record': str(path), 'scientific_id': scientific_id}, recipe='cell-behaviour-states', step=step)


def load_states(resolved):
    declaration = resolved.request.questions['states']['source']
    saved, binding = load_source(declaration, recipe='cell-behaviour-states', step='state-support')
    decision, model = read_support(saved)
    bindings = {'support': binding}
    if decision['status'] != 'accepted':
        candidates, candidate_binding = related(binding, 'candidate-models'); bindings['candidates'] = candidate_binding
        models = read_document(candidates.artifact('models'))
        if declaration['model_id'] not in models: raise ValueError('Requested unaccepted model is absent from the saved candidate definitions')
        candidate_provenance = read_document(candidates.artifact('provenance'))
        feature_source, bindings['features'] = related(binding, 'feature-inputs', candidate_provenance['input_id'])
        feature_provenance = matching_features(resolved, feature_source)
        return {'status': 'unaccepted_model', 'reason': decision['reason'], 'model_id': declaration['model_id'],
            'decision': decision, 'bindings': bindings, 'model': None, 'feature_provenance':feature_provenance}
    if declaration['model_id'] != model['model_id']:
        raise ValueError('Requested state labels belong to a different model; numeric component labels cannot establish comparability')
    assignment_source, bindings['assignments'] = related(binding, 'state-assignments')
    assignments, assignment_provenance = read_assignments(assignment_source, expected_model=model['model_id'])
    if assignment_provenance['support_id'] != saved.outcome.scientific_id or assignment_provenance['decision_id'] != decision['decision_id']:
        raise ValueError('State assignments refer to another acceptance decision')
    time_source, bindings['time'] = related(binding, 'durations-and-switches')
    time, time_provenance = read_statistics(time_source, expected_model=model['model_id'])
    if time_provenance['assignment_id'] != assignment_source.outcome.scientific_id or time_provenance['decision_id'] != decision['decision_id']:
        raise ValueError('State durations and switches refer to another assignment or acceptance result')
    feature_source, bindings['features'] = related(binding, 'feature-inputs', assignment_provenance['feature_input_id'])
    feature_provenance = matching_features(resolved, feature_source)
    source_window = time_provenance.get('time_range_hours')
    if (tuple(source_window) if source_window is not None else None) != resolved.request.time_range_hours:
        raise ValueError('State results have another analysis window; a matching saved assignment/time result is required')
    if assignment_provenance['transformation'] != model['transformation'] or assignment_provenance['assignment'] != model['assignment']:
        raise ValueError('Saved state assignment transformation differs from its accepted frozen model')
    definitions = read_table(assignment_source.artifact('state_definitions')); inventory = read_table(assignment_source.artifact('cell_inventory'))
    if not definitions.model_id.eq(model['model_id']).all() or not inventory.model_id.eq(model['model_id']).all():
        raise ValueError('State definition/inventory has incompatible model identities')
    if assignments.duplicated(['source_run','movie','identity','frame_index']).any(): raise ValueError('State source repeats an original cell/frame key')
    return {'status':'accepted', 'reason':decision['reason'], 'model_id':model['model_id'], 'decision':decision, 'model':model,
        'bindings':bindings, 'assignments':assignments, 'definitions':definitions, 'inventory':inventory, 'time':time,
        'assignment_provenance':assignment_provenance, 'time_provenance':time_provenance, 'feature_provenance':feature_provenance}


def matching_features(resolved, saved):
    provenance=read_document(saved.artifact('provenance'));original=provenance['resolved_request']
    if original['inputs']['source_run']!=resolved.inputs.source_run:raise ValueError('State results belong to another original source run')
    window=original['request'].get('time_range_hours')
    if (tuple(window) if window is not None else None)!=resolved.request.time_range_hours:
        raise ValueError('State features have another analysis window')
    original_samples={item['movie']:item for item in original['inputs']['samples']}
    for current in resolved.inputs.samples:
        previous=original_samples.get(current.movie)
        if current.confirmed and (previous is None or not previous['confirmed'] or previous['sample']!=current.sample):
            raise ValueError('Confirmed biological-sample mapping differs from the state-model source; matching saved state validation is required')
    for name,fingerprint in original['inputs']['table_hashes'].items():
        if name in resolved.inputs.table_hashes and resolved.inputs.table_hashes[name]!=fingerprint:
            raise ValueError('Original measured table changed since the accepted state analysis')
    return provenance


def cell_pairs(inventory, model_id, source_id):
    """State hypotheses refer to whole cells, not repeated measurement labels."""
    pairs = {}; memberships = []
    for row in inventory.to_dict('records'):
        key = {name: row[name] for name in ['source_run','movie','reference_identity','target_identity']}
        state_pair_id = content_id({**key,'model_id':model_id,'state_source_id':source_id})
        pair = pairs.setdefault(state_pair_id,{**key,'state_pair_id':state_pair_id,'model_id':model_id,
            'sample':row['sample'],'sample_confirmed':row['sample_confirmed'],'condition':row['condition'],
            'geometry_population_eligible':row['geometry_population_eligible'],'requested_pair_ids':[]})
        if pair['geometry_population_eligible'] != row['geometry_population_eligible']:
            raise ValueError('The same oriented cell pair has conflicting scientific geometry membership')
        pair['requested_pair_ids'].append(row['pair_id'])
        memberships.append({**row,'state_pair_id':state_pair_id,'model_id':model_id,
            'meaning':'Accepted whole-cell states; repeated measurement labels do not create repeated state hypotheses'})
    return list(pairs.values()), memberships


def joint_exposure(left, right):
    """Intersect saved interval allocations; unknown and unobserved stay distinct."""
    a = left.sort_values('start_hours').to_dict('records'); b = right.sort_values('start_hours').to_dict('records')
    for rows in [a,b]:
        if any(not np.isfinite(row['start_hours']) or not np.isfinite(row['end_hours']) or row['end_hours'] <= row['start_hours'] for row in rows):
            raise ValueError('Saved state exposure has invalid physical bounds')
        if any(second['start_hours'] < first['end_hours']-1e-9 for first,second in zip(rows,rows[1:])):
            raise ValueError('Saved state exposure overlaps itself')
    i=j=0; result=[]
    while i<len(a) and j<len(b):
        first,second=a[i],b[j];low,high=max(first['start_hours'],second['start_hours']),min(first['end_hours'],second['end_hours'])
        if high>low:
            observed=first['support'] in {'assigned','unknown'} and second['support'] in {'assigned','unknown'}
            assigned=first['support']==second['support']=='assigned'
            result.append({'reference_exposure_id':first['exposure_id'],'target_exposure_id':second['exposure_id'],
                'reference_observation_id':first['observation_id'],'target_observation_id':second['observation_id'],
                'start_hours':low,'end_hours':high,'duration_hours':high-low,
                'reference_support':first['support'],'target_support':second['support'],
                'reference_state_id':first['state_id'],'target_state_id':second['state_id'],
                'jointly_observed':observed,'jointly_assigned':assigned,
                'same_state':first['state_id']==second['state_id'] if assigned else None,
                'support':'assigned' if assigned else 'unknown' if observed else 'unobserved'})
        if first['end_hours']<=second['end_hours']:i+=1
        else:j+=1
    return result


def transition_support(left, right):
    """Keep intervals where both cells had an eligible observed transition step."""
    def intervals(frame):
        return frame.loc[frame.valid_transition_opportunity].rename(columns={'source_hours':'start_hours','target_hours':'end_hours'})
    a,b=intervals(left),intervals(right);rows=[];i=j=0
    aa=a.sort_values('start_hours').to_dict('records');bb=b.sort_values('start_hours').to_dict('records')
    while i<len(aa) and j<len(bb):
        first,second=aa[i],bb[j];low,high=max(first['start_hours'],second['start_hours']),min(first['end_hours'],second['end_hours'])
        if high>low:rows.append({'start_hours':low,'end_hours':high,'duration_hours':high-low,
            'reference_step_id':first['step_id'],'target_step_id':second['step_id']})
        if first['end_hours']<=second['end_hours']:i+=1
        else:j+=1
    return rows


def switch_events(steps, support):
    if steps.step_id.duplicated().any():raise ValueError('Saved original switch opportunities are duplicated')
    rows=[]
    for step in steps.loc[steps.counted_switch].to_dict('records'):
        low,high=step['source_hours'],step['target_hours']
        covered=sum(max(0.,min(high,row['end_hours'])-max(low,row['start_hours'])) for row in support)
        eligible=np.isclose(covered,high-low,rtol=0.,atol=1e-9)
        rows.append({**step,'event_id':step['step_id'],'event_hours':(low+high)/2.,
            'earliest_hours':low,'latest_hours':high,'eligible_for_coincidence':bool(eligible),
            'event_time_meaning':'Midpoint of the observed transition interval; the true switch time within these bounds is unresolved',
            'eligibility_reason':'Complete jointly eligible transition support' if eligible else 'Partner observation cannot resolve the whole transition interval'})
    return _table(rows,[*steps.columns,'event_id','event_hours','earliest_hours','latest_hours','eligible_for_coincidence'])
