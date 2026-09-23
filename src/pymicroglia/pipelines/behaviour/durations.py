"""Observed state time and unique bouts; no interpolation across missing support."""
from __future__ import annotations
from pymicroglia._results import read_document
from pymicroglia._sources import source_file

from pathlib import Path
import json
import math

import pandas as pd

from pymicroglia.pipelines.behaviour.assignments import read_assignments
from pymicroglia.pipelines.behaviour.options import BoutKey, ObservationKey, StateKey, time_settings
from pymicroglia.pipelines.behaviour.validation import read_support
from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


META = ['model_id', *KEYS, 'sample', 'sample_confirmed', 'condition', 'role']
EXPOSURE = [*META, 'exposure_id', 'interval_id', 'observation_id', 'start_hours', 'end_hours',
    'duration_hours', 'support', 'state_id', 'assignment_status', 'reason']
STEP = [*META, 'step_id', 'source_observation_id', 'target_observation_id', 'source_state_id', 'target_state_id',
    'source_hours', 'target_hours', 'elapsed_hours', 'valid_observed_interval', 'observed_state_change',
    'valid_transition_opportunity', 'counted_switch', 'interval_stratum_hours', 'reason']
BOUT = [*META, 'bout_id', 'state_id', 'component', 'observation_ids', 'first_frame', 'last_frame',
    'first_hours', 'last_hours', 'observed_span_hours', 'allocated_start_hours', 'allocated_end_hours',
    'allocated_hours', 'observed_duration_hours', 'duration_status', 'onset_observed', 'ending_observed',
    'onset_bounds_hours', 'ending_bounds_hours', 'onset_reason', 'ending_reason', 'complete']
CELL = [*META, 'status', 'observations', 'assigned_observations', 'reference_start_hours', 'reference_end_hours',
    'reference_hours', 'observed_hours', 'assigned_hours', 'unknown_hours', 'unobserved_hours',
    'unknown_fraction_observed', 'observed_fraction_reference', 'valid_transition_opportunities',
    'valid_transition_hours', 'switches', 'switches_per_valid_transition_hour', 'observed_endpoint_changes',
    'transition_interval_profile', 'bouts', 'complete_bouts', 'incomplete_bouts']
OCCUPANCY = [*META, 'state_id', 'component', 'assigned_observations', 'state_hours', 'observed_hours',
    'assigned_hours', 'fraction_observed', 'fraction_assigned', 'bouts', 'complete_bouts', 'incomplete_bouts']
TRANSITION = [*META, 'interval_stratum_hours', 'source_state_id', 'target_state_id', 'count',
    'source_opportunities', 'probability', 'is_switch', 'status']


def implementation_version():
    return content_id({'producer': file_hash(Path(__file__)), 'assignments': file_hash(source_file('behaviour_assignments.py'))})


def _finite(value):
    return value is not None and not pd.isna(value) and math.isfinite(float(value))


def _fraction(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _observed(row):
    return row['time_status'] == 'recorded' and row['within_range'] and _finite(row['hours'])


def _boundary(step, side):
    if step is None: return False, None, 'recording_or_analysis_boundary'
    if step['valid_observed_interval'] and step['source_state_id'] is not None and step['target_state_id'] is not None:
        return True, [step['source_hours'], step['target_hours']], 'observed_different_state_at_adjacent_endpoint'
    return False, None, 'unknown_assignment' if step['valid_observed_interval'] else step['reason']


def statistics(assignments, states, inventory, settings, time_range=None):
    """Split each supported adjacent interval at its midpoint, retaining unknown halves.

    Transition matrices are stratified by elapsed hours rounded to 1e-9 h solely
    for numerical clock precision. No probabilities across different spacings
    are pooled. A whole cell with nonincreasing finite clocks has no time result.
    """
    models = set(states.model_id)
    if len(models) != 1: raise ValueError('Time statistics require one shared model vocabulary')
    model_id = next(iter(models))
    if not assignments.model_id.eq(model_id).all() or not inventory.model_id.eq(model_id).all():
        raise ValueError('Time inputs use different model definitions')
    if assignments.observation_id.duplicated().any() or assignments.duplicated([*KEYS, 'frame_index']).any():
        raise ValueError('Time inputs repeat an original observation')
    if inventory.duplicated(KEYS).any(): raise ValueError('Cell inventory repeats full cell identities')
    vocabulary = states[['state_id', 'component']].to_dict('records')
    if states.state_id.duplicated().any() or any(row['state_id'] != StateKey(model_id, row['component']).record_id for row in vocabulary):
        raise ValueError('State vocabulary does not match its model components')
    assigned = assignments.status.eq('assigned')
    if not assignments.loc[assigned, 'state_id'].isin(states.state_id).all() or assignments.loc[~assigned, 'state_id'].notna().any():
        raise ValueError('Assignments contain an invalid state or label an unknown observation')
    if settings['interval_rule'] != 'adjacent_midpoint': raise ValueError('Unsupported observation interval convention')
    band = settings['transition_interval_hours']
    groups = {key: frame for key, frame in assignments.groupby(KEYS, sort=True)}
    if set(groups) - set(inventory[KEYS].itertuples(index=False, name=None)):
        raise ValueError('Observed cells are absent from the complete cell inventory')
    outputs = {name: [] for name in ['exposures', 'steps', 'bouts', 'cell_statistics', 'occupancy', 'transitions']}
    for cell in inventory.to_dict('records'):
        key = tuple(cell[name] for name in KEYS)
        meta = {name: _json_value(cell.get(name)) for name in META}
        frame = groups.get(key, assignments.iloc[:0]).sort_values('frame_index')
        rows = [_json_value(row) for row in frame.to_dict('records')]
        clocks = [row['hours'] for row in rows if _finite(row['hours'])]
        clock_ok = all(b > a for a, b in zip(clocks, clocks[1:]))
        start, end = (min(clocks), max(clocks)) if clocks else (None, None)
        if time_range is not None and clocks:
            start, end = max(start, time_range[0]), min(end, time_range[1])
            if end < start: start = end = None
        reference = end - start if start is not None and clock_ok else None
        steps, exposures, bouts, supported = [], [], [], []
        for left, right in zip(rows, rows[1:]):
            elapsed = right['hours'] - left['hours'] if _finite(left['hours']) and _finite(right['hours']) else None
            reason = 'supported'
            if not clock_ok: reason = 'invalid_clock_order'
            elif not _observed(left) or not _observed(right): reason = 'outside_range' if not left['within_range'] or not right['within_range'] else 'invalid_clock'
            elif right['frame_index'] != left['frame_index'] + 1: reason = 'missing_original_frame'
            elif elapsed is None or elapsed <= 0: reason = 'invalid_clock'
            elif elapsed > settings['max_gap_hours']: reason = 'excess_time_gap'
            valid = reason == 'supported'
            step_id = content_id({'model_id': model_id, 'left': left['observation_id'], 'right': right['observation_id']})
            both = left['status'] == right['status'] == 'assigned'
            opportunity = valid and both and band[0] <= elapsed <= band[1]
            step = {**meta, 'step_id': step_id, 'source_observation_id': left['observation_id'], 'target_observation_id': right['observation_id'],
                'source_state_id': left['state_id'], 'target_state_id': right['state_id'], 'source_hours': left['hours'], 'target_hours': right['hours'],
                'elapsed_hours': elapsed, 'valid_observed_interval': valid, 'observed_state_change': valid and both and left['state_id'] != right['state_id'],
                'valid_transition_opportunity': opportunity, 'counted_switch': opportunity and left['state_id'] != right['state_id'],
                'interval_stratum_hours': round(elapsed, 9) if opportunity else None,
                'reason': reason if not valid else 'unknown_assignment' if not both else 'outside_transition_interval' if not opportunity else 'supported'}
            steps.append(step)
            if valid:
                midpoint = (left['hours'] + right['hours']) / 2
                supported.append((left['hours'], right['hours']))
                for endpoint, a, b in [(left, left['hours'], midpoint), (right, midpoint, right['hours'])]:
                    exposures.append({**meta, 'exposure_id': content_id({'interval': step_id, 'endpoint': endpoint['observation_id']}),
                        'interval_id': step_id, 'observation_id': endpoint['observation_id'], 'start_hours': a, 'end_hours': b, 'duration_hours': b-a,
                        'support': 'assigned' if endpoint['status'] == 'assigned' else 'unknown', 'state_id': endpoint['state_id'],
                        'assignment_status': endpoint['status'], 'reason': 'adjacent_midpoint_observation_allocation'})
        if reference is not None and reference > 0:
            cursor = start
            for a, b in supported + [(end, end)]:
                if a > cursor:
                    exposures.append({**meta, 'exposure_id': content_id({'cell': key, 'model': model_id, 'gap': [cursor, a]}),
                        'interval_id': None, 'observation_id': None, 'start_hours': cursor, 'end_hours': a, 'duration_hours': a-cursor,
                        'support': 'unobserved', 'state_id': None, 'assignment_status': None, 'reason': 'unsupported_recording_or_analysis_interval'})
                cursor = max(cursor, b)
        # Each original assigned observation belongs to exactly one observed bout.
        member_groups = []
        for index, row in enumerate(rows):
            if row['status'] != 'assigned' or not _observed(row): continue
            if member_groups and member_groups[-1][-1] == index-1 and steps[index-1]['valid_observed_interval'] and rows[index-1]['state_id'] == row['state_id']:
                member_groups[-1].append(index)
            else: member_groups.append([index])
        cell_key = CellKey(*key)
        allocations = {}
        for exposure in exposures:
            if exposure['observation_id'] is not None: allocations.setdefault(exposure['observation_id'], []).append(exposure)
        for indices in member_groups:
            first, last = rows[indices[0]], rows[indices[-1]]
            members = [rows[index]['observation_id'] for index in indices]
            pieces = [piece for member in members for piece in allocations.get(member, [])]
            hours = sum(piece['duration_hours'] for piece in pieces)
            onset, onset_bounds, onset_reason = _boundary(steps[indices[0]-1] if indices[0] else None, 'onset')
            ending, ending_bounds, ending_reason = _boundary(steps[indices[-1]] if indices[-1] < len(steps) else None, 'ending')
            bout_id = BoutKey(StateKey(model_id, first['component']), ObservationKey(cell_key, first['frame_index'], first['hours']),
                ObservationKey(cell_key, last['frame_index'], last['hours'])).record_id
            bouts.append({**meta, 'bout_id': bout_id, 'state_id': first['state_id'], 'component': first['component'], 'observation_ids': members,
                'first_frame': first['frame_index'], 'last_frame': last['frame_index'], 'first_hours': first['hours'], 'last_hours': last['hours'],
                'observed_span_hours': last['hours']-first['hours'], 'allocated_start_hours': min(piece['start_hours'] for piece in pieces) if pieces else None,
                'allocated_end_hours': max(piece['end_hours'] for piece in pieces) if pieces else None, 'allocated_hours': hours,
                'observed_duration_hours': hours if hours > 0 else None,
                'duration_status': 'invalid_clock_order' if not clock_ok else 'point_only' if hours == 0 else 'complete_observed_bout' if onset and ending else 'incomplete_observed_bout',
                'onset_observed': onset, 'ending_observed': ending, 'onset_bounds_hours': onset_bounds, 'ending_bounds_hours': ending_bounds,
                'onset_reason': onset_reason, 'ending_reason': ending_reason, 'complete': onset and ending})
        assigned_hours = sum(item['duration_hours'] for item in exposures if item['support'] == 'assigned')
        unknown_hours = sum(item['duration_hours'] for item in exposures if item['support'] == 'unknown')
        observed_hours = assigned_hours + unknown_hours
        opportunities = [step for step in steps if step['valid_transition_opportunity']]
        transition_hours = sum(step['elapsed_hours'] for step in opportunities)
        switches = sum(step['counted_switch'] for step in opportunities)
        strata = sorted({step['interval_stratum_hours'] for step in opportunities})
        profile = [{'hours': spacing, 'opportunities': sum(step['interval_stratum_hours'] == spacing for step in opportunities)} for spacing in strata]
        outputs['cell_statistics'].append({**meta, 'status': 'invalid_clock_order' if not clock_ok else 'observed' if observed_hours > 0 else 'no_supported_time',
            'observations': len(rows), 'assigned_observations': sum(row['status'] == 'assigned' for row in rows),
            'reference_start_hours': start, 'reference_end_hours': end, 'reference_hours': reference,
            'observed_hours': observed_hours, 'assigned_hours': assigned_hours, 'unknown_hours': unknown_hours,
            'unobserved_hours': sum(item['duration_hours'] for item in exposures if item['support'] == 'unobserved') if reference is not None else None,
            'unknown_fraction_observed': _fraction(unknown_hours, observed_hours), 'observed_fraction_reference': _fraction(observed_hours, reference or 0),
            'valid_transition_opportunities': len(opportunities), 'valid_transition_hours': transition_hours, 'switches': switches,
            'switches_per_valid_transition_hour': _fraction(switches, transition_hours), 'observed_endpoint_changes': sum(step['observed_state_change'] for step in steps),
            'transition_interval_profile': profile, 'bouts': len(bouts), 'complete_bouts': sum(bout['complete'] for bout in bouts),
            'incomplete_bouts': sum(not bout['complete'] for bout in bouts)})
        for state in vocabulary:
            state_bouts = [bout for bout in bouts if bout['state_id'] == state['state_id']]
            state_hours = sum(item['duration_hours'] for item in exposures if item['state_id'] == state['state_id'])
            outputs['occupancy'].append({**meta, **state, 'assigned_observations': sum(row['state_id'] == state['state_id'] for row in rows),
                'state_hours': state_hours, 'observed_hours': observed_hours, 'assigned_hours': assigned_hours,
                'fraction_observed': _fraction(state_hours, observed_hours), 'fraction_assigned': _fraction(state_hours, assigned_hours),
                'bouts': len(state_bouts), 'complete_bouts': sum(bout['complete'] for bout in state_bouts),
                'incomplete_bouts': sum(not bout['complete'] for bout in state_bouts)})
        for spacing in strata or [None]:
            for source in vocabulary:
                source_steps = [step for step in opportunities if step['interval_stratum_hours'] == spacing and step['source_state_id'] == source['state_id']]
                for target in vocabulary:
                    count = sum(step['target_state_id'] == target['state_id'] for step in source_steps)
                    outputs['transitions'].append({**meta, 'interval_stratum_hours': spacing, 'source_state_id': source['state_id'], 'target_state_id': target['state_id'],
                        'count': count, 'source_opportunities': len(source_steps), 'probability': _fraction(count, len(source_steps)),
                        'is_switch': source['state_id'] != target['state_id'], 'status': 'observed' if source_steps else 'no_source_opportunities'})
        outputs['exposures'].extend(exposures); outputs['steps'].extend(steps); outputs['bouts'].extend(bouts)
    schemas = dict(exposures=EXPOSURE, steps=STEP, bouts=BOUT, cell_statistics=CELL, occupancy=OCCUPANCY, transitions=TRANSITION)
    return {name: pd.DataFrame(rows, columns=schemas[name]) for name, rows in outputs.items()}


def read_statistics(saved, *, expected_model=None):
    provenance = read_document(saved.artifact('provenance'))
    if provenance.get('schema_version') != 1 or provenance.get('scientific_id') != saved.outcome.scientific_id:
        raise ValueError('State-time provenance does not match its scientific identity')
    if expected_model is not None and provenance['model_id'] != expected_model: raise ValueError('Time statistics use a different model')
    tables = {name: read_table(saved.artifact(name)) for name in ['exposures', 'steps', 'bouts', 'cell_statistics', 'occupancy', 'transitions']}
    if any(not table.model_id.eq(provenance['model_id']).all() for table in tables.values()): raise ValueError('Time tables mix model definitions')
    return tables, provenance


def produce(context):
    decision, model = read_support(context.saved('state-support'), require_accepted=True)
    saved = context.saved('state-assignments')
    assignments, _ = read_assignments(saved, expected_model=model['model_id'])
    states = read_table(saved.artifact('state_definitions')); cells = read_table(saved.artifact('cell_inventory'))
    request = context.request.request
    tables = statistics(assignments, states, cells, request.statistics.as_dict(), request.time_range_hours)
    provenance = {'schema_version': 1, 'scientific_id': context.scientific_id, 'model_id': model['model_id'],
        'decision_id': decision['decision_id'], 'assignment_id': saved.outcome.scientific_id, 'settings': time_settings(request.statistics),
        'time_range_hours': request.time_range_hours, 'time_unit': 'hours', 'interval_rule': 'adjacent_midpoint',
        'exposure_scope': 'Each valid consecutive original-frame interval splits equally at its midpoint; unknown endpoints retain unknown time; unsupported gaps remain unobserved',
        'bout_scope': 'Unique contiguous observed assignments; midpoint allocations and sampling bounds are not exact latent dwell times; unseen within-interval switches are unresolved',
        'switch_scope': 'Observed endpoint changes per valid assigned transition-hour within the declared spacing band; self-steps contribute opportunities but not switches',
        'comparison_scope': 'Transition matrices are stratified by actual elapsed hours rounded to 1e-9 h; different interval strata cannot be pooled as common-step probabilities',
        'invalid_clock_scope': 'A nonincreasing finite clock in original frame order withholds all physical-time inference for that cell',
        'model_fitted': False, 'assignments_changed': False, 'inference_performed': False,
        'counts': {name: len(frame) for name, frame in tables.items()}}
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in tables.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    path = context.output/'provenance.json'; _write_json(path, provenance)
    refs.append(ArtifactRef('provenance', path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved physical-time occupancy, unique censored observed bouts and interval-stratified switching with complete denominators', tuple(refs))
