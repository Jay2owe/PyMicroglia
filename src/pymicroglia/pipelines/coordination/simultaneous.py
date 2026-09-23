"""Complete cross-cell temporal effects and reusable, original-clock pair views."""
from __future__ import annotations
from pymicroglia._results import read_document
from pymicroglia._sources import source_file

import json
from importlib.metadata import version as library_version
from pathlib import Path

import pandas as pd

import pymicroglia.measure.relationship_statistics as native
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import PAIR_KEYS, TRACE_COLUMNS, _adjacent, _matched, _table, joint_intervals, read_inputs
from pymicroglia.pipelines.coordination.temporal_inputs import ADJUSTMENT_MEANING, adjusted_pair, prepare_series
from pymicroglia.pipelines._runner import Unavailable
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table, write_table


TABLES = ('series', 'series_inventory', 'pair_views', 'pair_effects', 'matches', 'matching_intervals',
    'joint_intervals', 'reference_summary', 'reference_membership', 'shift_reference')
VIEW_KEYS = PAIR_KEYS + ['view_id', 'representation', 'adjustment', 'reference_series_id', 'target_series_id', 'is_hypothesis']


def implementation_version():
    return {'code': {name: file_hash(source_file(name)) for name in
        ['coordination_simultaneous.py', 'coordination_temporal_inputs.py', 'coordination_inputs.py', 'relationship_inputs.py']},
        'native': file_hash(Path(native.__file__)), 'libraries': {name: library_version(name) for name in ['numpy', 'pandas', 'scipy']}}


def identity(context):
    resolved = context.request; request = resolved.request
    return content_id({'pair_inputs': context.saved('pair-inputs').outcome.scientific_id,
        'question': request.questions['simultaneous'], 'temporal_requested': any(request.questions[q]['enabled'] for q in ['simultaneous', 'delay', 'proximity']),
        'representation': request.representation, 'increment': request.increment,
        'detrending': resolved.detrending, 'workbench_version': resolved.processing_version,
        'shared_reference': request.shared_reference, 'implementation': implementation_version()})


def validate_question(question):
    if not question['enabled']: return
    if question['statistic'] not in native.STATISTICS or question['evidence']['method'] not in {'none', 'truncated_time_shift'}:
        raise Unavailable('Simultaneous coordination supports Pearson/Spearman coefficients and the declared truncated time-shift independence test')
    native.validate_question(question)


def temporal_refusal(left, right, support):
    """Check original acquisition adjacency before the native uniform-clock test."""
    selected = [frame.loc[frame.within_range] for frame in [left, right]]
    if any(frame.empty for frame in selected): return 'No common observed acquisition window'
    low, high = max(frame.hours.min() for frame in selected), min(frame.hours.max() for frame in selected)
    for frame in [left, right]:
        if len(frame) and not frame.clock_valid.all(): return 'Invalid original acquisition clock'
        rows = frame.loc[frame.within_range & frame.hours.between(low, high)].sort_values('sequence_index').to_dict('records')
        for a, b in zip(rows, rows[1:]):
            valid, reason = _adjacent(a, b, support['max_gap_hours'])
            if not valid: return reason + '; temporal inference cannot compress the original acquisition'
    return None


def estimate(view, pair, left, right, question, support):
    """Keep the full-overlap coefficient separate from the tested central effect."""
    matches, intervals, coverage = _matched(left, right, support)
    effect, description = native.coefficient([row['reference_value'] for row in matches],
        [row['target_value'] for row in matches], question['statistic'])
    status, reason = ('descriptive', description) if effect is not None else ('unresolvable', description)
    if coverage['status'] != 'eligible': status, reason = 'insufficient', coverage['reason']
    if not pair['geometry_population_eligible']: status, reason = 'outside_geometry_population', 'Outside the predeclared geometric population'
    tested = {'status': status, 'reason': reason, 'p_value': None}
    formal = question['evidence']['method'] != 'none' and view['is_hypothesis']
    if status == 'descriptive' and formal:
        refusal = temporal_refusal(left, right, support)
        tested = {'status': 'untestable', 'reason': refusal, 'p_value': None} if refusal else native.same_time_evidence(
            left, right, question=question, support=support)
    if not view['is_hypothesis']:
        tested['reason'] = 'Original-level descriptive baseline; declared inference concerns the selected temporal representation'
    # Native reference is a paper URL, whereas view.reference is a measured
    # endpoint. Never merge native dictionaries over full endpoint keys.
    row = {**view, **coverage, 'question': 'simultaneous', 'evidence_level': 'pair',
        'statistic': question['statistic'], 'descriptive_effect': effect, 'descriptive_reason': description,
        'effect': tested.get('effect', effect), 'p_value': tested['p_value'], 'status': tested['status'], 'reason': tested['reason'],
        'confidence_low': None, 'confidence_high': None, 'uncertainty_status': 'not_provided_by_method',
        'effect_population': tested.get('effect_population', 'All finite matched original observations in the declared window'),
        'tested_observations': tested.get('tested_observations'), 'tested_start_hours': tested.get('tested_start_hours'),
        'tested_end_hours': tested.get('tested_end_hours'), 'minimum_attainable_p': tested.get('minimum_attainable_p'),
        'method': question['evidence']['method'] if formal else 'none', 'evidence_settings': question['evidence'] if formal else {'method': 'none'},
        'shared_reference_interpretation': ADJUSTMENT_MEANING if view['adjustment'] != 'none' else None,
        'native_details': {key: value for key, value in tested.items() if key not in {'shifts_observations', 'shift_statistics'}}}
    shifts = [{'view_id': view['view_id'], 'shift_observations': shift, 'shift_hours': shift*tested['sample_interval_hours'],
        'absolute_statistic': value} for shift, value in zip(tested.get('shifts_observations', []), tested.get('shift_statistics', []))]
    return row, matches, intervals, shifts


def analyse(resolved, prepared, scientific_id):
    request = resolved.request; question = request.questions['simultaneous']; validate_question(question)
    frames, inventory, processing = prepare_series(resolved, prepared)
    series = list(frames.values()); definitions = inventory.to_dict('records'); by_series = {row['series_id']: row for row in definitions}
    effects = []; views = []; matches = []; intervals = []; joint = []; summaries = []; memberships = []; shifts = []
    for pair in prepared['inventory'].to_dict('records'):
        variants = [('raw', 'none', {role: frames[(pair[role+'_endpoint_id'], 'raw')] for role in ['reference', 'target']})]
        if request.representation != 'raw':
            variants.append((request.representation, 'none', {role: frames[(pair[role+'_endpoint_id'], request.representation)] for role in ['reference', 'target']}))
        adjusted, reference_rows, member_rows, adjusted_definitions = adjusted_pair(resolved, pair, frames, inventory)
        summaries.extend(reference_rows); memberships.extend(member_rows); definitions.extend(adjusted_definitions)
        by_series.update({row['series_id']: row for row in adjusted_definitions})
        if adjusted:
            variants.append((request.representation, 'shared_reference', adjusted)); series.extend(adjusted.values())
        for representation, adjustment, selected in variants:
            # Empty endpoint traces still have a saved series definition.
            ids = {}
            for role in ['reference', 'target']:
                ids[role] = next(row['series_id'] for row in definitions if row['endpoint_id'] == pair[role+'_endpoint_id']
                    and row['representation'] == representation and row['adjustment'] == adjustment
                    and (adjustment == 'none' or row.get('pair_id') == pair['pair_id']))
            view_id = content_id({'pair': pair['pair_id'], 'series': ids})
            view = {**pair, 'view_id': view_id, 'representation': representation, 'adjustment': adjustment,
                'reference_series_id': ids['reference'], 'target_series_id': ids['target'],
                'is_hypothesis': representation == request.representation,
                'reference_series_status': by_series[ids['reference']]['status'], 'target_series_status': by_series[ids['target']]['status']}
            views.append(view)
            left, right = selected['reference'], selected['target']
            joint.extend({**view, **row} for row in joint_intervals(left, right, request.support))
            if question['enabled']:
                result, matched, elapsed, reference = estimate(view, pair, left, right, question, request.support)
                effects.append({**result, 'result_id': content_id({'scientific_id': scientific_id, 'view': view_id, 'question': 'simultaneous'})})
                matches.extend({**view, **row} for row in matched)
                intervals.extend({**view, **row} for row in elapsed); shifts.extend(reference)
    tables = {'series': pd.concat(series, ignore_index=True) if series else _table([], TRACE_COLUMNS+['series_id', 'representation', 'adjustment']),
        'series_inventory': _table(definitions, [*inventory.columns]), 'pair_views': _table(views, VIEW_KEYS),
        'pair_effects': _table(effects, VIEW_KEYS+['result_id', 'question', 'evidence_level', 'effect', 'descriptive_effect', 'p_value', 'status', 'reason']),
        'matches': _table(matches, VIEW_KEYS+['reference_observation', 'target_observation', 'reference_hours', 'target_hours', 'reference_value', 'target_value']),
        'matching_intervals': _table(intervals, VIEW_KEYS+['start_hours', 'end_hours', 'valid', 'jointly_observed_hours']),
        'joint_intervals': _table(joint, VIEW_KEYS+['start_hours', 'end_hours', 'valid', 'jointly_observed_hours']),
        'reference_summary': _table(summaries, ['pair_id', 'adjusted_series_id', 'endpoint_role', 'owner_observation_id', 'hours', 'reference_value', 'members', 'status']),
        'reference_membership': _table(memberships, ['pair_id', 'adjusted_series_id', 'owner_observation_id', 'donor_series_id', 'donor_source_observation_id']),
        'shift_reference': _table(shifts, ['view_id', 'shift_observations', 'shift_hours', 'absolute_statistic'])}
    return tables, processing


def produce(context):
    request = context.request.request
    if not any(request.questions[q]['enabled'] for q in ['simultaneous', 'delay', 'proximity']):
        return StepResult(context.step.name, context.scientific_id, 'skipped-empty', 'No temporal coordination question was requested')
    validate_question(request.questions['simultaneous'])
    prepared = read_inputs(context.saved('pair-inputs')); tables, processing = analyse(context.request, prepared, context.scientific_id)
    context.output.mkdir(parents=True); refs = []
    for name, frame in tables.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    documents = {'processing': processing, 'provenance': {'schema_version': 1, 'scientific_id': context.scientific_id,
        'pair_inputs_id': context.saved('pair-inputs').outcome.scientific_id, 'question': request.questions['simultaneous'],
        'representation': request.representation, 'increment': request.increment, 'detrending': context.request.detrending,
        'workbench_version': context.request.processing_version, 'shared_reference': request.shared_reference,
        'support': request.support, 'implementation': implementation_version(), 'temporal_method_reference': native.REFERENCE,
        'adjustment_interpretation': ADJUSTMENT_MEANING,
        'families': 'Complete requested pair/representation/reference hypotheses; raw baselines of transformed questions are descriptive only. Correction and selection occur later.',
        'uncertainty': 'The conservative time-shift method supplies a probability bound, not an effect confidence interval',
        'delay_selection': 'Every requested pair view remains available to delay/proximity; no simultaneous significance gate'}}
    for name, value in documents.items():
        path = context.output/(name+'.json'); _write_json(path, value)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved complete temporal representations, shared-reference contributors and separate pair effects', tuple(refs))


def read_temporal(saved):
    tables = {name: read_table(saved.artifact(name)) for name in TABLES}
    provenance = read_document(saved.artifact('provenance'))
    if provenance['scientific_id'] != saved.outcome.scientific_id: raise ValueError('Temporal provenance identity mismatch')
    definitions = tables['series_inventory']
    if definitions.series_id.duplicated().any(): raise ValueError('Duplicate temporal series identity')
    lookup = definitions.set_index('series_id').to_dict('index')
    if tables['series'].duplicated(['series_id', 'observation_id']).any(): raise ValueError('Duplicate original temporal observation')
    for row in tables['pair_views'].to_dict('records'):
        for role in ['reference', 'target']:
            definition = lookup[row[role+'_series_id']]
            if any(definition[key] != row[key] for key in ['source_run', 'movie']) or definition['endpoint_id'] != row[role+'_endpoint_id']:
                raise ValueError('Temporal view points to a different source/recording/endpoint')
    for series_id, frame in tables['series'].groupby('series_id', sort=False):
        definition = lookup[series_id]
        if any(not frame[key].eq(definition[key]).all() for key in ['source_run', 'movie', 'identity', 'endpoint_id', 'measurement_id']):
            raise ValueError('Temporal observations disagree with their full endpoint identity')
    return {**tables, 'provenance': provenance}
