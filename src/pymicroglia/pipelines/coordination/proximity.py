"""Windowed distance/coordination processes within the same measured cell pair."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path

import numpy as np
import pandas as pd

import pymicroglia.measure.relationship_statistics as native
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import _table, read_inputs
from pymicroglia.pipelines.coordination.simultaneous import VIEW_KEYS, read_temporal
from pymicroglia.pipelines.relationships.inputs import lag_values, match_observations, paired_support
from pymicroglia.pipelines._runner import Unavailable
from pymicroglia.pipelines._screening import _write_json, file_hash, write_table


MEANING = ('Association between observed distance summaries and measured window-coordination summaries within one pair. '
    'It is not a comparison of different pairs, an estimate of latent coupling strength, or a causal effect of approach.')


def window_question(question):
    evidence = dict(question['evidence'])
    if evidence['method'] != 'none':
        if evidence.get('stationary_series') not in {'distance', 'coordination'}:
            raise ValueError('Proximity evidence.stationary_series must explicitly be distance or coordination')
        evidence['stationary_series'] = {'distance': 'reference', 'coordination': 'target'}[evidence['stationary_series']]
    return {'enabled': True, 'statistic': question['statistic'], 'evidence': evidence}


def validate_question(question):
    if not question['enabled']: return
    if question['statistic'] not in native.STATISTICS or question['evidence']['method'] not in {'none', 'truncated_time_shift'}:
        raise Unavailable('Within-pair proximity supports Pearson/Spearman association of declared window summaries with conservative time-shift evidence')
    native.validate_question(window_question(question))


def implementation_version():
    return {'code': {path.name: file_hash(path) for path in [Path(__file__), Path(native.__file__),
        source_file('coordination_inputs.py'), source_file('coordination_simultaneous.py'),
        source_file('relationship_inputs.py')]},
        'libraries': {name: library_version(name) for name in ['numpy', 'pandas', 'scipy']}}


def identity(context):
    request = context.request.request; question = request.questions['proximity']
    relevant = request.questions[question['coordination']] if question['enabled'] else {}
    return content_id({'pair_inputs': context.saved('pair-inputs').outcome.scientific_id,
        'temporal_inputs': context.saved('simultaneous-coordination').outcome.scientific_id,
        'question': question, 'coordination_definition': relevant, 'implementation': implementation_version()})


def joint_observations(view, left, right, positions, support):
    """Align each cell's signal and position at its own original timestamps."""
    matches, _ = match_observations(left, right, 0., support)
    if matches.empty or positions.empty: return _table([], ['reference_hours', 'target_hours', 'distance', 'reference_value', 'target_value']), []
    # Match the reference clock and target clock independently. Only the same
    # original geometry pair is allowed to match both signal endpoints.
    position_rows = positions.to_dict('records')
    identifiers = [content_id({'view': view['view_id'], 'geometry': [row['reference_observation'], row['target_observation']]}) for row in position_rows]
    by_position = dict(zip(identifiers, position_rows)); chosen = {}
    for role in ['reference', 'target']:
        signal = pd.DataFrame({'hours': matches[role+'_hours'], 'observation_id': matches[role+'_observation'],
            'processed_value': matches[role+'_value'], 'within_range': True, 'processed_valid': True})
        geometry = pd.DataFrame({'hours': positions[role+'_hours'].to_numpy(), 'observation_id': identifiers,
            'processed_value': positions.distance.to_numpy(), 'within_range': True, 'processed_valid': True})
        selected, _ = match_observations(signal, geometry, 0., support)
        chosen[role] = {row['reference_observation']: row['target_observation'] for row in selected.to_dict('records')}
    rows = []; excluded = []
    for row in matches.to_dict('records'):
        a = chosen['reference'].get(row['reference_observation']); b = chosen['target'].get(row['target_observation'])
        if a is None or a != b:
            excluded.append({**row, 'reason': 'The two signal observations do not share an eligible original geometry pair'}); continue
        position = by_position[a]
        rows.append({**row, 'distance': position['distance'], 'distance_unit': position['unit'],
            'reference_geometry_observation': position['reference_observation'], 'target_geometry_observation': position['target_observation'],
            'reference_geometry_hours': position['reference_hours'], 'target_geometry_hours': position['target_hours'],
            'reference_visible_cells': position.get('reference_visible_cells'), 'target_visible_cells': position.get('target_visible_cells'),
            'reference_edge': position.get('reference_edge'), 'target_edge': position.get('target_edge'),
            'geometry_pair_observation_id': a, 'near': position.get('near')})
    return _table(rows, ['reference_hours', 'target_hours', 'distance', 'reference_value', 'target_value']), excluded


def windows(view, left, right, positions, request):
    q = request.questions['proximity']; coordination = request.questions[q['coordination']]
    observations, excluded = joint_observations(view, left, right, positions, request.support)
    original = [frame.loc[frame.within_range & np.isfinite(frame.hours.to_numpy(dtype=float, na_value=np.nan))] for frame in [left, right]]
    if any(frame.empty for frame in original): return [], [], [], excluded
    low, high = max(frame.hours.min() for frame in original), min(frame.hours.max() for frame in original)
    width, step = q['window_hours'], q['step_hours']
    count = max(0, int(np.floor((high-low-width)/step+1e-9))+1)
    if count > 100000: raise ValueError('Proximity window grid exceeds 100000 declared positions per pair')
    rows = []; members = []; profiles = []
    for index in range(count):
        start, end = float(low+index*step), float(low+index*step+width)
        selected = observations.loc[observations.reference_hours.ge(start) & observations.reference_hours.lt(end)
            & observations.target_hours.ge(start) & observations.target_hours.lt(end)]
        window_id = content_id({'view': view['view_id'], 'question': q, 'index': index, 'start': start, 'end': end})
        key = {**view, 'window_id': window_id, 'window_index': index, 'start_hours': start, 'end_hours': end,
            'hours': (start+end)/2., 'window_hours': width, 'step_hours': step}
        observed = len(selected); span = float(selected.reference_hours.max()-selected.reference_hours.min()) if observed else 0.
        distance = float(selected.distance.mean()) if observed else None
        enough = observed >= request.support['min_observations'] and span >= request.support['min_span_hours']
        status, reason = ('descriptive', 'Compatible original signal and position observations') if enough else ('insufficient', 'Insufficient jointly observed signal and distance support')
        value = None; local = []
        if q['coordination'] == 'simultaneous':
            value, diagnostic = native.coefficient(selected.reference_value, selected.target_value, coordination['statistic'])
            if value is None and enough: status, reason = 'unresolvable', diagnostic
            if value is not None and q['coordination_measure'] == 'absolute_coefficient': value = abs(value)
        else:
            # This predeclared transformation is the complete maximum over the
            # fixed grid in every window, not a chosen lag subsequently treated
            # as fixed. No window receives significance or a resolved delay.
            frames = []
            for role, source in [('reference', left), ('target', right)]:
                ids = set(selected[role+'_observation']) if len(selected) else set()
                frames.append(source.assign(within_range=source.observation_id.isin(ids)))
            for lag in lag_values(coordination):
                pairs, diagnostic = match_observations(*frames, lag, request.support)
                support = paired_support(pairs, diagnostic, request.support)
                coefficient, why = native.coefficient(pairs.reference_value, pairs.target_value, coordination['statistic'])
                good = support['status'] == 'eligible' and coefficient is not None
                local.append({**key, 'lag_hours': lag, 'effect': coefficient, 'paired_observations': len(pairs),
                    'status': 'descriptive' if good else 'unresolvable', 'reason': why if support['status'] == 'eligible' else support['reason'],
                    'p_value': None, 'delay_supported': False})
            if enough and all(row['status'] == 'descriptive' for row in local): value = max(abs(row['effect']) for row in local)
            elif enough: status, reason = 'unresolvable', 'Every lag of the predeclared window maximum requires adequate finite support'
            profiles.extend(local)
        if not view['geometry_population_eligible']: status, reason = 'outside_geometry_population', 'Outside the fixed scientific geometry population'
        if status != 'descriptive': value = None
        # Formal window-series inference requires complete original endpoint
        # coverage, not merely sufficient observations after dropping gaps.
        counts = [int((frame.hours.ge(start) & frame.hours.lt(end)).sum()) for frame in original]
        complete = bool(observed > 0 and counts == [observed, observed])
        if complete:
            for frame in original:
                points = frame.loc[frame.hours.ge(start) & frame.hours.lt(end)].sort_values('sequence_index').to_dict('records')
                from pymicroglia.pipelines.coordination.inputs import _adjacent
                if any(not _adjacent(a, b, request.support['max_gap_hours'])[0] for a, b in zip(points, points[1:])): complete = False
        rows.append({**key, 'distance': distance, 'distance_min': float(selected.distance.min()) if observed else None,
            'distance_max': float(selected.distance.max()) if observed else None, 'distance_summary': 'mean of original matched coordinates',
            'distance_unit': request.geometry['unit'], 'coordination': value, 'coordination_question': q['coordination'],
            'coordination_measure': q['coordination_measure'], 'coordination_statistic': coordination['statistic'],
            'joint_observations': observed, 'original_reference_observations': counts[0], 'original_target_observations': counts[1],
            'observed_span_hours': span, 'complete_original_support': complete, 'status': status, 'reason': reason,
            'p_value': None, 'window_is_independent_replicate': False})
        members.extend({**key, **row} for row in selected.to_dict('records'))
    return rows, members, profiles, excluded


def window_evidence(rows, question):
    """Test derived temporal processes, preserving the entire window sequence."""
    frame = _table(rows, ['hours', 'distance', 'coordination', 'complete_original_support', 'joint_observations'])
    valid = frame.loc[frame.distance.notna() & frame.coordination.notna()]
    effect, why = native.coefficient(valid.distance, valid.coordination, question['statistic'])
    span = float(valid.distance.max()-valid.distance.min()) if len(valid) else None
    result = {'effect': effect, 'descriptive_effect': effect, 'p_value': None, 'status': 'descriptive' if effect is not None else 'unresolvable',
        'reason': why, 'valid_windows': len(valid), 'requested_windows': len(frame), 'distance_range': span,
        'confidence_low': None, 'confidence_high': None, 'uncertainty_status': 'not_provided_by_method',
        'tested_start_hours': None, 'tested_end_hours': None, 'tested_windows': None,
        'interpretation': MEANING, 'window_overlap': 'Original overlapping windows remain a serial sequence; none are shuffled or treated as independent replicates'}
    if len(valid) < question['min_windows']:
        result.update(status='insufficient', reason='Too few jointly observed windows for the declared within-pair comparison'); return result
    if span is None or span <= question['min_distance_range']:
        result.update(status='unresolvable', effect=None, reason='Within-pair distance lacks the declared minimum variation'); return result
    if effect is None or question['evidence']['method'] == 'none': return result
    if not frame.complete_original_support.all() or frame.joint_observations.nunique() != 1:
        result.update(status='untestable', reason='Temporal inference requires complete equal original support in every declared window; gaps or changing observation membership cannot be compressed'); return result
    left = pd.DataFrame({'hours': frame.hours, 'processed_value': frame.distance, 'within_range': True})
    right = pd.DataFrame({'hours': frame.hours, 'processed_value': frame.coordination, 'within_range': True})
    settings = {'min_observations': question['min_windows'], 'min_span_hours': (question['min_windows']-1)*question['step_hours'],
        'max_gap_hours': question['step_hours']+1e-9}
    tested = native.same_time_evidence(left, right, question=window_question(question), support=settings)
    for key in ['effect', 'p_value', 'status', 'reason', 'tested_start_hours', 'tested_end_hours', 'minimum_attainable_p']:
        if key in tested: result[key] = tested[key]
    result.update(tested_windows=tested.get('tested_observations'), native_details=tested,
        derived_process_assumption='The declared distance or coordination window-summary process is stationary; overlapping windows need not be independent')
    return result


def analyse(resolved, prepared, temporal, scientific_id):
    question = resolved.request.questions['proximity']; validate_question(question)
    series = {key: frame for key, frame in temporal['series'].groupby('series_id', sort=False)}
    all_windows = []; members = []; profiles = []; excluded = []; effects = []
    for view in temporal['pair_views'].to_dict('records'):
        left, right = [series.get(view[role+'_series_id'], temporal['series'].iloc[:0]) for role in ['reference', 'target']]
        positions = prepared['pair_positions'].loc[prepared['pair_positions'].pair_id.eq(view['pair_id'])]
        rows, joined, curves, refusals = windows(view, left, right, positions, resolved.request)
        actual = dict(question)
        if not view['is_hypothesis']: actual['evidence'] = {'method': 'none'}
        result = window_evidence(rows, actual)
        if not view['geometry_population_eligible']:
            result.update(status='outside_geometry_population', p_value=None, reason='Outside the fixed scientific geometry population')
        effects.append({**view, **result, 'result_id': content_id({'scientific_id': scientific_id, 'view': view['view_id'], 'question': 'proximity'}),
            'question': 'proximity', 'evidence_level': 'pair', 'statistic': question['statistic'],
            'method': actual['evidence']['method'], 'evidence_settings': actual['evidence'],
            'coordination_question': question['coordination'], 'coordination_measure': question['coordination_measure'],
            'window_hours': question['window_hours'], 'step_hours': question['step_hours'],
            'mean_pair_distance': float(positions.distance.mean()) if len(positions) else None,
            'mean_pair_distance_meaning': 'Separate descriptive average position distance; never substituted for within-pair changes',
            'distance_unit': resolved.request.geometry['unit'],
            'shared_reference_interpretation': temporal['provenance']['adjustment_interpretation'] if view['adjustment'] != 'none' else None})
        all_windows.extend(rows); members.extend(joined); profiles.extend(curves); excluded.extend({**view, **row} for row in refusals)
    return {'pair_effects': _table(effects, VIEW_KEYS+['result_id', 'question', 'evidence_level', 'effect', 'p_value', 'status', 'reason']),
        'windows': _table(all_windows, VIEW_KEYS+['window_id', 'window_index', 'start_hours', 'end_hours', 'hours', 'distance', 'coordination', 'status', 'reason']),
        'window_observations': _table(members, VIEW_KEYS+['window_id', 'reference_observation', 'target_observation', 'reference_hours', 'target_hours', 'distance']),
        'window_lag_profiles': _table(profiles, VIEW_KEYS+['window_id', 'lag_hours', 'effect', 'status', 'p_value', 'delay_supported']),
        'excluded_observations': _table(excluded, VIEW_KEYS+['reference_observation', 'target_observation', 'reason'])}


def produce(context):
    question = context.request.request.questions['proximity']
    if not question['enabled']: return StepResult(context.step.name, context.scientific_id, 'skipped-empty', 'Changing-proximity question was explicitly disabled')
    validate_question(question)
    prepared = read_inputs(context.saved('pair-inputs')); temporal = read_temporal(context.saved('simultaneous-coordination'))
    outputs = analyse(context.request, prepared, temporal, context.scientific_id)
    context.output.mkdir(parents=True); refs = []
    for name, frame in outputs.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    path = context.output/'provenance.json'
    _write_json(path, {'schema_version': 1, 'scientific_id': context.scientific_id, 'question': question,
        'pair_inputs_id': context.saved('pair-inputs').outcome.scientific_id,
        'temporal_inputs_id': context.saved('simultaneous-coordination').outcome.scientific_id,
        'implementation': implementation_version(), 'method_reference': native.REFERENCE, 'interpretation': MEANING,
        'alignment': 'Both original signal timestamps must match the same original position pair under the declared unique matching rule',
        'windows': 'Fixed physical half-open windows entirely within the common recorded span; original signal/position membership and excluded observations retained',
        'lag_variant': 'Every declared lag is evaluated in every window before taking a maximum absolute coefficient. No per-window significance or certified lag; the derived sequence is tested as a whole.',
        'uncertainty': 'Time-shift evidence on the derived window processes supplies no latent-coupling interval',
        'selection': 'All requested pair views retained; full-family correction and report selection occur at stage 09'})
    refs.append(ArtifactRef('provenance', path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved original within-pair distance history, aligned coordination windows and complete temporal comparison outcomes', tuple(refs))
