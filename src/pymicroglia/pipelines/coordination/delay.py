"""Physical cross-cell lag profiles, complete-search tests and delay uncertainty."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path

import numpy as np

import pymicroglia.measure.relationship_lag_statistics as native
from pymicroglia.measure.relationship_statistics import coefficient, TIE_TOLERANCE
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import _adjacent, _table
from pymicroglia.pipelines.coordination.simultaneous import VIEW_KEYS, read_temporal, temporal_refusal
from pymicroglia.pipelines.relationships.inputs import lag_values, match_observations, paired_support
from pymicroglia.pipelines.relationships.options import LAG_CONVENTION
from pymicroglia.pipelines._runner import Unavailable
from pymicroglia.pipelines._screening import _write_json, file_hash, write_table


RESULT_COLUMNS = VIEW_KEYS + ['result_id', 'question', 'evidence_level', 'effect', 'p_value', 'status', 'reason',
    'empirical_peak_lags_hours', 'best_lag_hours', 'estimated_delay_hours', 'delay_hours', 'delay_supported',
    'delay_interval_hours', 'candidate_lags_hours', 'resolution_status', 'resolution_reason', 'association_sign']


def implementation_version():
    return {'code': {path.name: file_hash(path) for path in [Path(__file__), Path(native.__file__),
        source_file('coordination_simultaneous.py'), source_file('coordination_inputs.py'),
        source_file('relationship_inputs.py'), source_file('relationship_statistics.py')]},
        'libraries': {name: library_version(name) for name in ['numpy', 'pandas', 'scipy', 'arch']}}


def identity(context):
    return content_id({'pair_inputs': context.saved('pair-inputs').outcome.scientific_id,
        'temporal_inputs': context.saved('simultaneous-coordination').outcome.scientific_id,
        'question': context.request.request.questions['delay'], 'implementation': implementation_version()})


def validate_question(question):
    if not question['enabled']: return
    if question['statistic'] not in {'pearson', 'spearman'} or question['evidence']['method'] not in {'none', 'truncated_time_shift'}:
        raise Unavailable('Cross-cell delays support Pearson/Spearman complete-search time-shift evidence and stationary-bootstrap curve uncertainty')
    native.validate_lag_question(question)


def observed_profile(left, right, lag, question, support):
    matches, diagnostics = match_observations(left, right, lag, support)
    coverage = paired_support(matches, diagnostics, support)
    effect, reason = coefficient(matches.reference_value, matches.target_value, question['statistic'])
    good = coverage['status'] == 'eligible' and effect is not None
    profile = {'lag_hours': float(lag), **coverage, 'effect': effect, 'status': 'descriptive' if good else 'untestable',
        'reason': reason if coverage['status'] == 'eligible' else coverage['reason'],
        'tested_effect': None, 'tested_observations': None, 'coefficient_lower': None, 'coefficient_upper': None,
        'compatible_delay': None, 'interval_kind': None}
    lookup = {row['observation_id']: row for frame in [left, right] for row in frame.to_dict('records')}
    intervals = []; records = matches.to_dict('records')
    for previous, current in zip(records, records[1:]):
        a, b = [lookup[row['reference_observation']] for row in [previous, current]]
        c, d = [lookup[row['target_observation']] for row in [previous, current]]
        first, why_a = _adjacent(a, b, support['max_gap_hours']); second, why_b = _adjacent(c, d, support['max_gap_hours'])
        start, end = max(a['hours'], c['hours']+lag), min(b['hours'], d['hours']+lag)
        valid = bool(first and second and end > start)
        intervals.append({'lag_hours': float(lag), 'reference_start_hours': a['hours'], 'reference_end_hours': b['hours'],
            'target_start_hours': c['hours'], 'target_end_hours': d['hours'],
            'first_reference_observation': a['observation_id'], 'last_reference_observation': b['observation_id'],
            'first_target_observation': c['observation_id'], 'last_target_observation': d['observation_id'],
            'aligned_start_hours': start, 'aligned_end_hours': end, 'aligned_interval_hours': float(end-start) if valid else 0.,
            'valid': valid, 'reason': 'Original supported intervals aligned by the declared physical lag; not simultaneous exposure' if valid else why_a if not first else why_b if not second else 'No aligned interval'})
    profile['aligned_observed_hours'] = sum(row['aligned_interval_hours'] for row in intervals)
    return profile, records, intervals


def analyse(resolved, temporal, scientific_id):
    question = resolved.request.questions['delay']; validate_question(question)
    lags = lag_values(question); support = resolved.request.support
    series = {key: frame for key, frame in temporal['series'].groupby('series_id', sort=False)}
    rows = []; profiles = []; matches = []; intervals = []; shifts = []; details = []
    for view in temporal['pair_views'].to_dict('records'):
        left, right = [series.get(view[role+'_series_id'], temporal['series'].iloc[:0]) for role in ['reference', 'target']]
        local = []; result_id = content_id({'scientific_id': scientific_id, 'view': view['view_id'], 'question': 'delay'})
        key = {**view, 'result_id': result_id}
        for lag in lags:
            profile, matched, elapsed = observed_profile(left, right, lag, question, support)
            if not view['geometry_population_eligible']:
                profile.update(status='outside_geometry_population', reason='Outside the predeclared geometric population')
            local.append({**key, **profile})
            matches.extend({**key, 'lag_hours': float(lag), **row} for row in matched)
            intervals.extend({**key, **row} for row in elapsed)
        actual = question.as_dict() if hasattr(question, 'as_dict') else dict(question)
        if not view['is_hypothesis']:
            actual = {**actual, 'evidence': {'method': 'none'}, 'peak_resolution': {'method': 'none'}}
        refusal = temporal_refusal(left, right, support)
        inferential = actual['evidence']['method'] != 'none' or actual.get('peak_resolution', {}).get('method') == 'stationary_bootstrap'
        if not view['geometry_population_eligible']:
            tested = {'status': 'outside_geometry_population', 'reason': 'Outside the predeclared geometric population', 'p_value': None}
        elif refusal and inferential:
            tested = {'status': 'untestable', 'reason': refusal, 'p_value': None}
        else:
            tested = native.evaluate(left, right, actual, support, lags)
        valid = [item for item in local if item['status'] == 'descriptive']
        maximum = max((abs(item['effect']) for item in valid), default=None)
        empirical = [item for item in valid if abs(item['effect']) >= maximum-TIE_TOLERANCE] if maximum is not None else []
        raw_peaks = [item['lag_hours'] for item in empirical]
        row = {**key, 'question': 'delay', 'evidence_level': 'pair', 'statistic': actual['statistic'],
            'method': actual['evidence']['method'], 'evidence_settings': actual['evidence'],
            'status': tested['status'], 'reason': tested['reason'], 'p_value': tested['p_value'],
            'effect': tested.get('effect'), 'search_statistic': tested.get('search_statistic'),
            'effect_population': 'Fixed common reference window for the full tested lag search',
            'descriptive_peak_lags_hours': raw_peaks, 'descriptive_peak_effects': [item['effect'] for item in empirical],
            'empirical_peak_lags_hours': tested.get('empirical_peak_lags_hours', []),
            'best_lag_hours': None, 'estimated_delay_hours': tested.get('delay_hours'), 'delay_hours': None, 'delay_supported': False,
            'delay_interval_hours': tested.get('delay_interval_hours'), 'candidate_lags_hours': tested.get('candidate_lags_hours', []),
            'resolution_status': tested.get('resolution_status', 'unavailable'),
            'resolution_reason': tested.get('resolution_reason', 'Complete search or requested uncertainty is unavailable'),
            'tested_observations': tested.get('tested_observations'), 'tested_start_hours': tested.get('tested_start_hours'),
            'tested_end_hours': tested.get('tested_end_hours'), 'minimum_attainable_p': tested.get('minimum_attainable_p'),
            'uncertainty_method': actual.get('peak_resolution', {}).get('method', 'none'),
            'uncertainty_scope': tested.get('uncertainty', {}).get('coverage_scope'), 'lag_convention': LAG_CONVENTION,
            'selection_status': 'Unadjusted complete-search result; stage 09 decides supported association and delay',
            'shared_reference_interpretation': temporal['provenance']['adjustment_interpretation'] if view['adjustment'] != 'none' else None}
        uncertainty = tested.get('uncertainty', {}); bounds = uncertainty.get('coefficient_bounds')
        if tested.get('profile') is not None:
            for i, profile in enumerate(local):
                profile.update(tested_effect=tested['profile'][i], tested_observations=tested['tested_observations'])
                if bounds is not None:
                    profile.update(coefficient_lower=bounds[0][i], coefficient_upper=bounds[1][i],
                        compatible_delay=profile['lag_hours'] in row['candidate_lags_hours'], interval_kind=uncertainty['coefficient_interval_kind'])
        elif row['status'] == 'descriptive':
            if not valid: row.update(status='untestable', reason='No declared lag has sufficient finite nonconstant paired observations')
            else:
                row.update(effect=empirical[0]['effect'], search_statistic=maximum, empirical_peak_lags_hours=raw_peaks,
                    effect_population='Descriptive original per-lag overlaps; observation support can differ across the grid')
        peaks = row['empirical_peak_lags_hours']
        if len(peaks) == 1: row['best_lag_hours'] = peaks[0]
        values = tested.get('profile', [item['effect'] for item in local])
        peak_effects = [value for lag, value in zip(lags, values) if lag in peaks and value is not None]
        signs = set(np.sign(peak_effects)); row['peak_effects'] = peak_effects
        row['association_sign'] = 'mixed' if len(signs) > 1 else 'positive' if signs == {1} else 'negative' if signs == {-1} else 'undefined'
        if row['association_sign'] == 'mixed': row['effect'] = None
        if row['resolution_status'] == 'unavailable' and peaks:
            if min(lags) in peaks or max(lags) in peaks:
                row.update(resolution_status='search-boundary', resolution_reason='An empirical maximum reaches a search boundary; no supported interior delay')
            elif len(peaks) > 1:
                row.update(resolution_status='multiple-candidates', resolution_reason='Multiple empirical maxima remain; no supported unique delay')
        rows.append(row); profiles.extend(local); details.append({**key, 'result': tested})
        shifts.extend({'result_id': result_id, 'view_id': view['view_id'], 'shift_observations': shift,
            'shift_hours': shift*tested['sample_interval_hours'], 'maximum_absolute_statistic': value}
            for shift, value in zip(tested.get('shifts_observations', []), tested.get('shift_statistics', [])))
    return {'pair_effects': _table(rows, RESULT_COLUMNS),
        'profiles': _table(profiles, VIEW_KEYS+['result_id', 'lag_hours', 'effect', 'status', 'reason', 'paired_observations',
            'tested_effect', 'tested_observations', 'coefficient_lower', 'coefficient_upper', 'compatible_delay']),
        'matches': _table(matches, VIEW_KEYS+['result_id', 'lag_hours', 'reference_observation', 'target_observation', 'reference_hours', 'target_hours']),
        'matching_intervals': _table(intervals, VIEW_KEYS+['result_id', 'lag_hours', 'aligned_start_hours', 'aligned_end_hours', 'aligned_interval_hours', 'valid']),
        'shift_reference': _table(shifts, ['result_id', 'view_id', 'shift_observations', 'shift_hours', 'maximum_absolute_statistic'])}, details


def produce(context):
    question = context.request.request.questions['delay']
    if not question['enabled']: return StepResult(context.step.name, context.scientific_id, 'skipped-empty', 'Delay question was explicitly disabled')
    validate_question(question)
    temporal = read_temporal(context.saved('simultaneous-coordination'))
    tables, details = analyse(context.request, temporal, context.scientific_id)
    context.output.mkdir(parents=True); refs = []
    for name, frame in tables.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    for name, document in {'engine_details': details, 'provenance': {'schema_version': 1, 'scientific_id': context.scientific_id,
        'pair_inputs_id': context.saved('pair-inputs').outcome.scientific_id,
        'temporal_inputs_id': context.saved('simultaneous-coordination').outcome.scientific_id, 'question': question,
        'implementation': implementation_version(), 'lag_convention': LAG_CONVENTION,
        'method_reference': native.REFERENCE, 'uncertainty_reference': native.BOOTSTRAP_REFERENCE,
        'preprocessing_repeated': False, 'simultaneous_selection_applied': False,
        'observed_profile': 'All original per-lag matched observations and aligned support, with no interpolation across gaps',
        'tested_profile': 'One common original reference window at all lags and all complete-search surrogates',
        'uncertainty': 'Approximate simultaneous within-curve confidence set on the declared grid; no across-pair selection-adjusted coverage',
        'interpretation': 'Temporal association and grid-delay limits under declared stationarity; no causal ordering or rhythm phase claim'}}.items():
        path = context.output/(name+'.json'); _write_json(path, document)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved full cross-cell lag profiles, separate complete-search evidence and delay uncertainty without a simultaneous gate', tuple(refs))
