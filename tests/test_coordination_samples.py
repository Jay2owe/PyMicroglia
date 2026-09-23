"""Dependent pairs and repeated movies cannot manufacture biological replication."""
from pymicroglia._results import read_document
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import pymicroglia.measure.sample_contrasts as sample_contrasts
import pymicroglia.pipelines.coordination.samples as samples, pymicroglia.pipelines.coordination.sample_inputs as members
import pymicroglia.pipelines.coordination.sample_timing as timing, pymicroglia.pipelines.coordination.state_inputs as coordination_state_inputs
from pymicroglia.pipelines._contracts import SampleAssignment, Settings
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, _write_json
from tests.test_coordination_options import request

def fixture(tmp_path):
    rng = np.random.default_rng(761)
    parts = []
    biological = {}
    conditions = {}
    n = 40
    for condition, base in [('control', 0.2), ('treated', 0.8)]:
        for sample in range(4):
            for recording in range(2 if condition == 'control' and sample == 0 else 1):
                movie = f'{condition}-{sample}-{recording}'
                biological[movie] = f'{condition}-{sample}'
                conditions[movie] = condition
                x = rng.normal(size=n)
                x -= x.mean()
                x /= np.linalg.norm(x)
                z = rng.normal(size=n)
                z -= z.mean()
                z -= x * np.dot(x, z)
                z /= np.linalg.norm(z)
                rho = base + 0.01 * sample
                y = rho * x + np.sqrt(1 - rho * rho) * z
                for cell, values in [(7, x), (8, y)]:
                    parts.append(pd.DataFrame({'stem': movie, 'identity': cell, 'frame_index': np.arange(n), 'hours': 50.0 + np.arange(n) * 0.5, 'custom_signal': values, 'centroid_x': float(cell * 4), 'centroid_y': 0.0}))
    frame = pd.concat(parts, ignore_index=True)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {name: tmp_path / (name + '.csv') for name in tables}
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
    q = {'enabled': True, 'aggregation': 'mean', 'contrasts': [['control', 'treated']], 'evidence': {'method': 'independent_sample_permutation', 'min_samples': 4, 'resamples': 99, 'seed': 7, 'independent_samples': 'Independent controlled samples; recordings and pairs inside a sample are not independent'}}
    declared = request(target_measurements=['custom_signal'], biological_samples=biological, conditions=conditions, sample_summary=q, inference={'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'all'})
    return (resolve_request(declared, source_run='controlled-coordination-samples', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()}), paths)

def test_actual_sample_pipeline_preserves_repeated_recordings_and_reopens_without_comparison(tmp_path):
    resolved, paths = fixture(tmp_path)
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordination-samples'])
    assert first.successful, {key: value.outcome.reason for key, value in first.results.items()}
    result = samples.read_samples(first.results['coordination-samples'])
    units = result['unit_inventory']
    summaries = result['unit_summaries']
    assert len(units) == 8 and units.recordings.sum() == 9 and (units.cells.sum() == 18)
    repeated = units.loc[units['sample'].eq('control-0')].iloc[0]
    assert repeated.recordings == 2
    assert len(summaries) == 8 and summaries.formal_eligible.all()
    assert summaries.loc[summaries['sample'].eq('control-0'), 'value'].iloc[0] == pytest.approx(0.2)
    comparison = result['comparisons'].iloc[0]
    assert comparison.reference_samples == comparison.target_samples == 4
    assert comparison.effect == pytest.approx(0.6) and comparison.p_value == pytest.approx(2 / 70) and comparison.supported
    with patch.object(sample_contrasts, 'compare', side_effect=AssertionError('No new sample test')), patch.object(samples, 'aggregate', side_effect=AssertionError('No new aggregation')):
        second = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordination-samples'], presentation={'maps': {'edge_limit': 1}})
    assert second.successful and second.results['coordination-samples'].outcome.status == 'reused'
    assert samples.read_samples(second.results['coordination-samples'])['provenance'] == result['provenance']

def test_recording_balancing_and_model_or_baseline_exclusions_ignore_pair_significance():
    resolved = SimpleNamespace(inputs=SimpleNamespace(source_run='source', samples=(SampleAssignment('large', 'one', True), SampleAssignment('small', 'one', True))), request=SimpleNamespace(conditions={'large': 'control', 'small': 'control'}))
    prepared = {'cells': pd.DataFrame({'movie': ['large', 'large', 'small', 'small'], 'identity': [7, 8, 7, 8]}), 'inventory': pd.DataFrame({'movie': ['large'] * 100 + ['small']})}
    rows = [{'movie': movie, 'question_id': 'q', 'effect_id': movie + str(i), 'value': value, 'eligible': True, 'formal_eligible': True, 'reference_identity': 7, 'target_identity': 8, 'inference_exclusion': None, 'reason': 'Observed', 'original_pair_supported': False} for movie, count, value in [('large', 100, 1.0), ('small', 1, -1.0)] for i in range(count)]
    frame = pd.DataFrame(rows)
    definitions = pd.DataFrame([{'question_id': 'q'}])
    result = members.aggregate(resolved, prepared, frame, definitions, 'mean')
    assert result['unit_summaries'].value.iloc[0] == 0.0 and result['unit_summaries'].effects_eligible.iloc[0] == 101
    frame['original_pair_supported'] = True
    changed = members.aggregate(resolved, prepared, frame, definitions, 'mean')
    pd.testing.assert_frame_equal(result['unit_summaries'], changed['unit_summaries'])
    frame.loc[frame.movie.eq('small'), 'formal_eligible'] = False
    frame.loc[frame.movie.eq('small'), 'inference_exclusion'] = 'Model-development sample'
    changed = members.aggregate(resolved, prepared, frame, definitions, 'mean')
    assert changed['unit_summaries'].value.iloc[0] == 0.0 and (not changed['unit_summaries'].formal_eligible.iloc[0])

def test_state_source_rejects_newly_confirmed_or_changed_sample_membership(tmp_path):
    path = tmp_path / 'provenance.json'
    _write_json(path, {'resolved_request': {'inputs': {'source_run': 'source', 'samples': [{'movie': 'a', 'sample': 'trained', 'confirmed': True}], 'table_hashes': {}}, 'request': {'time_range_hours': None}}})
    source = SimpleNamespace(artifact=lambda _: path)
    resolved = SimpleNamespace(inputs=SimpleNamespace(source_run='source', samples=(SampleAssignment('a', 'trained', True),), table_hashes={}), request=SimpleNamespace(time_range_hours=None))
    assert coordination_state_inputs.matching_features(resolved, source)
    resolved.inputs.samples = (SampleAssignment('a', 'unused', True),)
    with pytest.raises(ValueError, match='biological-sample mapping'):
        coordination_state_inputs.matching_features(resolved, source)
    resolved.inputs.samples = (SampleAssignment('a', None, False),)
    assert coordination_state_inputs.matching_features(resolved, source)
    original = read_document(path)
    original['resolved_request']['inputs']['samples'][0].update(sample=None, confirmed=False)
    _write_json(path, original)
    resolved.inputs.samples = (SampleAssignment('a', 'newly-confirmed', True),)
    with pytest.raises(ValueError, match='biological-sample mapping'):
        coordination_state_inputs.matching_features(resolved, source)

def test_identical_measurements_cannot_be_renamed_into_comparable_phase_roles():
    resolved = SimpleNamespace(inputs=SimpleNamespace(source_run='source', samples=(SampleAssignment('a', 'one', True),)), request=SimpleNamespace(questions={'rhythm': {'enabled': True}}, measurement_pairs=[{'reference': 'signal', 'target': 'signal'}], conditions={'a': 'control'}, sample_summary={'contrasts': []}))
    effects = pd.DataFrame([{'question': 'rhythm', 'reference': 'signal', 'target': 'signal', 'movie': 'a', 'condition': 'control', 'source_run': 'source', 'effect_id': 'pair-result', 'pair_id': 'pair', 'reference_identity': 7, 'target_identity': 8, 'native_timing_evidence': {'status': 'eligible'}}])
    with patch.object(timing, 'load_source', side_effect=AssertionError('No invented role summary')):
        result, details, policy = timing.summarise(resolved, {'effects': effects}, {})
    assert result['timing_populations'].status.eq('incomparable_endpoint_roles').all()
    assert not result['timing_members'].summary_eligible.any() and all((not item['native_called'] for item in details))

def test_compatibility_strata_keep_adjustment_cadence_models_and_measurements_separate(tmp_path):
    resolved, _ = fixture(tmp_path)
    prepared = {'traces': pd.DataFrame([{'endpoint_id': endpoint, 'frame_index': i, 'hours': 50 + i * dt} for endpoint, dt in [('half-hour', 0.5), ('hourly', 1.0)] for i in range(4)])}
    base = {'source_run': 'source', 'movie': 'control-0-0', 'question': 'simultaneous', 'evidence_level': 'pair', 'reference': 'custom_signal', 'target': 'custom_signal', 'reference_measurement_id': 'signal', 'target_measurement_id': 'signal', 'reference_endpoint_id': 'half-hour', 'target_endpoint_id': 'half-hour', 'representation': 'raw', 'adjustment': 'unadjusted', 'statistic': 'pearson', 'descriptive_effect': 0.4, 'status': 'descriptive', 'reason': 'No pair test requested', 'decision': 'descriptive', 'supported': False, 'source_result_id': 'original', 'source_scientific_id': 'source', 'effect_id': 'base'}
    changes = [{}, {'adjustment': 'adjusted'}, {'target_endpoint_id': 'hourly'}, {'representation': 'changes'}, {'target': 'custom_shape', 'target_measurement_id': 'shape'}, {'question': 'states', 'model_id': 'model-one', 'evidence_kind': 'state_cooccupancy', 'reference_identity': 7, 'target_identity': 8}, {'question': 'states', 'model_id': 'model-two', 'evidence_kind': 'state_cooccupancy', 'reference_identity': 7, 'target_identity': 8}]
    frame = pd.DataFrame([{**base, **change, 'effect_id': str(i)} for i, change in enumerate(changes)])
    result, definitions = members.member_records(resolved, prepared, {'effects': frame})
    assert len(definitions) == len(changes) and result.question_id.nunique() == len(changes)
    assert result.eligible.all() and result.loc[result.question.eq('states'), 'formal_eligible'].eq(False).all()

def test_unconfirmed_insufficient_and_empty_populations_do_not_invent_replication(tmp_path):
    resolved, _ = fixture(tmp_path)
    unit = {'question_id': 'q', 'unit_id': 'one', 'sample_confirmed': False, 'condition': 'control', 'value': 0.2, 'formal_eligible': False, 'status': 'available', 'reason': 'Unconfirmed recording', 'inference_exclusions': ['Unconfirmed biological sample']}
    units = pd.DataFrame([unit, {**unit, 'unit_id': 'two', 'condition': 'treated', 'value': 0.8}])
    definitions = pd.DataFrame([{'question_id': 'q', 'question': 'simultaneous', 'value_kind': 'saved_association', 'is_hypothesis': True}])
    comparisons, families, _ = samples.comparisons(units, definitions, samples.policy(resolved.request), resolved.request.inference)
    result = comparisons.iloc[0]
    assert pd.isna(result.effect) and pd.isna(result.p_value) and (not result.supported)
    assert result.descriptive_effect_all_recorded_units == pytest.approx(0.6)
    assert result.reference_samples == result.target_samples == 0 and families.tested.iloc[0] == 0
    units['sample_confirmed'] = True
    units['formal_eligible'] = True
    comparisons, _, _ = samples.comparisons(units, definitions, samples.policy(resolved.request), resolved.request.inference)
    assert comparisons.effect.iloc[0] == pytest.approx(0.6) and pd.isna(comparisons.p_value.iloc[0])
    prepared = {'cells': pd.DataFrame(columns=['movie', 'identity']), 'inventory': pd.DataFrame(columns=['movie'])}
    empty, _ = members.member_records(resolved, {'traces': pd.DataFrame(columns=['endpoint_id'])}, {'effects': pd.DataFrame()})
    outputs = members.aggregate(resolved, prepared, empty, definitions.iloc[:0], 'mean')
    assert len(outputs['unit_inventory']) == 8 and outputs['unit_summaries'].empty
