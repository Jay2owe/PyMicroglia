"""Alterations are paired raw-data masks, never new biological replicates."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.audit.evaluator import evaluate_candidate, real_cases
from pymicroglia.pipelines.audit.options import resolve_request
from pymicroglia.pipelines.audit.stability import altered_cases, compare_pairs, summarize_pairs
from pymicroglia.pipelines.audit.workflow import run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_audit_options import declaration

@pytest.fixture
def fixture(tmp_path):
    table = pd.DataFrame({'stem': ['a'] * 96, 'identity': [1] * 96, 'frame_index': range(96), 'hours': [float(h) for h in range(96)], 'signal': [None if h in (20, 21) else float(h % 12) for h in range(96)]})
    path = tmp_path / 'cell_frame.csv'
    table.to_csv(path, index=False)
    raw = declaration(stability={'alterations': [{'start_fraction': 0.4, 'end_fraction': 0.5}, {'start_fraction': 0.0, 'end_fraction': 0.9}]}, biological_samples={'a': 'sample-1'}, candidates=[{'analysis_options': {'fit_method': 'mesa', 'significance_method': 'f'}}, {'filter': {'method': 'median', 'window_hours': 3.0, 'max_gap_hours': 1.5}, 'analysis_options': {'fit_method': 'mesa', 'significance_method': 'f'}}])
    resolved = resolve_request(parse([raw])[0], source_run='run', tables={'cell_frame': table}, input_hashes={'cell_frame': file_hash(path)})
    return (resolved, {'cell_frame': path})

@pytest.fixture
def methods(monkeypatch):
    calls = []

    def estimate(hours, values, params, method, **kwargs):
        calls.append((method, list(hours), np.asarray(values).copy()))
        return {'method': method, 'status': 'ok', 'period_hours': 12.0, 'p_value': 0.001, 'components': [{'period_hours': 12.0}, {'period_hours': 24.0}], 'diagnostics': {}}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    return calls

def test_identical_raw_omissions_preserve_original_gap_and_times(fixture):
    resolved, paths = fixture
    original = real_cases(resolved, paths)
    altered, manifest = altered_cases(original, resolved.request.stability)
    repeated, repeated_manifest = altered_cases(original, resolved.request.stability)
    assert altered == repeated
    pd.testing.assert_frame_equal(manifest, repeated_manifest)
    for case in altered:
        assert case['hours'] == original[0]['hours']
        assert case['values'][20:22] == [None, None]
    middle = next((row for row in altered if row['values'][0] is not None))
    assert middle['values'][38:48] == [None] * 10
    assert middle['hours'][48] == 48.0 and middle['values'][48] == 0.0
    assert manifest.operation.map(lambda row: bool(row['workbench_run_record'])).all()

@pytest.mark.parametrize('period,p,test_status,estimate_status,expected_period,expected_test', [(12.0, 0.001, 'ok', 'ok', 'stable_supported', 'retained_detection'), (24.0, 0.001, 'ok', 'ok', 'changed_supported', 'retained_detection'), (12.0, 0.8, 'ok', 'ok', 'stable_supported', 'lost_detection'), (None, 0.001, 'ok', 'failed', 'unevaluable', 'retained_detection'), (12.0, None, 'failed', 'ok', 'stable_supported', 'unevaluable')])
def test_independent_period_and_detection_pair_states(fixture, methods, period, p, test_status, estimate_status, expected_period, expected_test):
    resolved, paths = fixture
    original = real_cases(resolved, paths)
    candidate = resolved.candidates[0]
    baseline = evaluate_candidate(candidate, original)
    altered, manifest = altered_cases(original, resolved.request.stability)
    changed = evaluate_candidate(candidate, altered)
    usable = changed.results.observations.ge(24)
    changed.results.loc[usable, ['period_hours', 'p_value', 'q_value', 'test_status', 'estimate_status', 'period_available', 'period_underdetermined']] = [period, p, p, test_status, estimate_status, period is not None, period is None]
    pairs = compare_pairs(candidate, original, altered, manifest, baseline, changed, resolved.request.stability)
    row = pairs[pairs.altered_observations.ge(24)].iloc[0]
    assert row.period_comparison == expected_period and row.detection_comparison == expected_test
    if period == 24.0:
        assert row.component_switch and row.period_shift_hours == 12.0
    insufficient = pairs[pairs.altered_observations.lt(24)].iloc[0]
    assert insufficient.period_comparison == insufficient.detection_comparison == 'unevaluable'
    assert pd.isna(insufficient.period_shift_hours)

def test_recipe_reprocesses_each_mask_and_keeps_failed_pair_denominators(fixture, methods, tmp_path, monkeypatch):
    resolved, paths = fixture
    filtering_calls = []
    filter_trace = circadian.filter_rhythm_trace

    def filter_raw(hours, values, filtering):
        filtering_calls.append((list(hours), list(values), dict(filtering)))
        return filter_trace(hours, values, filtering)
    monkeypatch.setattr(circadian, 'filter_rhythm_trace', filter_raw)
    first = run_request(resolved, paths, tmp_path / 'audit', only=('real-stability',))
    assert first.successful, first.results['real-stability'].outcome.reason
    saved = first.results['real-stability']
    pairs = read_table(saved.artifact('stability_pairs'))
    summary = read_table(saved.artifact('stability_summary'))
    assert len(pairs) == 4 and pairs.altered_case_id.nunique() == 2
    assert len(filtering_calls) == 6
    assert all((len(call[0]) == 96 for call in filtering_calls))
    masks = [np.isnan(np.asarray(call[1], float)).tolist() for call in filtering_calls]
    assert masks[2:] == masks[2:4] * 2
    rows = summary[summary.scope.eq('measurement')]
    assert rows.pairs.eq(2).all() and rows.valid_period_pairs.eq(1).all()
    assert rows.stable_fraction_of_all_pairs.eq(0.5).all()
    assert rows.confirmed_sample_pairs.eq(2).all()
    count = len(methods)
    again = run_request(resolved, paths, tmp_path / 'audit', only=('real-stability',))
    assert again.results['real-stability'].outcome.status == 'reused' and len(methods) == count

def test_no_alterations_is_no_evidence_and_noop_masks_are_not_stability(fixture, methods):
    resolved, paths = fixture
    original = real_cases(resolved, paths)
    policy = {**resolved.request.stability.as_dict(), 'alterations': []}
    altered, manifest = altered_cases(original, policy)
    assert altered == () and manifest.empty
    from pymicroglia.pipelines.audit.stability import PAIR_COLUMNS
    from pymicroglia.pipelines._contracts import Settings
    summary = summarize_pairs(replace(resolved, request=replace(resolved.request, stability=Settings(policy))), pd.DataFrame(columns=PAIR_COLUMNS))
    assert summary.status.eq('not_requested').all()
    assert summary.stable_fraction_of_supported_pairs.isna().all()
    policy['alterations'] = [{'start_fraction': 20 / 95, 'end_fraction': 21 / 95}]
    altered, manifest = altered_cases(original, policy)
    assert not manifest.input_changed.any()
    candidate = resolved.candidates[0]
    pairs = compare_pairs(candidate, original, altered, manifest, evaluate_candidate(candidate, original), evaluate_candidate(candidate, altered), policy)
    assert pairs.period_comparison.eq('unchanged_input').all() and pairs.period_shift_hours.isna().all()
