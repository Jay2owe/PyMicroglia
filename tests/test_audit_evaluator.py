"""Candidate evidence keeps independent outcomes, original masks and fixed families."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines.audit.evaluator import evaluate_candidate, read_evaluation
from pymicroglia.pipelines._contracts import Settings, content_id
from pymicroglia.pipelines.audit.workflow import run_request
from pymicroglia.pipelines.audit.options import resolve_request
from pymicroglia.pipelines import parse
from pymicroglia.pipelines._screening import file_hash
from tests.test_audit_options import declaration

@pytest.fixture
def fixture(tmp_path):
    frame = pd.DataFrame([{'stem': movie, 'identity': 1, 'frame_index': hour, 'hours': float(hour), 'signal': float(hour % 12)} for movie in ('a', 'b') for hour in range(96)])
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    request = resolve_request(parse([declaration(candidates=[{'analysis_options': {'fit_method': 'mesa', 'significance_method': 'f'}}])])[0], source_run='run', tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(path)})
    from pymicroglia.pipelines.audit.evaluator import real_cases
    paths = {'cell_frame': path}
    return (request, paths, real_cases(request, paths))

@pytest.fixture
def methods(monkeypatch):
    calls = []

    def estimate(hours, values, params, method, **kwargs):
        calls.append((method, np.asarray(hours).copy(), np.asarray(values).copy(), params, kwargs))
        return {'method': method, 'status': 'ok', 'period_hours': 12.0, 'p_value': 0.01 if method == 'f' else 0.9, 'diagnostics': {'nested': [1, {'fit': 'saved'}]}, 'components': [{'period_hours': 12.0, 'amplitude': 2.0}], 'native_series': {}, 'native_result': {'row': {'period_hours': 12.0}}, 'workbench_run_record_json': json.dumps({'action': method})}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    return calls

def test_exact_independent_methods_and_full_settings(fixture, methods):
    request, _, cases = fixture
    result = evaluate_candidate(request.candidates[0], cases)
    assert [call[0] for call in methods] == ['mesa', 'f', 'mesa', 'f']
    assert all((call[4]['capture_details'] is True for call in methods))
    assert all((call[3]['period_search_hours'] == [2.0, 48.0] for call in methods))
    assert result.results.p_value.to_list() == [0.01, 0.01]
    assert result.results.estimator_p_value.to_list() == [0.9, 0.9]
    assert result.results.significant.all()
    assert result.families.requested.to_list() == [2]
    assert result.traces.waveform_status.to_list() == ['unavailable', 'unavailable']

@pytest.mark.parametrize('failed_method', ['mesa', 'f'])
def test_failure_does_not_erase_the_other_branch(fixture, monkeypatch, failed_method):
    request, _, cases = fixture

    def estimate(hours, values, params, method, **kwargs):
        if method == failed_method:
            raise ValueError('controlled independent failure')
        return {'method': method, 'status': 'ok', 'period_hours': 12.0, 'p_value': 0.01, 'diagnostics': {}, 'components': []}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    result = evaluate_candidate(request.candidates[0], cases)
    if failed_method == 'mesa':
        assert result.results.significant.all()
        assert not result.results.period_available.any()
        assert result.results.estimate_status.eq('failed').all()
    else:
        assert result.results.period_available.all()
        assert result.results.status.eq('untestable').all()
        assert result.results.q_value.isna().all()
    assert result.families.requested.iloc[0] == 2

def test_filtering_preserves_times_missing_rows_and_exact_inputs_to_both_methods(fixture, methods):
    request, _, cases = fixture
    case = cases[0].as_dict()
    case['values'][10] = None
    case['values'][30] = 1000.0
    case['hours'] = [hour if hour < 50 else hour + 20 for hour in case['hours']]
    candidate = replace(request.candidates[0], filtering=Settings({'method': 'median', 'window_hours': 3.0, 'max_gap_hours': 1.5, 'min_observations': 1}))
    result = evaluate_candidate(candidate, [Settings(case)])
    trace = result.traces.iloc[0]
    assert trace['hours'] == case['hours'] and trace['raw'][10] is None
    assert np.isnan(trace.filtered[10]) and trace.filtered[30] < 100
    assert trace.filter_result['observations'][49]['segment'] != trace.filter_result['observations'][50]['segment']
    assert 70 in methods[0][1] and 50 not in methods[0][1]
    np.testing.assert_array_equal(methods[0][1], methods[1][1])
    np.testing.assert_array_equal(methods[0][2], methods[1][2])
    assert trace.filter_result['workbench_run_record']

def test_adding_candidates_or_reordering_cases_does_not_change_family(fixture, methods):
    request, _, cases = fixture
    candidate = request.candidates[0]
    first = evaluate_candidate(candidate, cases)
    alternate = replace(candidate, candidate_id='alternate', analysis_options=Settings({**candidate.analysis_options.as_dict(), 'significance_method': 'lomb'}))
    evaluate_candidate(alternate, cases)
    repeated = evaluate_candidate(candidate, reversed(cases))
    assert first.families.family_id.to_list() == repeated.families.family_id.to_list()
    pd.testing.assert_frame_equal(first.results.sort_values('case_id').reset_index(drop=True), repeated.results.sort_values('case_id').reset_index(drop=True))

def test_insufficient_and_missing_cases_remain_in_complete_family(fixture, methods):
    request, _, cases = fixture
    missing, short = (cases[0].as_dict(), cases[1].as_dict())
    missing['values'] = [None] * 96
    short['hours'], short['values'] = (short['hours'][:5], short['values'][:5])
    result = evaluate_candidate(request.candidates[0], [Settings(missing), Settings(short)])
    assert not methods
    assert result.results.status.eq('untestable').all()
    assert result.families.requested.to_list() == [2]
    assert result.families.valid_tests.to_list() == [0]
    assert result.results.observations.to_list() == [0, 5]
    empty = evaluate_candidate(request.candidates[0], [])
    assert empty.results.empty and 'case_id' in empty.results

def test_saved_roundtrip_and_reuse_never_repeat_fits(fixture, methods, tmp_path):
    request, paths, cases = fixture
    first = run_request(request, paths, tmp_path / 'audit', only=('real-candidates',))
    saved = first.results['real-candidates']
    assert first.successful, saved.outcome.reason
    data = read_evaluation(saved)
    assert data.results.iloc[0].estimate_result['diagnostics']['nested'][1]['fit'] == 'saved'
    assert len(data.results) == 2
    fitted = len(methods)
    second = run_request(request, paths, tmp_path / 'audit', only=('real-candidates',), presentation={'columns': 2})
    assert second.results['real-candidates'].outcome.status == 'reused'
    assert len(methods) == fitted
    read_evaluation(saved)
    assert len(methods) == fitted

def test_actual_workbench_generated_non_daily_case(fixture):
    request, _, _ = fixture
    design = request.request.benchmark_design.as_dict()
    design.pop('justification')
    seed = design.pop('seed')
    design['replicates'] = 1
    generated = circadian.generate_benchmark_cases(design, [request.profiles[0].as_dict()], partition='development', seed=seed)
    trace = generated['traces'][0]
    case = Settings({'case_id': trace['case_id'], 'profile_id': request.profiles[0]['id'], 'measurement': 'signal', 'movie': 'a', 'identity': 1, 'family_block': 'known', 'hours': trace['hours'], 'values': trace['values'], 'metadata': {'constructed': True}})
    result = evaluate_candidate(request.candidates[0], [case])
    row = result.results.iloc[0]
    assert row['method'] == 'mesa' and row.significance_method == 'f'
    assert row.period_available and row.period_hours == pytest.approx(12.0, abs=1.5)
    assert row.estimate_result['workbench_run_record_json']
    assert row.significance_result['workbench_run_record_json']
