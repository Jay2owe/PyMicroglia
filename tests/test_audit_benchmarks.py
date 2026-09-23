"""Benchmarks retain known truth and reserve confirmation without opening it."""
from dataclasses import replace
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.audit.options import resolve_request
from pymicroglia.pipelines.audit.benchmarks import evaluation_inputs, generate_development, read_benchmarks
from pymicroglia.pipelines.audit.workflow import run_request
from pymicroglia.pipelines._screening import file_hash
from tests.test_audit_options import declaration

@pytest.fixture
def fixture(tmp_path):
    table = pd.DataFrame({'stem': ['a'] * 160, 'identity': [1] * 160, 'frame_index': range(160), 'hours': [float(h) for h in range(160)], 'signal': [None if h in (20, 21) else float(h) for h in range(160)]})
    path = tmp_path / 'cell_frame.csv'
    table.to_csv(path, index=False)
    raw = declaration(biological_samples={'a': 'sample-1'})
    design = raw['benchmark_design']
    design['replicates'] = 2
    design['scenarios'] += [{'id': 'long', 'components': [{'id': 'long', 'period_hours': 36.0, 'amplitude': 1.0, 'waveform': 'square', 'duty': 0.2}], 'noise': {'kind': 'exponential', 'sd': 0.2, 'correlation_hours': 3.0}, 'drift': {'linear_per_hour': 0.005}}, {'id': 'multiple', 'components': [{'id': 'short', 'period_hours': 6.0, 'amplitude': 1.0, 'waveform': 'sawtooth'}, {'id': 'long', 'period_hours': 30.0, 'amplitude': 0.5, 'waveform': 'cosine'}]}, {'id': 'short', 'components': [{'id': 'short', 'period_hours': 12.0, 'amplitude': 1.0, 'waveform': 'triangle'}], 'retain_observations': 8}, {'id': 'disturbed-negative', 'components': [], 'disturbances': [{'kind': 'pulse', 'start_hours': 30.0, 'duration_hours': 1.0, 'amplitude': 2.0}]}]
    resolved = resolve_request(parse([raw])[0], source_run='run', tables={'cell_frame': table}, input_hashes={'cell_frame': file_hash(path)})
    return (resolved, {'cell_frame': path})

def test_known_truth_and_original_profiles_are_preserved(fixture, monkeypatch):
    request, _ = fixture
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('generation fitted'))
    result = generate_development(request)
    assert set(result.cases.truth_kind) == {'positive', 'negative', 'insufficient'}
    assert len(result.cases) == 12
    assert set(result.cases.loc[result.cases.scenario.eq('long'), 'truth_kind']) == {'positive'}
    assert result.cases.loc[result.cases.scenario.eq('multiple'), 'components'].map(len).eq(2).all()
    assert result.cases.loc[result.cases.scenario.eq('short'), 'observations'].eq(8).all()
    assert not result.cases.loc[result.cases.scenario.eq('multiple'), 'fresh_confirmation_eligible'].any()
    for row in result.traces.to_dict('records'):
        assert row['hours'] == [float(h) for h in range(160)]
        assert row['source_missing'][20] and row['values'][20] is None
    assert result.cases.iloc[0].profile_metadata['sample_assignment']['confirmed']
    assert result.reservation['opened'] is False and len(result.reservation['case_keys']) == 12
    assert 'values' not in str(result.reservation.keys())

def test_candidate_choices_cannot_change_truth_or_realizations(fixture):
    request, _ = fixture
    first = generate_development(request)
    changed = replace(request, candidates=())
    second = generate_development(changed)
    pd.testing.assert_frame_equal(first.cases, second.cases)
    pd.testing.assert_frame_equal(first.traces, second.traces)
    assert first.reservation == second.reservation
    inputs = evaluation_inputs(first)
    assert len(inputs) == 12
    assert all((row['metadata']['truth']['partition'] == 'development' for row in inputs))
    assert all((row['case_id'] in set(first.cases.case_id) for row in inputs))

def test_development_producer_never_opens_confirmation_and_reuses(fixture, tmp_path, monkeypatch):
    request, paths = fixture
    called = []
    original = circadian.generate_benchmark_cases

    def generate(*args, **kwargs):
        called.append(kwargs['partition'])
        assert kwargs['partition'] == 'development'
        return original(*args, **kwargs)
    monkeypatch.setattr(circadian, 'generate_benchmark_cases', generate)

    def run():
        return run_request(request, paths, tmp_path / 'audit', only=('development-cases',))
    first = run()
    assert first.successful, first.results['development-cases'].outcome.reason
    saved = read_benchmarks(first.results['development-cases'])
    assert saved.manifest['confirmation_opened'] is False
    assert run().results['development-cases'].outcome.status == 'reused'
    assert called == ['development']

def test_reserved_partition_is_fresh_but_deterministic_cases_are_not(fixture):
    request, _ = fixture
    development = generate_development(request)
    design = request.request.benchmark_design
    confirmation = circadian.generate_benchmark_cases({key: design[key] for key in ('replicates', 'truth_policy', 'scenarios')}, [request.profiles[0].as_dict()], partition='confirmation', seed=design['seed'])
    old = {(r['scenario'], r['replicate']): r for r in development.cases.to_dict('records')}
    for case in confirmation['cases']:
        previous = old[case['scenario'], case['replicate']]
        assert case['case_id'] != previous['case_id']
        assert case['targets'] == previous['targets']
        if case['fresh_confirmation_eligible']:
            assert case['observation_sha256'] != previous['observation_sha256']
        else:
            assert case['observation_sha256'] == previous['observation_sha256']
