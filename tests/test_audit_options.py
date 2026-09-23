"""The audit freezes input-only populations and complete, distinct recipes."""
from pymicroglia._results import read_document
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.audit.options import resolve_request
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.audit.workflow import run_request
from pymicroglia.pipelines._screening import file_hash

def declaration(**overrides):
    return {'pipeline': 'method-selection-audit', 'measurements': ['signal'], 'candidates': [{}], 'analysis_options': {'detrend': 'none'}, 'benchmark_design': {'replicates': 3, 'seed': 912, 'justification': 'Test fixtures span non-daily periods with matched original sampling.', 'truth_policy': {'min_observations': 24, 'min_cycles': 3, 'period_min_hours': 2, 'period_max_hours': 48, 'relative_tolerance': 0.1, 'absolute_tolerance_hours': 0.5, 'target': 'all', 'extra_components': 'penalize'}, 'scenarios': [{'id': 'positive', 'components': [{'id': 'short', 'period_hours': 12, 'amplitude': 1, 'waveform': 'triangle'}], 'noise': {'kind': 'white', 'sd': 0.5}}, {'id': 'negative', 'components': [], 'noise': {'kind': 'white', 'sd': 0.5}}]}, 'score_policy': {'false_alarm_limit': 0.1, 'confidence': 0.95, 'min_positive': 30, 'min_negative': 30, 'min_valid_fraction': 0.9, 'recovery_margin': 0.03, 'uncertainty': 'stratified_independent_realizations'}, **overrides}

@pytest.fixture
def tables():
    frame = pd.DataFrame([{'stem': movie, 'identity': cell, 'frame_index': hour, 'hours': float(hour), 'signal': np.nan if hour in (9, 10, 30) else float(hour + cell), 'new_measurement': float(cell - hour)} for movie in ('a', 'b') for cell in (1, 2, 3) for hour in range(96)])
    return {'cell_frame': frame}

def resolved(tables, **overrides):
    return resolve_request(parse([declaration(**overrides)])[0], source_run='run', tables=tables, input_hashes={name: content_id(name) for name in tables})

def test_config_parse_has_no_engine_or_renderer_import():
    code = "\nimport json, sys\nfrom pymicroglia.pipelines import parse\nassert parse([json.loads(sys.argv[1])])[0].pipeline == 'method-selection-audit'\nassert not any(n.startswith(('pymicroglia.workbench', 'circadian_workbench', 'matplotlib')) for n in sys.modules)\n"
    subprocess.run([sys.executable, '-c', code, json.dumps(declaration())], check=True, capture_output=True, text=True)

def test_all_live_methods_and_arbitrary_measurements_are_selectable(tables):
    from pymicroglia import workbench as circadian
    request = resolved(tables, measurements=['signal', 'new_measurement'], candidates=[{'analysis_options': {'fit_method': method, 'significance_method': 'lomb'}} for method in circadian.PERIOD_METHODS])
    assert {c.analysis_options['fit_method'] for c in request.candidates} == set(circadian.PERIOD_METHODS)
    assert len(request.profiles) == 12
    assert all((c.analysis_options['significance_method'] == 'lomb' for c in request.candidates))
    assert all((set(c.analysis_options) == set(circadian.CIRCADIAN_ANALYSIS_OPTIONS) for c in request.candidates))
    assert request.request.recommendation_scope == 'measurement'

def test_complete_recipe_identity_aliases_and_labels(tables):
    request = resolved(tables, candidates=[{'label': 'first', 'analysis_options': {'detrend': 'loess'}}, {'label': 'same', 'analysis_options': {'detrend': 'lowess'}}, {'analysis_options': {'detrend': 'lowess', 'detrend_lowess_fraction': 0.3}}, {'analysis_options': {'detrend': 'lowess', 'significance_method': 'f'}}, {'analysis_options': {'detrend': 'lowess'}, 'filter': {'method': 'median', 'window_hours': 3, 'max_gap_hours': 1.5}}])
    assert len(request.candidates) == 4
    assert request.candidates[0].labels == ('first', 'same')
    assert len({c.candidate_id for c in request.candidates}) == 4
    assert request.candidates[-1].filtering['window_hours'] == 3

def test_polynomial_alias_retains_its_degree(tables):
    request = resolved(tables, candidates=[{'analysis_options': {'detrend': 'poly6'}}, {'analysis_options': {'detrend': 'polynomial', 'detrend_polynomial_degree': 6}}, {'analysis_options': {'detrend': 'polynomial', 'detrend_polynomial_degree': 3}}])
    assert len(request.candidates) == 2
    assert request.candidates[0].analysis_options['detrend_polynomial_degree'] == 6

def test_method_specific_settings_are_frozen(tables):
    request = resolved(tables, recommendation_scope='dataset', candidates=[{'analysis_options': {'fit_method': 'jtk', 'significance_method': 'f', 'period_config': {'jtk_periods': [8.0, 12.0, 16.0], 'jtk_seed': 713}, 'multiple_testing': 'sidak', 'min_cycles': 4, 'period_min_hours': 3}}])
    candidate = request.candidates[0]
    assert candidate.rhythm_params['jtk_periods'] == [8.0, 12.0, 16.0]
    assert candidate.engine_settings['jtk_seed'] == 713
    assert candidate.analysis_options['multiple_testing'] == 'sidak'
    assert candidate.analysis_options['min_cycles'] == 4
    assert request.evaluation_design['recommendation_scope'] == 'dataset'

def test_representative_selection_uses_only_coverage(tables, monkeypatch):
    from pymicroglia import workbench as circadian
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('resolution fitted'))
    population = {'mode': 'representative', 'per_stratum': 1, 'seed': 6, 'duration_edges_hours': [48, 120], 'missing_edges': [0.1, 0.5]}
    first = resolved(tables, population=population, biological_samples={'a': 'mouse', 'b': 'mouse'})
    tables['cell_frame'].loc[tables['cell_frame'].signal.notna(), 'signal'] = -12345.0
    second = resolved(tables, population=population, biological_samples={'a': 'mouse', 'b': 'mouse'})
    assert [p['id'] for p in first.profiles] == [p['id'] for p in second.profiles]
    assert len(first.profiles) == 2 and len(first.population_inventory) == 6
    assert {p['metadata']['cell']['movie'] for p in first.profiles} == {'a', 'b'}
    assert all((p['metadata']['sample_assignment']['confirmed'] for p in first.profiles))
    assert all((len(p['hours']) == 96 and sum(p['missing']) == 3 for p in first.profiles))

def test_explicit_selection_preserves_movie_cell_identity(tables):
    request = resolved(tables, population={'mode': 'explicit', 'cells': [{'movie': 'a', 'identity': 1}, {'movie': 'b', 'identity': 1}]})
    assert len(request.profiles) == 2
    assert {p['metadata']['cell']['movie'] for p in request.profiles} == {'a', 'b'}

@pytest.mark.parametrize('overrides, message', [({'candidates': []}, 'candidates'), ({'measurements': ['hours']}, 'identity/time'), ({'candidates': [{'filter': {'method': 'median', 'window_hours': 3}}]}, 'max_gap_hours'), ({'population': {'mode': 'significant'}}, 'population.mode'), ({'score_policy': {}}, 'score_policy requires'), ({'recommendation_scope': 'cell'}, 'recommendation_scope'), ({'candidates': [{'analysis_options': {'period_config': {'period_min_hours': 7}}}]}, 'conflicts'), ({'candidates': [{'analysis_options': {'period_config': {'bin_minutes': 30}}}]}, 'actual sampling'), ({'candidates': [{'analysis_options': {'fit_method': 'unknown'}}]}, 'unknown'), ({'candidates': [{'analysis_options': {'significance_method': 'fft_nlls'}}]}, 'cannot test rhythmicity')])
def test_invalid_choices_explain_problem(tables, overrides, message):
    with pytest.raises(ValueError, match=message):
        resolved(tables, **overrides)

def test_design_freeze_reuses_without_generating_or_fitting(tables, tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    paths = {name: tmp_path / f'{name}.csv' for name in tables}
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
    request = resolve_request(parse([declaration()])[0], source_run='run', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    monkeypatch.setattr(circadian, 'generate_benchmark_cases', lambda *a, **k: pytest.fail('opened cases'))
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('fitted'))
    first = run_request(request, paths, tmp_path / 'audit', only=('audit-design',))
    assert first.successful, first.results['audit-design'].outcome.reason
    record = read_document(first.results['audit-design'].artifact('audit_design'))
    assert record['confirmation_opened'] is False
    assert record['profile_count'] == 6
    repeated = run_request(request, paths, tmp_path / 'audit', presentation={'columns': 4}, only=('audit-design',))
    assert repeated.results['audit-design'].outcome.status == 'reused'

def test_command_runs_audit_from_original_measurement_tables(tables, tmp_path):
    from pymicroglia.cli import main
    run = tmp_path / 'run'
    folder = run / 'pooled' / 'tables'
    folder.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(folder / f'{name}.csv', index=False)
    (run / 'manifest.json').write_text(json.dumps({'movies': [{'stem': 'a', 'modules': []}, {'stem': 'b', 'modules': []}]}), encoding='utf-8')
    request_path = tmp_path / 'audit-request.json'
    request_path.write_text(json.dumps(declaration()), encoding='utf-8')
    assert main(['run','method_audit',f'run={run}',f'pipeline_request={request_path}',"only=['audit-design']",'--claim','Verify the saved audit design command']) == 0
    from pymicroglia.pipelines import _records
    records = _records.read(run / 'pipelines' / 'method-selection-audit')['invocations']
    assert len(records) == 1 and next(iter(records.values()))['successful'] is True
