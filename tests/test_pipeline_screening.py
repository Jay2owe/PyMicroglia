"""Saved screens preserve the full family, evidence and original observations."""
from __future__ import annotations
from pymicroglia._results import read_document
import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
from pymicroglia.pipelines._contracts import CellKey, Settings
from pymicroglia.pipelines.rhythm.discovery import resolve_request
from pymicroglia.pipelines._screening import file_hash, read_screen, read_verified_tables, screen, screen_identity

def inputs(tmp_path, *, scope='all', empty=False, options=None, **request_options):
    observations = []
    if not empty:
        for stem, left, right in (('a', 10.0, 20.0), ('b', 30.0, 80.0)):
            for frame in range(8):
                observations.append(dict(stem=stem, subject='same-label', identity=7, frame_index=frame, hours=float(frame), signal=left + frame, another=right + frame))
    tables = {'cell_frame': pd.DataFrame(observations, columns=['stem', 'subject', 'identity', 'frame_index', 'hours', 'signal', 'another']), 'cell_summary': pd.DataFrame([] if empty else [{'stem': 'a', 'subject': 'same-label', 'identity': 7}, {'stem': 'a', 'subject': 'same-label', 'identity': 9}, {'stem': 'b', 'subject': 'same-label', 'identity': 7}], columns=['stem', 'subject', 'identity'])}
    paths = {name: tmp_path / (name + '.csv') for name in tables}
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
    hashes = {name: file_hash(path) for name, path in paths.items()}
    loaded = read_verified_tables(paths, hashes)
    request = parse([{'pipeline': 'rhythm-discovery', 'test_measurements': ['signal', 'another'], 'correction_scope': scope, 'analysis_options': {'fit_method': 'mesa', 'significance_method': 'f', 'detrend': 'none', 'min_observations': 6, **(options or {})}, **request_options}])[0]
    resolved = resolve_request(request, source_run='run-one', tables=loaded, input_hashes=hashes)
    return (resolved, paths)

@pytest.fixture
def fake_methods(monkeypatch):
    calls = []

    def estimate(hours, values, params, method, **kwargs):
        calls.append((np.asarray(hours).copy(), np.asarray(values).copy(), params, method, kwargs))
        return {'status': 'ok', 'period_hours': 12.0, 'phase_hours': 2.0, 'p_value': float(values[0]) / 1000 if method == 'f' else None, 'significant': True if method == 'f' else None, 'components': [{'period_hours': 12.0, 'amplitude': 3.0}], 'diagnostics': {'status': 'ok', 'nested': {'missing': None}}, 'detrend': params['detrend'], 'native_result': {'source': method}, 'native_series': {}, 'requested_settings': {'method': method}, 'effective_settings': [{'method': method}], 'workbench_run_record_json': json.dumps({'method': method})}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    return calls

@pytest.mark.parametrize('scope', ['all', 'measurement', 'movie_measurement'])
def test_complete_families_and_repeated_movie_cell_keys(tmp_path, fake_methods, scope):
    resolved, paths = inputs(tmp_path, scope=scope)
    saved = screen(resolved, paths, tmp_path / 'screen')
    results = saved.results
    assert len(results) == 6
    assert set(zip(results.movie, results.identity)) == {('a', 7), ('a', 9), ('b', 7)}
    absent = results[results.identity.eq(9)]
    assert set(absent.status) == {'untestable'}
    assert set(absent.reason) == {'no_observations'}
    assert results.sample_confirmed.eq(False).all()
    assert results.observed_subject.eq('same-label').all()
    for family in saved.families.to_dict('records'):
        members = results[results.family_id.eq(family['family_id'])]
        raw = pd.to_numeric(members.p_value).to_numpy(float)
        np.testing.assert_allclose(pd.to_numeric(members.q_value), circadian.adjust_pvalues(raw, 'bh'), equal_nan=True)
        assert family['requested'] == len(members) == len(family['members'])
        assert family['valid_tests'] == family['correction_denominator'] == np.isfinite(raw).sum()
    assert len(fake_methods) == 8
    assert all((call[4]['capture_details'] for call in fake_methods))
    assert all((call[3] in {'mesa', 'f'} for call in fake_methods))
    assert results[results.test_status.eq('ok')].period_underdetermined.all()
    assert results.status.eq('significant').any()

def test_round_trip_and_selections_do_not_fit_or_correct(tmp_path, fake_methods, monkeypatch):
    resolved, paths = inputs(tmp_path)
    original = screen(resolved, paths, tmp_path / 'screen')
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('read-back fitted'))
    monkeypatch.setattr(circadian, 'adjust_pvalues', lambda *a, **k: pytest.fail('read-back corrected'))
    loaded = read_screen(tmp_path / 'screen', expected_id=original.outcome.scientific_id)
    pd.testing.assert_frame_equal(original.results, loaded.results)
    pd.testing.assert_frame_equal(original.traces, loaded.traces)
    assert loaded.outcome == original.outcome
    assert loaded.results.iloc[0].estimate_result['diagnostics']['nested']['missing'] is None
    selections = {s.name: s for s in loaded.outcome.selections}
    for metric in ('signal', 'another'):
        for status in ('significant', 'not-significant', 'untestable'):
            rows = loaded.results[loaded.results.measurement.eq(metric) & loaded.results.status.eq(status)]
            members = selections[f'{status}:{metric}'].members
            assert {(m.cell.movie, m.cell.identity) for m in members} == set(zip(rows.movie, rows.identity))
    union = {m.cell for s in loaded.outcome.selections if s.name.startswith('significant:') for m in s.members}
    assert set(selections['any-significant'].members) == union
    assert selections['any-significant'].rule['cell_level_significance_test'] is False

def test_empty_population_has_schema_valid_artifacts_and_no_fit(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path, empty=True)
    saved = screen(resolved, paths, tmp_path / 'screen')
    assert saved.results.empty and saved.traces.empty and saved.display_inputs.empty
    assert {'p_value', 'q_value', 'status', 'family_id'} <= set(saved.results)
    assert saved.outcome.status == 'completed'
    assert all((not selection.members for selection in saved.outcome.selections))
    assert not fake_methods

def test_changed_input_fails_before_evaluation(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    paths['cell_frame'].write_text(paths['cell_frame'].read_text() + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='fingerprint changed'):
        screen(resolved, paths, tmp_path / 'screen')
    assert not fake_methods and (not (tmp_path / 'screen').exists())

def test_missing_artifact_and_incompatible_identity_are_not_reusable(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    saved = screen(resolved, paths, tmp_path / 'screen')
    changed = replace(resolved, request=replace(resolved.request, correction_scope='measurement'))
    assert screen_identity(changed) != screen_identity(resolved)
    with pytest.raises(ValueError, match='identity does not match'):
        read_screen(tmp_path / 'screen', expected_id=screen_identity(changed))
    (tmp_path / 'screen' / 'trace_inputs.csv').unlink()
    with pytest.raises(ValueError, match='missing or changed'):
        read_screen(tmp_path / 'screen', expected_id=saved.outcome.scientific_id)

@pytest.mark.parametrize('failed_method', ['mesa', 'f'])
def test_estimator_and_significance_fail_independently(tmp_path, fake_methods, monkeypatch, failed_method):
    resolved, paths = inputs(tmp_path)
    original = circadian.estimate_one

    def fail_one(hours, values, params, method, **kwargs):
        if method == failed_method:
            return {'status': 'failed', 'diagnostics': {'reason': 'controlled failure'}, 'p_value': 0.001}
        return original(hours, values, params, method, **kwargs)
    monkeypatch.setattr(circadian, 'estimate_one', fail_one)
    saved = screen(resolved, paths, tmp_path / 'screen')
    observed = saved.results[saved.results.identity.eq(7)]
    if failed_method == 'mesa':
        assert observed.status.eq('significant').any()
        assert not observed.period_available.any()
        assert observed.estimate_status.eq('failed').all()
    else:
        assert observed.status.eq('untestable').all()
        assert observed.period_available.all()
        assert observed.p_value.isna().all()
        assert not next((s for s in saved.outcome.selections if s.name == 'any-significant')).members

def test_zero_significant_is_completed_and_unavailable_curve_is_explicit(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path, options={'rhythmic_alpha': 1e-05})
    saved = screen(resolved, paths, tmp_path / 'screen')
    assert saved.outcome.status == 'completed'
    assert saved.results.status.eq('not-significant').sum() == 4
    assert saved.results.status.eq('untestable').sum() == 2
    assert saved.display_inputs.waveform_status.eq('unavailable').all()
    assert saved.display_inputs.waveform_reason.str.len().gt(0).all()

def test_real_non_daily_fit_and_independent_test_keep_native_provenance(tmp_path):
    hours = np.arange(0, 96.0, 1.0)
    frame = pd.DataFrame({'stem': 'movie', 'identity': 7, 'frame_index': np.arange(len(hours)), 'hours': hours, 'signal': 10 + np.cos(2 * np.pi * hours / 12)})
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    hashes = {'cell_frame': file_hash(path)}
    req = parse([{'pipeline': 'rhythm-discovery', 'test_measurements': ['signal'], 'analysis_options': {'fit_method': 'mesa', 'significance_method': 'f', 'detrend': 'none', 'period_config': {'mesa_model_length': 12}}}])[0]
    resolved = resolve_request(req, source_run='run', tables={'cell_frame': frame}, input_hashes=hashes)
    saved = screen(resolved, {'cell_frame': path}, tmp_path / 'screen')
    row = saved.results.iloc[0]
    assert row.status == 'significant' and row.period_available
    assert row.period_hours == pytest.approx(12, abs=0.6)
    assert row.estimate_result['native_result']['row']['method'] == 'mesa'
    assert row.significance_result['native_result']['row']['method'] == 'f'
    assert row.estimate_result['effective_settings'][0]['mesa_model_length'] == 12
    assert row.estimate_result['effective_settings'][0]['period_min_hours'] == 2
    assert row.significance_result['effective_settings'][0]['period_max_hours'] == 48
    assert row.estimate_result['workbench_version'] == circadian.WORKBENCH_VERSION

def test_gap_and_nonfinite_observations_remain_in_display_inventory(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    frame = pd.read_csv(paths['cell_frame'])
    frame.loc[1, 'signal'] = np.nan
    frame.loc[2, 'another'] = np.inf
    frame.loc[frame.stem.eq('a') & frame.frame_index.ge(4), 'frame_index'] += 3
    frame.loc[frame.stem.eq('a') & frame.hours.ge(4), 'hours'] += 3
    frame.to_csv(paths['cell_frame'], index=False)
    hashes = {name: file_hash(path) for name, path in paths.items()}
    loaded = read_verified_tables(paths, hashes)
    resolved = resolve_request(resolved.request, source_run='run-one', tables=loaded, input_hashes=hashes)
    saved = screen(resolved, paths, tmp_path / 'screen')
    assert len(saved.traces) == 32
    assert saved.traces.value_kind.eq('missing').sum() == 1
    assert saved.traces.value_kind.eq('positive_infinity').sum() == 1
    assert np.isposinf(saved.traces.loc[saved.traces.value_kind.eq('positive_infinity'), 'value']).all()
    assert saved.traces.eligible_input.eq(False).sum() == 2
    assert saved.results[saved.results.movie.eq('a') & saved.results.identity.eq(7)].nonconsecutive_steps.eq(1).all()
    assert all((np.isfinite(values).all() for _, values, *_ in fake_methods))

def test_explicit_sample_mapping_remains_confirmed(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path, biological_samples={'a': 'animal', 'b': 'animal'})
    saved = screen(resolved, paths, tmp_path / 'screen')
    assert saved.results['sample'].eq('animal').all()
    assert saved.results.sample_confirmed.all()
    assert {member for member in saved.outcome.selections[-1].members} <= {CellKey('run-one', 'a', 7), CellKey('run-one', 'b', 7)}

def test_selection_corruption_is_not_reused(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    screen(resolved, paths, tmp_path / 'screen')
    from pymicroglia._results import document
    path = document(tmp_path / 'screen' / 'result.json')
    result = read_document(path)
    result['selections'][-1]['members'] = []
    path.write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(ValueError, match='selections disagree'):
        read_screen(tmp_path / 'screen')

def test_duplicate_time_is_untestable_without_losing_raw_rows(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    frame = pd.read_csv(paths['cell_frame'])
    frame.loc[1, 'hours'] = frame.loc[0, 'hours']
    frame.to_csv(paths['cell_frame'], index=False)
    hashes = {name: file_hash(path) for name, path in paths.items()}
    resolved = resolve_request(resolved.request, source_run='run-one', tables=read_verified_tables(paths, hashes), input_hashes=hashes)
    saved = screen(resolved, paths, tmp_path / 'screen')
    rejected = saved.results[saved.results.movie.eq('a') & saved.results.identity.eq(7)]
    assert rejected.status.eq('untestable').all()
    assert rejected.reason.eq('duplicate_time').all()
    assert not rejected.input_eligible.any()
    assert saved.families.usable.sum() == 2
    assert len(saved.traces) == 32
