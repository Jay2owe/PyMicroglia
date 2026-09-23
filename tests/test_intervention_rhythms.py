"""Independent original windows, native direct contrasts and immutable reuse."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
import pymicroglia.pipelines.intervention.rhythms as rhythms
import pymicroglia.pipelines.intervention.rhythm_windows as rhythm_windows
from pymicroglia.pipelines.intervention.options import resolve_request, run_request
from pymicroglia.pipelines.intervention.windows import prepare
from pymicroglia.pipelines._screening import file_hash, _json_value

def declaration():
    return {'enabled': True, 'measurements': ['custom_signal'], 'analysis_options': {'fit_method': 'fft_nlls', 'significance_method': 'lomb', 'detrend': 'none', 'period_min_hours': 2.0, 'period_max_hours': 16.0, 'min_observations': 24, 'min_cycles': 3.0, 'multiple_testing': 'bh', 'rhythmic_alpha': 0.05, 'period_config': {'nlls_max_components': 2, 'nlls_circadian_min': 6.0, 'nlls_circadian_max': 12.0}}, 'comparisons': [{'measurement': 'custom_signal', 'baseline': 'baseline', 'target_window': 'followup', 'component_period_band': [6.0, 12.0]}], 'direct_change': {'method': 'native_fit_covariance', 'properties': ['period', 'amplitude', 'phase'], 'independent_window_errors': 'Original software windows have independently generated residual errors', 'residual_model_justification': 'Two-component software waveform with independent homoskedastic Gaussian errors', 'phase_reference': 'recording_anchor', 'period_equivalence_fraction': 0.05, 'max_phase_extrapolation_cycles': 0.05}}

def fixture(tmp_path, cases=None):
    cases = cases or ['amplitude-phase', 'period', 'same', 'short', 'missing', 'gapped']
    rows = []
    clocks = []
    anchors = {}
    overrides = {}
    for movie in cases:
        short = movie == 'short'
        width = 8.0 if short else 96.0
        anchor = width
        anchors[movie] = {'hours': anchor, 'kind': 'intervention', 'label': 'Software event'}
        if short:
            overrides[movie] = [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -8.0, 'end': 0.0}, {'name': 'followup', 'coordinate': 'relative_hours', 'start': 0.0, 'end': 8.0, 'baseline': 'baseline'}]
        for side in range(2):
            t = side * width + np.arange(int(width / 0.25)) * 0.25
            phase = 1.0 if movie == 'amplitude-phase' and side else 0.0
            amplitude = 4.0 if movie == 'amplitude-phase' and side else 3.0
            period = 10.0 if movie == 'period' and side else 8.0
            y = 10.0 + amplitude * np.cos(2 * np.pi * (t - phase) / period) + np.cos(4 * np.pi * (t - phase) / period) + np.random.default_rng(17809 + side).normal(scale=0.25, size=len(t))
            for index, (hours, value) in enumerate(zip(t, y)):
                frame = index + side * len(t)
                clocks.append({'stem': movie, 'frame_index': frame, 'hours': hours})
                if movie == 'missing' and side:
                    continue
                if movie == 'gapped' and side and (index == 20):
                    value = np.nan
                rows.append({'stem': movie, 'identity': 7, 'frame_index': frame, 'hours': hours, 'custom_signal': value, 'other_metric': 2.0 * value})
    frame = pd.DataFrame(rows)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates(), 'frame_summary': pd.DataFrame(clocks)}
    paths = {}
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        table.to_csv(paths[name], index=False)
    request = {'pipeline': 'intervention-response', 'measurements': ['custom_signal', 'other_metric'], 'summary': 'mean', 'anchors': anchors, 'windows': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -96.0, 'end': 0.0}, {'name': 'followup', 'coordinate': 'relative_hours', 'start': 0.0, 'end': 96.0, 'baseline': 'baseline'}], 'recording_windows': overrides, 'support': {'max_gap_hours': 0.3}, 'rhythms': declaration()}
    return (request, tables, paths, resolve(request, tables, paths))

def resolve(request, tables, paths):
    return resolve_request(parse([request])[0], source_run='rhythm-window-software', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})

@pytest.fixture(scope='module')
def actual(tmp_path_factory):
    request, tables, paths, resolved = fixture(tmp_path_factory.mktemp('rhythm-windows'))
    return (request, tables, paths, resolved, rhythms.analyse(prepare(resolved, tables), resolved))

def test_actual_nondaily_multicomponent_changes_use_native_direct_evidence(actual):
    _, _, _, resolved, data = actual
    window = data['window_results']
    direct = data['direct_comparisons']
    selected = direct.loc[direct.movie.eq('amplitude-phase')].set_index('property')
    assert len(window) == 12 and len(direct) == 18 and (set(window.measurement) == {'custom_signal'})
    assert set(window.method) == {'fft_nlls'} and set(window.significance_method) == {'lomb'}
    assert selected.loc['amplitude', 'effect'] == pytest.approx(1.0, abs=0.08) and selected.loc['amplitude', 'change_supported']
    assert selected.loc['phase', 'effect'] == pytest.approx(0.125, abs=0.01) and selected.loc['phase', 'change_supported']
    changed = direct.loc[direct.movie.eq('period')].set_index('property')
    assert changed.loc['period', 'effect'] == pytest.approx(2.0, abs=0.05) and changed.loc['period', 'change_supported']
    assert pd.isna(changed.loc['phase', 'p_value']) and 'period' in changed.loc['phase', 'reason']
    assert all((len(row['estimate_result'].get('components', [])) == 2 for row in data['window_details'].to_dict('records') if row['window_id'] in set(window.loc[window.movie.eq('amplitude-phase'), 'window_id'])))
    assert data['window_families'].requested.sum() == 12 and data['direct_families'].requested.sum() == 18
    assert set(data['window_families'].family_id).isdisjoint(data['direct_families'].family_id)

def test_unresolved_missing_and_gapped_windows_keep_full_requested_comparisons(actual):
    *_, data = actual
    window = data['window_results']
    direct = data['direct_comparisons']
    assert not window.loc[window.movie.eq('short'), 'period_supported'].any()
    assert direct.loc[direct.movie.isin(['short', 'missing', 'gapped']), 'p_value'].isna().all()
    assert len(data['comparisons']) == 6 and data['comparisons'].rhythm_comparison_id.is_unique
    assert len(direct) == 18 and data['direct_families'].iloc[0].requested == 18
    gap = data['window_traces'].loc[lambda x: x.movie.eq('gapped') & x.window.eq('followup')]
    assert len(gap) == 384 and gap.raw_value.isna().sum() == 1 and (gap.filtered_value.isna().sum() == 1)

def test_windows_retain_original_clock_and_independent_native_input(actual):
    _, _, _, resolved, data = actual
    traces = data['window_traces']
    details = data['window_details'].set_index('window_id')
    for row in data['window_results'].to_dict('records'):
        original = traces.loc[traces.window_id.eq(row['window_id'])]
        assert set(details.loc[row['window_id'], 'source_observation_ids']) == set(original.observation_id)
        assert (original.hours - original.relative_hours).eq(row['anchor_hours']).all()
        native = details.loc[row['window_id'], 'estimate_result']
        if native:
            assert native['method'] == 'fft_nlls'
            assert details.loc[row['window_id'], 'applied_recipe']['params']['period_search_hours'] == [2.0, 16.0]
    assert traces.loc[traces.window.eq('baseline'), 'relative_hours'].lt(0).all()
    assert traces.loc[traces.window.eq('followup'), 'relative_hours'].ge(0).all()

def test_significance_label_change_cannot_supply_direct_change(actual, monkeypatch):
    _, tables, _, resolved, data = actual
    captured = {name: table.copy(deep=True) for name, table in data.items() if name.startswith('window_')}
    mask = captured['window_results'].movie.eq('amplitude-phase') & captured['window_results'].window.eq('followup')
    captured['window_results'].loc[mask, 'detected'] = False
    captured['window_results'].loc[mask, 'detection_outcome'] = 'not-significant'
    monkeypatch.setattr(rhythm_windows, 'analyse', lambda *args: captured)
    result = rhythms.analyse(prepare(resolved, tables), resolved)
    row = result['comparisons'].loc[lambda x: x.movie.eq('amplitude-phase')].iloc[0]
    assert row.detection_transition == 'significant to not-significant'
    assert result['direct_comparisons'].loc[lambda x: x.movie.eq('amplitude-phase'), 'p_value'].isna().all()

def test_configured_other_estimator_stays_selected_without_implicit_covariance_fit(tmp_path):
    request, tables, paths, _ = fixture(tmp_path, ['amplitude-phase'])
    request['rhythms']['analysis_options']['fit_method'] = 'lomb'
    resolved = resolve(request, tables, paths)
    data = rhythms.analyse(prepare(resolved, tables), resolved)
    assert set(data['window_results'].method) == {'lomb'} and data['direct_comparisons'].p_value.isna().all()
    assert data['direct_comparisons'].reason.str.contains('no substitute estimator').all()

def test_baseline_preprocessing_cannot_consume_changed_followup(tmp_path):
    request, tables, paths, _ = fixture(tmp_path, ['amplitude-phase'])
    request['rhythms']['direct_change'] = {'method': 'none'}
    request['rhythms']['analysis_options']['detrend'] = 'linear'
    first = resolve(request, tables, paths)
    a = rhythms.analyse(prepare(first, tables), first)
    tables['cell_frame'].loc[tables['cell_frame'].hours.ge(96.0), 'custom_signal'] += 1000.0
    tables['cell_frame'].to_csv(paths['cell_frame'], index=False)
    second = resolve(request, tables, paths)
    b = rhythms.analyse(prepare(second, tables), second)
    for name in ['window_results', 'window_details', 'window_traces']:
        ids = set(a['window_results'].loc[a['window_results'].window.eq('baseline'), 'window_id'])
        left = a[name].loc[a[name].window_id.isin(ids)]
        right = b[name].loc[b[name].window_id.isin(ids)]
        if name == 'window_results':
            for column in ['period_hours', 'amplitude', 'p_value', 'components']:
                assert _json_value(left[column].tolist()) == _json_value(right[column].tolist())
        elif name == 'window_traces':
            assert _json_value(left.to_dict('records')) == _json_value(right.to_dict('records'))
        else:
            assert left.iloc[0].source_observation_ids == right.iloc[0].source_observation_ids

def test_runtime_reuses_saved_science_and_disabled_branch_calls_no_rhythm_engine(tmp_path, monkeypatch):
    request, tables, paths, resolved = fixture(tmp_path / 'source', ['amplitude-phase'])
    output = tmp_path / 'pipeline'
    first = run_request(resolved, paths, output, only=['rhythm-changes'])
    assert first.successful
    saved = first.results['rhythm-changes']
    data = rhythms.read_rhythms(saved)
    assert len(data['direct_comparisons']) == 3
    fingerprints = {str(path): file_hash(path) for path in paths.values()}
    artifacts = {str(saved.artifact(item.name)): file_hash(saved.artifact(item.name)) for item in saved.outcome.artifacts}

    def forbidden(*args, **kwargs):
        raise AssertionError('Saved rhythm science was unexpectedly rerun')
    monkeypatch.setattr(rhythm_windows, 'analyse', forbidden)
    monkeypatch.setattr(circadian, 'rhythm_window_comparison', forbidden)
    second = run_request(resolved, paths, output, presentation={'report': {'title': 'Changed saved display'}}, only=['rhythm-changes'])
    assert second.successful and second.results['rhythm-changes'].outcome.status == 'reused'
    assert all((file_hash(Path(path)) == digest for path, digest in {**fingerprints, **artifacts}.items()))
    request['rhythms'] = {'enabled': False}
    disabled = resolve(request, tables, paths)
    result = run_request(disabled, paths, output, only=['rhythm-changes'])
    assert result.successful and result.results['rhythm-changes'].outcome.status == 'skipped-empty'

@pytest.mark.parametrize('change', [{'measurements': ['custom_signal', 'custom_signal']}, {'comparisons': [{'measurement': 'custom_signal', 'baseline': 'baseline', 'target_window': 'followup', 'component_period_band': [12.0, 6.0]}]}, {'direct_change': {'method': 'native_fit_covariance'}}, {'direct_change': {**declaration()['direct_change'], 'phase_reference': 'fixed_day'}}])
def test_unsafe_or_ambiguous_comparison_declarations_are_refused(change):
    with pytest.raises(ValueError):
        rhythms.policy({**declaration(), **change}, ['custom_signal', 'other_metric'])
