"""All-cell grid keeps the full inventory and separates display from inference."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.measure.modules.rhythms import DEFAULTS
from pymicroglia.visualisation.figures import load
from pymicroglia.figure_tables import all_cell_traces as preparation
from pymicroglia.visualisation.panels import all_cell_traces
from pymicroglia.visualisation.panels import colour

def load_all():
    return {spec.slug: spec for spec in load().values()}

def _functions():
    return {**vars(preparation), 'all_cell_traces': all_cell_traces}

def _frame():
    return pd.DataFrame([{'identity': identity, 'frame_index': i, 'hours': float(i), 'corrected_mean': float(value), 'area_px': float(10 + value)} for identity, multiplier in ((2, 1), (5, 8), (9, 1)) for i in range(8) for value in [multiplier * (1 + np.sin(2 * np.pi * i / 4))]])

def _resolved(**overrides):
    params = {**DEFAULTS, 'min_observations': 6, **overrides}
    defaults = {**circadian.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS, 'min_observations': params['min_observations'], 'multiple_testing': 'none'}
    return circadian.resolve_analysis_options(params, defaults.__getitem__)

def test_package_default_is_single_source_for_fresh_downstream_analyses():
    from pymicroglia.measure.modules import coupling, recurrence, rhythms
    assert circadian.DETREND_DEFAULTS['detrend'] == 'robust_linear'
    assert rhythms.DEFAULTS['detrend'] == coupling.DEFAULTS['detrend'] == recurrence.DEFAULTS['detrend']
    assert circadian.PERIOD_ANALYSIS_DEFAULTS['period_estimation_method'] == 'lomb'
    assert circadian.PERIOD_ANALYSIS_DEFAULTS['primary_rhythm_test'] == 'lomb'

def test_all_cells_have_separate_normalized_traces_and_one_axis_per_row():
    helpers = _functions()
    points, tests = helpers['trace_data'](_frame(), ['corrected_mean', 'area_px'], [2, 5, 9], _resolved(), 'raw', 'minmax', {})
    assert set(points.identity) == {2, 5, 9}
    assert len(tests) == 6
    for _, series in points.groupby(['identity', 'metric']):
        assert np.isclose(series.value.min(), -1.0)
        assert np.isclose(series.value.max(), 1.0)
    assert not points.loc[points.identity.eq(2), 'raw_value'].equals(points.loc[points.identity.eq(5), 'raw_value'])
    rows, columns = helpers['grid_shape'](9, None, None)
    assert (rows, columns) == (3, 3)
    assert helpers['grid_shape'](9, 2, None) == (2, 5)
    with pytest.raises(ValueError, match='hold all'):
        helpers['grid_shape'](9, 2, 4)
    settings = {'identities': [2, 5, 9], 'rows': 2, 'columns': 2, 'metrics': ['corrected_mean', 'area_px'], 'labels': ['Intensity', 'Area'], 'unit': 'Self-normalized', 'xlim': [0.0, 7.0], 'title': 'All cells'}
    figure, axes = helpers['all_cell_traces'].draw(points, settings, canvas=__import__('matplotlib.pyplot', fromlist=['figure']).figure())
    try:
        assert len(axes) == 3
        assert [axis.get_title(loc='left') for axis in axes] == ['Cell 2', 'Cell 5', 'Cell 9']
        assert len(figure.legends) == 1
        assert [label.get_text() for label in figure.legends[0].get_texts()] == ['Intensity', 'Area']
        assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
        assert not axes[0].get_shared_y_axes().joined(axes[0], axes[2])
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)

def test_grid_uses_twelve_hour_ticks_and_draws_period_results_on_the_right():
    helpers = _functions()
    points, tests = helpers['trace_data'](_frame(), ['corrected_mean'], [2, 5], _resolved(), 'raw', 'minmax', {})
    tests.loc[tests.identity.eq(2), 'period_hours'] = 4.0
    tests.loc[tests.identity.eq(2), 'q_value'] = 0.01
    tests.loc[tests.identity.eq(2), 'estimate_status'] = 'ok'
    tests.loc[tests.identity.eq(2), 'supported_period'] = True
    settings = {'identities': [2, 5], 'rows': 1, 'columns': 2, 'metrics': ['corrected_mean'], 'labels': ['Intensity'], 'unit': 'Self-normalized', 'xlim': [0.0, 36.0], 'title': 'All cells', 'hour_ticks': 12.0, 'period_testing': True}
    figure, axes = helpers['all_cell_traces'].draw(points, settings, tests, canvas=__import__('matplotlib.pyplot', fromlist=['figure']).figure())
    try:
        assert list(axes[0].get_xticks()) == [0.0, 12.0, 24.0, 36.0]
        assert axes[0].get_title(loc='left') == 'Cell 2'
        assert axes[0].get_title(loc='right') == 'Intensity: 4 h, q=0.01'
        right_title = axes[0]._right_title.get_fontsize()
        assert right_title < axes[0]._left_title.get_fontsize()
        assert axes[0]._right_title.get_color() == colour('nan_text')
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)

def test_missing_and_flat_cells_remain_visible_without_inventing_significance():
    helpers = _functions()
    frame = _frame()
    frame.loc[frame.identity.eq(9), 'corrected_mean'] = 4.0
    frame.loc[frame.identity.eq(5) & frame.frame_index.eq(2), 'corrected_mean'] = np.nan
    points, tests = helpers['trace_data'](frame, ['corrected_mean'], [2, 5, 9], _resolved(min_observations=24), 'raw', 'minmax', {})
    assert len(points) == len(frame)
    assert points.loc[points.identity.eq(5) & points.frame_index.eq(2), 'value'].isna().all()
    assert set(points.loc[points.identity.eq(9), 'value']) == {0.0}
    assert 'fallback' in tests.loc[tests.identity.eq(9), 'display_note'].iloc[0]
    assert set(tests.rhythm_status) == {'unknown'}

def test_single_observation_is_a_marked_dot_not_a_detrended_trace():
    helpers = _functions()
    frame = _frame()
    frame.loc[frame.identity.eq(9) & frame.frame_index.ne(0), 'corrected_mean'] = np.nan
    points, tests = helpers['trace_data'](frame, ['corrected_mean'], [9], _resolved(min_observations=24), 'detrended', 'minmax', {})
    assert points.value.notna().sum() == 1
    assert points.processed_value.isna().all()
    assert points.loc[points.value.notna(), 'value'].iloc[0] == 0.0
    assert 'single raw observation' in tests.display_note.iloc[0]
    assert tests.rhythm_status.iloc[0] == 'unknown'

def test_period_testing_can_be_disabled_without_calling_workbench(monkeypatch):
    helpers = _functions()

    def fail(*args, **kwargs):
        raise AssertionError('period testing was called while disabled')
    monkeypatch.setattr(circadian, 'estimate_one', fail)
    _, tests = helpers['trace_data'](_frame(), ['corrected_mean'], [2], _resolved(), 'raw', 'minmax', {}, period_testing=False)
    assert tests.period_testing.tolist() == [False]
    assert tests.reason.iloc[0] == 'period testing disabled'

def test_grid_exposes_shared_science_options_without_changing_saved_selected_grid():
    specs = load_all()
    grid = specs['all-cell-trace-grid']
    names = {option.name for option in grid.options}
    assert set(circadian.CIRCADIAN_ANALYSIS_OPTIONS) <= names
    assert {'metrics', 'cells', 'grid_rows', 'grid_columns', 'trace_view', 'trace_normalization', 'trace_normalization_config', 'hour_ticks', 'period_testing'} <= names
    assert next((o.default for o in grid.options if o.name == 'metrics')) == ['signal_mean']
    assert next((o.default for o in grid.options if o.name == 'multiple_testing')) == 'none'
    assert next((o.default for o in grid.options if o.name == 'hour_ticks')) == 12.0
    assert next((o.default for o in grid.options if o.name == 'period_testing')) is True
    assert next((o.default for o in specs['rhythm-trace-grid'].options if o.name == 'grid_columns')) == 3

def test_registered_builder_fits_original_values_and_keeps_display_separate(monkeypatch):
    from types import SimpleNamespace
    spec = load_all()['all-cell-trace-grid']
    options = {option.name: option.default for option in spec.options}
    options['min_observations'] = 6
    options['metrics'] = ['corrected_mean']
    options['cells'] = '2,5'
    options['grid_columns'] = 2
    original = _frame()
    received = []

    def estimate(hours, values, params, method):
        received.append((method, np.asarray(values, float).copy(), params['detrend']))
        return {'status': 'ok', 'period_hours': 4.0, 'components': [{'period_hours': 4.0, 'selected': True}], 'diagnostics': {}, 'workbench_run_record_json': '{}'}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    monkeypatch.setattr(circadian, 'detrend_trace', lambda hours, values, params: {'values': list(values)})
    monkeypatch.setattr(preparation, 'test_cell_components', lambda *args, **kwargs: ([{'component': 1, 'status': 'ok', 'period_hours': 4.0, 'empirical_p_uncorrected': 0.01}], {'component': 1, 'status': 'ok', 'period_hours': 4.0, 'empirical_p_uncorrected': 0.01, 'selected_fft_component': True}))
    ctx = SimpleNamespace(spec=spec, name='all-cell-trace-grid', table=lambda _: original.copy(), option=options.__getitem__, module_params=lambda _: {})
    prepared, evidence = preparation.prepare(ctx, options)
    data = prepared['traces']
    fig, axes = all_cell_traces.draw(data['table'], data['settings'], data['annotations'], canvas=__import__('matplotlib.pyplot', fromlist=['figure']).figure())
    result = SimpleNamespace(figure=fig, axes=axes, figure_data=data['table'], auxiliary=prepared.auxiliary)
    try:
        assert len(result.axes) == 2
        assert len(result.figure_data) == 16
        assert set(result.auxiliary['statistics.csv'].rhythm_status) == {'rhythmic'}
        assert [method for method, _, _ in received] == ['fft_nlls', 'fft_nlls']
        assert all((detrend == 'robust_linear' for _, _, detrend in received))
        np.testing.assert_array_equal(received[0][1], preparation.median_then_mean(original.loc[original.identity.eq(2), 'corrected_mean'].to_numpy(float)))
        assert not np.allclose(received[0][1], result.figure_data.loc[result.figure_data.identity.eq(2), 'value'].to_numpy(float))
    finally:
        import matplotlib.pyplot as plt
        plt.close(result.figure)

def test_fft_grid_reports_the_period_tested_by_its_displayed_p_value(monkeypatch):
    resolved = _resolved()
    resolved['method'] = 'fft_nlls'
    resolved['multiple_testing'] = 'none'
    monkeypatch.setattr(circadian, 'estimate_one', lambda *args, **kwargs: {
        'status': 'ok', 'period_hours': 30.0,
        'components': [{'period_hours': 30.0, 'selected': True},
                       {'period_hours': 12.0, 'selected': False}],
        'diagnostics': {},
    })
    monkeypatch.setattr(circadian, 'detrend_trace', lambda hours, values, params: {'values': list(values)})
    monkeypatch.setattr(preparation, 'test_cell_components', lambda *args, **kwargs: (
        [{'component': 1, 'period_hours': 30.0, 'status': 'ok', 'empirical_p_uncorrected': .2},
         {'component': 2, 'period_hours': 12.0, 'status': 'ok', 'empirical_p_uncorrected': .1}],
        None,
    ))
    _, evidence = preparation.trace_data(
        _frame(), ['corrected_mean'], [2], resolved, 'raw', 'none', {},
        fft_component_test=True, recording='test',
    )
    assert evidence.period_hours.iloc[0] == 12.0
    assert evidence.p_value.iloc[0] == .1
    assert evidence.rhythm_status.iloc[0] == 'not rhythmic'


def test_reference_period_style_changes_display_only(monkeypatch):
    helpers = _functions()
    points, tests = helpers['trace_data'](_frame(), ['corrected_mean'], [2], _resolved(), 'raw', 'minmax', {})
    tests.loc[:, 'period_hours'] = 4.0
    tests.loc[:, 'p_value'] = 0.01
    tests.loc[:, 'rhythm_status'] = 'rhythmic'
    tests.loc[:, 'significance_status'] = 'ok'
    monkeypatch.setattr(circadian, 'descriptive_cosinor_fitted_values', lambda hours, values, period: np.full(len(hours), 0.25))
    styled = helpers['add_period_display'](points, tests)
    pd.testing.assert_series_equal(styled.raw_value, points.raw_value)
    assert styled.period_test_significant.all()
    assert styled.descriptive_fit.eq(0.25).all()
    settings = {'identities': [2], 'rows': 1, 'columns': 1, 'metrics': ['corrected_mean'], 'labels': ['Intensity'], 'unit': 'Self-normalized', 'xlim': [0.0, 7.0], 'title': 'Test', 'period_display_style': True}
    figure, axes = helpers['all_cell_traces'].draw(styled, settings, tests, canvas=__import__('matplotlib.pyplot', fromlist=['figure']).figure())
    try:
        assert axes[0].lines[0].get_color() == colour('circadian_red')
        assert axes[0].lines[1].get_linestyle() == ':'
        assert len(figure.legends[0].get_texts()) == 3
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_fft_nlls_period_style_uses_the_actual_multicomponent_fit(monkeypatch):
    helpers = _functions()
    points, tests = helpers['trace_data'](_frame(), ['corrected_mean'], [2], _resolved(), 'detrended', 'minmax', {})
    points.loc[points.frame_index.eq(0), ['processed_value', 'value']] = np.nan
    points.loc[points.frame_index.eq(3), ['processed_value', 'value']] = np.nan
    tests.loc[:, 'estimation_method'] = 'fft_nlls'
    tests.loc[:, 'period_hours'] = 4.0
    tests.loc[:, 'estimate_status'] = 'ok'
    tests.loc[:, 'estimate_diagnostics_json'] = json.dumps({'status': 'ok', 'mesor': 0.0, 'phase_zero_timestamp': '2000-01-01T00:00:00'})
    components = [{'period_hours': 4.0, 'amplitude': 0.4, 'phase_hours': 0.5}, {'period_hours': 2.0, 'amplitude': 0.2, 'phase_hours': 0.0}]
    tests.loc[:, 'estimate_components_json'] = json.dumps(components)
    monkeypatch.setattr(circadian, 'descriptive_cosinor_fitted_values', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('substitute cosine used')))
    styled = helpers['add_period_display'](points, tests)
    segment = styled.sort_values('frame_index')
    native = circadian.fft_nlls_fitted_values(segment.hours.to_numpy(float), {'method': 'fft_nlls', 'diagnostics': json.loads(tests.estimate_diagnostics_json.iloc[0]), 'components': components})
    processed = segment.processed_value.to_numpy(float)
    displayed = segment.value.to_numpy(float)
    usable = np.isfinite(processed) & np.isfinite(displayed)
    slope, intercept = np.linalg.lstsq(np.column_stack([processed[usable], np.ones(usable.sum())]), displayed[usable], rcond=None)[0]
    expected = slope * native + intercept
    expected[segment.hours.to_numpy(float) < segment.loc[usable, 'hours'].min()] = np.nan
    expected[~usable] = np.nan
    np.testing.assert_allclose(segment.descriptive_fit, expected, equal_nan=True)
    assert np.isnan(segment.descriptive_fit.iloc[0])
    assert np.isnan(segment.loc[segment.frame_index.eq(3), 'descriptive_fit']).all()
    assert set(segment.fit_kind) == {'actual summed FFT-NLLS fit'}


def test_fft_nlls_never_substitutes_a_cosine_when_actual_fit_cannot_share_the_view(monkeypatch):
    helpers = _functions()
    points, tests = helpers['trace_data'](_frame(), ['corrected_mean'], [2], _resolved(), 'raw', 'minmax', {})
    tests.loc[:, 'estimation_method'] = 'fft_nlls'
    tests.loc[:, 'period_hours'] = 4.0
    tests.loc[:, 'estimate_status'] = 'ok'
    tests.loc[:, 'estimate_diagnostics_json'] = json.dumps({'status': 'ok', 'mesor': 0.0, 'phase_zero_timestamp': '2000-01-01T00:00:00'})
    tests.loc[:, 'estimate_components_json'] = json.dumps([{'period_hours': 4.0, 'amplitude': 0.4, 'phase_hours': 0.5}])
    monkeypatch.setattr(circadian, 'descriptive_cosinor_fitted_values', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('substitute cosine used')))
    styled = helpers['add_period_display'](points, tests)
    assert styled.descriptive_fit.isna().all()
    assert set(styled.fit_kind) == {'none'}
    settings = {'identities': [2], 'rows': 1, 'columns': 1, 'metrics': ['corrected_mean'], 'labels': ['Intensity'], 'unit': 'Self-normalized', 'xlim': [0.0, 7.0], 'title': 'Test', 'period_display_style': True, 'fit_label': 'Actual summed FFT-NLLS fit'}
    figure, _ = helpers['all_cell_traces'].draw(styled, settings, tests, canvas=__import__('matplotlib.pyplot', fromlist=['figure']).figure())
    try:
        assert [label.get_text() for label in figure.legends[0].get_texts()] == ['Significant period test (uncorrected p < 0.05)', 'Other cells']
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
