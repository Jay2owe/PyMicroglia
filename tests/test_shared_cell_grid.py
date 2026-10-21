"""Public grid integration: shared layout, native inputs and reusable science."""
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from pymicroglia import workbench
from pymicroglia.figure_tables import cell_grid
from pymicroglia.figure_tables.cell_grid_inputs import frame_mean, process, recipe
from pymicroglia.visualisation.figures import load
from pymicroglia.visualisation.panels.cell_analysis_grid import analyses


def source():
    hours = 4 + np.arange(96) * .52732
    table = pd.DataFrame([{'identity': identity, 'frame_index': i, 'hours': hour,
                           'integrated_density': 100 + .3 * hour + 8 * np.sin(2 * np.pi * hour / 24)}
                          for identity in (3, 8) for i, hour in enumerate(hours)])
    return SimpleNamespace(table=lambda _: table.copy(), module_params=lambda _: {}, figure_theme='classic')


def options(**changes):
    return {**{o.name: o.default for o in load()['cell_grid'].options}, **changes}


def test_centered_means_reuse_measurement_helper_and_do_not_bridge_missing_frames():
    from pymicroglia.measure.signal import centered_frame_mean
    values = np.array([10., 20., 30., 40., 50., 60.])
    frames = np.array([0, 1, 2, 4, 5, 6])
    np.testing.assert_allclose(frame_mean(values, frames), centered_frame_mean(values, frames), equal_nan=True)
    np.testing.assert_allclose(frame_mean(values, frames), [np.nan, 20., np.nan, np.nan, 50., np.nan], equal_nan=True)


def test_preprocessing_matches_native_workbench_on_fractional_clock():
    table = source().table('cell_frame.csv').query('identity == 3')
    hours, raw = table.hours.to_numpy(), table.integrated_density.to_numpy()
    resolved = recipe({'mean_frames': 3, 'detrend': 'robust_linear'})
    with workbench.analysis_session():
        mean, actual, records = process(hours, raw, table.frame_index, resolved)
        expected = workbench.cw.trace(hours, mean, settings={'bin_minutes': 31.6392}).detrend('robust_linear')
    np.testing.assert_allclose(actual, expected.trace_data.values, atol=1e-10, rtol=1e-10, equal_nan=True)
    assert records[0]['action'] == 'detrend'
    assert len(actual) == len(raw)


def test_three_wavelet_views_compute_one_spectrum_for_each_cell(monkeypatch):
    import circadian_workbench.client as client
    original = client.call
    calls = []
    def counted(action, **kwargs):
        if action == 'wavelet_spectrum':
            calls.append(action)
        return original(action, **kwargs)
    monkeypatch.setattr(client, 'call', counted)
    views = ['wavelet_spectrum/' + view for view in ('wavelet', 'wavelet_phase', 'wavelet_amplitude')]
    progress = []
    prepared, evidence = cell_grid.prepare(source(), options(analysis=views[0], analysis_views=views,
        analysis_parameters={'wavelet_options': {'n_sim': 0}}), progress=progress.append)
    assert len(calls) == 2
    assert len(evidence) == len(prepared['analyses']['panels']) == 6
    assert set(evidence.status) == {'ok'}
    assert evidence.analysis.tolist() == views * 2
    assert progress[-1]['completed'] == progress[-1]['total'] == 6
    for row in evidence.itertuples():
        assert json.loads(row.preprocessing_records_json)
        assert json.loads(row.run_records_json)[0]['action'] == 'wavelet_spectrum'
    inputs = prepared.auxiliary['input_traces.csv']
    assert len(inputs) == 192
    assert {'raw_value', 'filtered_value', 'value'} <= set(inputs)


def test_trace_and_native_panels_share_layout_and_drawing_does_not_compute(monkeypatch):
    trace, _ = cell_grid.prepare(source(), options(analysis_parameters={'period_testing': False}))
    lag, _ = cell_grid.prepare(source(), options(analysis='autocorrelation/autocorrelation'))
    def refuse(*args, **kwargs):
        raise AssertionError('Drawing reran an analysis')
    monkeypatch.setattr(workbench, 'analyze_cell_view', refuse)
    monkeypatch.setattr(workbench, 'estimate_one', refuse)
    first, second = Figure(), Figure()
    a, b = analyses(first, trace['analyses']), analyses(second, lag['analyses'])
    np.testing.assert_equal(first.get_size_inches(), second.get_size_inches())
    assert len(a.axes) == len(b.axes) == 2
    for x, y in zip(a.axes, b.axes):
        # Native panels may letterbox internally to retain their physical aspect.
        np.testing.assert_allclose(x.get_position(original=True).bounds, y.get_position(original=True).bounds)
    assert all(len(axis.lines) == 1 for axis in a.axes)


def test_trace_recipe_and_missing_native_slots_are_retained():
    src = source()
    table = src.table('cell_frame.csv')
    table = table.loc[~((table.identity == 3) & table.frame_index.between(30, 32))]
    src.table = lambda _: table.copy()
    prepared, _ = cell_grid.prepare(src, options(analysis_parameters={'period_testing': False, 'trace_normalization': 'none'}))
    inputs = prepared.auxiliary['input_traces.csv'].query('identity == 3')
    assert len(inputs) == 96
    assert inputs.query('30 <= frame_index <= 32').raw_value.isna().all()
    assert inputs.query('29 <= frame_index <= 33').filtered_value.isna().all()
    assert prepared['analyses']['settings']['detrend'] == 'asymmetric_least_squares'


@pytest.mark.parametrize('setting', [{'mean_frames': 2}, {'unknown': 1}, {'detrend': 'unknown'},
                                    {'detrend_options': {'unknown': 1}}])
def test_invalid_preprocessing_is_rejected_before_any_analysis(setting):
    with pytest.raises(ValueError):
        cell_grid.prepare(source(), options(preprocessing=setting))


def test_analysis_registry_and_ui_share_parameters_and_one_canonical_home():
    from pymicroglia.knowledge import describe
    from pymicroglia.ui.catalog import load_map, validate_map
    spec = describe('cell_grid')
    assert spec['ok'] and not spec['pending']
    analysis = next(row for row in spec['params'] if row['name'] == 'analysis')
    entries = analysis['analysis_entries']
    assert entries.keys() == cell_grid.catalogue().keys()
    assert 'period_testing' in {row['name'] for row in entries['trace']['parameters']}
    assert 'wavelet_options' in {row['name'] for row in entries['wavelet_spectrum/wavelet']['parameters']}
    assert next(row for row in spec['params'] if row['name'] == 'preprocessing')['properties']['mean_frames']['units'] == 'frames'
    data = load_map()
    assert validate_map(data) == []
    canonical = [row for row in data['inventory'] if row['name'] in {'cell_grid','cell_analysis_grid','all_cell_trace_grid'} and not row.get('alias_of')]
    assert len(canonical) == 1 and canonical[0]['name'] == 'cell_grid'


def test_public_example_uses_the_same_registered_exporter(monkeypatch):
    from pymicroglia.figure_tables import actions, saved
    from pymicroglia.visualisation import panels
    from pymicroglia.pipelines import _saved_figures
    src = source()
    src.manifest, src.sources, src.stem = {}, [], 'synthetic'
    monkeypatch.setattr(saved, 'Tables', lambda *args: src)
    captured = {}
    monkeypatch.setattr(panels, 'save', lambda figure, path, **kwargs: captured.update(kwargs) or path)
    monkeypatch.setattr(_saved_figures, 'save_auxiliary', lambda tables, *args: captured.update(auxiliary=tables))
    result = actions.cell_grid('unused', cells='all', metrics=['integrated_density'],
        analysis='wavelet_spectrum/wavelet', preprocessing={'mean_frames': 3, 'detrend': 'robust_linear'},
        grid_columns=8, analysis_parameters={'wavelet_options': {'n_sim': 0}})
    assert result.name == 'synthetic'
    assert len(captured['statistics']) == 2
    assert captured['settings']['preprocessing']['mean_frames'] == 3
    assert {'raw_value','filtered_value','value'} <= set(captured['auxiliary']['input_traces.csv'])


def test_validation_rejects_unknown_analysis_arguments_before_a_job():
    from pymicroglia.knowledge import validate
    check = validate('cell_grid', {'run': 'unused', 'analysis': 'wavelet_spectrum/wavelet',
                                  'analysis_parameters': {'typo': 1}})
    assert not check['ok'] and 'typo' in check['message']


def test_insufficient_native_clock_keeps_an_unavailable_panel_and_raw_evidence():
    src = source()
    table = src.table('cell_frame.csv').query('frame_index == 0')
    src.table = lambda _: table.copy()
    prepared, evidence = cell_grid.prepare(src, options(analysis='autocorrelation/autocorrelation'))
    assert len(prepared['analyses']['panels']) == 2
    assert set(evidence.status) == {'failed'}
    inputs = prepared.auxiliary['input_traces.csv']
    assert inputs.raw_value.notna().all() and inputs.value.isna().all()


def test_job_publishes_cell_progress_and_clears_the_callback():
    from pymicroglia.figure_tables.actions import _figure_progress
    from pymicroglia.ui.jobs import JobManager
    observed = []
    def runner(action, **kwargs):
        callback = _figure_progress.get()
        observed.append(callable(callback))
        if callback:
            callback({'completed': 2, 'total': 2, 'identity': 8})
        return {'record': {}, 'result': None}
    manager = JobManager(runner)
    row = manager.submit('cell_grid', {'run': 'unused'}, claim='Inspect all measured cells')
    manager.pool.shutdown(wait=True)
    done = manager.get(row['id'])
    assert done['state'] == 'completed' and done['progress']['completed'] == 2
    assert observed == [True] and _figure_progress.get() is None
