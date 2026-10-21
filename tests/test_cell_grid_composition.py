"""Composition must preserve completed values and independent scientific axes."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pymicroglia.figure_tables import cell_grid
from pymicroglia.visualisation.figures import load
from pymicroglia.visualisation.panels.cell_analysis_grid import analyses
from pymicroglia import workbench


def options(**changes):
    return {**{row.name: row.default for row in load()['cell_grid'].options}, **changes}


def source():
    hours = np.arange(96) * .5
    table = pd.DataFrame([{'identity': identity, 'frame_index': index, 'hours': hour,
                          'integrated_density': 100 + hour + 8*np.sin(2*np.pi*hour/24),
                          'area_px': 300 + 40*np.cos(2*np.pi*hour/24)}
                         for identity in (3, 8) for index, hour in enumerate(hours)])
    table.loc[(table.identity == 3) & table.frame_index.eq(20), 'integrated_density'] = np.nan
    return SimpleNamespace(table=lambda _: table.copy(), module_params=lambda _: {}, figure_theme='classic')


@pytest.mark.parametrize('mode,count', [('overlay', 2), ('stack', 4), ('separate', 4)])
def test_trace_measurement_composition_preserves_values_and_drawing_is_data_only(mode, count, monkeypatch):
    prepared, evidence = cell_grid.prepare(source(), options(metrics=['integrated_density', 'area_px'],
        analysis_parameters={'period_testing': False}, grid_presentation={'layout': mode}))
    data = prepared['analyses']
    before = data['table'].copy(deep=True)
    def refuse(*args, **kwargs):
        raise AssertionError('Presentation reran science')
    monkeypatch.setattr(workbench, 'estimate_one', refuse)
    monkeypatch.setattr(workbench, 'detrend_trace', refuse)
    result = analyses(Figure(), data)
    assert len(result.axes) == count
    expected = data['plot_table'].sort_values(['identity', 'plot_series', 'frame_index'])
    actual = sorted((line for axis in result.axes for line in axis.lines if len(line.get_ydata()) == 96), key=lambda line: line.get_label())
    assert len(actual) == 4
    pd.testing.assert_frame_equal(data['table'], before)
    assert len(evidence) == 4
    for row in data['series_specs']:
        one = expected.loc[expected.plot_series.eq(row['key'])]
        np.testing.assert_allclose(one.plot_value, one.value, equal_nan=True)
    assert json.loads(prepared.auxiliary['grid_presentation.csv'].settings_json.iloc[0])['layout'] == mode


def test_raw_mean_and_processed_lanes_have_explicit_units_and_keep_gaps():
    prepared, _ = cell_grid.prepare(source(), options(analysis_parameters={'period_testing': False},
        grid_presentation={'layout': 'stack', 'trace_layers': ['raw', 'mean', 'processed']}))
    data = prepared['analyses']
    assert len(analyses(Figure(), data).axes) == 6
    table = data['plot_table'].query('identity == 3')
    assert table.query("trace_layer == 'raw' and frame_index == 20").plot_value.isna().all()
    assert table.query("trace_layer == 'mean' and 19 <= frame_index <= 21").plot_value.isna().all()
    for layer, column in [('raw', 'raw_value'), ('mean', 'filtered_value'), ('processed', 'value')]:
        one = table.query('trace_layer == @layer')
        np.testing.assert_allclose(one.plot_value, one[column], equal_nan=True)
    assert len({row['unit'] for row in data['series_specs']}) == 2


def test_overlay_refuses_mixed_scales_and_invalid_preferences_before_job():
    with pytest.raises(ValueError, match='different units or normalization'):
        cell_grid.prepare(source(), options(analysis_parameters={'period_testing': False},
            grid_presentation={'layout': 'overlay', 'trace_layers': ['raw', 'processed']}))
    from pymicroglia.knowledge import validate
    for value in ({'alpha': 2}, {'line_width': -1}, {'trace_layers': ['typo']}, {'layout': 'typo'}):
        assert not validate('cell_grid', {'run': 'unused', 'grid_presentation': value})['ok']


def test_native_curve_overlays_preserve_series_and_stack_preserves_each_panel():
    request = options(analysis='autocorrelation/autocorrelation', metrics=['integrated_density', 'area_px'])
    separate, evidence = cell_grid.prepare(source(), request)
    overlaid, second = cell_grid.prepare(source(), {**request, 'grid_presentation': {'layout': 'overlay'}})
    stacked, third = cell_grid.prepare(source(), {**request, 'grid_presentation': {'layout': 'stack'}})
    assert len(overlaid['analyses']['panels']) == 2
    assert len(analyses(Figure(), overlaid['analyses']).axes) == 2
    assert [len(group) for group in stacked['analyses']['panels']] == [2, 2]
    assert len(analyses(Figure(), stacked['analyses']).axes) == 4
    for index, (_, definition) in enumerate(overlaid['analyses']['panels']):
        originals = [panel[1].series for panel in separate['analyses']['panels'][index*2:index*2+2]]
        originals = [series for group in originals for series in group]
        assert len(definition.series) == len(originals)
        for actual, expected in zip(definition.series, originals):
            assert actual['x'] == expected['x'] and actual['y'] == expected['y']
    def numerical_results(table):
        return [[{key: result[key] for key in ('action', 'answer', 'data')}
                 for result in json.loads(text)] for text in table.results_json]
    assert numerical_results(evidence) == numerical_results(second) == numerical_results(third)
    exported = overlaid['analyses']['table']
    assert set(exported.series) == {series['key'] for _, definition in overlaid['analyses']['panels'] for series in definition.series}
    assert 'source_plotted_data.csv' in overlaid.auxiliary


def test_missing_fits_are_visible_without_estimating_them():
    prepared, _ = cell_grid.prepare(source(), options(analysis_parameters={'period_testing': False},
        grid_presentation={'layout': 'stack', 'trace_layers': ['processed', 'fit']}))
    drawn = analyses(Figure(), prepared['analyses'])
    assert sum(text.get_text() == 'No completed fit' for axis in drawn.axes for text in axis.texts) == 2
