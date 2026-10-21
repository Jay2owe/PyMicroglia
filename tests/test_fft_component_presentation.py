"""Component display retains native amplitudes and individual test evidence."""
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pymicroglia import workbench
from pymicroglia.figure_tables.prepared import PreparedViews
from pymicroglia.figure_tables.grid_presentation import apply
from pymicroglia.visualisation.panels.cell_analysis_grid import analyses
from pymicroglia.visualisation.panels import colour


def prepared():
    points, evidence = [], []
    for identity, count in ((3, 3), (8, 1), (11, 0)):
        hours = np.arange(96.) + 5
        processed = 2 + np.cos(2*np.pi*hours/24)
        displayed = .25 * processed + .4
        processed[20], displayed[20] = np.nan, np.nan
        points.extend(dict(identity=identity, metric='integrated_density', frame_index=i,
                           hours=hour, raw_value=y+100, filtered_value=y+100,
                           processed_value=y, value=v) for i, (hour, y, v) in enumerate(zip(hours, processed, displayed)))
        components = [dict(period_hours=p, amplitude=abs(a), phase_hours=phase % p,
                           native_parameters=[a, p, phase], selected=i == 0)
                      for i, (a, p, phase) in enumerate([(-4., 12., 51.), (2., 24., 29.), (.5, 36., -4.)][:count])]
        tests = [dict(component=i+1, period_hours=c['period_hours'], status='ok' if i != 2 else 'unresolved_frequency',
                      empirical_p_uncorrected=[.2, .01, .001][i], selected_fft_component=i == 0)
                 for i, c in enumerate(components)]
        evidence.append(dict(identity=identity, metric='integrated_density', alpha=.05,
                             estimate_components_json=json.dumps(components), component_results_json=json.dumps(tests),
                             estimate_diagnostics_json=json.dumps(dict(status='ok', mesor=2., phase_zero_timestamp='2000-01-01T05:00:00'))))
    table, statistics = pd.DataFrame(points), pd.DataFrame(evidence)
    settings = dict(identities=[3, 8, 11], metrics=['integrated_density'], labels=['Total light'],
                    trace_normalization='minmax', rows=1, columns=3, xlim=[5, 100],
                    period_testing=False, hour_ticks=24, rhythmic_alpha=.05)
    return PreparedViews({'analyses': dict(table=table, annotations=statistics, settings=settings)},
                         auxiliary={'statistics.csv': statistics.copy()})


@pytest.mark.parametrize('layout,axes_count', [('overlay', 3), ('stack', 8), ('separate', 8)])
def test_all_components_include_secondary_significance_and_keep_gaps(layout, axes_count, monkeypatch):
    result = prepared()
    before = result.auxiliary['statistics.csv'].copy(deep=True)
    apply(result, {'grid_presentation': dict(fft_components=True, layout=layout, show_fit=False)})
    data = result['analyses']
    components = data['plot_table'].query("trace_layer == 'component' and identity == 3")
    for component, expected in ((1, False), (2, True), (3, False)):
        assert components.query('component == @component').component_significant.eq(expected).all()
    assert components.query('frame_index == 20').plot_value.isna().all()
    wave = components.query('component == 1').plot_value.to_numpy(float)
    hours = components.query('component == 1').hours.to_numpy(float)
    expected = .25 * -4*np.cos(2*np.pi*(hours-5-51)/12)
    expected[20] = np.nan
    np.testing.assert_allclose(wave, expected, equal_nan=True, atol=1e-12)
    def refuse(*args, **kwargs):
        raise AssertionError('Drawing reran fitting or testing')
    monkeypatch.setattr(workbench, 'estimate_one', refuse)
    monkeypatch.setattr(workbench, 'test_fft_components', refuse)
    drawn = analyses(Figure(), data)
    assert len(drawn.axes) == axes_count
    component_lines = [line for axis in drawn.axes for line in axis.lines if line.get_label().startswith('Component ')]
    red = [line for line in component_lines if line.get_color() == colour('circadian_red')]
    assert len(red) == 1 and red[0].get_label().startswith('Component 2:')
    assert any('No completed components' in text.get_text() for axis in drawn.axes for text in axis.texts) if layout != 'overlay' else True
    pd.testing.assert_frame_equal(before, result.auxiliary['statistics.csv'])


@pytest.mark.parametrize('p,status,expected', [(.01, 'ok', True), (.05, 'ok', False),
    (float('nan'), 'ok', False), (.001, 'insufficient_blocks', False), (None, 'not_tested', False)])
def test_significance_uses_strict_unadjusted_component_test(p, status, expected):
    assert workbench.fft_component_test_significant({'status': status, 'empirical_p_uncorrected': p}, .05) is expected


def test_component_options_fail_before_reading_data():
    from pymicroglia.figure_tables.cell_grid import validate_options
    for params in ({'fit_method': 'lomb'}, {'period_testing': False}, {'fft_component_test': False}):
        with pytest.raises(ValueError, match='requires'):
            validate_options({'analysis': 'trace', 'analysis_parameters': params, 'grid_presentation': {'fft_components': True}})
    with pytest.raises(ValueError, match='component view'):
        validate_options({'analysis': 'autocorrelation/autocorrelation', 'grid_presentation': {'fft_components': True}})


def test_non_affine_component_display_is_explicitly_unavailable():
    result = prepared()
    result['analyses']['table']['value'] = result['analyses']['table']['processed_value'] ** 2
    apply(result, {'grid_presentation': {'fft_components': True, 'layout': 'stack', 'show_fit': False}})
    components = result['analyses']['plot_table'].query("trace_layer == 'component' and identity == 3")
    assert components.plot_value.isna().all()
    assert components.component_display_note.eq('Component display requires affine normalization').all()


def test_native_component_grid_tests_once_and_stacks_every_wave(monkeypatch):
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from pymicroglia.figure_tables import cell_grid
    from pymicroglia.visualisation.figures import load
    from circadian_workbench.figure_definitions import fft_nlls_component_figure
    stamps = [(datetime(2000, 1, 1) + timedelta(hours=i)).isoformat() for i in range(96)]
    estimate = dict(diagnostics=dict(mesor=0, phase_zero_timestamp=stamps[0]),
                    components=[dict(native_parameters=[2., 12., 0.], selected=True),
                                dict(native_parameters=[1., 24., 0.], selected=False)])
    definition = fft_nlls_component_figure({'timestamps': stamps, 'values': [10.]*96}, estimate)
    completed = SimpleNamespace(definition=definition, records=(), scripts=(), warnings=(), results=())
    monkeypatch.setattr(workbench, 'analyze_cell_view', lambda *args, **kwargs: completed)
    calls = []
    def testing(definition, **settings):
        calls.append(settings)
        return dict(settings=settings, tests=[dict(component=1, status='ok', empirical_p_uncorrected=.2),
                                              dict(component=2, status='ok', empirical_p_uncorrected=.01)])
    monkeypatch.setattr(workbench, 'test_fft_component_view', testing)
    frame = prepared()['analyses']['table'].query('identity == 3').rename(columns={'raw_value': 'integrated_density'})
    source = SimpleNamespace(table=lambda _: frame.copy(), figure_theme='classic', stem='MCG_04')
    options = {option.name: option.default for option in load()['cell_grid'].options}
    options.update(analysis='compare_periods/fft_nlls_components', preprocessing=None,
                   grid_presentation={'fft_components': True, 'layout': 'stack'})
    result, evidence = cell_grid.prepare(source, options)
    assert len(calls) == 1 and len(result['analyses']['panels'][0]) == 3
    assert len(analyses(Figure(), result['analyses']).axes) == 3
    assert len(calls) == 1
    assert len(result.auxiliary['component_tests.csv']) == 1
    assert len(result['analyses']['table'].series.unique()) == 4
    assert json.loads(evidence.definition_json.iloc[0])['series'][3]['colour'] == colour('circadian_red')
