"""The cell grid keeps measured frames and uses one tested FFT component."""
import numpy as np

from pymicroglia.figure_tables import fft_component_grid as grid
from pymicroglia.visualisation.figures import load


def test_default_grid_declares_accepted_analysis():
    spec = next(spec for spec in load().values() if spec.slug == 'all-cell-trace-grid')
    options = {option.name: option.default for option in spec.options}
    assert options['fft_component_test'] is True
    assert options['fit_method'] == 'fft_nlls'
    assert options['detrend'] == 'robust_linear'
    assert options['multiple_testing'] == 'none'
    assert options['period_min_hours'] == 2.0
    assert options['period_max_hours'] == 48.0
    assert options['min_cycles'] == 2.0
    assert options['component_surrogates'] == 199
    assert options['component_block_hours'] == 4.0


def test_filters_preserve_missing_frames_and_order():
    raw = np.array([0., 2., 100., 4., 6., np.nan, 1., 9., 3.])
    result = grid.median_then_mean(raw)
    expected = np.array([0., 2., 4., 16/3, 6., np.nan, 1., 7/3, 3.])
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_component_choice_prefers_selected_fft_component(monkeypatch):
    passed = {}

    def component_test(hours, values, periods, **settings):
        passed.update(settings)
        return [
            {'component': 1, 'period_hours': 25., 'status': 'ok',
             'empirical_p_uncorrected': .01, 'conditional_partial_r2': .3},
            {'component': 2, 'period_hours': 20., 'status': 'ok',
             'empirical_p_uncorrected': .005, 'conditional_partial_r2': .4},
        ]

    monkeypatch.setattr(grid.workbench, 'test_fft_components', component_test)
    rows, selected = grid.test_cell_components(
        np.arange(10., dtype=float), np.arange(10., dtype=float), np.arange(10),
        [{'period_hours': 25., 'selected': True}, {'period_hours': 20., 'selected': False}],
        recording='MCG_04', identity=7,
    )
    assert len(rows) == 2
    assert selected['period_hours'] == 25.
    assert passed['seed'] == 20260923 + 100000 + 700 + 3
    assert passed['n_surrogates'] == 199
    assert passed['block_hours'] == 4.
