"""The approved lower baseline is shared, executable and explicitly overridable."""
import numpy as np
import pytest

from pymicroglia import workbench
from pymicroglia.figure_tables.cell_grid import DEFAULT_PREPROCESSING
from pymicroglia.figure_tables.cell_grid_inputs import process, recipe
from pymicroglia.measure.modules.rhythms import DEFAULTS
from pymicroglia.visualisation.cell_period_recipe import resolve_cell_period_recipe


def test_default_baseline_matches_independent_reference_and_preserves_native_clock():
    Baseline = pytest.importorskip('pybaselines').Baseline
    hours = np.arange(0, 120, .53)
    raw = 100 + .2 * hours + 12 * np.cos(2 * np.pi * hours / 24)
    raw[55:58] += 30
    with workbench.analysis_session():
        result = workbench.detrend_trace(hours, raw, {})
    baseline, _ = Baseline(x_data=hours).asls(raw)
    np.testing.assert_allclose(result['baseline'], baseline, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(result['values'], raw - baseline, rtol=1e-7, atol=1e-6)
    assert result['method'] == 'asymmetric_least_squares'
    assert len(result['values']) == len(hours)
    assert result['asls_smoothness'] == 1_000_000
    assert result['asls_asymmetry'] == .01


def test_shared_defaults_and_explicit_legacy_recipe():
    assert DEFAULTS['detrend'] == DEFAULT_PREPROCESSING['detrend'] == 'asymmetric_least_squares'
    assert resolve_cell_period_recipe(None)['detrend'] == 'asymmetric_least_squares'
    assert resolve_cell_period_recipe({'detrend': 'robust_linear'})['detrend'] == 'robust_linear'
    assert workbench.detrend_settings({'detrend': 'none'})['detrend'] == 'none'
    assert workbench.detrend_settings({'detrend': 'robust_linear'})['detrend'] == 'robust_linear'


def test_default_preprocessing_keeps_gaps_and_matches_workbench():
    hours = np.arange(80) * .53
    raw = 100 + 12 * np.cos(2 * np.pi * hours / 24)
    raw[22] = np.nan
    with workbench.analysis_session():
        mean, residual, records = process(hours, raw, np.arange(80), recipe(DEFAULT_PREPROCESSING))
        native = workbench.cw.trace(hours, mean, settings={'bin_minutes': 31.8}).detrend('asymmetric_least_squares')
    np.testing.assert_allclose(residual, native.trace_data.values, rtol=1e-7, atol=1e-6, equal_nan=True)
    assert np.isnan(residual[[0, 21, 22, 23, 79]]).all()
    assert len(residual) == 80
    assert records[0]['action'] == 'detrend'


def test_trace_extraction_and_legacy_pipelines_share_the_new_default():
    from inspect import signature
    from pymicroglia import tracing
    from pymicroglia.pipelines import bioluminescence, dluc_single_cell
    for function in (tracing.extract_traces, bioluminescence.run, dluc_single_cell.run):
        assert signature(function).parameters['detrends'].default == ('asymmetric_least_squares',)
    hours = np.arange(0, 120, .53)
    raw = 100 + .2 * hours + 12 * np.cos(2 * np.pi * hours / 24)
    actual = tracing.detrend(raw, hours)
    with workbench.analysis_session():
        native = workbench.detrend_trace(hours, raw, {})
    np.testing.assert_allclose(actual[0], np.asarray(native['values']) / raw.mean(), rtol=1e-7, atol=1e-6)
    assert np.isfinite(tracing.detrend(raw, hours, 'cubic')).all()
