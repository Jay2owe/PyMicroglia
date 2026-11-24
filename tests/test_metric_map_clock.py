"""Actual-time cell-value maps, independently checked with trapezoidal weights."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid

from pymicroglia.figure_tables.metric_maps import cell_metric


def mapped(labels, values, **kwargs):
    return cell_metric(labels, pd.Series(values), metric='saved_cell_value',
                       assignment=kwargs.pop('assignment', 'occupancy_weighted_mean'),
                       range_mode='full', **kwargs)


def test_missing_acquisition_uses_hours_not_frame_count():
    labels = np.array([1, 1, 2, 2], dtype=np.uint16)[:, None, None]
    hours = np.array([1, 1.5, 2.5, 3.])
    # Endpoint trapezoids: cell 1 holds .5 + .5 = 1 h; cell 2 the same.
    result, pixels, _, _ = mapped(labels, {1: 10, 2: 30}, hours=hours)
    assert result[0, 0] == 20
    assert pixels.total_weight.iloc[0] == hours[-1] - hours[0]
    assert pixels.weight_unit.iloc[0] == 'occupied hours'


def test_clock_weighting_differs_when_acquisition_intervals_differ():
    labels = np.array([1, 2, 2], dtype=np.uint16)[:, None, None]
    hours = np.array([1., 1.5, 3.])
    result, pixels, _, _ = mapped(labels, {1: 10, 2: 30}, hours=hours)
    assert result[0, 0] == 27.5  # .25 h for cell 1; 1.75 h for cell 2.
    assert pixels.total_weight.iloc[0] == 2.
    legacy, legacy_pixels, _, _ = mapped(labels, {1: 10, 2: 30})
    assert legacy[0, 0] == 70 / 3
    assert legacy_pixels.total_weight.iloc[0] == 3
    assert legacy_pixels.weight_unit.iloc[0] == 'occupied frames'


def test_time_maps_match_independent_integrator_and_keep_missing_values():
    rng = np.random.default_rng(142)
    labels = rng.integers(0, 4, size=(9, 5, 6), dtype=np.uint16)
    labels[:, 0, 0] = 0
    hours = np.cumsum([1, .5, .5, 1, .5, .5, .5, 1, .5])
    values = {1: 7., 2: 19., 3: np.nan}
    numerator = np.zeros((5, 6)); denominator = numerator.copy()
    for identity, value in values.items():
        if np.isfinite(value):
            weights = trapezoid((labels == identity).astype(float), x=hours, axis=0)
            numerator += value * weights; denominator += weights
    expected = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    result, pixels, cells, _ = mapped(labels, values, hours=hours)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0, equal_nan=True)
    assert np.isnan(result[0, 0])
    assert cells.loc[cells.identity.eq(3), 'contributing_pixels'].iloc[0] == 0
    assert pixels.weight_unit.eq('occupied hours').all()


@pytest.mark.parametrize('method', ['equal_mean', 'first', 'last', 'most_frequent'])
def test_other_overlap_choices_unchanged(method):
    labels = np.array([1, 2, 2], dtype=np.uint16)[:, None, None]
    before = mapped(labels, {1: 10, 2: 30}, assignment=method)
    after = mapped(labels, {1: 10, 2: 30}, assignment=method, hours=[1, 1.5, 3])
    np.testing.assert_array_equal(before[0], after[0])
    pd.testing.assert_frame_equal(before[1], after[1])
    pd.testing.assert_frame_equal(before[2], after[2])


@pytest.mark.parametrize('hours', [[1, 1], [1, 1, 2], [1, np.nan, 3], [3, 2, 1]])
def test_invalid_clock_rejected(hours):
    with pytest.raises(ValueError, match='finite increasing time'):
        mapped(np.ones((3, 1, 1), dtype=np.uint16), {1: 7}, hours=hours)
