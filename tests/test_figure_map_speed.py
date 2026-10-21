"""Exact map values when only the final cumulative field is requested."""
import numpy as np
import pytest
from pymicroglia.figure_tables.territory_values import cumulative_occupancy


@pytest.mark.parametrize('shape', [(2, 1, 1), (48, 12, 16), (193, 32, 21)])
def test_final_occupancy_matches_full_history_exactly(shape):
    rng = np.random.default_rng(20261019)
    labels = rng.integers(0, 5, shape, dtype=np.uint16)
    hours = np.cumsum(rng.uniform(.01, 1.7, shape[0]))
    before = labels.copy()
    np.testing.assert_array_equal(cumulative_occupancy(labels, hours, final_only=True),
                                  cumulative_occupancy(labels, hours)[-1])
    np.testing.assert_array_equal(labels, before)


@pytest.mark.parametrize('labels,hours', [
    (np.zeros((1, 2, 2)), [0]), (np.zeros((2, 2, 2)), [1, 1]),
    (np.full((2, 2, 2), -1), [0, 1]), (np.full((2, 2, 2), np.nan), [0, 1])])
def test_final_occupancy_keeps_input_refusals(labels, hours):
    with pytest.raises(ValueError):
        cumulative_occupancy(labels, hours, final_only=True)
