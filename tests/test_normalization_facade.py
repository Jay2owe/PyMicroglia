"""The legacy PyMicroglia rhythm address retains all normalisations."""

from __future__ import annotations

import numpy as np
import pytest

import circadian_workbench as cw
from pymicroglia import rhythm


EXPECTED = {
    "none", "mean_center", "median_center", "zscore", "robust_zscore",
    "robust_scale", "own_mean", "own_daily_total", "minmax", "max_abs",
    "reference_delta", "fold_change", "delta_over_reference",
    "log2_fold_change", "percent_of_reference", "percent_change", "envelope",
    "pre_treatment_cycle",
}


@pytest.mark.parametrize("method", sorted(EXPECTED))
def test_legacy_address_matches_the_workbench(method) -> None:
    hours = np.arange(0.0, 7 * 24.0, 0.5)
    values = 20.0 + 5.0 * np.cos(2.0 * np.pi * (hours - 4.0) / 24.0)
    kwargs = {}
    if method in {
        "reference_delta", "fold_change", "delta_over_reference",
        "log2_fold_change", "percent_of_reference", "percent_change",
        "pre_treatment_cycle",
    }:
        kwargs["reference_value"] = 20.0

    borrowed = rhythm.normalize(hours, values, method, **kwargs)
    direct = cw.trace(hours, values).normalize(method, **kwargs).data

    assert {row["key"] for row in rhythm.available_normalization_methods()} == EXPECTED
    assert borrowed["method"] == method
    assert borrowed["values"] == pytest.approx(direct["values"], nan_ok=True)
