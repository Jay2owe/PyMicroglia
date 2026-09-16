"""The legacy rhythm address reaches the moved public-faÃ§ade adapter."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from auto_organotypic import rhythm as moved_rhythm
from pymicroglia import rhythm


def test_legacy_module_is_still_the_moved_module_object():
    assert rhythm is moved_rhythm


def test_legacy_address_reaches_the_public_circadian_facade():
    hours = np.arange(0.0, 8 * 24.0, 0.5)
    values = 20.0 + 5.0 * np.cos(2.0 * np.pi * (hours - 4.0) / 24.0)

    result = rhythm.estimate_period(hours, values, method="lomb")

    assert result["method"] == "lomb"
    assert result["status"] == "ok"
    assert result["period_hours"] == pytest.approx(24.0)
    assert result["source"] == "circadian_workbench.period_methods.estimate_period"


@pytest.mark.parametrize("method", [
    "lomb", "chi_square", "f", "fft_nlls", "mesa", "mfourfit",
    "spectrum_resampling", "jtk", "ejtk",
])
def test_legacy_address_reaches_every_period_method(method):
    expected = {
        "lomb", "chi_square", "f", "fft_nlls", "mesa", "mfourfit",
        "spectrum_resampling", "jtk", "ejtk",
    }
    assert {item["key"] for item in rhythm.available_period_methods()} == expected

    hours = np.arange(0.0, 7 * 24.0, 0.5)
    values = 20.0 + 5.0 * np.cos(2.0 * np.pi * (hours - 4.0) / 24.0)
    result = rhythm.estimate_period(
        hours, values, method=method,
        config={
            "period_detrend": "none", "nlls_max_components": 2,
            "mesa_model_length": 20, "mfourfit_harmonics": 2,
            "mfourfit_step_hours": 0.2, "sr_iterations": 25,
            "sr_grid_points": 64, "sr_seed": 19,
            "jtk_periods": [20.0, 24.0, 28.0], "ejtk_permutations": 25,
            "jtk_correction": "none", "jtk_seed": 19, "jtk_max_points": 48,
        })

    assert result["method"] == method
    assert result["status"] == "ok"
    assert result["period_hours"] == pytest.approx(24.0, abs=0.7)


def test_legacy_address_exposes_every_new_workbench_detrend():
    required = {
        "lowess", "first_difference", "moving_median", "savitzky_golay",
        "robust_linear", "asymmetric_least_squares",
    }
    assert required <= set(rhythm.available_detrend_methods())

    hours = np.arange(0.0, 8 * 24.0, 0.5)
    values = 20.0 + 0.03 * hours + 5.0 * np.cos(
        2.0 * np.pi * (hours - 4.0) / 24.0)
    result = rhythm.detrend(
        hours, values, method="lowess", lowess_fraction=0.25,
        lowess_iterations=1, asls_smoothness=50_000.0,
        asls_asymmetry=0.05, asls_iterations=4)

    assert len(result["values"]) == len(values)
    assert result["method"] == "lowess"
    assert result["lowess_fraction"] == 0.25
    assert result["source"] == "circadian_workbench.analysis.detrend"


def test_real_adapter_imports_only_the_workbench_package_root():
    path = Path(moved_rhythm.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert [name for name in imports if name.startswith("circadian_workbench")] == [
        "circadian_workbench"
    ]
