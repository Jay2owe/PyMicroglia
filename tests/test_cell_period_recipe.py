"""The tracked-cell grids use the trace figure's scientific default and profiles."""

import json

import numpy as np
import pandas as pd

from pymicroglia import workbench
from pymicroglia.figure_tables.all_cell_traces import trace_data
from pymicroglia.measure.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from pymicroglia.visualisation.cell_period_recipe import (
    DEFAULT_CELL_PERIOD_RECIPE, resolve_cell_period_recipe)
from pymicroglia.visualisation.figures import get_figure


def test_all_cell_trace_figure_uses_the_grid_period_default():
    options = {option.name: option.default for option in
               get_figure("all_cell_trace_grid").options}
    for name, value in DEFAULT_CELL_PERIOD_RECIPE.items():
        assert options[name] == value


def test_exported_method_profile_preserves_its_complete_recipe(tmp_path, monkeypatch):
    from pymicroglia.pipelines.audit import profiles

    path = tmp_path / "saved-profile.json"
    path.write_text(json.dumps({"kind": "motion-rhythm-settings"}),
                    encoding="utf-8")
    analysis = {**workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
                "fit_method": "lomb", "significance_method": "f",
                "detrend": "none", "multiple_testing": "bh"}
    filtering = {"method": "median", "window_hours": 3.0,
                 "max_gap_hours": 1.5, "min_observations": 1}
    candidate = {"analysis_options": analysis,
                 "rhythm_params": dict(RHYTHM_DEFAULTS),
                 "filtering": filtering, "candidate_id": "candidate-id"}
    def validated_profile(file, measurements):
        assert file == path
        assert measurements == ["signal_mean"]
        return {"profile_id": "profile-id",
                "measurement_recipes": {"signal_mean": candidate}}
    monkeypatch.setattr(profiles, "load_profile", validated_profile)
    recipe = resolve_cell_period_recipe(path)
    assert recipe["fit_method"] == "lomb"
    assert recipe["significance_method"] == "f"
    assert recipe["multiple_testing"] == "bh"
    assert recipe["filtering"] == filtering
    assert recipe["rhythm_params"] == candidate["rhythm_params"]
    assert recipe["fft_component_test"] is False
    assert recipe["profile_provenance"]["candidate_id"] == "candidate-id"


def test_trace_evidence_uses_the_selected_workbench_filter():
    hours = np.arange(30, dtype=float)
    values = np.full(30, 10.0)
    values[15] = 500.0
    frame = pd.DataFrame({"identity": 1, "frame_index": range(30),
                          "hours": hours, "signal_mean": values})
    options = {**workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
               "fit_method": "lomb"}
    resolved = workbench.resolve_analysis_options(
        RHYTHM_DEFAULTS, lambda name: options.get(name))
    points, evidence = trace_data(
        frame, ["signal_mean"], [1], resolved, "raw", "none", {},
        period_testing=False, filtering={"method": "median",
            "window_hours": 3.0, "max_gap_hours": 1.5,
            "min_observations": 1})
    spike = points.loc[points.frame_index.eq(15)].iloc[0]
    assert spike.raw_value == 500.0
    assert spike.filtered_value == 10.0
    assert evidence.iloc[0].observations == 30
