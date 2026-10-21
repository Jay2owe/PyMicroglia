"""Existing cell-grid period preparation, separated from rendering."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np

def _period_decisions(tiles, recipe: Mapping[str, Any], recording: str
                      ) -> dict[int, dict[str, Any]]:
    import pandas as pd

    from .. import workbench
    from ..figure_tables.all_cell_traces import trace_data
    from ..measure.modules.rhythms import DEFAULTS as rhythm_defaults

    rows = []
    for tile in tiles:
        times, values = (np.asarray(part, float) for part in tile.trace)
        rows.extend({"identity": int(tile.key), "frame_index": index,
                     "hours": float(hour), "integrated_density": float(value)}
                    for index, (hour, value) in enumerate(zip(times, values)))
    resolved = workbench.resolve_analysis_options(
        recipe.get("rhythm_params", rhythm_defaults),
        lambda name: recipe.get(name))
    component_test = recipe["fft_component_test"]
    if component_test:
        if (resolved["method"] != "fft_nlls" or
                resolved["detrend"] not in workbench.FFT_COMPONENT_DETRENDS or
                resolved["multiple_testing"] != "none"):
            raise ValueError("FFT component testing needs fft_nlls, asymmetric least-squares or robust linear detrending "
                             "and no multiple-testing correction")
        resolved["significance_method"] = "fft_component"
        resolved["params"] = {
            **resolved["params"], "primary_rhythm_test": "fft_component",
            "period_methods": ["fft_nlls"]}
    _, evidence = trace_data(
        pd.DataFrame(rows), ["integrated_density"], [int(tile.key) for tile in tiles],
        resolved, "raw", "none", {}, fft_component_test=component_test,
        recording=recording,
        component_surrogates=recipe["component_surrogates"],
        component_block_hours=recipe["component_block_hours"],
        component_seed=recipe["component_seed"],
        filtering=recipe.get("filtering"))
    decisions = {}
    for row in evidence.itertuples(index=False):
        period = float(row.period_hours) if row.period_hours is not None else np.nan
        p_value = float(row.p_value) if row.p_value is not None else np.nan
        q_value = float(row.q_value) if row.q_value is not None else np.nan
        significant = (row.significance_status == "ok" and
                       row.rhythm_status == "rhythmic" and
                       bool(row.supported_period) and np.isfinite(period))
        decisions[int(row.identity)] = {
            "significant_period": bool(significant),
            "period_hours": period if np.isfinite(period) else None,
            "p_value": p_value if np.isfinite(p_value) else None,
            "q_value": q_value if np.isfinite(q_value) else None,
            "test_status": str(row.significance_status),
            "rhythm_status": str(row.rhythm_status),
            "supported_period": bool(row.supported_period),
        }
    return decisions



def motion_profile_recipe(path, metric):
    from ..pipelines.audit.profiles import load_profile
    profile = load_profile(path, [metric])
    return profile, profile["measurement_recipes"][metric]
