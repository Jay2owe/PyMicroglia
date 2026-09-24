"""One default scientific recipe for measured-cell period displays and grids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .. import workbench


# The all-cell trace grid and the tracked-cell image/video selectors use this
# same recipe. A caller can save an override as JSON and pass its path to each.
DEFAULT_CELL_PERIOD_RECIPE: dict[str, Any] = {
    "fit_method": "fft_nlls",
    "detrend": "robust_linear",
    "multiple_testing": "none",
    "min_cycles": 2.0,
    "period_config": {"nlls_max_components": 5,
                      "nlls_improvement_alpha": 0.05},
    "fft_component_test": True,
    "component_surrogates": 199,
    "component_block_hours": 4.0,
    "component_seed": 20260923,
}


def resolve_cell_period_recipe(value: Mapping[str, Any] | str | Path | None
                               ) -> dict[str, Any]:
    """Resolve and validate the complete recipe used for a grid selection."""
    provenance = None
    if isinstance(value, (str, Path)):
        path = Path(value)
        with path.open(encoding="utf-8") as stream:
            loaded = json.load(stream)
        if isinstance(loaded, Mapping) and loaded.get("kind") == "motion-rhythm-settings":
            from ..pipelines.audit.profiles import load_profile

            profile = load_profile(path, ["signal_mean"])
            candidate = profile["measurement_recipes"]["signal_mean"]
            value = {**candidate["analysis_options"],
                     "rhythm_params": candidate["rhythm_params"],
                     "filtering": candidate["filtering"],
                     "fft_component_test": False}
            provenance = {"profile_id": profile["profile_id"],
                          "candidate_id": candidate["candidate_id"],
                          "path": str(path.resolve())}
        else:
            value = loaded
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("period_recipe must be a settings mapping or JSON path")
    supplied = dict(value or {})
    allowed = set(workbench.CIRCADIAN_ANALYSIS_OPTIONS) | {
        "fft_component_test", "component_surrogates",
        "component_block_hours", "component_seed", "filtering",
        "rhythm_params", "profile_provenance"}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError("period_recipe has unknown settings: " + ", ".join(unknown))
    recipe = {**DEFAULT_CELL_PERIOD_RECIPE, **supplied}
    if not isinstance(recipe["fft_component_test"], bool):
        raise ValueError("period_recipe.fft_component_test must be true or false")
    if recipe["fft_component_test"] and supplied.get("significance_method") is not None:
        raise ValueError("set fft_component_test=false to choose significance_method")
    if (isinstance(recipe["component_surrogates"], bool) or
            not isinstance(recipe["component_surrogates"], int) or
            recipe["component_surrogates"] < 1):
        raise ValueError("period_recipe.component_surrogates must be positive")
    if float(recipe["component_block_hours"]) <= 0:
        raise ValueError("period_recipe.component_block_hours must be positive")
    if isinstance(recipe["component_seed"], bool) or not isinstance(recipe["component_seed"], int):
        raise ValueError("period_recipe.component_seed must be an integer")
    if not isinstance(recipe["period_config"], Mapping):
        raise ValueError("period_recipe.period_config must be a mapping")
    recipe["period_config"] = dict(recipe["period_config"])
    for name in ("filtering", "rhythm_params"):
        if name in recipe and not isinstance(recipe[name], Mapping):
            raise ValueError(f"period_recipe.{name} must be a mapping")
        if name in recipe:
            recipe[name] = dict(recipe[name])
    if provenance is not None:
        recipe["profile_provenance"] = provenance
    return recipe
