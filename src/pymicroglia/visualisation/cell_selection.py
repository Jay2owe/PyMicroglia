"""Auditable cell inclusion for tracked-cell image and video grids."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..eligibility import audit_presence
from .cell_period_recipe import resolve_cell_period_recipe


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
                     "hours": float(hour), "signal_mean": float(value)}
                    for index, (hour, value) in enumerate(zip(times, values)))
    resolved = workbench.resolve_analysis_options(
        recipe.get("rhythm_params", rhythm_defaults),
        lambda name: recipe.get(name))
    component_test = recipe["fft_component_test"]
    if component_test:
        if (resolved["method"] != "fft_nlls" or
                resolved["detrend"] != "robust_linear" or
                resolved["multiple_testing"] != "none"):
            raise ValueError("FFT component testing needs fft_nlls, robust_linear "
                             "and no multiple-testing correction")
        resolved["significance_method"] = "fft_component"
        resolved["params"] = {
            **resolved["params"], "primary_rhythm_test": "fft_component",
            "period_methods": ["fft_nlls"]}
    _, evidence = trace_data(
        pd.DataFrame(rows), ["signal_mean"], [int(tile.key) for tile in tiles],
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


def select_cell_tiles(tiles, *, raw, significant_period_only: bool = False,
                      period_recipe: Mapping[str, Any] | str | Path | None = None,
                      period_decisions: Mapping[int | str, Mapping[str, Any]] | None = None,
                      max_gap_frames: int | None = None,
                      max_missing_frames: int | None = None,
                      max_missing_fraction: float | None = None,
                      ) -> tuple[list[int], dict[str, Any]]:
    """Choose identities once; the renderer later sizes crops from those cells."""
    if not isinstance(significant_period_only, bool):
        raise ValueError("significant_period_only must be true or false")
    if period_recipe is not None and not significant_period_only:
        raise ValueError("period_recipe needs significant_period_only=true")
    if period_decisions is not None and not significant_period_only:
        raise ValueError("period_decisions needs significant_period_only=true")
    if not tiles:
        raise ValueError("the selected label view contains no tracked cells")
    quality = []
    for tile in tiles:
        times, _values = tile.trace
        interval = (float(np.median(np.diff(times))) if len(times) > 1 else
                    float(tile.provenance["frame_interval_h"]))
        present = np.ones(len(times), bool)
        present[list(tile.unavailable_frames)] = False
        row = audit_presence(
            present, frame_interval_h=interval, max_gap_hours=None,
            max_gap_frames=max_gap_frames,
            max_missing_frames=max_missing_frames,
            max_missing_fraction=max_missing_fraction)
        quality.append({"identity": int(tile.key), **row})
    candidates = [tile for tile, row in zip(tiles, quality) if row["eligible"]]
    recipe = (resolve_cell_period_recipe(period_recipe)
              if significant_period_only else None)
    if recipe is None:
        periods = {}
    elif period_decisions is None:
        # The test family is the complete label view, before quality limits.
        periods = _period_decisions(tiles, recipe, Path(raw).stem)
    else:
        periods = {int(identity): dict(result)
                   for identity, result in period_decisions.items()}
        missing = sorted({int(tile.key) for tile in tiles} - set(periods))
        if missing:
            raise ValueError("period_decisions omit tracked cells: " +
                             ", ".join(map(str, missing)))
    selected = [int(tile.key) for tile in candidates
                if recipe is None or periods[int(tile.key)]["significant_period"]]
    rows = []
    for row in quality:
        identity = row["identity"]
        period = periods.get(identity)
        rows.append({**row, "period": period,
                     "included": identity in selected,
                     "selection_reason": (row["exclusion_reasons"] if not row["eligible"]
                                          else "period_not_significant_or_unsupported"
                                          if recipe is not None and identity not in selected
                                          else "included")})
    report = {
        "significant_period_only": significant_period_only,
        "period_recipe": recipe,
        "period_source": ("shared automated-chain verdicts"
                          if period_decisions is not None else
                          "tested from this complete label view"
                          if recipe is not None else None),
        "quality_thresholds": {
            "max_gap_frames": max_gap_frames,
            "max_missing_frames": max_missing_frames,
            "max_missing_fraction": max_missing_fraction},
        "selected_identities": selected,
        "excluded_identities": [row["identity"] for row in rows
                                if not row["included"]],
        "cells": rows,
    }
    if not selected:
        raise ValueError("no cells meet the cell-grid selection; inspect the "
                         "period recipe and tracking-gap limits")
    return selected, report


def period_evidence(raw, labels, *, source_frame_offset: int = 0,
                    frame_interval_h: float | None = None,
                    trace_channel: int = 1,
                    period_recipe: Mapping[str, Any] | str | Path | None = None
                    ) -> dict[str, Any]:
    """Test the complete tracked population once for automated grid consumers."""
    from .cell_tiles import cell_tiles

    recipe = resolve_cell_period_recipe(period_recipe)
    tiles = cell_tiles(
        raw, labels, source_frame_offset=source_frame_offset,
        frame_interval_h=frame_interval_h, crop_basis="own_cell",
        trace_channel=trace_channel)
    return {
        "period_recipe": recipe,
        "source": str(Path(raw)), "labels": str(Path(labels)),
        "raw_fingerprint": tiles[0].provenance["raw"],
        "labels_fingerprint": tiles[0].provenance["labels"],
        "decisions": _period_decisions(tiles, recipe, Path(raw).stem),
    }
