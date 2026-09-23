"""Auditable state-count searches with whole-group or whole-cell holdouts."""
from __future__ import annotations

import json
import numpy as np

from pymicroglia.states.features import CELL

UNIT = "state_selection_unit"


def selection_units(meta, options):
    """Use biological groups when possible; never split a cell across sets."""
    groups = meta[options.split_by].astype(str)
    if meta[options.split_by].isna().any() or groups.str.strip().eq("").any():
        raise ValueError("Every frame needs a group identifier for validation")
    counts = meta.groupby(CELL).frame_index.transform("size")
    enough = counts >= options.min_cell_frames
    cells = meta.loc[enough, CELL].drop_duplicates()
    use_cells = options.selection_scope == "cells" or (
        options.selection_scope == "auto" and groups[enough].nunique() < 3 and len(cells) >= 5)
    if use_cells:
        keys = meta[CELL].apply(lambda row: json.dumps([str(v) for v in row], separators=(",", ":")), axis=1)
        scope = "within_recording_cells_exploratory"
    else:
        keys, scope = groups, "held_out_" + options.split_by
    return keys, scope, enough


def score_unit(meta, split_by):
    return UNIT if UNIT in meta else split_by


def split_report(meta, splits, options):
    keys, scope, _ = selection_units(meta, options)
    cell_scope = scope == "within_recording_cells_exploratory"
    result = {"selection_scope": scope, "selection_unit": "cell" if cell_scope else options.split_by}
    for split in ("training", "selection", "test"):
        result[split + "_groups"] = sorted(keys[splits == split].unique().tolist())
        result[split + "_cells"] = int(meta.loc[splits == split, CELL].drop_duplicates().shape[0])
    result["independent_test_groups"] = 0 if cell_scope else len(result["test_groups"])
    result["test_scope"] = (
        "whole cells withheld from fitting and count selection; same recording, not independent biological validation"
        if cell_scope else "whole groups withheld from preprocessing, fitting and count selection"
        if result["test_groups"] else "no independent test groups available")
    return result


def candidate_counts(options, records, score, selection_available, feasible_max):
    """Expand while the search edge remains competitive; retain every attempt."""
    if options.candidate_states is not None:
        requested = sorted(set(options.candidate_states))
        yield from requested if selection_available else requested[:1]
        return
    if not selection_available:
        yield 1
        return
    ceiling = max(1, min(options.auto_max_states, int(feasible_max)))
    stop = min(options.auto_initial_states, ceiling)
    start = 1
    while True:
        yield from range(start, stop + 1)
        if stop >= ceiling:
            return
        valid = [r for r in records if r.get("status") == "ok" and
                 r.get(score) is not None and np.isfinite(r[score])]
        if not valid:
            return
        best = max(valid, key=lambda r: r[score])
        cutoff = best[score] - (best.get("selection_standard_error") or 0.)
        edge = [r for r in valid if r["states"] >= stop - 1]
        if not any(r[score] >= cutoff for r in edge):
            return
        start, stop = stop + 1, min(2 * stop, ceiling)


def search_report(options, records, score, selected):
    tried = sorted({int(r["states"]) for r in records})
    valid = [r for r in records if r.get("status") == "ok" and
             r.get(score) is not None and np.isfinite(r[score])]
    competitive = []
    if valid:
        best = max(valid, key=lambda r: r[score])
        cutoff = best[score] - (best.get("selection_standard_error") or 0.)
        competitive = sorted(r["states"] for r in valid if r[score] >= cutoff)
    return {"count_selection_mode": "adaptive" if options.candidate_states is None else
            "fixed" if len(set(options.candidate_states)) == 1 else "requested_candidates",
            "counts_tested": tried, "competitive_counts": competitive,
            "search_upper_bound": max(tried) if tried else None,
            "computational_state_ceiling": options.auto_max_states if options.candidate_states is None else None,
            "search_boundary_competitive": bool(competitive and max(competitive) >= max(tried) - 1),
            "selected_at_search_boundary": bool(tried and selected == max(tried)),
            "count_uncertainty": "descriptive one-standard-error set across held-out units; not a confidence interval for biological state count"}
