"""Run all six options on identical source measurements, preserving each result.

python -m pymicroglia.states.comparison RUN --out NEW_FOLDER --options SETTINGS.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from .._results import document
import time
import numpy as np
import pandas as pd

from pymicroglia.states.mixture import StateOptions
from pymicroglia.states.features import CELL, read_inputs, frame_features
from pymicroglia.states.engine import run_states, apply_dictionary, _write_json

METHODS = ("gaussian_mixture", "kmeans", "agglomerative", "hdbscan", "hidden_markov", "trajectory_dtw")


def run_comparison(run, out, options=None, resume=False):
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits
    out = Path(out).resolve()
    if out.exists() and not resume:
        raise FileExistsError(f"{out} already exists; use a new comparison folder")
    out.mkdir(parents=True, exist_ok=True)
    options = options or StateOptions(rhythm_enabled=False, persistence_surrogates=0)
    if options.rhythm_enabled:
        raise ValueError("The comparison displays clustering; run rhythm estimation separately with its complete settings")
    tables, source = read_inputs(run)
    meta, snapshot, _, _, _ = frame_features(tables)
    summaries, outputs, replays = [], {}, []
    for method in METHODS:
        trajectory = method == "trajectory_dtw"
        settings = replace(options, state_method="gaussian_mixture" if trajectory else method,
                           group_dynamic_cells=trajectory, dynamic_method="trajectory_dtw" if trajectory else "descriptor_hdbscan")
        target = out / method
        serial_options = json.loads(json.dumps(asdict(settings)))
        if target.exists():
            manifest_path = document(target / "manifest.json")
            if not resume or not manifest_path.exists():
                raise FileExistsError(f"Incomplete or unapproved existing comparison output: {target}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["status"] != "complete" or manifest["options"] != serial_options or manifest["source"] != source:
                raise ValueError(f"Existing {method} result does not match complete requested settings and input provenance")
            print(f"Using completed {method} result.", flush=True)
        else:
            print(f"Fitting {method}...", flush=True)
            start = time.monotonic()
            with threadpool_limits(limits=2):
                manifest = run_states(run, target, settings, model=out / "gaussian_mixture" if trajectory else None)
            _write_json(target / "comparison_timing.json", {"seconds": time.monotonic() - start})
            print(f"Completed {method}: {manifest['snapshot_model']['states']} frame states; {manifest['dynamic_grouping']['status']} cell grouping.", flush=True)
        _write_json(out / f"options_{method}.json", serial_options)
        frame = pd.read_csv(target / "frame_states.csv")
        outputs[method] = frame
        cells = pd.read_csv(target / "cell_dynamics.csv")
        # Replay every frozen frame dictionary from numeric disk artifacts.
        with threadpool_limits(limits=2):
            replay = apply_dictionary(meta, snapshot, target, source)["assignments"]
        equal = np.array_equal(replay.state.to_numpy(), frame.state.to_numpy())
        if not equal:
            raise AssertionError(f"Frozen {method} labels do not replay")
        numeric = [c for c in replay.select_dtypes(include="number") if c in frame and c not in {"state"}]
        for column in numeric:
            np.testing.assert_allclose(replay[column], frame[column], rtol=1e-6, atol=1e-8, equal_nan=True,
                                       err_msg=f"{method}: {column}")
        replays.append({"method": method, "rows": len(frame), "labels_identical": equal,
                        "numeric_columns_checked": len(numeric), "refitted": False})
        model = manifest["snapshot_model"]
        groups = manifest["dynamic_grouping"]
        summaries.append({"method": method, "observation_level": "whole_cell_trajectory" if trajectory else "cell_frame",
            "states_or_groups": groups.get("groups", 0) if trajectory else model["states"],
            "cells": manifest["cells"], "frames": manifest["frames"],
            "assigned_frames": None if trajectory else int(frame.state.ge(0).sum()),
            "unassigned_frames": None if trajectory else int(frame.state.eq(-1).sum()),
            "insufficient_frames": None if trajectory else int(frame.state.eq(-2).sum()),
            "observed_switches": None if trajectory else int(cells.switches.sum()),
            "switches_per_observed_hour": None if trajectory else float(cells.switches.sum() / cells.confident_observed_hours.sum()) if cells.confident_observed_hours.sum() else None,
            "cells_compared": groups.get("cells_compared") if trajectory else None,
            "window_start_hours": groups.get("window_start_hours") if trajectory else None,
            "window_end_hours": groups.get("window_end_hours") if trajectory else None,
            "features": len(groups.get("features", [])) if trajectory else len(model["features"]),
            "score_kind": "distance_to_medoid" if trajectory else model["score_kind"],
            "independent_test_groups": model.get("independent_test_groups", len(model["test_groups"])),
            "count_selection_mode": groups.get("count_selection_mode") if trajectory else model.get("count_selection_mode"),
            "selection_scope": groups.get("scope") if trajectory else model.get("selection_scope"),
            "counts_tested": json.dumps([] if trajectory else model.get("counts_tested", [])),
            "competitive_counts": json.dumps([] if trajectory else model.get("competitive_counts", [])),
            "search_boundary_competitive": False if trajectory else model.get("search_boundary_competitive", False),
            "grouped_cells": groups.get("grouped_cells") if trajectory else None,
            "unassigned_cells": groups.get("unassigned_cells") if trajectory else None,
            "status": groups["status"] if trajectory else "complete"})
        pd.DataFrame(summaries).to_csv(out / "comparison_summary.csv", index=False)
        pd.DataFrame(replays).to_csv(out / "replay_checks.csv", index=False)
    agreements = []
    for i, a in enumerate(METHODS[:-1]):
        for b in METHODS[:-1][i + 1:]:
            both = outputs[a].state.ge(0) & outputs[b].state.ge(0)
            agreements.append({"method_a": a, "method_b": b, "shared_assigned_frames": int(both.sum()),
                "adjusted_rand": float(adjusted_rand_score(outputs[a].loc[both, "state"], outputs[b].loc[both, "state"])) if both.sum() > 1 else None,
                "interpretation": "descriptive partition agreement on jointly assigned frames; not accuracy or biological validation"})
    pd.DataFrame(agreements).to_csv(out / "partition_agreement.csv", index=False)
    _write_json(out / "comparison_manifest.json", {"status": "complete", "methods": list(METHODS),
        "source": source, "options": json.loads(json.dumps(asdict(options))), "replay_verified": True,
        "rhythms_refitted": False, "method_selected_as_keeper": None,
        "interpretation": "Same measurement source; each method selects its count under its recorded rule unless an explicit count is requested. Cell holdouts within one recording are exploratory, not independent biological validation. Trajectory groups compare cells on a common window, so frame-level agreement is inapplicable."})
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--options")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = StateOptions(**json.loads(Path(args.options).read_text(encoding="utf-8"))) if args.options else None
    summary = run_comparison(args.run, args.out, settings, args.resume)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
