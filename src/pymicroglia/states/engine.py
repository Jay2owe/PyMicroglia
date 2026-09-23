"""Frame states, temporal phenotypes and broad-period rhythm evidence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from .._results import document, read_document, write_document, register_outputs
import platform

import numpy as np
import pandas as pd

from pymicroglia.clustering.cohort import _hash
from pymicroglia.clustering.fingerprint import ClusteringOptions, prepare_features
from pymicroglia.states.dynamics import describe_dynamics
from pymicroglia.states.features import CELL, META, frame_features, read_inputs
from pymicroglia.states.mixture import StateOptions, fit_states, predict_states


def _write_json(path, value):
    # Fail on non-finite provenance instead of writing nonstandard JSON.
    write_document(path, value)


def state_profiles(assignments, snapshots, features):
    rows = []
    selected = assignments.split.eq("training") & assignments.state.ge(0)
    for state, group in assignments.loc[selected].groupby("state"):
        for feature in features:
            values = snapshots.loc[group.index, feature].dropna()
            rows.append({"state": int(state), "feature": feature, "observed_frames": len(values),
                         "cells": len(group[CELL].drop_duplicates()), "subjects": group.subject.nunique(),
                         "median": values.median(), "q25": values.quantile(.25), "q75": values.quantile(.75)})
    return pd.DataFrame(rows, columns=["state", "feature", "observed_frames", "cells", "subjects", "median", "q25", "q75"])


def group_dynamics(cells, features, rhythm_results, options):
    """Exploratory whole-cell groups; static state discovery remains independent."""
    from sklearn.cluster import HDBSCAN
    features = features.set_index(CELL).copy()
    if not rhythm_results.empty:
        selected = rhythm_results.loc[rhythm_results.supported_period_for_grouping]
        if not selected.empty:
            periods = selected.pivot(index=CELL, columns="trace_key", values="period_hours")
            periods.columns = [f"supported_period_hours|{column}" for column in periods]
            features = features.join(periods)
    assignments = cells[[*CELL, "subject", "condition"]].copy()
    assignments["dynamic_group"], assignments["membership_score"] = -2, np.nan
    report = {"scope": "exploratory grouping of this cohort; independent-animal reproducibility not established",
              "status": "disabled" if not options.group_dynamic_cells else "insufficient_data",
              "features": [], "excluded": {}}
    eligible_cells = cells.get("state_observed_hours", cells.probability_observed_hours).gt(0).to_numpy() & cells.frames.ge(options.min_cell_frames).to_numpy()
    indices = np.flatnonzero(eligible_cells)
    if options.group_dynamic_cells and len(indices) >= max(options.dynamic_min_cluster_size, options.dynamic_min_samples + 1):
        try:
            families = {c: c.split("|")[0] for c in features}
            values, _, eligible, prep = prepare_features(features.reset_index(drop=True), families, indices,
                ClusteringOptions(max_feature_missing=options.max_feature_missing,
                                  max_cell_missing=options.max_frame_missing,
                                  correlation_cutoff=options.correlation_cutoff))
        except ValueError as error:
            report["reason"] = str(error)
        else:
            indices = np.flatnonzero(eligible & eligible_cells)
            report.update(features=prep["columns"], excluded=prep["excluded"], preprocessing=prep)
            if len(indices) >= max(options.dynamic_min_cluster_size, options.dynamic_min_samples + 1):
                cluster = HDBSCAN(min_cluster_size=options.dynamic_min_cluster_size,
                                  min_samples=options.dynamic_min_samples, copy=True).fit(values[indices])
                assignments.loc[indices, "dynamic_group"] = cluster.labels_
                assignments.loc[indices, "membership_score"] = cluster.probabilities_
                report["status"] = "fitted"
                report["groups"] = len(set(cluster.labels_) - {-1})
    assignments["dynamic_group_status"] = np.select(
        [assignments.dynamic_group.ge(0), assignments.dynamic_group.eq(-1)], ["grouped", "unassigned"],
        default="disabled" if not options.group_dynamic_cells else "insufficient_data")
    return assignments, features.reset_index(), report


def _calibration(source):
    return sorted({json.dumps(movie.get("provenance", {}).get("scale", {}), sort_keys=True)
                   for movie in source.get("measurement_runs", [])})


def apply_dictionary(meta, features, directory, source):
    directory = Path(directory).resolve()
    dictionary = read_document(directory / "state_dictionary.json")
    if dictionary["calibration"] != _calibration(source):
        raise ValueError("The saved state dictionary and new recordings have different calibration metadata")
    report = dictionary["report"]
    with np.load(directory / "state_model.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    checkpoint = None
    if report["representation"] == "neural":
        import torch
        checkpoint = torch.load(directory / "state_encoder.pt", map_location="cpu", weights_only=True)
    if report.get("state_method", "gaussian_mixture") != "gaussian_mixture":
        from pymicroglia.states.methods import replay_alternative
        assignments = replay_alternative(meta, features, dictionary["preprocessing"], arrays, report, checkpoint)
    else:
        probability, density, coverage = predict_states(features, dictionary["preprocessing"], arrays, report, checkpoint)
        assignment_options = dictionary["assignment_options"]
        eligible = coverage >= 1 - assignment_options["max_frame_missing"]
        outlier = density < report["density_threshold"] if report["density_threshold"] is not None else np.zeros(len(meta), bool)
        uncertain = probability.max(axis=1) < assignment_options["state_probability_min"]
        labels = probability.argmax(axis=1)
        labels[outlier | uncertain], labels[~eligible] = -1, -2
        probability[~eligible | outlier] = np.nan
        assignments = meta.copy()
        assignments["split"], assignments["eligible"], assignments["state"] = "application", eligible, labels
        assignments["state_status"] = np.select([~eligible, outlier, uncertain],
            ["insufficient_measurements", "outside_training_distribution", "uncertain"], default="assigned")
        assignments["observed_feature_fraction"], assignments["log_density"] = coverage, np.where(eligible, density, np.nan)
        for state in range(probability.shape[1]):
            assignments[f"state_probability_{state}"] = probability[:, state]
    model_inputs = [document(directory / "state_dictionary.json"), directory / "state_model.npz"]
    if checkpoint is not None:
        model_inputs.append(directory / "state_encoder.pt")
    return {"assignments": assignments, "preprocessing": dictionary["preprocessing"], "model": arrays,
            "checkpoint": checkpoint, "report": report, "neural_history": pd.DataFrame(),
            "candidates": pd.DataFrame(), "validation": pd.DataFrame(), "dictionary": dictionary,
            "source_dictionary": [{"path": str(p), "sha256": _hash(p)} for p in model_inputs]}


def run_states(run, out, options=None, model=None, progress=None):
    """Write a new result; replaying a dictionary never refits the snapshot states."""
    import sklearn
    from pymicroglia.states.rhythms import analyse_rhythms

    options = options or StateOptions()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new state analysis output")
    tables, source = read_inputs(run)
    meta, snapshots, changes, families, audit = frame_features(tables)
    fitted = apply_dictionary(meta, snapshots, model, source) if model else fit_states(meta, snapshots, families, options)
    assignments = fitted["assignments"]
    cells, transitions, bouts, dynamic_features = describe_dynamics(assignments, snapshots, changes, options.max_gap_hours)
    profiles = state_profiles(assignments, snapshots, fitted["report"]["features"])
    if model:
        profiles = pd.read_csv(Path(model) / "state_profiles.csv")
    for index, row in audit.iterrows():
        if row.get("kind") == "snapshot" and row.reason == "candidate":
            audit.loc[index, "reason"] = ("used_in_state_dictionary" if row.feature in fitted["report"]["features"] else
                fitted["preprocessing"]["excluded"].get(row.feature, "not_selected_for_state_dictionary"))
    if source["input_audit"]:
        audit = pd.concat([audit, pd.DataFrame(source["input_audit"])], ignore_index=True)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "options": asdict(options), "mode": "apply_saved_dictionary" if model else "fit_dictionary",
                "source": source, "snapshot_model": fitted["report"], "frames": len(meta), "cells": len(cells),
                "calibration_metadata": "recorded" if _calibration(source) else "not_recorded; comparability must be established from acquisition records",
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit_learn": sklearn.__version__},
                "implementation": {str(p.relative_to(Path(__file__).parents[1])): _hash(p)
                    for p in [*Path(__file__).parent.glob("*.py"),
                              *Path(__file__).parents[1].joinpath("clustering").glob("*.py"),
                              *Path(__file__).parents[1].joinpath("_workbench").glob("*.py"),
                              Path(__file__).parents[1] / "workbench.py"]}}
    _write_json(out / "manifest.json", manifest)
    try:
        # Retain the useful histories even if a later rhythm method fails.
        outputs = {"frame_states": assignments, "cell_dynamics": cells, "transitions": transitions,
                   "dwell_bouts": bouts, "state_profiles": profiles, "feature_audit": audit,
                   "snapshot_features": pd.concat([meta, snapshots], axis=1),
                   "change_features": pd.concat([meta, changes], axis=1),
                   "state_candidates": fitted["candidates"], "state_validation": fitted["validation"]}
        for name, table in outputs.items():
            table.to_csv(out / f"{name}.csv", index=False)
        dictionary = fitted.get("dictionary") or {"schema_version": 1, "report": fitted["report"],
            "preprocessing": fitted["preprocessing"], "calibration": _calibration(source),
            "assignment_options": {key: getattr(options, key) for key in ("max_frame_missing", "state_probability_min")}}
        _write_json(out / "state_dictionary.json", dictionary)
        np.savez(out / "state_model.npz", **fitted["model"])
        if "merge_children" in fitted["model"]:
            children = fitted["model"]["merge_children"]
            pd.DataFrame({"left": children[:, 0], "right": children[:, 1],
                          "distance": fitted["model"]["merge_distances"]}).to_csv(out / "state_hierarchy.csv", index=False)
        if "transmat" in fitted["model"]:
            matrix = fitted["model"]["transmat"]
            pd.DataFrame([{"from_state": a, "to_state": b, "probability": matrix[a, b]}
                          for a in range(len(matrix)) for b in range(len(matrix))]).to_csv(out / "model_transitions.csv", index=False)
        if fitted["checkpoint"] is not None:
            import torch
            torch.save(fitted["checkpoint"], out / "state_encoder.pt")
            fitted["neural_history"].to_csv(out / "encoder_training.csv", index=False)
            manifest["versions"]["torch"] = torch.__version__
        rhythm = analyse_rhythms(assignments, snapshots, changes, source, options, progress=progress)
        if options.group_dynamic_cells and options.dynamic_method == "trajectory_dtw":
            from pymicroglia.states.trajectory import group_trajectories
            trajectory = group_trajectories(meta, snapshots, changes, options)
            dynamic_groups, dynamic_report = trajectory["assignments"], trajectory["report"]
            trajectory["distances"].to_csv(out / "trajectory_distances.csv", index=False)
            trajectory["trajectories"].to_csv(out / "trajectory_features.csv", index=False)
        else:
            dynamic_groups, dynamic_features, dynamic_report = group_dynamics(cells, dynamic_features, rhythm["results"], options)
            dynamic_report["method"] = "descriptor_hdbscan"
        for name, table in {"rhythms": rhythm["results"], "rhythm_traces": rhythm["traces"],
                            "persistence_null_results": rhythm["null_results"], "persistence_null_summary": rhythm["null_summary"],
                            "dynamic_features": dynamic_features, "dynamic_groups": dynamic_groups}.items():
            table.to_csv(out / f"{name}.csv", index=False)
        manifest.update(status="complete", rhythms=rhythm["report"], dynamic_grouping=dynamic_report,
                        source_dictionary=fitted.get("source_dictionary", []),
                        biological_validation="not established",
                        state_codes={"-2": "insufficient measurements", "-1": "uncertain or outside training distribution",
                                     "0 and above": "shared snapshot state"})
        _write_json(out / "manifest.json", manifest)
        (out / "README.md").write_text(
            "# Shared snapshot states and changing cell behaviour\n\n"
            "frame_states.csv contains method-specific scores and state assignments. Posterior probabilities, hard indicators, density strengths and distance margins have distinct names and meanings. "
            "state_profiles.csv describes training frames in original measurement units; these are descriptive frame medians, not animal-level inference. "
            "state_candidates.csv and state_validation.csv record count selection and held-out unit scores; the dictionary states whether units are animals, recordings or exploratory whole-cell holdouts.\n\n"
            "transitions.csv contains observed consecutive-frame pairs, including self transitions. "
            "dwell_bouts.csv records duration bounds and censoring at recording boundaries, gaps and uncertain frames. "
            "cell_dynamics.csv records coverage; dynamic_features.csv contains occupancy, changes, transition and dwell descriptions. "
            "dynamic_groups.csv adds exploratory whole-cell groups (-1 unassigned; -2 unavailable).\n\n"
            "rhythms.csv separates period estimates, significance, fit and data sufficiency. "
            "rhythm_traces.csv contains the exact tested probability/original traces. "
            "persistence_null_results.csv and persistence_null_summary.csv diagnose how often first-order persistence "
            "alone passes the selected tests; they do not establish a biological null p-value. "
            "No fixed daily rhythm or cross-period phase comparison is used.\n\n"
            "The fitted state dictionary is frozen and can be applied to another comparable run using --model. See snapshot_model.transfer_rule for method-specific prediction extensions; hidden Markov inference requires compatible cadence and uses complete supplied sequences. "
            "Dynamic groups are rediscovered for each cohort; their numbers are not transferable identities. "
            "manifest.json records settings, methods, versions, input hashes, validation scope and limitations. "
            "Subjects must identify animals for animal-level validation. Biological states and rhythm claims need independent validation.\n",
            encoding="utf-8")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        _write_json(out / "manifest.json", manifest)
        raise
    register_outputs(out, stage="states", inputs=source["inputs"], settings=asdict(options))
    return manifest


def command(args):
    try:
        settings = json.loads(Path(args.options).read_text(encoding="utf-8")) if args.options else {}
        report = run_states(args.run, args.out, StateOptions(**settings), model=args.model,
                            progress=lambda message: print(message, flush=True))
    except (ValueError, OSError, TypeError, ImportError) as error:
        print(f"Cell states: {error}")
        return 1
    print(f"Cell states: {report['cells']} cells, {report['frames']} frames, "
          f"{report['snapshot_model']['states']} shared snapshot components. Saved to {Path(args.out).resolve()}")
    return 0
