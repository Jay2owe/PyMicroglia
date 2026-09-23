"""Shared snapshot states, fitted without frame order or biological labels."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

from pymicroglia.clustering.fingerprint import (ClusteringOptions, prepare_features,
                                      learn_fingerprints, replay_fingerprints,
                                      transform_features)
from pymicroglia.states.features import CELL


from .options import StateOptions


def group_split(meta, options):
    from pymicroglia.states.selection import selection_units
    groups, _, enough = selection_units(meta, options)
    unique = np.array(sorted(groups[enough].unique()))
    np.random.default_rng(options.seed).shuffle(unique)
    count = len(unique)
    n_test = max(1, int(np.ceil(count * options.test_fraction))) if count >= 3 and options.test_fraction else 0
    n_select = max(1, int(np.ceil(count * options.selection_fraction))) if count - n_test >= 2 and options.selection_fraction else 0
    n_select = min(n_select, count - n_test - 1)
    labels = np.full(len(meta), "training", dtype=object)
    labels[groups.isin(unique[:n_test])] = "test"
    labels[groups.isin(unique[n_test:n_test + n_select])] = "selection"
    return labels


def balanced_sample(meta, indices, options, seed):
    """Bootstrap equally by group, then cell, then observed frame within cell."""
    subset = meta.iloc[indices]
    if subset.empty:
        return np.array([], dtype=int)
    keys = list(dict.fromkeys([options.split_by, *CELL]))
    cells = subset.groupby(keys, sort=False).size().rename("frames").reset_index()
    cells["cells_in_group"] = cells.groupby(options.split_by)["frames"].transform("size")
    weights = subset.merge(cells, on=keys, how="left", validate="many_to_one", sort=False)
    probability = 1 / (weights.frames.to_numpy(float) * weights.cells_in_group.to_numpy(float))
    probability /= probability.sum()
    largest = max(options.candidate_states) if options.candidate_states else options.auto_initial_states
    size = min(options.training_samples, max(len(indices), 2 * largest))
    return np.random.default_rng(seed).choice(indices, size=size, replace=True, p=probability)


def _group_score(meta, values, indices, estimator, split_by):
    from pymicroglia.states.selection import score_unit
    split_by = score_unit(meta, split_by)
    keys = list(dict.fromkeys([split_by, *CELL]))
    table = meta.iloc[indices][keys].copy()
    table["score"] = estimator.score_samples(values[indices])
    # One animal, one vote, independent of movie/cell/track lengths.
    return table.groupby(keys).score.mean().groupby(level=0).mean()


def prepare_state_inputs(meta, features, families, options):
    """Identical training-only preprocessing for each selectable state estimator."""
    if options.snapshot_features is not None:
        missing = set(options.snapshot_features) - set(features)
        if missing:
            raise ValueError(f"Unknown snapshot features: {sorted(missing)}")
        features = features[list(options.snapshot_features)]
    meta, features = meta.reset_index(drop=True), features.reset_index(drop=True)
    from pymicroglia.states.selection import UNIT, selection_units
    units, scope, _ = selection_units(meta, options)
    if scope == "within_recording_cells_exploratory":
        meta = meta.assign(**{UNIT: units})
    splits = group_split(meta, options)
    counts = meta.groupby(CELL)["frame_index"].transform("size").to_numpy()
    train = np.flatnonzero((splits == "training") & (counts >= options.min_cell_frames))
    if len(train) < 12:
        raise ValueError("At least twelve training frames from sufficiently observed cells are needed")
    balanced = balanced_sample(meta, train, options, options.seed)
    preprocessing_options = ClusteringOptions(max_feature_missing=options.max_feature_missing,
                                               max_cell_missing=options.max_frame_missing,
                                               correlation_cutoff=options.correlation_cutoff)
    values, observed, eligible, preprocessing = prepare_features(features, families, balanced, preprocessing_options)
    train = train[eligible[train]]
    balanced = balanced_sample(meta, train, options, options.seed)
    selection = np.flatnonzero((splits == "selection") & eligible)
    test = np.flatnonzero((splits == "test") & eligible)
    if len(train) < 12:
        raise ValueError("Fewer than twelve sufficiently measured training frames remain")
    checkpoint, history = None, []
    if options.representation == "neural":
        neural_options = ClusteringOptions(**options.neural)
        values, _, checkpoint, history = learn_fingerprints(values, observed, balanced, selection, neural_options, options.seed)
    return meta, values, observed, eligible, preprocessing, splits, train, balanced, selection, test, checkpoint, history


def fit_states(meta, features, families, options):
    if options.state_method != "gaussian_mixture":
        from pymicroglia.states.methods import fit_alternative
        return fit_alternative(meta, features, families, options)
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score
    (meta, values, observed, eligible, preprocessing, splits, train, balanced,
     selection, test, checkpoint, history) = prepare_state_inputs(meta, features, families, options)
    candidates, fitted = [], {}
    from pymicroglia.states.selection import candidate_counts, search_report, split_report, score_unit
    searched = candidate_counts(options, candidates, "selection_log_density", len(selection), len(np.unique(balanced)) // 3)
    for count in searched:
        print(f"Gaussian mixture: assessing {count} states", flush=True)
        row = {"states": count, "selected": False, "status": "not_fitted"}
        if len(np.unique(balanced)) < max(12, count * 3):
            row["status"] = "insufficient_unique_training_frames"
            candidates.append(row)
            continue
        model = GaussianMixture(n_components=count, covariance_type=options.covariance_type,
                                reg_covar=options.regularization, n_init=options.initializations,
                                max_iter=options.max_iterations, random_state=options.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(values[balanced])
        if not model.converged_:
            row["status"] = "not_converged"
            candidates.append(row)
            continue
        row["status"] = "ok"
        row["training_log_density"] = float(_group_score(meta, values, train, model, options.split_by).mean())
        if len(selection):
            scores = _group_score(meta, values, selection, model, options.split_by)
            row["selection_log_density"] = float(scores.mean())
            row["selection_standard_error"] = float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.
        else:
            row["selection_log_density"], row["selection_standard_error"] = None, None
        fitted[count] = model
        candidates.append(row)
    accepted = [r for r in candidates if r["status"] == "ok"]
    if not accepted:
        raise ValueError("No candidate state model converged with sufficient training observations")
    if len(selection):
        best = max(accepted, key=lambda r: r["selection_log_density"])
        cutoff = best["selection_log_density"] - best["selection_standard_error"]
        chosen = min((r for r in accepted if r["selection_log_density"] >= cutoff), key=lambda r: r["states"])
    else:
        chosen = min(accepted, key=lambda r: r["states"])
    chosen["selected"] = True
    model = fitted[chosen["states"]]
    # Stabilise display numbering; never interpret numeric distance between IDs.
    order = np.array(sorted(range(model.n_components), key=lambda k: tuple(model.means_[k])))
    probabilities = model.predict_proba(values)[:, order]
    density = model.score_samples(values)
    lower = float(np.quantile(density[balanced], options.outlier_fraction)) if options.outlier_fraction else None
    labels = probabilities.argmax(axis=1)
    uncertain = probabilities.max(axis=1) < options.state_probability_min
    outlier = density < lower if lower is not None else np.zeros(len(density), dtype=bool)
    labels[uncertain | outlier] = -1
    labels[~eligible] = -2
    probabilities[~eligible | outlier] = np.nan
    assignments = meta.copy()
    assignments["split"] = splits
    assignments["eligible"] = eligible
    assignments["state"] = labels
    assignments["state_status"] = np.select([~eligible, outlier, uncertain],
                                             ["insufficient_measurements", "outside_training_distribution", "uncertain"], default="assigned")
    assignments["observed_feature_fraction"] = observed.mean(axis=1)
    assignments["log_density"] = np.where(eligible, density, np.nan)
    for state in range(model.n_components):
        assignments[f"state_probability_{state}"] = probabilities[:, state]
    stability = []
    for repeat in range(1, options.stability_repeats):
        repeat_seed = (options.seed + repeat) % 2**32
        other = GaussianMixture(n_components=model.n_components, covariance_type=options.covariance_type,
                                reg_covar=options.regularization, n_init=options.initializations,
                                max_iter=options.max_iterations, random_state=repeat_seed)
        sample = balanced_sample(meta, train, options, repeat_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            other.fit(values[sample])
        audit_indices = test if len(test) else selection if len(selection) else train
        agreement = float(adjusted_rand_score(model.predict(values[audit_indices]), other.predict(values[audit_indices]))) if other.converged_ else None
        stability.append({"repeat": repeat, "converged": bool(other.converged_),
                          "adjusted_rand": agreement,
                          "evaluated_on": "test" if len(test) else "selection" if len(selection) else "training"})
    diagnostics = []
    for split in ("training", "selection", "test"):
        indices = np.flatnonzero((splits == split) & eligible)
        if len(indices):
            for group, score in _group_score(meta, values, indices, model, options.split_by).items():
                group_rows = (splits == split) & meta[score_unit(meta, options.split_by)].eq(group).to_numpy()
                diagnostics.append({"split": split, "group": group, "frames": int(group_rows.sum()),
                                    "eligible_frames": int((group_rows & eligible).sum()), "mean_log_density": float(score),
                                    "assigned_fraction": float(np.mean(labels[group_rows] >= 0))})
    model_state = {"weights": model.weights_, "means": model.means_, "covariances": model.covariances_,
                   "precisions_cholesky": model.precisions_cholesky_, "order": order}
    report = {"state_method": "gaussian_mixture", "score_kind": "posterior_probability",
              "states": model.n_components, "representation": options.representation,
              "training_groups": sorted(meta.loc[splits == "training", options.split_by].unique().tolist()),
              "selection_groups": sorted(meta.loc[splits == "selection", options.split_by].unique().tolist()),
              "test_groups": sorted(meta.loc[splits == "test", options.split_by].unique().tolist()),
              "selection_rule": "smallest within one group-level standard error of best held-out density" if len(selection) else "smallest requested model; no independent selection groups",
              "test_scope": "untouched by preprocessing, model fitting and state-count selection" if len(test) else "no independent test groups available",
              "state_evidence": "single continuous component" if model.n_components == 1 else "exploratory mixture components; discreteness not established",
              "covariance_type": options.covariance_type, "density_threshold": lower,
              "stability_scope": "mixture refits with balanced bootstrap samples; representation held fixed",
              "features": preprocessing["columns"], "seed_and_balancing_sensitivity": stability}
    report.update(split_report(meta, splits, options))
    report.update(search_report(options, candidates, "selection_log_density", model.n_components))
    return {"assignments": assignments, "preprocessing": preprocessing, "model": model_state,
            "checkpoint": checkpoint, "neural_history": pd.DataFrame(history),
            "candidates": pd.DataFrame(candidates), "validation": pd.DataFrame(diagnostics), "report": report}


def predict_states(features, preprocessing, arrays, report, checkpoint=None):
    """Apply a saved dictionary to new frames without redefining its states."""
    if report.get("state_method", "gaussian_mixture") != "gaussian_mixture":
        raise ValueError("Use apply_dictionary for method-aware assignments and temporal metadata")
    from sklearn.mixture import GaussianMixture
    values, observed = transform_features(features, preprocessing)
    if report["representation"] == "neural":
        if checkpoint is None:
            raise ValueError("The saved neural state dictionary needs its checkpoint")
        values, _ = replay_fingerprints(features, preprocessing, checkpoint)
    model = GaussianMixture(n_components=len(arrays["weights"]), covariance_type=report["covariance_type"])
    model.weights_, model.means_ = arrays["weights"], arrays["means"]
    model.covariances_, model.precisions_cholesky_ = arrays["covariances"], arrays["precisions_cholesky"]
    return model.predict_proba(values)[:, arrays["order"]], model.score_samples(values), observed.mean(axis=1)
