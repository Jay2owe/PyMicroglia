"""Retrospective hidden Markov states on genuinely consecutive cell observations."""
from __future__ import annotations

import warnings
from importlib.metadata import version
import numpy as np
import pandas as pd

from pymicroglia.states.features import CELL
from pymicroglia.states.dynamics import adjacent_edges


def sequences(meta, eligible, max_gap_hours=None, cadence=None):
    """Restart at every cell, absence, unusable observation or large time gap."""
    result, deltas = [], []
    for _, part in meta.groupby(CELL, sort=True):
        part = part.sort_values("frame_index")
        rows = part.index.to_numpy()
        edges, delta = adjacent_edges(part, max_gap_hours)
        edges &= eligible[rows[:-1]] & eligible[rows[1:]]
        deltas.extend(delta[edges].tolist())
        current = []
        for j, row in enumerate(rows):
            if current and (j == 0 or not edges[j - 1]):
                result.append(np.asarray(current, dtype=int))
                current = []
            if eligible[row]:
                current.append(row)
        if current:
            result.append(np.asarray(current, dtype=int))
    inferred = float(np.median(deltas)) if deltas else cadence
    expected = cadence if cadence is not None else inferred
    if deltas and not np.allclose(deltas, expected, rtol=.001, atol=1e-8):
        raise ValueError("Hidden Markov states require a common sampling cadence; irregular intervals need a continuous-time model")
    return result, expected


def _restore(arrays, report):
    from hmmlearn.hmm import GaussianHMM
    model = GaussianHMM(n_components=report["states"], covariance_type=report["covariance_type"], init_params="", params="")
    model.n_features = arrays["means"].shape[1]
    model.startprob_ = arrays["startprob"]
    model.transmat_ = arrays["transmat"]
    model.means_ = arrays["means"]
    model.covars_ = arrays["covariances"]
    return model


def _density(values, arrays, report):
    from sklearn.mixture import GaussianMixture
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    model = GaussianMixture(n_components=report["states"], covariance_type=report["covariance_type"])
    model.weights_, model.means_, model.covariances_ = arrays["emission_weights"], arrays["means"], arrays["covariances"]
    # hmmlearn's fitted spherical variances repeat over features; sklearn's
    # equivalent mixture stores one scalar per component.
    if report["covariance_type"] == "spherical" and model.covariances_.ndim == 2:
        model.covariances_ = model.covariances_.mean(axis=1)
    model.precisions_cholesky_ = _compute_precision_cholesky(model.covariances_, model.covariance_type)
    return model.score_samples(values)


def infer_hidden_markov(meta, values, coverage, arrays, report, splits="application"):
    meta = meta.reset_index(drop=True)
    eligible = coverage >= 1 - report["max_frame_missing"]
    density = _density(values, arrays, report)
    threshold = report["density_threshold"]
    outlier = density < threshold if threshold is not None else np.zeros(len(meta), bool)
    parts, _ = sequences(meta, eligible & ~outlier, report["max_gap_hours"], report["cadence_hours"])
    probability = np.full((len(meta), report["states"]), np.nan)
    if parts:
        rows = np.concatenate(parts)
        probability[rows] = _restore(arrays, report).predict_proba(values[rows].astype(float), [len(p) for p in parts])
    uncertain = np.nan_to_num(probability).max(axis=1) < report["state_probability_min"]
    labels = np.nan_to_num(probability).argmax(axis=1)
    labels[outlier | uncertain], labels[~eligible] = -1, -2
    assignments = meta.copy()
    assignments["split"], assignments["eligible"], assignments["state"] = splits, eligible, labels
    assignments["state_status"] = np.select([~eligible, outlier, uncertain],
        ["insufficient_measurements", "outside_training_distribution", "uncertain"], default="assigned")
    assignments["observed_feature_fraction"] = coverage
    assignments["emission_log_density"] = np.where(eligible, density, np.nan)
    for state in range(report["states"]):
        assignments[f"state_probability_{state}"] = probability[:, state]
    return assignments


def _scores(meta, values, parts, model, split_by):
    from pymicroglia.states.selection import score_unit
    split_by = score_unit(meta, split_by)
    rows = []
    for part in parts:
        key = meta.iloc[part[0]]
        rows.append({**{name: key[name] for name in dict.fromkeys([split_by, *CELL])},
                     "score": model.score(values[part].astype(float)), "frames": len(part)})
    if not rows:
        return pd.Series(dtype=float)
    keys = list(dict.fromkeys([split_by, *CELL]))
    cells = pd.DataFrame(rows).groupby(keys)[["score", "frames"]].sum()
    return (cells.score / cells.frames).groupby(level=0).mean()


def fit_hidden_markov(meta, features, families, options):
    from hmmlearn.hmm import GaussianHMM
    from sklearn.mixture import GaussianMixture
    from pymicroglia.states.mixture import prepare_state_inputs
    (meta, values, observed, eligible, prep, splits, train, balanced,
     selection, test, checkpoint, history) = prepare_state_inputs(meta, features, families, options)
    train_mask = np.zeros(len(meta), bool)
    train_mask[train] = True
    parts, cadence = sequences(meta, train_mask, options.max_gap_hours)
    if cadence is None:
        raise ValueError("Hidden Markov training requires consecutive observed frames")
    # Preserve sequence order, limit by whole sequences; a final prefix is still
    # contiguous. This objective weights observed frames, unlike mixture bootstrap.
    rng = np.random.default_rng(options.seed)
    rng.shuffle(parts)
    kept, remaining = [], options.training_samples
    for part in parts:
        if remaining <= 0:
            break
        kept.append(part[:remaining])
        remaining -= len(kept[-1])
    rows = np.concatenate(kept)
    if len(rows) < 12 or not any(len(p) > 1 for p in kept):
        raise ValueError("Insufficient ordered training data for hidden Markov states")
    selection_parts, _ = sequences(meta, (splits == "selection") & eligible, options.max_gap_hours, cadence)
    candidates, models = [], {}
    from pymicroglia.states.selection import candidate_counts, search_report, split_report
    searched = candidate_counts(options, candidates, "selection_log_likelihood_per_frame", len(selection), len(np.unique(values[rows], axis=0)) // 3)
    for count in searched:
        print(f"Hidden Markov: assessing {count} states", flush=True)
        if len(np.unique(values[rows], axis=0)) < max(12, 3 * count):
            candidates.append({"states": count, "selected": False, "status": "insufficient_unique_training_frames"})
            continue
        best = None
        for repeat in range(options.initializations):
            seed = (options.seed + repeat) % 2**32
            mixture = GaussianMixture(n_components=count, covariance_type=options.covariance_type,
                    reg_covar=options.regularization, n_init=1, max_iter=options.max_iterations, random_state=seed).fit(values[balanced])
            model = GaussianHMM(n_components=count, covariance_type=options.covariance_type,
                    min_covar=options.regularization, covars_prior=options.regularization,
                    n_iter=options.max_iterations, tol=options.hmm_tolerance, random_state=seed,
                    startprob_prior=1.1, transmat_prior=1.1, init_params="", params="stmc")
            model.means_, model.covars_ = mixture.means_.copy(), mixture.covariances_.copy()
            labels = mixture.predict(values)
            start, trans = np.ones(count), np.ones((count, count))
            for part in kept:
                start[labels[part[0]]] += 1
                np.add.at(trans, (labels[part[:-1]], labels[part[1:]]), 1)
            model.startprob_, model.transmat_ = start / start.sum(), trans / trans.sum(axis=1, keepdims=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(values[rows].astype(float), [len(p) for p in kept])
            trace = np.asarray(model.monitor_.history)
            score = float(model.score(values[rows].astype(float), [len(p) for p in kept]))
            converged = len(trace) >= 2 and abs(float(trace[-1] - trace[-2])) < options.hmm_tolerance
            if np.isfinite(score) and converged and (best is None or score > best[0]):
                best = score, model, trace
        row = {"states": count, "selected": False, "status": "not_converged"}
        if best is not None:
            score, model, trace = best
            held = _scores(meta, values, selection_parts, model, options.split_by)
            row.update(status="ok", training_sequence_log_likelihood=score,
                selection_log_likelihood_per_frame=float(held.mean()) if len(held) else None,
                selection_standard_error=float(held.std(ddof=1) / np.sqrt(len(held))) if len(held) > 1 else 0.,
                iterations=int(model.monitor_.iter), final_likelihood_change=float(trace[-1] - trace[-2]))
            models[count] = model
        candidates.append(row)
    accepted = [r for r in candidates if r["status"] == "ok"]
    if not accepted:
        raise ValueError("No hidden Markov candidate converged; increase max_iterations or simplify features/covariance")
    if selection_parts:
        best = max(accepted, key=lambda r: r["selection_log_likelihood_per_frame"])
        chosen = min((r for r in accepted if r["selection_log_likelihood_per_frame"] >= best["selection_log_likelihood_per_frame"] - best["selection_standard_error"]), key=lambda r: r["states"])
    else:
        chosen = min(accepted, key=lambda r: r["states"])
    chosen["selected"] = True
    model, count = models[chosen["states"]], chosen["states"]
    order = np.array(sorted(range(count), key=lambda k: tuple(model.means_[k])))
    weight = model.predict_proba(values[rows].astype(float), [len(p) for p in kept]).mean(axis=0)
    arrays = {"means": model.means_[order], "covariances": model._covars_ if options.covariance_type == "tied" else model._covars_[order],
              "startprob": model.startprob_[order], "transmat": model.transmat_[np.ix_(order, order)], "emission_weights": weight[order]}
    report = {"state_method": "hidden_markov", "states": count, "representation": options.representation,
        "features": prep["columns"], "score_kind": "retrospective_state_posterior",
        "covariance_type": options.covariance_type, "cadence_hours": cadence, "max_gap_hours": options.max_gap_hours,
        "max_frame_missing": options.max_frame_missing, "state_probability_min": options.state_probability_min,
        "library_version": version("hmmlearn"), "training_rows": len(rows), "training_sequences": len(kept),
        "training_sampling": "ordered contiguous sequences; observed-frame-weighted likelihood, seeded sequence order up to training_samples",
        "selection_rule": "smallest within one group standard error of held-out sequence likelihood per frame" if selection_parts else "smallest requested model; no independent selection groups",
        "transfer_rule": "frozen emissions and transitions; complete-sequence posterior inference restarts at every cell, gap or excluded frame",
        "inference_scope": "retrospective; later frames within a supplied sequence can influence earlier state probabilities",
        "density_interpretation": "emission mixture density weighted by training posterior occupancy; not sequence likelihood",
        "state_evidence": "exploratory persistent measurement states; no periodic drive or circadian claim",
        "stability_scope": "best converged training likelihood across initializations; bootstrap sensitivity not assessed",
        "training_groups": sorted(meta.loc[splits == "training", options.split_by].unique().tolist()),
        "selection_groups": sorted(meta.loc[splits == "selection", options.split_by].unique().tolist()),
        "test_groups": sorted(meta.loc[splits == "test", options.split_by].unique().tolist()),
        "test_scope": "untouched by fitting and selection" if len(test) else "no independent test groups available"}
    report.update(split_report(meta, splits, options))
    report.update(search_report(options, candidates, "selection_log_likelihood_per_frame", count))
    density = _density(values, arrays, report)
    report["density_threshold"] = float(np.quantile(density[rows], options.outlier_fraction)) if options.outlier_fraction else None
    assignments = infer_hidden_markov(meta, values, observed.mean(axis=1), arrays, report, splits)
    validation = []
    for split in ("training", "selection", "test"):
        sequences_in_split, _ = sequences(meta, (splits == split) & eligible, options.max_gap_hours, cadence)
        for group, score in _scores(meta, values, sequences_in_split, _restore(arrays, report), options.split_by).items():
            validation.append({"split": split, "group": group, "mean_sequence_log_likelihood_per_frame": float(score)})
    return {"assignments": assignments, "preprocessing": prep, "model": arrays, "checkpoint": checkpoint,
            "neural_history": pd.DataFrame(history), "candidates": pd.DataFrame(candidates),
            "validation": pd.DataFrame(validation), "report": report}
