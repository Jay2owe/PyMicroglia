"""Distance and density state estimators, with explicit frozen transfer rules."""
from __future__ import annotations

from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from pymicroglia.clustering.fingerprint import transform_features, replay_fingerprints
from pymicroglia.states.features import CELL


def unique_training(meta, train, options, cap=None):
    """No bootstrap duplicates: repeated points must not manufacture density."""
    cap = min(options.training_samples, cap or options.training_samples)
    if len(train) <= cap:
        return train
    keys = list(dict.fromkeys([options.split_by, *CELL]))
    part = meta.iloc[train]
    size = part.groupby(keys)["frame_index"].transform("size").to_numpy()
    cells = part[keys].drop_duplicates().groupby(options.split_by).size()
    weight = 1 / (size * part[options.split_by].map(cells).to_numpy())
    return np.sort(np.random.default_rng(options.seed).choice(train, size=cap, replace=False, p=weight / weight.sum()))


def _canonical(labels, centres):
    order = np.array(sorted(range(len(centres)), key=lambda k: tuple(centres[k])), dtype=int)
    inverse = np.argsort(order)
    return np.where(labels >= 0, inverse[np.maximum(labels, 0)], -1) if len(order) else labels, centres[order], inverse


def _fit(values, indices, method, count, options):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    x = values[indices]
    if method == "kmeans":
        estimator = KMeans(n_clusters=count, n_init=options.initializations,
                          max_iter=options.max_iterations, random_state=options.seed).fit(x)
        labels, centres, _ = _canonical(estimator.labels_, estimator.cluster_centers_)
        return {"centres": centres}, {"iterations": int(estimator.n_iter_)}
    if method == "agglomerative":
        # Identical measurements cannot carry contradictory frozen native labels.
        x = np.unique(x, axis=0)
        if len(x) < count:
            raise ValueError("Fewer unique measurement vectors than requested states")
        estimator = AgglomerativeClustering(n_clusters=count, linkage=options.agglomerative_linkage,
                                            compute_distances=True).fit(x)
        centres = np.array([x[estimator.labels_ == k].mean(axis=0) for k in range(count)])
        labels, centres, _ = _canonical(estimator.labels_, centres)
        return {"centres": centres, "anchor_values": x, "anchor_labels": labels,
                "merge_children": estimator.children_, "merge_distances": estimator.distances_}, {}
    if method == "hdbscan":
        import hdbscan
        # Retain real repeated observations, but never artificial bootstrap copies.
        if len(x) < max(options.density_min_cluster_size, 2 * options.density_min_samples):
            raise ValueError("Density clustering needs at least max(min_cluster_size, 2 * min_samples) unique training rows")
        estimator = hdbscan.HDBSCAN(min_cluster_size=options.density_min_cluster_size,
                    min_samples=options.density_min_samples, cluster_selection_method=options.density_selection_method,
                    prediction_data=True, core_dist_n_jobs=1).fit(x.astype(float))
        count = len(set(estimator.labels_) - {-1})
        centres = np.array([x[estimator.labels_ == k].mean(axis=0) for k in range(count)]).reshape(count, x.shape[1])
        labels, centres, inverse = _canonical(estimator.labels_, centres)
        return {"centres": centres, "anchor_values": x, "anchor_labels": labels,
                "native_labels": estimator.labels_, "anchor_strength": estimator.probabilities_,
                "condensed_tree": estimator.condensed_tree_.to_numpy(), "label_map": inverse}, {
                    "library_version": version("hdbscan"), "min_cluster_size": options.density_min_cluster_size,
                    "min_samples": options.density_min_samples, "cluster_selection_method": options.density_selection_method}
    raise ValueError(f"Unsupported state method: {method}")


def _density_model(arrays, report):
    import hdbscan
    # The condensed hierarchy and labels are frozen. Rebuild only a neighbour
    # search cache, never fit or reselect clusters. Version pin guards internals.
    if version("hdbscan") != report["library_version"]:
        raise ValueError(f"Frozen density replay requires hdbscan {report['library_version']}")
    model = hdbscan.HDBSCAN(min_cluster_size=report["min_cluster_size"],
        min_samples=report["min_samples"], cluster_selection_method=report["cluster_selection_method"])
    model._raw_data = arrays["anchor_values"].astype(float)
    model._condensed_tree = arrays["condensed_tree"]
    model.labels_ = arrays["native_labels"]
    model.generate_prediction_data()
    return model


def predict_values(values, arrays, report):
    """Return labels, method score and transfer origin, never fake posteriors."""
    method = report["state_method"]
    count = len(arrays["centres"])
    if method == "hdbscan":
        import hdbscan
        if count:
            labels, strength = hdbscan.approximate_predict(_density_model(arrays, report), values.astype(float))
            labels = np.where(labels >= 0, arrays["label_map"][np.maximum(labels, 0)], -1)
        else:
            labels, strength = np.full(len(values), -1), np.zeros(len(values))
        origin = np.full(len(values), "frozen_density_approximation", dtype=object)
        score = {"membership_strength": strength}
    else:
        distances = cdist(values, arrays["centres"])
        labels = distances.argmin(axis=1)
        origin = np.full(len(values), "nearest_centroid_extension" if method == "agglomerative" else "native_nearest_centroid", dtype=object)
        score = {}
    if "anchor_values" in arrays:
        distance, nearest = cKDTree(arrays["anchor_values"]).query(values, k=1)
        same = distance == 0
        labels[same] = arrays["anchor_labels"][nearest[same]]
        origin[same] = "native_training_partition"
        if method == "hdbscan":
            score["membership_strength"][same] = arrays["anchor_strength"][nearest[same]]
    if method != "hdbscan":
        own = distances[np.arange(len(values)), labels]
        other = distances.copy()
        other[np.arange(len(values)), labels] = np.inf
        second = other.min(axis=1)
        margin = np.maximum(0, (second - own) / np.maximum(second, 1e-12)) if count > 1 else np.full(len(values), np.nan)
        score.update(distance_to_centre=own, distance_margin=margin)
    return labels, score, origin


def assignment_table(meta, labels, scores, origin, coverage, report, splits="application"):
    eligible = coverage >= 1 - report["max_frame_missing"]
    outlier = labels < 0
    uncertain = np.zeros(len(meta), bool)
    if "distance_to_centre" in scores:
        threshold = report.get("distance_threshold")
        if threshold is not None:
            outlier |= scores["distance_to_centre"] > threshold
        uncertain = scores["distance_margin"] < report["distance_margin_min"]
    labels = labels.copy()
    labels[outlier | uncertain], labels[~eligible] = -1, -2
    result = meta.copy()
    result["split"], result["eligible"], result["state"] = splits, eligible, labels
    result["state_status"] = np.select([~eligible, outlier, uncertain],
        ["insufficient_measurements", "density_noise" if report["state_method"] == "hdbscan" else "outside_training_distribution", "uncertain"], default="assigned")
    result["observed_feature_fraction"], result["assignment_origin"] = coverage, origin
    for name, score in scores.items():
        result[name] = np.where(eligible, score, np.nan)
    for state in range(report["states"]):
        result[f"state_indicator_{state}"] = np.where(labels >= 0, (labels == state).astype(float), np.nan)
    return result


def _selection_score(meta, values, indices, labels, split_by):
    from sklearn.metrics import silhouette_samples
    from pymicroglia.states.selection import score_unit
    split_by = score_unit(meta, split_by)
    if len(indices) > 2000:
        # Deterministic bounded diagnostic, never used to fit preprocessing.
        indices = np.random.default_rng(0).choice(indices, 2000, replace=False)
    y = labels[indices]
    good = y >= 0
    indices, y = indices[good], y[good]
    if len(set(y)) < 2 or len(set(y)) >= len(y):
        return None, None
    keys = list(dict.fromkeys([split_by, *CELL]))
    table = meta.iloc[indices][keys].copy()
    table["score"] = silhouette_samples(values[indices], y)
    scores = table.groupby(keys).score.mean().groupby(level=0).mean()
    return float(scores.mean()), float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.


def fit_alternative(meta, features, families, options):
    if options.state_method == "hidden_markov":
        from pymicroglia.states.hmm import fit_hidden_markov
        return fit_hidden_markov(meta, features, families, options)
    from pymicroglia.states.mixture import prepare_state_inputs
    (meta, values, observed, eligible, prep, splits, train, balanced,
     selection, test, checkpoint, history) = prepare_state_inputs(meta, features, families, options)
    method = options.state_method
    indices = balanced if method == "kmeans" else unique_training(meta, train, options,
                options.hierarchical_max_samples if method == "agglomerative" else None)
    candidates, models = [], {}
    from pymicroglia.states.selection import candidate_counts, search_report, split_report, score_unit
    searched = [0] if method == "hdbscan" else candidate_counts(options, candidates, "selection_silhouette", len(selection), len(np.unique(values[indices], axis=0)) - 1)
    for count in searched:
        print(f"{method}: assessing {'density-selected' if method == 'hdbscan' else count} states", flush=True)
        if method != "hdbscan" and len(np.unique(values[indices], axis=0)) < count:
            candidates.append({"states": count, "status": "insufficient_unique_vectors", "selected": False})
            continue
        arrays, extra = _fit(values, indices, method, count, options)
        report = {"state_method": method, **extra}
        labels, scores, origin = predict_values(values, arrays, report)
        mean, se = _selection_score(meta, values, selection, labels, options.split_by) if len(selection) and method != "hdbscan" else (None, None)
        candidates.append({"states": len(arrays["centres"]), "status": "ok", "selected": False,
                           "selection_silhouette": mean, "selection_standard_error": se})
        models[len(arrays["centres"])] = (arrays, report, labels, scores, origin)
    accepted = [row for row in candidates if row["status"] == "ok"]
    if not accepted:
        raise ValueError("No candidate has enough unique training measurements")
    scored = [row for row in accepted if row["selection_silhouette"] is not None]
    if scored:
        best = max(scored, key=lambda row: row["selection_silhouette"])
        chosen = min((row for row in scored if row["selection_silhouette"] >= best["selection_silhouette"] - best["selection_standard_error"]), key=lambda row: row["states"])
    else:
        chosen = min(accepted, key=lambda row: row["states"])
    chosen["selected"] = True
    arrays, report, labels, scores, origin = models[chosen["states"]]
    threshold = float(np.quantile(scores["distance_to_centre"][indices], 1 - options.outlier_fraction)) if method != "hdbscan" and options.outlier_fraction else None
    report.update(states=chosen["states"], representation=options.representation, features=prep["columns"],
        score_kind="density_membership_strength" if method == "hdbscan" else "distance_margin",
        score_interpretation="membership strength, not a posterior probability" if method == "hdbscan" else "relative distance advantage over next centre, not a probability; undefined with one state",
        max_frame_missing=options.max_frame_missing, distance_margin_min=options.distance_margin_min,
        distance_threshold=threshold, density_threshold=None,
        selection_rule="density hierarchy; candidate_states ignored" if method == "hdbscan" else
            "smallest within one group standard error of held-out silhouette (up to 2000 rows; cannot validate one continuous component)" if scored else
            "smallest requested feasible count; no usable independent selection score",
        training_groups=sorted(meta.loc[splits == "training", options.split_by].unique().tolist()),
        selection_groups=sorted(meta.loc[splits == "selection", options.split_by].unique().tolist()),
        test_groups=sorted(meta.loc[splits == "test", options.split_by].unique().tolist()),
        training_rows=len(indices), unique_training_rows=len(np.unique(indices)),
        training_sampling="group/cell balanced bootstrap" if method == "kmeans" else "unique rows; all eligible training rows up to cap, then weighted sampling without replacement",
        transfer_rule="native nearest centroid" if method == "kmeans" else "native training labels; nearest-centroid extension for novel measurements" if method == "agglomerative" else "native training labels; frozen HDBSCAN approximate_predict for novel measurements",
        test_scope="untouched by fitting and selection" if len(test) else "no independent test groups available",
        state_evidence="exploratory measurement clusters; biological discreteness not established",
        stability_scope="not assessed; stability_repeats currently applies to Gaussian mixture only",
        linkage=options.agglomerative_linkage if method == "agglomerative" else None)
    report.update(split_report(meta, splits, options))
    if method == "hdbscan":
        report.update(count_selection_mode="density_hierarchy", counts_tested=[], competitive_counts=[],
                      computational_state_ceiling=None, selected_at_search_boundary=False,
                      search_boundary_competitive=False)
    else:
        report.update(search_report(options, candidates, "selection_silhouette", chosen["states"]))
        report["single_population_assessment"] = "silhouette is undefined for one group; it does not establish that multiple populations exist"
    assignments = assignment_table(meta, labels, scores, origin, observed.mean(axis=1), report, splits)
    validation = []
    for (split, group), part in assignments.groupby(["split", score_unit(meta, options.split_by)]):
        validation.append({"split": split, "group": group, "frames": len(part),
                           "eligible_frames": int(part.eligible.sum()), "assigned_fraction": float(part.state.ge(0).mean())})
    return {"assignments": assignments, "preprocessing": prep, "model": arrays, "checkpoint": checkpoint,
            "neural_history": pd.DataFrame(history), "candidates": pd.DataFrame(candidates),
            "validation": pd.DataFrame(validation), "report": report}


def replay_alternative(meta, features, preprocessing, arrays, report, checkpoint=None):
    values, observed = transform_features(features, preprocessing)
    if report["representation"] == "neural":
        if checkpoint is None:
            raise ValueError("Saved neural dictionary needs its checkpoint")
        values, _ = replay_fingerprints(features, preprocessing, checkpoint)
    if report["state_method"] == "hidden_markov":
        from pymicroglia.states.hmm import infer_hidden_markov
        return infer_hidden_markov(meta, values, observed.mean(axis=1), arrays, report)
    labels, scores, origin = predict_values(values, arrays, report)
    return assignment_table(meta, labels, scores, origin, observed.mean(axis=1), report)
