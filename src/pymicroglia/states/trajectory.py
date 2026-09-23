"""Whole-cell multimetric trajectories on a shared, fully observed time window."""
from __future__ import annotations

from importlib.metadata import version
import numpy as np
import pandas as pd

from pymicroglia.clustering.fingerprint import ClusteringOptions, prepare_features
from pymicroglia.states.features import CELL
from pymicroglia.states.hmm import sequences


def bounded_distances(series, cadence_hours, max_warp_hours):
    """No resampling or phase alignment: warp never exceeds the recorded bound."""
    from tslearn.metrics import cdist_dtw
    radius = int(np.floor(max_warp_hours / cadence_hours + 1e-9))
    distance = cdist_dtw(np.asarray(series, dtype=float), global_constraint="sakoe_chiba",
                         sakoe_chiba_radius=radius, n_jobs=1)
    # All sequences have equal duration and cadence; common normalization leaves
    # the clustering unchanged and removes its trivial square-root length scale.
    return distance / np.sqrt(series.shape[1]), radius


def _window(meta, parts, cadence, options):
    segments = [(meta.hours.iloc[p[0]], meta.hours.iloc[p[-1]], p) for p in parts
                if len(p) >= options.min_cell_frames]
    if not segments:
        raise ValueError("No sufficiently observed contiguous trajectory segments")
    if options.trajectory_window_start_hours is not None:
        candidates = [(options.trajectory_window_start_hours, options.trajectory_window_end_hours)]
    else:
        starts = sorted({float(s) for s, _, _ in segments})
        ends = sorted({float(e) for _, e, _ in segments}, reverse=True)
        candidates = [(start, end) for start in starts for end in ends
                      if (end - start) / cadence + 1 >= options.min_cell_frames - 1e-7]
        candidates.sort(key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
    minimum = max(options.trajectory_min_cells, (options.trajectory_groups or 0) + 1)
    best = None
    for start, end in candidates:
        if best is not None and end - start < best[1] - best[0] - 1e-8:
            break
        steps = (end - start) / cadence
        if not np.isclose(steps, round(steps), atol=1e-6) or steps + 1 < options.min_cell_frames:
            continue
        grid = start + np.arange(round(steps) + 1) * cadence
        selected = []
        for left, right, rows in segments:
            if left > start + 1e-8 or right < end - 1e-8:
                continue
            t = meta.hours.iloc[rows].to_numpy()
            take = rows[(t >= start - 1e-8) & (t <= end + 1e-8)]
            if len(take) == len(grid) and np.allclose(meta.hours.iloc[take], grid, rtol=0, atol=1e-6):
                selected.append(take)
        if len(selected) >= minimum and (best is None or len(selected) > len(best[2])):
            best = start, end, selected
    if best is None:
        raise ValueError(f"No common contiguous time window covers at least {minimum} cells and {options.min_cell_frames} frames")
    return best


def group_trajectories(meta, snapshots, changes, options):
    from sklearn.cluster import AgglomerativeClustering
    meta = meta.reset_index(drop=True)
    features = pd.concat([snapshots.reset_index(drop=True), changes.reset_index(drop=True)], axis=1)
    families = {column: column.split("|")[0] for column in features}
    prep_options = ClusteringOptions(max_feature_missing=options.max_feature_missing,
                                     max_cell_missing=options.max_frame_missing,
                                     correlation_cutoff=options.correlation_cutoff)
    counts = meta.groupby(CELL).frame_index.transform("size").to_numpy()
    candidate = np.flatnonzero(counts >= options.min_cell_frames)
    identities = meta[[*CELL, "subject", "condition"]].drop_duplicates(CELL).reset_index(drop=True)
    identities["dynamic_group"], identities["distance_to_medoid"] = -2, np.nan
    identities["medoid_identity"], identities["medoid_stem"] = None, None
    identities["dynamic_group_status"] = "insufficient_common_window_coverage"
    report = {"method": "trajectory_dtw", "status": "insufficient_data",
              "scope": "exploratory whole-cell groups within this cohort; no independent validation or transferable group identities",
              "distance_interpretation": "bounded multivariate dynamic time warping, divided by square root of common frame count; preserves pooled-scaled levels and temporal order",
              "rhythm_interpretation": "warped similarity does not establish period, shared phase, synchrony or rhythmicity",
              "library_version": version("tslearn"), "max_warp_hours_requested": options.trajectory_max_warp_hours}
    empty = {"assignments": identities, "features": pd.DataFrame(columns=CELL), "distances": pd.DataFrame(),
             "trajectories": pd.DataFrame(), "report": report}
    if not len(candidate):
        report["reason"] = "No cells meet min_cell_frames"
        return empty
    try:
        _, _, eligible, coverage_prep = prepare_features(features, families, candidate, prep_options)
        parts, cadence = sequences(meta, eligible, options.max_gap_hours)
        if cadence is None:
            raise ValueError("No consecutive observations establish a common cadence")
        start, end, selected = _window(meta, parts, cadence, options)
        rows = np.concatenate(selected)
        # Final scaling uses precisely the compared cells and window.
        while True:
            values, observed, final_eligible, prep = prepare_features(features, families, rows, prep_options)
            remaining = [p for p in selected if final_eligible[p].all()]
            if len(remaining) < max(options.trajectory_min_cells, (options.trajectory_groups or 0) + 1):
                raise ValueError("Too few complete cells after common-window feature coverage filtering")
            if len(remaining) == len(selected):
                break
            selected = remaining
            rows = np.concatenate(selected)
    except ValueError as error:
        report["reason"] = str(error)
        return empty
    series = np.stack([values[part] for part in selected])
    distance, radius = bounded_distances(series, cadence, options.trajectory_max_warp_hours)
    if options.trajectory_groups is None:
        import hdbscan
        cluster = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=options.dynamic_min_cluster_size,
            min_samples=options.dynamic_min_samples, cluster_selection_method=options.density_selection_method,
            allow_single_cluster=True).fit(distance.astype(float))
        labels = cluster.labels_
        report.update(grouping_algorithm="hdbscan_on_bounded_dtw_distances", count_selection_mode="density_hierarchy",
            requested_groups=None, density_library_version=version("hdbscan"),
            min_cluster_size=options.dynamic_min_cluster_size, min_samples=options.dynamic_min_samples,
            cluster_selection_method=options.density_selection_method, allow_single_cluster=True,
            selection_rule="density hierarchy chooses count; single cluster and unassigned cells allowed",
            linkage="density_hierarchy")
    else:
        labels = (AgglomerativeClustering(n_clusters=options.trajectory_groups, metric="precomputed", linkage="average").fit_predict(distance)
                  if options.trajectory_groups > 1 else np.zeros(len(selected), int))
        report.update(grouping_algorithm="average_linkage_on_bounded_dtw_distances", count_selection_mode="fixed",
            requested_groups=options.trajectory_groups, selection_rule="explicitly requested group count", linkage="average")
    medoids = {int(label): int(indices[np.argmin(distance[np.ix_(indices, indices)].sum(axis=1))])
               for label in sorted(set(labels) - {-1}) for indices in [np.flatnonzero(labels == label)]}
    ordered = sorted(medoids, key=lambda label: tuple(series[medoids[label]].mean(axis=0)))
    label_map = {label: j for j, label in enumerate(ordered)}
    keys = [tuple(meta.iloc[p[0]][CELL]) for p in selected]
    lookup = {tuple(row): i for i, row in enumerate(identities[CELL].itertuples(index=False, name=None))}
    for i, key in enumerate(keys):
        index, label = lookup[key], int(labels[i])
        if label < 0:
            identities.loc[index, ["dynamic_group", "dynamic_group_status"]] = [-1, "density_noise"]
            continue
        identities.loc[index, ["dynamic_group", "distance_to_medoid", "dynamic_group_status"]] = [label_map[label], distance[i, medoids[label]], "grouped"]
        identities.loc[index, "medoid_identity"] = str(keys[medoids[label]][1])
        identities.loc[index, "medoid_stem"] = str(keys[medoids[label]][0])
    rows = np.concatenate(selected)
    trajectories = pd.concat([meta.iloc[rows].reset_index(drop=True),
                              features.iloc[rows][prep["columns"]].reset_index(drop=True)], axis=1)
    trajectories = trajectories.merge(identities[[*CELL, "dynamic_group"]], on=CELL, validate="many_to_one")
    pairs = [{"stem_a": keys[i][0], "identity_a": keys[i][1], "stem_b": keys[j][0], "identity_b": keys[j][1],
              "distance": float(distance[i, j])} for i in range(len(keys)) for j in range(i, len(keys))]
    report.update(status="fitted", groups=len(medoids), cells_compared=len(selected),
        grouped_cells=int((labels >= 0).sum()), unassigned_cells=int((labels < 0).sum()),
        cells_excluded=len(identities) - len(selected), frames_per_cell=series.shape[1],
        window_start_hours=float(start), window_end_hours=float(end), cadence_hours=cadence,
        window_selection="explicit common recorded-time window" if options.trajectory_window_start_hours is not None else
            "longest common fully observed window with sufficient cells; ties favour more cells then earlier start",
        max_warp_frames=radius, max_warp_hours_applied=radius * cadence,
        preprocessing=prep, coverage_screen_preprocessing=coverage_prep, features=prep["columns"],
        imputed_feature_fraction=float(1 - observed[rows].mean()),
        missing_policy="No absent or insufficiently measured frame is bridged; limited missing features use pooled training medians, never temporal interpolation")
    return {"assignments": identities, "features": identities[CELL].copy(), "distances": pd.DataFrame(pairs),
            "trajectories": trajectories, "report": report}
