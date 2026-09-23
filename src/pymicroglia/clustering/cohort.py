"""Discover cell groups from saved measurements, without predefined cell labels.

This is a pooled post-analysis step, not a per-movie measurement module. It
never changes accepted measurements or fits rhythms. Group numbers describe
this cohort only; neither membership scores nor reconstruction errors are
probabilities of a biological cell type.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from .._results import pooled_paths, write_document, register_outputs
import platform
import re

import numpy as np
import pandas as pd

from pymicroglia.clustering.fingerprint import (ClusteringOptions, learn_fingerprints,
                                       prepare_features, split_cells)

KEY = ["stem", "identity"]
METADATA = [*KEY, "subject", "condition"]
# Detailed per-frame/lag/ring measurements retain their declared axes; pairwise
# and population tables do not have one cell per row and cannot be folded here.
TABLES = {
    "cell_summary": (), "cell_frame": ("frame_index",),
    "trend": ("metric",), "rhythms": ("metric",),
    "recurrence_quantification": (), "recurrence": ("lag_frames",),
    "msd_curves": ("lag_frames",), "channel_tracks": ("channel",),
    "channels": ("frame_index", "channel"),
    "cell_object_tracks": ("object_set",),
    "cell_objects": ("frame_index", "object_set"),
    "sholl": ("frame_index", "scaling", "ring"),
    "branches": ("frame_index", "branch"),
}
DISALLOWED_MODULES = {"history", "presence", "provenance", "regimes", "coupling"}
SUPPORT = re.compile(
    r"(^|_)(identity|frame|frames|observations|valid|invalid|saturated|"
    r"inferred|unresolved|background|phase|peak|onset|offset|centroid|handoff|"
    r"x|y|angle|orientation|p_value|alpha|regime)(_|$)"
)
RHYTHM_FEATURES = {"best_period_hours", "best_goodness_of_fit", "rhythmic"}
RHYTHM_PROVENANCE = ["period_estimation_method", "primary_rhythm_test",
                     "workbench_version", "detrend", "detrend_window_hours"]


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry():
    from pymicroglia.measure import modules
    modules.load()
    from pymicroglia.measure.declare import list_derived, list_modules

    columns = {}
    for module in (*list_modules(), *list_derived()):
        for column in module.produces:
            columns.setdefault(column.name, (module.name, column))
    return columns


def _measurement(name, declared):
    if name in declared:
        return declared[name]
    for suffix in ("_median", "_mean", "_iqr", "_sd"):
        if name.endswith(suffix) and name[:-len(suffix)] in declared:
            return declared[name[:-len(suffix)]]
    return None


def _true(values):
    return values.astype(str).str.lower().isin(["true", "1", "1.0"])


def build_features(tables: dict[str, pd.DataFrame]):
    """One row per (movie, cell), with explicit source and exclusion records."""
    if "cell_summary" not in tables:
        raise ValueError("Clustering needs cell_summary.csv")
    summary = tables["cell_summary"]
    if any(c not in summary for c in METADATA):
        raise ValueError(f"cell_summary must contain {METADATA}")
    if summary[KEY].isna().any().any() or summary.duplicated(KEY).any():
        raise ValueError("cell_summary needs unique, nonmissing (stem, identity) keys")
    metadata = summary[METADATA].reset_index(drop=True).copy()
    index = pd.MultiIndex.from_frame(metadata[KEY])
    declared = _registry()
    features, families, audit = {}, {}, []
    rhythm_provenance = []
    for table_name, axes in TABLES.items():
        if table_name not in tables:
            continue
        table = tables[table_name].copy()
        required = [*KEY, *axes]
        if any(c not in table for c in required):
            raise ValueError(f"{table_name} lacks its declared keys: {required}")
        if table[required].isna().any().any() or table.duplicated(required).any():
            raise ValueError(f"{table_name} has missing or duplicate keys {required}")
        if not pd.MultiIndex.from_frame(table[KEY]).isin(index).all():
            raise ValueError(f"{table_name} includes cells absent from cell_summary")
        for name in ("subject", "condition"):
            if name in table:
                expected = metadata.set_index(KEY)[name].reindex(pd.MultiIndex.from_frame(table[KEY]))
                if not np.array_equal(table[name].fillna("").astype(str).to_numpy(),
                                      expected.fillna("").astype(str).to_numpy()):
                    raise ValueError(f"{table_name} has inconsistent {name} metadata")
        if table_name == "rhythms":
            required_rhythm = [*RHYTHM_PROVENANCE, "rhythm_status",
                               "period_underdetermined", "best_period_at_search_edge"]
            if any(c not in table for c in required_rhythm):
                audit.append({"table": table_name, "column": "*", "reason": "missing_rhythm_provenance_or_sufficiency"})
                continue
            for metric, group in table.groupby("metric"):
                if any(group[c].nunique(dropna=False) != 1 for c in RHYTHM_PROVENANCE):
                    raise ValueError(f"{metric}: incompatible saved rhythm methods/settings; cluster comparable runs")
            rhythm_provenance = json.loads(table[[*KEY, "metric", *RHYTHM_PROVENANCE,
                                       "rhythm_status", "period_underdetermined",
                                       "best_period_at_search_edge"]].to_json(orient="records"))
            known = table["rhythm_status"].isin(["rhythmic", "arrhythmic"])
            for name in RHYTHM_FEATURES & set(table):
                table[name] = pd.to_numeric(table[name], errors="coerce").astype(float)
                table.loc[~known, name] = np.nan
            # Unsupported periods remain missing, never zero or an assumed day.
            supported = (table["rhythm_status"].eq("rhythmic")
                         & table["period_underdetermined"].notna()
                         & table["best_period_at_search_edge"].notna()
                         & ~_true(table["period_underdetermined"])
                         & ~_true(table["best_period_at_search_edge"]))
            if "best_period_hours" in table:
                table.loc[~supported, "best_period_hours"] = np.nan
        for column in table.columns:
            if column in METADATA or column in axes:
                continue
            match = _measurement(column, declared)
            reason = None
            if match is None:
                reason = "not_a_declared_measurement"
            else:
                family, declaration = match
                if family in DISALLOWED_MODULES or SUPPORT.search(column):
                    reason = "identifier_timing_or_quality_measure"
                elif family == "rhythms":
                    reason = "rhythm_features_require_the_saved_rhythms_table"
                elif declaration.role in ("reference", "invalid", "significant", "inferred", "unclaimed"):
                    reason = "support_or_significance_column"
            if table_name == "rhythms":
                reason = None if column in RHYTHM_FEATURES else "rhythm_support_or_noncomparable_timing"
                family = "rhythms"
            numeric = pd.to_numeric(table[column], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            if reason is None and numeric.notna().sum() == 0:
                reason = "no_numeric_values"
            if reason:
                audit.append({"table": table_name, "column": column, "reason": reason})
                continue
            if table_name == "cell_frame" and any(column + suffix in summary for suffix in ("_median", "_mean")):
                audit.append({"table": table_name, "column": column, "reason": "already_in_cell_summary"})
                continue
            table[column] = numeric
            dimensions = [a for a in axes if a not in ("frame_index", "branch")]
            parts = table.groupby(dimensions, sort=True) if dimensions else [((), table)]
            for axis_values, part in parts:
                if not isinstance(axis_values, tuple):
                    axis_values = (axis_values,)
                # JSON preserves punctuation in channel/metric names without collisions.
                qualifier = json.dumps(dict(zip(dimensions, axis_values)), sort_keys=True, default=str)
                prefix = f"{table_name}|{qualifier}|{column}"
                if "frame_index" in axes:
                    grouped = part.groupby(KEY)[column]
                    reductions = {"median": grouped.median(),
                                  "iqr": grouped.quantile(0.75) - grouped.quantile(0.25)}
                else:
                    reductions = {"value": part.set_index(KEY)[column]}
                for statistic, values in reductions.items():
                    name = f"{prefix}|{statistic}"
                    features[name] = values.reindex(index).to_numpy(float)
                    families[name] = family
                    audit.append({"feature": name, "table": table_name, "column": column,
                                  "family": family, "label": match[1].label if match else column,
                                  "statistic": statistic, "reason": "candidate"})
    if not features:
        raise ValueError("No suitable measured features found")
    return metadata, pd.DataFrame(features), families, audit, rhythm_provenance


def load_tables(run: Path):
    """Read pooled availability records as well as the actual saved measurements."""
    run = Path(run).resolve()
    source = run / "pooled" if (run / "pooled").is_dir() else run
    manifest_path, root_path, table_folder = pooled_paths(run)
    if not manifest_path.is_file():
        raise ValueError("A saved analysis manifest is required alongside the tables")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = [{"path": str(manifest_path), "sha256": _hash(manifest_path)}]
    measurement_runs = []
    if manifest_path == root_path:
        root = manifest
        manifest = manifest.get("pooled") or {}
        measurement_runs = [m for m in root.get("movies", []) if isinstance(m, dict)]
    if root_path.is_file() and root_path != manifest_path:
        root = json.loads(root_path.read_text(encoding="utf-8"))
        inputs.append({"path": str(root_path), "sha256": _hash(root_path)})
        measurement_runs = [m for m in root.get("movies", []) if isinstance(m, dict)]
    # Distances and lag axes are still in saved pixel/frame units. Scaling each
    # feature statistically cannot make two different calibrations comparable.
    for setting in ("minutes_per_frame", "microns_per_pixel"):
        scales = [m.get("provenance", {}).get("scale", {}).get(setting) for m in measurement_runs]
        if len({json.dumps(s) for s in scales}) > 1:
            raise ValueError(f"Movies have different {setting}; use measurements with comparable calibration")
    tables = {}
    table_audit = []
    for name in TABLES:
        path = table_folder / f"{name}.csv"
        if path.is_file():
            tables[name] = pd.read_csv(path, dtype={"stem": str, "subject": str, "condition": str})
            inputs.append({"path": str(path), "sha256": _hash(path)})
            availability = manifest.get("tables", {}).get(name, {})
            if availability.get("movies_absent"):
                if name == "cell_summary":
                    raise ValueError("Some movies lack the required cell summaries")
                del tables[name]
                table_audit.append({"table": name, "column": "*", "reason": "table_not_measured_in_every_movie"})
                continue
            absent_columns = set().union(*map(set, availability.get("columns_missing", {}).values()))
            if absent_columns.intersection([*METADATA, *TABLES[name]]):
                raise ValueError(f"{name}: pooled manifest reports missing identity or measurement axes")
            tables[name] = tables[name].drop(columns=list(absent_columns), errors="ignore")
            table_audit.extend({"table": name, "column": c, "reason": "column_not_measured_in_every_movie"}
                               for c in sorted(absent_columns))
    rhythm_settings = {}
    if "rhythms" in tables:
        for movie in measurement_runs:
            records = [m for m in movie.get("modules", []) if m.get("module") == "rhythms" and m.get("status") == "done"]
            if len(records) == 1 and records[0].get("parameters"):
                rhythm_settings[movie["stem"]] = records[0]["parameters"]
        contributing = set(tables["rhythms"]["stem"])
        if not contributing.issubset(rhythm_settings):
            del tables["rhythms"]
            table_audit.append({"table": "rhythms", "column": "*", "reason": "complete_saved_rhythm_settings_unavailable"})
        else:
            # Preserve all applied settings; only the list of measurements may
            # differ because the table itself carries its measurement axis.
            signatures = {json.dumps({k: v for k, v in rhythm_settings[s].items() if k != "metrics"}, sort_keys=True)
                          for s in contributing}
            if len(signatures) > 1:
                raise ValueError("Movies have incompatible saved rhythm settings; use a comparable measurement run")
    return tables, {"inputs": inputs, "source_manifest": manifest,
                    "measurement_provenance": [{"stem": m.get("stem"), "provenance": m.get("provenance", {})}
                                               for m in measurement_runs],
                    "saved_rhythm_settings": rhythm_settings, "table_audit": table_audit}


def _clusters(embedding, eligible, options):
    from sklearn.cluster import HDBSCAN

    labels = np.full(len(embedding), -2, dtype=int)
    membership = np.full(len(embedding), np.nan)
    if eligible.sum() < max(options.min_cluster_size, options.min_samples):
        raise ValueError("Too few eligible cells for the configured cluster size/minimum samples")
    fitted = HDBSCAN(min_cluster_size=options.min_cluster_size,
                     min_samples=options.min_samples, n_jobs=1, copy=True).fit(embedding[eligible])
    labels[eligible] = fitted.labels_
    membership[eligible] = fitted.probabilities_
    return labels, membership


def _agreement(a, b):
    from sklearn.metrics import adjusted_rand_score

    common = (a >= 0) & (b >= 0)
    enough = common.sum() >= 3 and len(set(a[common])) > 1 and len(set(b[common])) > 1
    eligible = (a != -2) & (b != -2)
    return {"jointly_assigned_cells": int(common.sum()),
            "adjusted_rand_on_jointly_assigned": float(adjusted_rand_score(a[common], b[common])) if enough else None,
            "assigned_status_agreement": float(np.mean((a[eligible] >= 0) == (b[eligible] >= 0))) if eligible.any() else None}


def fit_cells(metadata, features, families, options):
    """Train fingerprints, cluster the cohort, and report a linear reference."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    if len(metadata) != len(features):
        raise ValueError("Cell metadata and measurements must have the same number of rows")
    metadata, features = metadata.reset_index(drop=True), features.reset_index(drop=True)
    train, validation = split_cells(metadata, options)
    values, observed, eligible, preprocessing = prepare_features(features, families, train, options)
    train, validation = train[eligible[train]], validation[eligible[validation]]
    if len(train) < max(10, options.min_cluster_size):
        raise ValueError("At least ten eligible training cells are needed; add cells or revise explicit settings")
    neural, error, checkpoint, history = learn_fingerprints(values, observed, train, validation, options, options.seed)
    reference = PCA(n_components=checkpoint["latent"], svd_solver="full").fit(values[train])
    linear = reference.transform(values)
    labels, membership = _clusters(neural, eligible, options)
    baseline, baseline_membership = _clusters(linear, eligible, options)
    stability = []
    for repeat in range(1, options.repeats):
        seed = options.seed + repeat
        repeated, _, _, _ = learn_fingerprints(values, observed, train, validation, options, seed)
        other, _ = _clusters(repeated, eligible, options)
        stability.append({"seed": seed, **_agreement(labels, other)})
    assignments = metadata.copy()
    assignments["eligible"] = eligible
    assignments["observed_feature_fraction"] = observed.mean(axis=1)
    assignments["split"] = "excluded"
    assignments.loc[train, "split"] = "training"
    assignments.loc[validation, "split"] = "validation"
    assignments["cluster"] = labels
    assignments["membership_score"] = membership
    assignments["linear_cluster"] = baseline
    assignments["linear_membership_score"] = baseline_membership
    assignments["reconstruction_error"] = np.where(eligible, error, np.nan)
    embeddings = metadata[KEY].copy()
    for method, coordinates in (("neural", neural), ("linear", linear)):
        for dimension in range(coordinates.shape[1]):
            embeddings[f"{method}_{dimension + 1}"] = np.where(eligible, coordinates[:, dimension], np.nan)
    diagnostics = {}
    for method, lab, coordinates in (("neural", labels, neural), ("linear", baseline, linear)):
        assigned = lab >= 0
        count = len(set(lab[assigned]))
        silhouette = None
        if 1 < count < assigned.sum():
            silhouette = float(silhouette_score(coordinates[assigned], lab[assigned],
                                                sample_size=min(2000, int(assigned.sum())), random_state=options.seed))
        diagnostics[method] = {"clusters": count, "assigned_cells": int(assigned.sum()),
                               "unassigned_cells": int(((lab == -1) & eligible).sum()),
                               "silhouette_assigned_only": silhouette}
    # Descriptive differences, in original units and against the cohort median.
    # These are not independent significance tests of discovered groups.
    profile_rows = []
    used = features[preprocessing["columns"]].replace([np.inf, -np.inf], np.nan)
    for label in sorted(set(labels[labels >= 0])):
        for i, column in enumerate(used):
            group = used.loc[labels == label, column].dropna()
            cohort = used.loc[eligible, column].dropna()
            difference = (group.median() - cohort.median()) / preprocessing["scale"][i]
            profile_rows.append({"cluster": int(label), "feature": column,
                                 "family": families[column], "observed_cells": len(group),
                                 "median": group.median(), "cohort_median": cohort.median(),
                                 "scaled_median_difference": difference})
    profiles = pd.DataFrame(profile_rows, columns=["cluster", "feature", "family", "observed_cells",
                                                  "median", "cohort_median", "scaled_median_difference"])
    report = {"cells": len(metadata), "eligible_cells": int(eligible.sum()),
              "training_cells": len(train), "validation_cells": len(validation),
              "training_groups": sorted(metadata.iloc[train][options.split_by].unique().tolist()),
              "validation_groups": sorted(metadata.iloc[validation][options.split_by].unique().tolist()),
              "features_used": values.shape[1], "diagnostics": diagnostics,
              "neural_linear_agreement": _agreement(labels, baseline),
              "neural_seed_sensitivity": stability,
              "validation_scope": "held-out-group reconstruction only; clustering uses the entire eligible cohort",
              "biological_validation": "not established",
              "limitations": [
                  "Cluster numbers are descriptive groups, not validated cell types.",
                  "Membership scores are not calibrated probabilities of a biological category.",
                  "Seed sensitivity is not validation in independent animals.",
                  "The linear reference is not an automatic model-selection rule.",
                  "Subject metadata must name the animal; defaults may name each movie separately.",
                  "Calibration, acquisition settings and recording lengths must be comparable.",
              ]}
    if not len(validation):
        report["limitations"].append("No independent group was held out; training reconstruction only.")
    return {"assignments": assignments, "embeddings": embeddings, "profiles": profiles,
            "preprocessing": preprocessing, "checkpoint": checkpoint, "history": history,
            "report": report, "linear_model": {"components": reference.components_, "mean": reference.mean_}}


def run_clustering(run, out, options=None):
    """Write a new clustering result directory; accepted run files stay immutable."""
    import sklearn
    import torch

    options = options or ClusteringOptions()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new clustering output")
    tables, source = load_tables(Path(run))
    metadata, features, families, audit, rhythms = build_features(tables)
    audit.extend(source["table_audit"])
    fitted = fit_cells(metadata, features, families, options)
    for row in audit:
        if row.get("reason") == "candidate":
            row["reason"] = fitted["preprocessing"]["excluded"].get(row["feature"], "used")
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "options": asdict(options), **source, **fitted["report"],
                "saved_rhythm_provenance": rhythms,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__},
                "implementation": {p.name: _hash(p) for p in
                                   (Path(__file__), Path(__file__).with_name("fingerprint.py"))},
                "cluster_codes": {"-2": "insufficient measurements", "-1": "unassigned", "0 and above": "cohort group"}}
    out.mkdir(parents=True, exist_ok=False)
    for name in ("assignments", "embeddings", "profiles"):
        fitted[name].to_csv(out / f"{name}.csv", index=False)
    pd.concat([metadata, features], axis=1).to_csv(out / "features.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "feature_audit.csv", index=False)
    pd.DataFrame(fitted["history"]).to_csv(out / "training.csv", index=False)
    composition = fitted["assignments"].groupby(["cluster", "subject", "stem", "condition"], dropna=False).size().rename("cells").reset_index()
    composition.to_csv(out / "composition.csv", index=False)
    for name, content in (("manifest", manifest), ("preprocessing", fitted["preprocessing"])):
        write_document(out / f"{name}.json", content)
    torch.save(fitted["checkpoint"], out / "neural_model.pt")
    np.savez(out / "linear_model.npz", **fitted["linear_model"])
    (out / "README.md").write_text(
        "# Cell groups from combined measurements\n\n"
        "assignments.csv identifies every cell by movie and identity. Cluster -1 is unassigned; "
        "-2 lacks enough measurements. Membership is a clustering score, not a probability of cell type.\n\n"
        "profiles.csv describes the measurements separating groups; composition.csv shows their movies, "
        "subjects and conditions. These are descriptive, not hypothesis tests. features.csv contains candidate "
        "measurements; feature_audit.csv records exclusions.\n\n"
        "The neural model compresses whole-recording measurements. The linear model is principal component "
        "analysis using the same training cells and preprocessing. Both group the entire eligible cohort "
        "with hierarchical density clustering. No rhythm is fitted here.\n\n"
        "manifest.json records the input fingerprints, settings, versions, seed sensitivity and limits. "
        "Held-out-group reconstruction does not establish reproducible biological groups. The saved neural "
        "weights and preprocessing can encode new cells; assigning new cells requires a separately validated classifier.\n",
        encoding="utf-8")
    register_outputs(out, stage="cluster", inputs=source["inputs"], settings=asdict(options))
    return manifest


def command(args):
    try:
        settings = json.loads(Path(args.options).read_text(encoding="utf-8")) if args.options else {}
        report = run_clustering(args.run, args.out, ClusteringOptions(**settings))
    except (ValueError, OSError, TypeError, ImportError) as error:
        print(f"Cell clustering: {error}")
        return 1
    print(f"Cell clustering: {report['eligible_cells']} eligible cells, {report['features_used']} measurements, "
          f"{report['diagnostics']['neural']['clusters']} neural groups; biological validation not established.")
    print(f"Saved to {Path(args.out).resolve()}")
    return 0
