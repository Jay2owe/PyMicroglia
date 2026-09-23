"""Frame-level measurements for a shared, time-blind cell-state vocabulary."""

from __future__ import annotations

import json
from pathlib import Path
from .._results import pooled_paths

import numpy as np
import pandas as pd

from pymicroglia.clustering.cohort import _hash, _registry, SUPPORT

CELL = ["stem", "identity"]
FRAME = [*CELL, "frame_index"]
META = [*FRAME, "hours", "subject", "condition"]
TABLE_AXES = {"cell_frame": (), "channels": ("channel",),
              "cell_objects": ("object_set",), "sholl": ("scaling", "ring"),
              "branches": ("branch",)}
SNAPSHOT_MODULES = {"morphology", "intensity", "channels", "neighbours", "sholl", "object_geometry"}
CHANGE_MODULES = {"motility", "surveillance", "motion_evidence", "walk", "territory_shape", "territory"}
HISTORY_NORMALISED = {"dff", "channel_dff", "channel_detrended", "channel_detrended_mean"}


def _read_manifest(path, audit):
    # Legacy measurement manifests encode unavailable summaries as NaN. Keep
    # them unavailable in strict JSON while retaining the original file hash.
    constants = {}

    def unavailable(token):
        constants[token] = constants.get(token, 0) + 1
        return None

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=unavailable)
    if constants:
        audit.append({"table": "manifest", "path": str(path),
                      "reason": "nonfinite_provenance_replaced_with_null", "constants": constants})
    return value


def read_inputs(run):
    root = Path(run).resolve()
    pooled = root / "pooled" if (root / "pooled").is_dir() else root
    manifest_path, parent_path, table_folder = pooled_paths(run)
    if not manifest_path.is_file():
        raise ValueError("State analysis needs saved pooled tables and their manifest")
    audit = []
    pool = _read_manifest(manifest_path, audit)
    if manifest_path == parent_path:
        pool = pool.get("pooled") or {}
    source_paths = [manifest_path]

    parent = _read_manifest(parent_path, audit) if parent_path.is_file() else {}
    if parent_path.is_file():
        source_paths.append(parent_path)
    movies = [m for m in parent.get("movies", []) if isinstance(m, dict)]
    for setting in ("minutes_per_frame", "microns_per_pixel"):
        values = [m.get("provenance", {}).get("scale", {}).get(setting) for m in movies]
        if len({json.dumps(v) for v in values}) > 1:
            raise ValueError(f"Different {setting} across movies; frame measurements need comparable calibration")
    tables = {}
    for name in TABLE_AXES:
        path = table_folder / f"{name}.csv"
        if not path.is_file():
            continue
        source_paths.append(path)
        availability = pool.get("tables", {}).get(name, {})
        if availability.get("movies_absent"):
            if name == "cell_frame":
                raise ValueError("The pooled manifest reports movies without cell-frame measurements")
            audit.append({"table": name, "column": "*", "reason": "not_measured_in_every_movie"})
            continue
        table = pd.read_csv(path, dtype={"stem": str, "subject": str, "condition": str})
        absent = set().union(*map(set, availability.get("columns_missing", {}).values()))
        if absent.intersection([*META, *TABLE_AXES[name]]):
            raise ValueError(f"{name}: identity, time or measurement axes missing in some movies")
        table = table.drop(columns=sorted(absent), errors="ignore")
        audit.extend({"table": name, "column": c, "reason": "not_measured_in_every_movie"} for c in sorted(absent))
        tables[name] = table
    return tables, {"inputs": [{"path": str(p), "sha256": _hash(p)} for p in source_paths],
                    "pooled_manifest": pool, "measurement_runs": movies, "input_audit": audit}


def frame_features(tables):
    if "cell_frame" not in tables:
        raise ValueError("State analysis requires cell_frame.csv, not whole-recording cell summaries")
    base = tables["cell_frame"].copy()
    if any(c not in base for c in META):
        raise ValueError(f"cell_frame needs {META}")
    if base[META].isna().any().any() or base.duplicated(FRAME).any():
        raise ValueError("Every cell-frame needs a unique key, time, subject and condition")
    for name in ("hours", "frame_index", "identity"):
        values = pd.to_numeric(base[name], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (name != "hours" and (values != np.floor(values)).any()):
            raise ValueError(f"Invalid numeric {name}")
        base[name] = values if name == "hours" else values.astype(np.int64)
    if any(base[name].astype(str).str.strip().eq("").any() for name in ("stem", "subject", "condition")):
        raise ValueError("Movie, subject and condition metadata cannot be blank")
    base = base.sort_values(FRAME, kind="stable").reset_index(drop=True)
    for _, group in base.groupby(CELL, sort=False):
        if (np.diff(group.hours.to_numpy(float)) <= 0).any():
            raise ValueError("Recorded hours must increase with frame_index within each cell")
        if any(group[c].nunique() != 1 for c in ("subject", "condition")):
            raise ValueError("A cell cannot change subject or condition metadata during a recording")
    if base.groupby("stem")["subject"].nunique().gt(1).any():
        raise ValueError("Every movie must identify a single subject")
    meta = base[META].copy()
    index = pd.MultiIndex.from_frame(meta[FRAME])
    declared = _registry()
    snapshot, changes, families, rows = {}, {}, {}, []
    for name, axes in TABLE_AXES.items():
        if name not in tables:
            continue
        table = base if name == "cell_frame" else tables[name].copy()
        keys = [*FRAME, *axes]
        if any(c not in table for c in keys) or table[keys].isna().any().any() or table.duplicated(keys).any():
            raise ValueError(f"{name} needs unique, complete keys {keys}")
        table_index = pd.MultiIndex.from_frame(table[FRAME])
        if not table_index.isin(index).all():
            raise ValueError(f"{name} refers to cell-frames absent from cell_frame")
        for column in ("subject", "condition", "hours"):
            if column in table:
                expected = meta.set_index(FRAME)[column].reindex(table_index).to_numpy()
                if not np.array_equal(table[column].to_numpy(), expected):
                    raise ValueError(f"{name} has inconsistent {column} metadata")
        for column in table:
            if column in META or column in axes:
                continue
            declaration = declared.get(column)
            reason = None
            family, spec = declaration if declaration else ("unknown", None)
            if declaration is None:
                reason = "not_a_registered_measurement"
            elif SUPPORT.search(column) or spec.role in {"reference", "invalid", "inferred", "unclaimed", "significant"}:
                reason = "identity_timing_or_quality"
            elif family not in SNAPSHOT_MODULES | CHANGE_MODULES:
                reason = "not_a_snapshot_or_change_measurement"
            numeric = pd.to_numeric(table[column], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            if reason is None and not numeric.notna().any():
                reason = "no_numeric_values"
            if reason:
                rows.append({"table": name, "column": column, "reason": reason})
                continue
            kind = "snapshot" if family in SNAPSHOT_MODULES and column not in HISTORY_NORMALISED and not column.endswith("_scale_global") else "change"
            table[column] = numeric
            dimensions = [a for a in axes if a != "branch"]
            parts = table.groupby(dimensions, sort=True) if dimensions else [((), table)]
            for values, part in parts:
                values = values if isinstance(values, tuple) else (values,)
                qualifier = json.dumps(dict(zip(dimensions, values)), sort_keys=True, default=str)
                part_kind = "change" if name == "sholl" and dict(zip(dimensions, values)).get("scaling") == "global" else kind
                if name == "branches":
                    grouped = part.groupby(FRAME)[column]
                    summaries = {"median": grouped.median(), "iqr": grouped.quantile(.75) - grouped.quantile(.25)}
                else:
                    summaries = {"value": part.set_index(FRAME)[column]}
                for statistic, series in summaries.items():
                    feature = f"{name}|{qualifier}|{column}|{statistic}"
                    target = snapshot if part_kind == "snapshot" else changes
                    target[feature] = series.reindex(index).to_numpy(float)
                    families[feature] = family
                    rows.append({"feature": feature, "table": name, "column": column, "kind": part_kind,
                                 "family": family, "label": spec.label, "statistic": statistic, "reason": "candidate"})
    if len(snapshot) < 2:
        raise ValueError("At least two registered snapshot measurements are needed")
    return meta, pd.DataFrame(snapshot), pd.DataFrame(changes, index=meta.index), families, pd.DataFrame(rows)
