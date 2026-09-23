"""Run the measurement for one or more movies into an immutable run folder.

A run is written once, never edited, and carries a manifest that fingerprints
both what went in and what came out. Re-running with the same inputs and the
same settings must produce the same numbers, and the manifest is how that is
checked rather than assumed.

Ported from Motion's ``analysis/run.py`` on 2026-09-21. The per-movie
orchestration -- which module runs, what folds into which roll-up, which
side columns a derived module may see -- is unchanged. What moved is where
things land::

    <output_dir>/<run>/
      measure/<stem>/   cell_frame.csv cell_summary.csv frame_summary.csv <module tables>.csv
      tracker/<stem>/   the tracker's decision tables copied through
      stacks/<stem>/    per-pixel images a module measured
      windows/<stem>/   the roll-ups again, per declared window
      pooled/           every movie's tables concatenated (``pool``)
      .auto-organotypic/  manifest.json conditions.json

Kind first, then recording, and one ledger per folder: every table goes
through :func:`pymicroglia.store.put`, which writes the folder's
``artefacts.json`` -- this package never writes a record beside a table
itself. The manifest names files and never contains one; Motion's
``manifest.json``, ``conditions.json``, ``theme.json``, ``figures.json``,
``movie_summary.json`` and ``pooled/manifest.json`` are its sections.
"""

from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from auto_organotypic import io as _io
from auto_organotypic import layout as _layout

from .. import __version__, store
from ..pipelines import append_runs_index, run_folder, slug
from ..tracking.contract import sha256_of
from .context import MeasurementContext
from .declare import (DERIVATIONS, MEASUREMENTS, Output, declared_tables,
                      get_derived, get_module, list_derived, list_modules)
from .inputs import load_movie
from .spec import MeasureConfig, MovieSpec
from .summarise import (ROLLUPS, build_cell_summary, build_frame_summary,
                        build_movie_summary, join_side, side_column_names)
from .windows import WINDOWED, window_extent, windowed_summaries

__all__ = ["METHOD_VERSION", "STAGE", "MEASURE_FOLDER", "TRACKER_FOLDER",
           "STACKS_FOLDER", "WINDOWS_FOLDER", "POOLED_FOLDER", "MANIFEST",
           "CONDITIONS", "table_values", "write_table", "stamp", "folder_for",
           "read_manifest", "write_manifest", "write_windows", "analyse_movie",
           "run", "check_module_options"]

#: Bumped when the arithmetic of the chassis changes. A table stored under an
#: older version is not reused for a newer one.
METHOD_VERSION = "2026-09-21-measure-chassis-v1"
#: The store stage every table this package writes is keyed under.
STAGE = "measure"

MEASURE_FOLDER = "measure"
TRACKER_FOLDER = "tracker"
STACKS_FOLDER = "stacks"
WINDOWS_FOLDER = "windows"
POOLED_FOLDER = "pooled"
MANIFEST = "manifest.json"
CONDITIONS = "conditions.json"

_ROLLUP_BY_GRAIN = {output.grain: output.name for output in ROLLUPS}
_ROLLUP_BY_NAME = {output.name: output for output in ROLLUPS}
_WINDOWED_BY_NAME = {output.name: output for output in WINDOWED}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- writing
def _nine(value: float) -> float | None:
    """One float at nine significant figures, or ``None`` for a blank.

    Nine significant figures, as Motion wrote. A mean of 16-bit pixel values
    printed to seventeen digits records the arithmetic rather than the
    measurement; nine holds every value to 5e-9 of itself, which no downstream
    derivation can amplify into anything drawable. Rounded here, before the
    store writes the file, so the precision of the file is this package's
    decision and the ledger is the store's.
    """
    if value != value:                       # NaN
        return None
    if value in (float("inf"), float("-inf")):
        return value
    return float(f"{value:.9g}")


def table_values(table: pd.DataFrame) -> dict[str, list[Any]]:
    """A table as columns of plain Python values, ready for the store.

    Floats are rounded to nine significant figures and a missing value is
    ``None``; integers, booleans and text pass through as themselves.
    """
    out: dict[str, list[Any]] = {}
    for column in table.columns:
        series = table[column]
        kind = series.dtype.kind
        if kind == "f":
            out[str(column)] = [_nine(v) for v in series.to_numpy(dtype=float).tolist()]
        elif kind in "iu":
            # A nullable integer column (pandas ``Int64``) reports kind "i"
            # and holds ``pd.NA`` where a value is missing; Motion wrote those
            # as blanks, and so does this.
            out[str(column)] = [None if v is pd.NA else int(v) for v in series.tolist()]
        elif kind == "b":
            out[str(column)] = [None if v is pd.NA else bool(v) for v in series.tolist()]
        else:
            values = []
            for v in series.tolist():
                if v is None or v is pd.NA or (isinstance(v, float) and v != v):
                    values.append(None)
                elif isinstance(v, (np.floating, float)):
                    values.append(_nine(float(v)))
                elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
                    values.append(int(v))
                elif isinstance(v, (np.bool_, bool)):
                    values.append(bool(v))
                else:
                    values.append(str(v))
            out[str(column)] = values
    return out


def stamp(table: pd.DataFrame, movie: MovieSpec, condition: str) -> pd.DataFrame:
    """Put who and what at the front of every table.

    Pooling ten movies then becomes a concatenation rather than a join against
    the configuration. ``subject`` defaults to the stem: two movies are
    separate subjects unless the configuration says they are the same animal.
    """
    stamped = table.copy()
    for column, value in (
        ("subject", movie.subject or movie.stem),
        ("condition", condition),
        ("stem", movie.stem),
    ):
        if column in stamped.columns:
            stamped = stamped.drop(columns=[column])
        stamped.insert(0, column, value)
    return stamped


def folder_for(output: Output | None) -> str:
    """Which kind folder a table lands in, by its declared origin."""
    return TRACKER_FOLDER if output is not None and output.origin == "tracker" else MEASURE_FOLDER


def write_table(table: pd.DataFrame, *, name: str, folder: Path, source,
                params: Mapping[str, Any], output: Output | None = None,
                upstream: Sequence[str] = ()) -> dict[str, Any]:
    """Store one table in ``folder`` and describe what was written.

    The store writes the CSV and the folder's ledger; this records the grain
    and the origin as well as the fingerprint, so a run folder says what one
    row of each of its tables is without anyone having to open the package
    that made it.
    """
    extra: dict[str, Any] = {"table": name}
    if output is not None:
        extra["grain"] = list(output.grain)
        extra["origin"] = output.origin
    stored = store.put(STAGE, source, dict(params), kind="table",
                       value=table_values(table), name=name, output_dir=folder,
                       method_version=METHOD_VERSION, upstream=upstream,
                       extra=extra)
    record = {
        "path": stored.path.name,
        "folder": str(Path(folder).name),
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "sha256": sha256_of(stored.path),
        "digest": stored.digest,
    }
    if output is not None:
        record["grain"] = list(output.grain)
        record["origin"] = output.origin
    return record


def _store_stack(stack: np.ndarray, *, name: str, folder: Path, source,
                 params: Mapping[str, Any]) -> dict[str, Any]:
    """One per-pixel image a module measured, in the store's array form."""
    array = np.asarray(stack)
    if array.dtype == bool:
        kind = "mask"
    elif array.dtype.kind in "iu":
        kind = "labels"
    else:
        kind = "array"
    stored = store.put(STAGE, source, dict(params), kind=kind, value=array,
                       name=name, output_dir=folder, method_version=METHOD_VERSION,
                       extra={"stack": name})
    return {
        "path": stored.path.name,
        "folder": str(Path(folder).name),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_of(stored.path),
        "digest": stored.digest,
    }


# --------------------------------------------------------------- documents
def manifest_path(run: Path) -> Path:
    return _layout.workings_path(Path(run)) / MANIFEST


def read_manifest(run: Path) -> dict[str, Any]:
    """The run's manifest, or a refusal naming the folder."""
    found = _io.read_json(Path(run) / MANIFEST, default=None)
    if not isinstance(found, dict):
        raise FileNotFoundError(
            f"{run} holds no {MANIFEST} in its workings; it is not a run this "
            "package wrote, or the run did not finish")
    return found


def write_manifest(run: Path, payload: Mapping[str, Any]) -> Path:
    return _io.write_json(Path(run) / MANIFEST, dict(payload), workings=True)


# ------------------------------------------------------------- the join
def _fold_target(name: str, declared: dict[str, Output]) -> str | None:
    """The roll-up this table's columns belong in, or ``None`` if it is a file."""
    output = declared.get(name)
    if output is None or not output.fold:
        return None
    return _ROLLUP_BY_GRAIN.get(output.grain)


def _join_cell_frame(tables: dict[str, pd.DataFrame], context: MeasurementContext) -> pd.DataFrame:
    declared = declared_tables()
    keys = ["identity", "frame_index"]
    joined: pd.DataFrame | None = None
    for name in sorted(tables):
        if _fold_target(name, declared) != "cell_frame":
            continue
        table = tables[name]
        if table.empty:
            continue
        if joined is None:
            joined = table.copy()
            continue
        overlap = [c for c in table.columns if c in joined.columns and c not in keys]
        joined = joined.merge(table.drop(columns=overlap), on=keys, how="outer")

    if joined is None:
        joined = pd.DataFrame(columns=keys)

    frames = context.frame_table()
    shared = [c for c in (*frames.columns, "stem") if c != "frame_index" and c in joined.columns]
    joined = joined.drop(columns=shared)
    joined = joined.merge(frames, on="frame_index", how="left")
    joined.insert(0, "stem", context.stem)
    ordered = ["stem", "identity", "frame_index", "imagej_frame", "source_imagej_frame", "hours"]
    rest = [c for c in joined.columns if c not in ordered]
    return joined[ordered + rest].sort_values(["identity", "frame_index"]).reset_index(drop=True)


def _fold_derived(cell_frame: pd.DataFrame, tables: dict[str, pd.DataFrame],
                  names: set[str]) -> pd.DataFrame:
    """Merge a derived module's cell-frame-grain columns into the joined table."""
    declared = declared_tables()
    keys = ["identity", "frame_index"]
    for name in sorted(names):
        if _fold_target(name, declared) != "cell_frame":
            continue
        table = tables.get(name)
        if table is None or table.empty or any(key not in table.columns for key in keys):
            continue
        extra = [c for c in table.columns if c not in cell_frame.columns]
        if not extra:
            continue
        cell_frame = cell_frame.merge(table[keys + extra], on=keys, how="left")
    return cell_frame


def _granted_side_columns(name: str, context: MeasurementContext,
                          supplied: set[str]) -> set[str]:
    """The side columns one derived module is allowed to read."""
    asked = context.module_params(name).get("side_columns")
    if not asked:
        return set()
    if not isinstance(asked, (list, tuple)):
        raise TypeError(
            f"{name}: side_columns must be a list of column names, not "
            f"{type(asked).__name__}"
        )
    granted = {str(column) for column in asked}
    unknown = sorted(granted - supplied)
    if unknown:
        offered = sorted(supplied)
        tables = sorted(context.side)
        raise ValueError(
            f"{name} is granted the side column(s) {unknown}, which no side "
            f"table supplied. This movie declared the side table(s) {tables}, "
            f"between them providing {offered}. A side column is written under "
            "its table's name, so a `dose` column in a table called `schedule` "
            "is `schedule_dose`."
        )
    return granted


def _side_columns_used(granted: set[str], produced: dict[str, pd.DataFrame]) -> set[str]:
    """Which of the granted side columns actually reached the module's output."""
    if not granted:
        return set()
    used: set[str] = set()
    for table in produced.values():
        if table is None or table.empty:
            continue
        used |= granted & set(table.columns)
        for column in table.columns:
            if table[column].dtype == object:
                used |= granted & set(table[column].dropna().astype(str).unique())
    return used


# -------------------------------------------------------------- options
def check_module_options(module_options: Mapping[str, Any] | None) -> dict[str, dict]:
    """Refuse an option no module declares, before anything runs.

    Each module states its settings in ``defaults``; a derived module may also
    be granted ``side_columns``. An option outside that is a typo, and a typo
    that ran would produce exactly the result the reader wanted with the
    setting they asked for quietly ignored.
    """
    if not module_options:
        return {}
    if not isinstance(module_options, Mapping):
        raise TypeError("module_options must be a mapping of module name -> "
                        f"{{option: value}}, not {type(module_options).__name__}")
    checked: dict[str, dict] = {}
    for name, options in module_options.items():
        module = MEASUREMENTS.get(name) or DERIVATIONS.get(name)
        if module is None:
            known = sorted({*MEASUREMENTS, *DERIVATIONS}) or ["none registered"]
            raise ValueError(f"module_options names {name!r}, which is not a "
                             f"registered module; registered: {', '.join(known)}")
        if not isinstance(options, Mapping):
            raise TypeError(f"module_options[{name!r}] must be a mapping of "
                            f"option -> value, not {type(options).__name__}")
        allowed = set(module.defaults)
        if name in DERIVATIONS:
            allowed.add("side_columns")
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(
                f"{name} does not declare {unknown}; its settings are "
                f"{sorted(allowed) or 'none'}")
        checked[name] = dict(options)
    return checked


# --------------------------------------------------------------- one movie
def write_windows(run: Path, movie: MovieSpec, context: MeasurementContext,
                  cell_frame: pd.DataFrame, windows, condition: str, *,
                  source, params: Mapping[str, Any]) -> dict[str, dict]:
    """The roll-ups again, per declared window, into ``windows/<stem>/``."""
    written: dict[str, dict] = {}
    folder = run / WINDOWS_FOLDER / movie.stem
    for name, table in sorted(windowed_summaries(cell_frame, context, windows).items()):
        written[name] = write_table(
            stamp(table, movie, condition), name=name, folder=folder, source=source,
            params={**params, "table": name}, output=_WINDOWED_BY_NAME[name])
    return written


def analyse_movie(config: MeasureConfig, movie: MovieSpec, run: Path, *,
                  run_label: str, modules: list[str] | None = None) -> dict:
    started = time.perf_counter()
    context, provenance = load_movie(config, movie)
    source = store.fingerprint(movie.labels)
    key_params = {"run": run_label, "stem": movie.stem}

    requested = list(modules or config.enabled_modules or [m.name for m in list_modules()])
    measurement_names = [m.name for m in list_modules()]
    derived_names = [m.name for m in list_derived()]
    unregistered = [name for name in requested
                    if name not in measurement_names and name not in derived_names]

    tables: dict[str, pd.DataFrame] = {}
    stacks: dict[str, np.ndarray] = {}
    module_records: list[dict] = []

    for name in requested:
        if name not in measurement_names:
            continue
        module = get_module(name)
        available, reason = module.available(context)
        if not available:
            module_records.append({"module": name, "status": "skipped", "reason": reason})
            continue
        module_started = time.perf_counter()
        produced = module.measure(context)
        table_outputs = {
            key: value for key, value in produced.items() if isinstance(value, pd.DataFrame)
        }
        stack_outputs = {
            key: np.asarray(value) for key, value in produced.items()
            if isinstance(value, np.ndarray)
        }
        unknown = set(produced) - set(table_outputs) - set(stack_outputs)
        if unknown:
            raise TypeError(
                f"{name} returned unsupported output(s) {sorted(unknown)}; "
                "modules may return pandas tables or numpy image stacks"
            )
        tables.update(table_outputs)
        stacks.update(stack_outputs)
        module_records.append(
            {
                "module": name,
                "status": "done",
                "seconds": round(time.perf_counter() - module_started, 3),
                "tables": {key: int(len(value)) for key, value in table_outputs.items()},
                "stacks": {key: list(value.shape) for key, value in stack_outputs.items()},
                "parameters": {**module.defaults, **context.module_params(name)},
                "method_version": module.method_version,
            }
        )

    cell_frame = _join_cell_frame(tables, context)

    # Side tables before the derived modules, so a user's own series can be
    # something the package computes with. A derived module is handed a table
    # with every side column removed except the ones named for it.
    cell_frame = join_side(cell_frame, context, "frame_index")
    cell_frame = join_side(cell_frame, context, "identity")
    supplied = side_column_names(context) & set(cell_frame.columns)

    side_use: dict[str, dict] = {}
    derived_produced: set[str] = set()
    for name in requested:
        if name not in derived_names:
            continue
        module = get_derived(name)
        # Fold what the derived modules before this one wrote, so a derived
        # module can read a column an earlier one added.
        cell_frame = _fold_derived(cell_frame, tables, derived_produced)

        granted = _granted_side_columns(name, context, supplied)
        hidden = sorted(supplied - granted)
        visible = cell_frame.drop(columns=hidden) if hidden else cell_frame

        missing = [c for c in module.needs_columns if c not in visible.columns]
        if missing:
            module_records.append(
                {"module": name, "status": "skipped", "reason": f"cell_frame is missing {missing}"}
            )
            continue

        module_started = time.perf_counter()
        produced = module.derive(visible, context)
        tables.update(produced)
        derived_produced.update(produced)
        side_use[name] = {
            "offered": sorted(granted),
            "used": sorted(_side_columns_used(granted, produced)),
        }
        module_records.append(
            {
                "module": name,
                "status": "done",
                "seconds": round(time.perf_counter() - module_started, 3),
                "tables": {key: int(len(value)) for key, value in produced.items()},
                "parameters": module.parameters(context),
                "method_version": module.method_version,
            }
        )

    cell_frame = _fold_derived(cell_frame, tables, derived_produced)

    # The user's own columns keep their place at the end of the table.
    if supplied:
        cell_frame = cell_frame[[c for c in cell_frame.columns if c not in supplied]
                                + [c for c in cell_frame.columns if c in supplied]]

    cell_summary = build_cell_summary(cell_frame, tables, context)
    frame_summary = build_frame_summary(cell_frame, context, tables)
    movie_summary = build_movie_summary(cell_frame, cell_summary, frame_summary, tables, context)

    assignment = config.assignment(movie)
    movie_summary["condition"] = assignment.condition
    movie_summary["condition_label"] = assignment.label
    movie_summary["subject"] = movie.subject or movie.stem

    declared = declared_tables()
    measure_dir = run / MEASURE_FOLDER / movie.stem
    written = {
        name: write_table(stamp(table, movie, assignment.condition), name=name,
                          folder=measure_dir, source=source,
                          params={**key_params, "table": name},
                          output=_ROLLUP_BY_NAME[name])
        for name, table in (("cell_frame", cell_frame),
                            ("cell_summary", cell_summary),
                            ("frame_summary", frame_summary))
    }

    windows = config.windows_for(movie)
    written.update(write_windows(run, movie, context, cell_frame, windows,
                                 assignment.condition, source=source, params=key_params))

    for name, table in sorted(tables.items()):
        # A folded table is already inside a roll-up; writing it again would be
        # two files holding one set of numbers.
        if _fold_target(name, declared) is not None or table is None or table.empty:
            continue
        output = declared.get(name)
        written[name] = write_table(
            stamp(table, movie, assignment.condition), name=name,
            folder=run / folder_for(output) / movie.stem, source=source,
            params={**key_params, "table": name}, output=output)

    written_stacks: dict[str, dict] = {}
    for name, stack in sorted(stacks.items()):
        written_stacks[name] = _store_stack(
            stack, name=name, folder=run / STACKS_FOLDER / movie.stem,
            source=source, params={**key_params, "stack": name})

    return {
        "stem": movie.stem,
        "spec": movie.as_dict(),
        "condition": assignment.as_dict(),
        "subject": movie.subject or movie.stem,
        "provenance": provenance,
        "source": source.as_dict(),
        "modules": module_records,
        "modules_unregistered": unregistered,
        "side_column_use": side_use,
        "windows": [
            {"name": w.name, "baseline": w.baseline, "description": w.description,
             "frames": window_extent(w, context)[0],
             "hours": window_extent(w, context)[1]}
            for w in windows
        ],
        "tables": written,
        "stacks": written_stacks,
        "summary": movie_summary,
        "seconds": round(time.perf_counter() - started, 3),
    }


# ------------------------------------------------------------------ the run
def _registered_modules() -> list[dict]:
    return [
        {"name": m.name, "kind": "measurement", "description": m.description,
         "requires": list(m.requires), "method_version": m.method_version}
        for m in list_modules()
    ] + [
        {"name": m.name, "kind": "derived", "description": m.description,
         "method_version": m.method_version}
        for m in list_derived()
    ]


def run(config: MeasureConfig, output_dir, *, run_label: str | None = None,
        if_exists: str = "version", stems: Sequence[str] | None = None,
        modules: Sequence[str] | None = None, claim: str = "") -> dict:
    """Measure every configured movie into ``<output_dir>/<run>/``.

    The run label is a slug of the settings when none is given, so a repeat
    run finds its own previous output and ``if_exists`` has something to act
    on -- ``version`` keeps both, ``error`` refuses, ``overwrite`` replaces,
    ``skip`` hands back the stored manifest.
    """
    from . import modules as _modules

    _modules.load()                     # importing a module is what registers it

    output_dir = Path(output_dir)
    selected = [m for m in config.movies if not stems or m.stem in stems]
    if not selected:
        raise ValueError(f"no configured movies match {stems}")

    settings = config.settings()
    settings["recording_windows"] = {name: windows for name, windows in settings["recording_windows"].items()
                                     if name in {m.stem for m in selected}}
    label = run_label or slug("measure", {
        **settings, "movies": [m.as_dict() for m in selected],
        "modules": list(modules or []), "method_version": METHOD_VERSION})
    folder = run_folder(output_dir.parent, output_dir.name, label, if_exists)
    run_dir = Path(folder.path)
    if folder.reuse:
        return read_manifest(run_dir)
    # Empty directory shells survive a delete on a Dropbox-backed folder for a
    # while, so the guard asks whether any *file* is present.
    if run_dir.exists() and any(p.is_file() for p in run_dir.rglob("*")):
        raise FileExistsError(
            f"{run_dir} already holds results; analysis runs are immutable, "
            "choose a new run name"
        )

    check_module_options(config.module_params)

    # A movie in the wrong group is the one error no figure will reveal, so it
    # stops the run rather than being measured and sorted out later.
    assignments = [config.assignment(movie) for movie in selected]
    broken = [f"{a.stem}: {a.problem}" for a in assignments if a.problem is not None]
    if broken:
        raise ValueError(
            "the experimental design cannot be resolved:\n  " + "\n  ".join(broken))

    run_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    clock = time.perf_counter()
    results = [analyse_movie(config, movie, run_dir, run_label=folder.label,
                             modules=list(modules) if modules else None)
               for movie in selected]

    design = {
        **config.conditions.as_dict(),
        "assignments": [a.as_dict() for a in assignments],
        "subjects": {m.stem: (m.subject or m.stem) for m in selected},
        "warnings": config.conditions.audit(assignments),
    }
    _io.write_json(run_dir / CONDITIONS, design, workings=True)

    manifest = {
        "run": {
            "pipeline": "measure",
            "run_label": folder.label,
            "folder": str(run_dir),
            "claim": claim,
            "method_version": METHOD_VERSION,
            "pymicroglia_version": __version__,
            "generated_at_utc": started,
            "finished_at_utc": _now(),
            "elapsed_seconds": round(time.perf_counter() - clock, 3),
            "config_path": str(config.source_path) if config.source_path else None,
            "config_sha256": (sha256_of(config.source_path)
                              if config.source_path and Path(config.source_path).is_file()
                              else None),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        "settings": settings,
        "theme": dict(config.theme),
        "figures": dict(config.figures),
        "design": design,
        "registered_modules": _registered_modules(),
        "movies": results,
        "pooled": None,
        "statistics": None,
    }
    write_manifest(run_dir, manifest)
    append_runs_index(output_dir.parent, output_dir.name, {
        "run_label": folder.label, "pipeline": "measure", "folder": str(run_dir),
        "movies": [m.stem for m in selected], "dataset": config.dataset,
        "finished_at_utc": manifest["run"]["finished_at_utc"], "claim": claim,
    })
    return manifest
