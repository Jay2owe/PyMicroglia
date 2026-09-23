"""Resolve workflow requests against existing, unmodified measurement tables."""
from __future__ import annotations
from pymicroglia._results import read_document

import importlib
import json
from pathlib import Path

FAMILIES = {
    "rhythm_discovery": ("rhythm-discovery", "rhythm.discovery", "RhythmDiscoveryRequest", "rhythm.discovery"),
    "method_audit": ("method-selection-audit", "audit.options", "AuditRequest", "audit.workflow"),
    "measurement_relationships": ("measurement-relationships", "relationships.options", "RelationshipRequest", "relationships.options"),
    "behaviour_states": ("cell-behaviour-states", "behaviour.options", "BehaviourRequest", "behaviour.options"),
    "spatial_coordination": ("spatial-coordination", "coordination.options", "CoordinationRequest", "coordination.options"),
    "intervention_response": ("intervention-response", "intervention.options", "InterventionRequest", "intervention.options"),
}


def parse(entries, groups=None):
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("pipelines must be a list of request objects")
    available = {row[0]: row for row in FAMILIES.values()}
    requests, names = [], set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("pipeline") not in available:
            raise ValueError(f"pipelines[{i}]: unknown pipeline")
        _, module, kind, _ = available[entry["pipeline"]]
        cls = getattr(importlib.import_module(f"pymicroglia.pipelines.{module}"), kind)
        request = cls.from_dict(entry, groups or {}, where=f"pipelines[{i}]")
        if request.name in names:
            raise ValueError(f"duplicate pipeline request: {request.name}")
        names.add(request.name)
        requests.append(request)
    return requests


def inputs(run):
    from .._results import document, read_document, measurement_identity
    from ._screening import file_hash, read_verified_tables
    from ._contracts import content_id
    root = Path(run).resolve()
    manifest_path = document(root / "manifest.json")
    manifest = read_document(manifest_path)
    pooled = root / "pooled"
    if pooled.is_dir():
        folders = [pooled / "tables" if (pooled / "tables").is_dir() else pooled]
    else:
        movies = manifest.get("movies", [])
        if len(movies) != 1:
            raise ValueError("Multiple movies require pooled measurement tables")
        stem = movies[0]["stem"]
        folders = [root / kind / stem for kind in ("measure", "tracker", "windows")]
        if not any(p.is_dir() for p in folders):
            folders = [root / stem / kind for kind in ("tables", "derived", "tracker", "windows")]
    paths = {}
    for folder in folders:
        for path in sorted(folder.glob("*.csv")):
            if path.stem in paths:
                raise ValueError(f"Ambiguous measured table: {path.stem}")
            paths[path.stem] = path
    if not paths:
        raise ValueError(f"No saved measurement tables in {root}")
    hashes = {name: file_hash(path) for name, path in paths.items()}
    tables = read_verified_tables(paths, hashes)
    inherited = [next((r.get("parameters", {}) for r in movie.get("modules", [])
                      if r["module"] == "rhythms"), {}) for movie in manifest.get("movies", [])]
    # Adding a display plan must not change the scientific input identity.
    identity = measurement_identity(manifest_path)
    return manifest, identity, paths, hashes, tables, inherited


def execute(action, run, request, *, output_dir=None, only=None, presentation=None,
            if_exists="version", run_label=None, claim=""):
    from ..registry import require_claim
    from . import run_folder
    require_claim(action, claim)
    if isinstance(request, (str, Path)):
        request = read_document(Path(request))
    if not isinstance(request, dict):
        raise ValueError("request must be a request object or a JSON file")
    pipeline, _, _, workflow = FAMILIES[action]
    declared = {"pipeline": pipeline, **request}
    if declared["pipeline"] != pipeline:
        raise ValueError(f"{action} requires pipeline {pipeline}")
    manifest, identity, paths, hashes, tables, inherited = inputs(run)
    from ..measure.metric_groups import MetricGroup
    settings = manifest.get("settings", {})
    saved_groups = manifest.get("metric_groups", settings.get("metric_groups", {}))
    groups = {name: MetricGroup(name, columns, tuple(columns))
              for name, columns in saved_groups.items()}
    parsed = parse([declared], groups)[0]
    needs_rhythms = action in {"rhythm_discovery", "method_audit"} or (
        action == "intervention_response" and parsed.rhythms["enabled"])
    if needs_rhythms and inherited and any(value != inherited[0] for value in inherited[1:]):
        raise ValueError("Movie rhythm settings differ; declare a comparable measurement run")
    module = importlib.import_module(f"pymicroglia.pipelines.{workflow}")
    extra = {}
    if action == "intervention_response":
        # Only declared windows travel across this boundary. Resolved movie
        # windows also contain whole-recording fallbacks, which are not designs.
        extra = {"recording_windows": settings.get("recording_windows"),
                 "default_windows": settings.get("windows")}
    resolved = module.resolve_request(parsed, source_run=identity, tables=tables,
        input_hashes=hashes, rhythm_params=inherited[0] if inherited else {}, **extra)
    root = Path(output_dir) if output_dir else Path(run) / "pipelines"
    target = run_folder(root, "", run_label or parsed.name, if_exists)
    return module.run_request(resolved, paths, target.path, presentation=presentation,
                              only=tuple(only) if only else None)
