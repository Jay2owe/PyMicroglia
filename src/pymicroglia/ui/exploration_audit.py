"""Selected-cell comparisons using the existing period-method audit engine.

The browser, background queue and replay script use this same entry point.
Normalisation affects displayed traces, as in the main rhythm analysis; it
does not replace the input supplied to the period estimator or its test.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .exploration_data import _observations, read_rows, recording
from .exploration_rhythms import _atomic_json, _fingerprint, rhythm_options


class AuditCancelled(RuntimeError):
    """A cancellation request reached a safe comparison boundary."""


def _folder(root, stem):
    return root / "AI_Exports" / "rhythm_audits" / hashlib.sha256(stem.encode()).hexdigest()[:16]


def audit_options(run, stem):
    from pymicroglia import workbench
    from pymicroglia.figure_tables.method_audit import DETREND_LABELS
    shared = rhythm_options(run, stem)
    root, _, movie = recording(run, stem)
    _, observations, _, _ = _observations(root, movie)
    shared.update(identities=sorted(observations),
                  detrends=[{"key": key, "label": DETREND_LABELS.get(key, row["label"])}
                            for row in shared["detrends"] for key in [row["key"]]],
                  scientific_options=workbench.scientific_options(),
                  audit_defaults={"detrend_methods": [workbench.DEFAULT_DETREND_METHOD, "none", "linear", "robust_linear"],
                                  "period_methods": ["lomb", "fft_nlls", "mesa"],
                                  "significance_method": "lomb", "input_filters": [{"method": "none"}],
                                  "trace_normalization": "none"})
    shared["input_filters"] = [{"key": key, **value} for key, value in workbench.filter_methods().items()]
    # Preserve the legacy grid's polynomial-degree comparisons as distinct rows.
    shared["detrends"].extend({"key": key, "label": DETREND_LABELS[key]} for key in ("poly3", "poly6"))
    return shared


def validate_request(run, stem, options):
    from pymicroglia import workbench
    from pymicroglia.measure.modules.rhythms import DEFAULTS
    from .exploration_previews import _module_settings
    if not isinstance(options, dict):
        raise ValueError("Audit settings must be an object.")
    allowed = {*workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS, "identity", "metric",
               "detrend_methods", "period_methods", "median_window_points", "input_filters",
               "trace_normalization", "trace_normalization_config"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError("Unknown audit settings: " + ", ".join(sorted(unknown)))
    root, saved, movie = recording(run, stem)
    path, observations, _, metrics = _observations(root, movie)
    identity = options.get("identity")
    if isinstance(identity, bool) or not isinstance(identity, int) or identity not in observations:
        raise ValueError("Choose one saved cell for the audit.")
    metric = options.get("metric")
    if metric not in {row["key"] for row in metrics}:
        raise ValueError("Choose one saved measurement for the audit.")
    config = {**workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
              "detrend_methods": [workbench.DEFAULT_DETREND_METHOD, "none", "linear", "robust_linear"],
              "period_methods": ["lomb", "fft_nlls", "mesa"], "significance_method": "lomb",
              "trace_normalization": "none",
              "trace_normalization_config": {}, **options}
    for key, choices in (("detrend_methods", workbench.DETREND_METHODS),
                         ("period_methods", workbench.PERIOD_METHODS)):
        chosen = config[key]
        if (not isinstance(chosen, list) or not chosen or len(chosen) != len(set(chosen))
                or any(not isinstance(value, str) or value not in choices for value in chosen)):
            raise ValueError("Choose distinct available " + key.replace("_", " ") + ".")
    config["input_filters"] = workbench.audit_input_filters(config.get("input_filters"), median_window_points=config.pop("median_window_points", None))
    if config["trace_normalization"] not in workbench.NORMALIZATION_METHODS:
        raise ValueError("Unknown trace normalisation.")
    norm = config["trace_normalization_config"]
    if not isinstance(norm, dict):
        raise ValueError("Normalisation options must be an object.")
    import inspect
    norm_allowed = set(inspect.signature(workbench.normalize_trace).parameters) - {"hours", "values", "method", "detrended"}
    if set(norm) - norm_allowed:
        raise ValueError("Unknown normalisation option.")
    inherited = {**DEFAULTS, **_module_settings(saved, movie, "rhythms")}
    # No primary method is silently added to the requested comparison grid.
    config["fit_method"] = config["period_methods"][0]
    config["detrend"] = config["detrend_methods"][0]
    resolved = workbench.resolve_analysis_options(inherited, config.get)
    config["significance_method"] = resolved["significance_method"]
    import shutil
    if shutil.disk_usage(root).free < max(64 * 1024**2, path.stat().st_size * 8):
        raise ValueError("Insufficient space to save this audit.")
    return root, movie, path, config, resolved


def _clean_frame(frame):
    # pandas handles numpy scalars and converts missing/nonfinite values to null.
    return json.loads(frame.to_json(orient="records"))


def _signature(path, movie, resolved):
    from pymicroglia import workbench
    from pymicroglia.figure_tables import method_audit
    from pymicroglia._workbench import estimates, processing, options
    return {"input_sha256": _fingerprint(path), "recording": movie,
            "inherited_parameters": resolved["params"], "environment": workbench.rhythm_environment(),
            "producer_sha256": {module.__name__: _fingerprint(Path(module.__file__))
                                for module in (workbench, method_audit, estimates, processing, options,
                                               __import__(__name__, fromlist=["_"]))}}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


def _save_comparisons(path, signature, pending):
    if _fingerprint(path) != signature["input_sha256"]:
        raise ValueError("Saved measurements changed during this audit. Run it again.")
    for cache, row_data in pending:
        _atomic_json(cache, {"schema_version": 1, "tables": row_data}, indent=None)


def run_audit(run, stem, options, *, progress=None, cancelled=None, reuse=False):
    """Run and record one cell's comparison; never attach it as population data."""
    import pandas as pd
    import numpy as np
    from pymicroglia import workbench
    from pymicroglia.figure_tables.method_audit import analyse_traces
    root, movie, path, config, resolved = validate_request(run, stem, options)
    signature = _signature(path, movie, resolved)
    audit_id = _digest({"config": config, "signature": signature})
    parent = _folder(root, stem)
    folder = parent / audit_id
    total = len(config["detrend_methods"]) * len(config["period_methods"]) * len(config["input_filters"])
    pending_caches = []
    def checkpoint(completed, stage, **extra):
        if cancelled and cancelled():
            # Keep completed rows for resume, after checking the same
            # environment boundary used by a fully completed sweep.
            if pending_caches:
                workbench.verify_rhythm_environment(signature["environment"])
                _save_comparisons(path, signature, pending_caches)
            raise AuditCancelled("Audit cancelled after the current comparison.")
        if progress:
            progress({"stage": stage, "completed": completed, "total": total, **extra})
    checkpoint(0, "preparing")
    if reuse:
        cached = load_audit(root, stem, audit_id, required=False)
        if cached:
            checkpoint(total, "reused")
            return {**cached, "reused": True}
    # Cached results need no new analysis capture. A fresh sweep captures
    # one identity and checks it again before publishing.
    with workbench.analysis_session() as capture:
        if capture.environment != signature["environment"]:
            raise ValueError("Scientific environment changed before this audit. Run it again.")
        frame = pd.DataFrame(read_rows(path))
        frame = frame.loc[pd.to_numeric(frame.identity, errors="coerce").eq(config["identity"])].copy()
        for column in ("identity", "hours", config["metric"]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame[config["metric"]].notna().any():
            raise ValueError("This cell has no finite observations for the selected measurement.")
        parent.mkdir(parents=True, exist_ok=True)
        pieces = [[] for _ in range(5)]
        completed = 0
        reused_rows = 0
        for detrend in config["detrend_methods"]:
            checkpoint(completed, "comparing", detrend=detrend)
            row_config = {**config, "detrend_methods": [detrend], "detrend": detrend,
                          "multiple_testing": "none", "trace_normalization": "none",
                          "trace_normalization_config": {}}
            # Family correction is reapplied below for the currently requested grid.
            row_signature = {**signature, "inherited_parameters": {**resolved["params"], "detrend": detrend, "multiple_testing": "none"}}
            row_key = _digest({"config": row_config, "signature": row_signature})
            cache = parent / "comparisons" / (row_key + ".json")
            row_data = None
            if reuse and cache.is_file():
                try:
                    row_data = json.loads(cache.read_text(encoding="utf-8"))["tables"]
                except (ValueError, KeyError, OSError):
                    row_data = None
            if row_data is None:
                tables = analyse_traces(frame, identities=[config["identity"]], metrics=[config["metric"]],
                                        detrends=[detrend], estimators=config["period_methods"],
                                        input_filters=config["input_filters"], params=resolved["params"],
                                        fallback_significance_method=config["significance_method"], correction="none",
                                        alpha=resolved["rhythmic_alpha"], min_observations=resolved["min_observations"],
                                        min_cycles=resolved["min_cycles"])
                row_data = [_clean_frame(table) for table in tables]
                cache.parent.mkdir(exist_ok=True)
                pending_caches.append((cache, row_data))
            else:
                reused_rows += 1
            for piece, rows in zip(pieces, row_data):
                piece.extend(rows)
            completed += len(config["input_filters"]) * len(config["period_methods"])
            checkpoint(completed, "comparing", detrend=detrend)
        traces, results, components, spectra, evidence = [pd.DataFrame(rows) for rows in pieces]
        traces = traces.drop_duplicates(subset=["display_panel", "preprocessor", "detrend", "estimator", "hours"])
        evidence = evidence.drop_duplicates("evidence_id")
        evidence["q_value"] = workbench.adjust_pvalues(pd.to_numeric(evidence.p_value, errors="coerce"), resolved["multiple_testing"])
        evidence["significant"] = evidence.status.eq("ok") & evidence.q_value.lt(resolved["rhythmic_alpha"])
        evidence["rhythm_status"] = np.where(evidence.status.eq("ok") & evidence.q_value.notna(),
                                             np.where(evidence.significant, "significant", "not significant"), "not tested")
        evidence["correction_method"] = resolved["multiple_testing"]
        evidence["alpha"] = resolved["rhythmic_alpha"]
        lookup = evidence.set_index("evidence_id")
        for key in ("q_value", "significant", "rhythm_status", "correction_method", "alpha"):
            results[key] = results.evidence_id.map(lookup[key])
        traces["display_value"] = traces.plotted_value
        traces["normalization_message"] = ""
        normalization_records = []
        if config["trace_normalization"] != "none":
            for _, group in traces.groupby(["display_panel", "preprocessor", "detrend", "estimator"], sort=False):
                try:
                    normalized = workbench.normalize_trace(group.hours.to_numpy(float), group.plotted_value.to_numpy(float),
                                                           config["trace_normalization"],
                                                           detrended=group.iloc[0].display_panel != "raw_reference",
                                                           **config["trace_normalization_config"])
                    traces.loc[group.index, "display_value"] = normalized["values"]
                    normalization_records.append(normalized["workbench_run_record"])
                except (ValueError, RuntimeError) as exc:
                    traces.loc[group.index, "display_value"] = np.nan
                    traces.loc[group.index, "normalization_message"] = str(exc)
        checkpoint(total, "saving")
    # Reject a source edit during fitting rather than publish stale evidence.
    _save_comparisons(path, signature, pending_caches)
    folder.mkdir(exist_ok=True)
    tables = dict(zip(("traces", "results", "components", "spectra", "evidence"),
                      (traces, results, components, spectra, evidence)))
    for name, table in tables.items():
        table.to_csv(folder / (name + ".csv"), index=False)
    selected_input = frame[["identity", "hours", config["metric"]]]
    selected_input.to_csv(folder / "input.csv", index=False)
    script = ("# Reproduce this saved single-cell audit without opening the UI.\n"
              "from pathlib import Path\nimport json\nimport hashlib\n"
              "from pymicroglia.ui.exploration_audit import run_audit\n"
              "from pymicroglia import workbench\n"
              f"OPTIONS = {config!r}\nSTEM = {stem!r}\n"
              "RUN = Path(__file__).resolve().parents[4]\n"
              f"assert workbench.rhythm_environment() == {signature['environment']!r}, 'Scientific environment changed'\n"
              f"INPUT = RUN / {str(path.relative_to(root))!r}\n"
              f"assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == {signature['input_sha256']!r}, 'Input changed'\n"
              "result = run_audit(RUN, STEM, OPTIONS, reuse=False)\n"
              "print(json.dumps(result['results'], allow_nan=False))\n")
    (folder / "reproduce.py").write_text(script, encoding="utf-8")
    outputs = {file.name: _fingerprint(file) for file in folder.iterdir() if file.name != "result.json" and file.is_file()}
    result = {"schema_version": 1, "audit_id": audit_id, "run": str(root), "stem": stem,
              "created_at": datetime.now(timezone.utc).isoformat(), "options": config,
              "signature": signature, "input_table": str(path.relative_to(root)), "output_sha256": outputs,
              "producer": "pymicroglia.figure_tables.method_audit.analyse_traces",
              "correction_family": {"scope": "selected cell, measurement, detrending and selected input-filter comparisons",
                                    "independent_tests": len(evidence), "method": resolved["multiple_testing"]},
              "normalization_scope": "display only; period estimation and testing use unnormalised inputs",
              "normalization_records": normalization_records, "reused_comparison_rows": reused_rows,
              **{name: _clean_frame(table) for name, table in tables.items()}}
    _atomic_json(folder / "result.json", result, indent=None)
    checkpoint(total, "completed")
    return result


def load_audit(run, stem, audit_id, *, required=True):
    if not isinstance(audit_id, str) or not re.fullmatch(r"[0-9a-f]{24}", audit_id):
        raise ValueError("Invalid saved audit identifier.")
    root, _, movie = recording(run, stem)
    folder = _folder(root, stem) / audit_id
    try:
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        path = root / result["input_table"]
        if _fingerprint(path) != result["signature"]["input_sha256"]:
            raise ValueError("The saved audit's input measurements changed.")
        for name, digest in result["output_sha256"].items():
            if Path(name).name != name or _fingerprint(folder / name) != digest:
                raise ValueError("Saved audit outputs changed.")
        return {**result, "run": str(root)}
    except (OSError, KeyError, ValueError):
        if required:
            raise ValueError("This saved audit is unavailable or its recorded inputs or outputs changed.") from None
        return None


def audit_history(run, stem):
    root, _, _ = recording(run, stem)
    parent = _folder(root, stem)
    rows = []
    if parent.is_dir():
        for folder in parent.iterdir():
            if re.fullmatch(r"[0-9a-f]{24}", folder.name) and (folder / "result.json").is_file():
                try:
                    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                    rows.append({key: result[key] for key in ("audit_id", "created_at", "options")})
                except (ValueError, KeyError, OSError):
                    continue
    return {"audits": sorted(rows, key=lambda row: row["created_at"], reverse=True)}


def audit_files(run, stem, audit_id, *, result=None):
    result = result if result is not None else load_audit(run, stem, audit_id)
    root = Path(result["run"])
    folder = _folder(root, stem) / audit_id
    return folder, [folder / name for name in ("result.json", *result["output_sha256"])]


def browser_result(result):
    """Keep UI values exact; full repeated provenance stays in recorded exports."""
    return {**result,
            "traces": [{key: value for key, value in row.items() if key != "filter_run_record_json"}
                       for row in result["traces"]],
            "results": [{key: value for key, value in row.items() if key != "workbench_comparison_json"}
                        for row in result["results"]]}
