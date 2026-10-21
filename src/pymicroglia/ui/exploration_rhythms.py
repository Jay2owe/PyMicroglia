"""Run the accepted trace-grid analysis and attach immutable cell rhythm results."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .exploration_data import _observations, number, read_rows, recording, table_path

OPTION_NAMES = ("fit_method", "significance_method", "fft_component_test",
                "period_min_hours", "period_max_hours", "min_observations",
                "min_cycles", "multiple_testing", "detrend", "rhythmic_alpha",
                "component_surrogates", "component_block_hours", "component_seed",
                "trace_view", "trace_normalization", "trace_normalization_config")
ADVANCED_OPTION_NAMES = ("trace_filter", "period_config", "detrend_window_hours", "detrend_polynomial_degree",
                         "detrend_min_valid_fraction", "detrend_bandwidth_hours",
                         "detrend_low_cut_hours", "detrend_high_cut_hours", "detrend_filter_order",
                         "detrend_lowess_fraction", "detrend_lowess_iterations",
                         "detrend_asls_smoothness", "detrend_asls_asymmetry", "detrend_asls_iterations")
OPTION_NAMES = (*OPTION_NAMES, *ADVANCED_OPTION_NAMES)


def _defaults():
    from pymicroglia.visualisation.figures import load
    return {o.name: o.default for o in load()["all_cell_trace_grid"].options}


def _folder(root, stem):
    # Names cannot escape the output folder, even for unusual saved stems.
    key = hashlib.sha256(stem.encode()).hexdigest()[:16]
    return root / "AI_Exports" / "exploration_rhythms" / key


def _fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _producer_hashes():
    from pymicroglia.figure_tables import all_cell_traces, fft_component_grid, trace_workers
    return {name: _fingerprint(Path(module.__file__))
            for name, module in (("all_cell_traces", all_cell_traces),
                                 ("fft_component_grid", fft_component_grid),
                                 ("trace_workers", trace_workers))}


def reusable_rhythm_result(root, movie, config, analysis_id=None, *, producer_hashes=None):
    from pymicroglia import workbench
    from .exploration_previews import _module_settings
    _, saved, declared = recording(root, movie["stem"])
    result = load_attached_rhythms(root, declared, analysis_id)
    if (result and result.get("resolved_options") == config
            and result.get("source_module_settings") == _module_settings(saved, declared, "rhythms")
            and result.get("producer_sha256") == (producer_hashes or _producer_hashes())
            and result.get("environment") == workbench.rhythm_environment()):
        return result
    return None


def rhythm_options(run, stem):
    from pymicroglia import workbench
    root, _, movie = recording(run, stem)
    _, observations, _, metrics = _observations(root, movie)
    defaults = _defaults()
    keys = {m["key"] for m in metrics}
    chosen = [key for key in defaults["metrics"] if key in keys]
    if not chosen:
        chosen = [next((key for key in ("integrated_density", "signal_mean") if key in keys), metrics[0]["key"])]
    def method_label(key):
        return {"lomb": "Lomb–Scargle", "fft_nlls": "FFT–NLLS"}.get(key, workbench.PERIOD_METHODS[key]["label"])
    recipes = [{"key": "fft_component", "label": "Fourier components",
                "fit_method": "fft_nlls", "test": "fft_component",
                "required": {"detrend": workbench.DEFAULT_DETREND_METHOD, "multiple_testing": "none"},
                "compatible_detrends": list(workbench.FFT_COMPONENT_DETRENDS),
                "description": "Fit Fourier components with nonlinear least squares and test each fitted component. Use asymmetric least-squares or robust linear detrending and uncorrected component tests."}]
    for key in ["lomb", "fft_nlls", *workbench.PERIOD_METHODS]:
        if any(recipe["key"] == key for recipe in recipes):
            continue
        label = method_label(key)
        test = key if key in workbench.SIGNIFICANCE_METHODS else "lomb"
        recipes.append({"key": key, "label": label if test == key else label + " + Lomb–Scargle",
                        "fit_method": key, "test": test, "required": {},
                        "description": f"Estimate each cell's period using {label}; test rhythmicity using {method_label(test)}. Detrending is selected separately."})
    detrend_labels = {"none": "No detrending", "robust_linear": "Robust linear",
                      "asymmetric_least_squares": "Asymmetric least squares",
                      "linear": "Linear", "polynomial": "Polynomial",
                      "running_mean": "Running mean", "moving_median": "Moving median",
                      "lowess": "Locally weighted smooth (LOWESS)"}
    return {"input_filters": [{"key": key, **row} for key, row in workbench.filter_methods().items()], "defaults": {key: defaults[key] for key in OPTION_NAMES}, "advanced_option_names": list(ADVANCED_OPTION_NAMES), "recipes": recipes,
            "metrics": metrics, "selected_metrics": chosen, "cells": len(observations),
            "normalizations": [{"key": key, "label": row["label"]}
                               for key, row in workbench.NORMALIZATION_METHODS.items()],
            "methods": [{"key": key, "label": method_label(key)}
                        for key, row in workbench.PERIOD_METHODS.items()],
            "tests": [{"key": "fft_component", "label": "Fourier component test"},
                      *[{"key": key, "label": method_label(key)}
                        for key in workbench.SIGNIFICANCE_METHODS]],
            "detrends": [{"key": key, "label": detrend_labels.get(key, key.replace("_", " ").capitalize())}
                         for key in workbench.scientific_options()["detrend"]["allowed"]]}


def validate_request(run, stem, options):
    if not isinstance(options, dict) or set(options) - {*OPTION_NAMES, "metrics"}:
        raise ValueError("Unknown rhythm analysis settings.")
    root, saved, movie = recording(run, stem)
    path, observations, frames, metrics = _observations(root, movie)
    defaults = _defaults()
    config = {**defaults, **options}
    keys = {m["key"] for m in metrics}
    if "metrics" not in options:
        config["metrics"] = [key for key in defaults["metrics"] if key in keys]
        if not config["metrics"]:
            config["metrics"] = rhythm_options(run, stem)["selected_metrics"]
    chosen = config["metrics"]
    if (not isinstance(chosen, list) or not chosen or len(chosen) != len(set(chosen))
            or set(chosen) - keys):
        raise ValueError("Choose distinct saved measurements for rhythm analysis.")
    from pymicroglia import workbench
    from pymicroglia.measure.modules.rhythms import DEFAULTS
    from .exploration_previews import _module_settings
    inherited = {**DEFAULTS, **_module_settings(saved, movie, "rhythms")}
    resolved = workbench.resolve_analysis_options(inherited, config.get)
    if config["trace_view"] not in {"raw", "detrended"}:
        raise ValueError("Choose raw or detrended trace display.")
    if config["trace_normalization"] not in workbench.NORMALIZATION_METHODS:
        raise ValueError("Unknown trace normalization.")
    if not isinstance(config["trace_normalization_config"], dict):
        raise ValueError("Trace normalization settings must be an object.")
    filtering = config.get('trace_filter')
    if filtering is not None:
        config['trace_filter'] = workbench.validate_trace_filter(filtering)
    if resolved["method"] not in workbench.PERIOD_METHODS:
        raise ValueError("Unknown period estimator.")
    if not isinstance(config["fft_component_test"], bool):
        raise ValueError("Fourier component testing must be true or false.")
    if config["fft_component_test"] and (resolved["method"] != "fft_nlls"
            or resolved["detrend"] not in workbench.FFT_COMPONENT_DETRENDS
            or resolved["multiple_testing"] != "none"
            or config["significance_method"] is not None):
        raise ValueError("Fourier component testing uses FFT-NLLS, asymmetric least-squares or robust linear detrending and no correction. Select a periodogram test for other settings.")
    if (isinstance(config["component_surrogates"], bool)
            or not isinstance(config["component_surrogates"], int)
            or not 1 <= config["component_surrogates"] <= 10000
            or not 0 < float(config["component_block_hours"]) < float("inf")):
        raise ValueError("Component surrogate count and block duration must be positive.")
    if shutil.disk_usage(root).free < max(64 * 1024**2, path.stat().st_size * 8):
        raise ValueError("There is insufficient space to save the rhythm analysis.")
    return root, saved, movie, path, observations, frames, config


def _atomic_json(path, value, *, indent=2):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=indent), encoding="utf-8")
    os.replace(temporary, path)


def run_rhythm_analysis(run, stem, options=None, *, analysis_id=None, progress=None, reuse=False, workers=None):
    """One recording, every observed identity, through the accepted preparation."""
    import pandas as pd
    from pymicroglia import workbench
    from pymicroglia.figure_tables import all_cell_traces, fft_component_grid, trace_workers
    prepare = all_cell_traces.prepare
    from .exploration_previews import _module_settings
    root, saved, movie, path, observations, frames, config = validate_request(run, stem, options or {})
    if workers is None:
        # Small recordings finish sooner without process startup. A bounded
        # pool keeps simultaneous fitting and memory use predictable.
        cap = 8 if len(config['metrics']) > 1 else 4
        workers = min(cap, max(1, (os.cpu_count() or 1) // 2)) if len(observations) * len(config['metrics']) >= 8 else 1
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError('Use between one and eight rhythm analysis workers.')
    workers = min(workers, len(observations) * len(config['metrics']))
    if progress is not None:
        progress({'stage': 'preparing', 'completed': 0,
                  'total': len(observations) * len(config['metrics']),
                  'completed_cells': 0, 'total_cells': len(observations)})
    fingerprint = _fingerprint(path)
    movie_hash = hashlib.sha256(json.dumps(movie, sort_keys=True).encode()).hexdigest()
    inherited_settings = _module_settings(saved, movie, 'rhythms')
    producer_hashes = _producer_hashes()
    if reuse:
        # Only complete results with matching inputs, settings and scientific
        # code can skip a rerun. Direct calls/reproduction still calculate anew.
        attached = reusable_rhythm_result(root, movie, config, producer_hashes=producer_hashes)
        if attached:
            if progress is not None:
                progress({'stage': 'reused', 'completed': len(attached['rows']),
                          'total': len(attached['rows']), 'completed_cells': len(observations),
                          'total_cells': len(observations)})
            return {**attached, 'reused': True}
    frame = pd.DataFrame(read_rows(path))
    for column in {"identity", "frame_index", "hours", *config["metrics"]}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    class Source:
        def table(self, name):
            if name != "cell_frame.csv":
                raise ValueError("Rhythm analysis needs only saved cell-frame observations.")
            return frame

        def module_params(self, name):
            return _module_settings(saved, movie, name)

    source = Source()
    source.stem = stem
    with workbench.analysis_session() as session:
        prepared, tests = prepare(source, config, progress=progress, workers=workers)
        environment = dict(session.environment)
    rows = []
    for row in tests.to_dict("records"):
        period, phase = number(row.get("period_hours")), number(row.get("phase_hours"))
        if row.get("phase_units") not in {None, "hours", "h"}:
            phase = None
        metric = row["metric"]
        rows.append({"identity": int(row["identity"]), "metric": metric,
                     "status": row["rhythm_status"], "period_hours": period,
                     "phase_hours": phase,
                     "phase_fraction": (phase / period) % 1 if phase is not None and period and period > 0 else None,
                     "period_error_hours": number(row.get("period_error_hours")),
                     "p_value": number(row.get("p_value")), "q_value": number(row.get("q_value")),
                     "alpha": number(row.get("alpha")), "correction": row.get("correction"),
                     "method": row["estimation_method"], "significance_method": row["significance_method"],
                     "reason": row.get("reason") or ("Insufficient observations" if row["estimate_status"] == "not_tested" else ""),
                     "observations": int(row["observations"]), "cycles_observed": number(row.get("cycles_observed")),
                     "period_underdetermined": not bool(row["supported_period"]),
                     "time_mean": 3 if metric in {"integrated_density", "integrated_density_mean3"} else 1})
    if _fingerprint(path) != fingerprint:
        raise ValueError("Saved measurements changed during rhythm analysis; no results were attached.")
    _, current_saved, current_movie = recording(root, stem)
    if (hashlib.sha256(json.dumps(current_movie, sort_keys=True).encode()).hexdigest() != movie_hash
            or _module_settings(current_saved, current_movie, 'rhythms') != inherited_settings):
        raise ValueError("Saved alignment or rhythm settings changed during analysis; no results were attached.")
    if progress is not None:
        progress({'stage': 'saving', 'completed': len(tests), 'total': len(tests),
                  'completed_cells': len(observations), 'total_cells': len(observations)})
    identifier = analysis_id or uuid4().hex
    if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
        raise ValueError("Invalid rhythm analysis identifier.")
    folder = _folder(root, stem) / identifier
    folder.mkdir(parents=True, exist_ok=False)
    result = {"schema": "pymicroglia.exploration-rhythms.v1", "analysis_id": identifier,
              "run": str(root), "stem": stem, "rows": rows, "metrics": list(config["metrics"]),
              "preview": False, "source": str(folder / "result.json"),
              "source_sha256": fingerprint, "movie_sha256": movie_hash,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "settings": prepared["traces"]["settings"], "requested_options": options or {},
              "resolved_options": config, "workbench_version": prepared["traces"]["settings"]["workbench_version"],
              "source_module_settings": inherited_settings,
              "environment": environment,
              "execution": {"workers": workers, "executor": "process" if workers > 1 else "serial",
                            "task": "cell_measurement" if workers > 1 else "cell",
                            "scheduler": "spectral_entropy_descending" if workers > 1 else "identity_order"},
              "producer_sha256": producer_hashes,
              "meaning": "Tested rhythm results; peak position is a fraction of each cell's fitted period. Short traces retain their period-support flag.",
              "message": "", "producer": "pymicroglia.figure_tables.all_cell_traces.prepare"}
    tests.to_csv(folder / "statistics.csv", index=False)
    prepared["traces"]["table"].to_csv(folder / "traces.csv", index=False)
    result['traces_sha256'] = _fingerprint(folder / 'traces.csv')
    result['statistics_sha256'] = _fingerprint(folder / 'statistics.csv')
    _atomic_json(folder / "result.json", result)
    (folder / "reproduce.py").write_text(
        "import json\nfrom pathlib import Path\nfrom pymicroglia.ui.exploration_rhythms import run_rhythm_analysis\n"
        "if __name__ == '__main__':\n"
        "    result = json.loads(Path(__file__).with_name('result.json').read_text(encoding='utf-8'))\n"
        "    run_rhythm_analysis(result['run'], result['stem'], {k: v for k, v in result['resolved_options'].items() "
        f"if k in {list(OPTION_NAMES)!r} or k == 'metrics'}}, workers=result.get('execution', {{}}).get('workers', 1))\n", encoding="utf-8")
    _atomic_json(folder.parent / "latest.json", {"result": f"{identifier}/result.json"})
    return result


def load_attached_rhythms(root, movie, analysis_id=None):
    """Only load completed results whose saved measurement and alignment match."""
    folder = _folder(root, movie["stem"])
    pointer = folder / "latest.json"
    if analysis_id is not None and (len(analysis_id) != 32 or any(c not in "0123456789abcdef" for c in analysis_id)):
        raise ValueError("Invalid rhythm analysis identifier.")
    if analysis_id is None and not pointer.is_file():
        return None
    try:
        path = (folder / (f"{analysis_id}/result.json" if analysis_id else json.loads(pointer.read_text(encoding="utf-8"))["result"])).resolve()
        if not path.is_relative_to(folder.resolve()) or path.stat().st_size > 16 * 1024**2:
            return None
        result = json.loads(path.read_text(encoding="utf-8"))
        source = table_path(root, movie, "cell_frame")
        # The actual movie declaration includes its alignment and module recipe.
        _, saved, declared = recording(root, movie["stem"])
        alignment = hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest()
        if source is None or result.get("source_sha256") != _fingerprint(source) or result.get("movie_sha256") != alignment:
            return None
        if "source_module_settings" in result:
            from .exploration_previews import _module_settings
            if result["source_module_settings"] != _module_settings(saved, declared, 'rhythms'):
                return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def rhythm_history(run, stem):
    root, _, movie = recording(run, stem)
    folder = _folder(root, stem)
    # Bound history scanning; files are immutable and newest folders sort by mtime.
    paths = sorted(folder.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]
    rows = []
    for path in paths:
        result = load_attached_rhythms(root, movie, path.parent.name)
        if result:
            rows.append({key: result.get(key) for key in ("analysis_id", "metrics", "created_at", "resolved_options")})
    return {"analyses": rows}


def load_rhythm_traces(run, stem, metric, analysis_id, identities):
    """Read the saved displayed samples and exact model without recomputation."""
    import csv
    root, _, movie = recording(run, stem)
    result = load_attached_rhythms(root, movie, analysis_id)
    if not result or metric not in result["metrics"]:
        raise ValueError("The selected saved analysis is unavailable for this measurement.")
    wanted = {int(i) for i in identities.split(",") if i}
    if not wanted or len(wanted) > 128:
        raise ValueError("Choose between one and 128 cells for saved trace display.")
    path = Path(result["source"]).with_name("traces.csv")
    if not path.is_file():
        return {"analysis_id": analysis_id, "metric": metric, "cells": [], "message": "This older result has no saved display traces; raw measurements are shown."}
    if _fingerprint(path) != result.get("traces_sha256"):
        raise ValueError("Saved display traces changed after analysis.")
    cells = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = int(float(row["identity"]))
            if identity not in wanted or row["metric"] != metric:
                continue
            cell = cells.setdefault(identity, {"identity": identity, "indices": [], "values": [], "fitted": []})
            cell["indices"].append(int(float(row["frame_index"])))
            cell["values"].append(number(row["value"]))
            cell["fitted"].append(number(row.get("descriptive_fit")))
    return {"analysis_id": analysis_id, "metric": metric, "cells": list(cells.values()), "settings": result["settings"]}
