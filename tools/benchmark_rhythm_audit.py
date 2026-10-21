"""Compare uncached audit execution with a frozen implementation on one workload.

The workload JSON supplies recording, stem, options and the input SHA256.
All results are written to that isolated recording, never the user's source run.
Example: python tools/benchmark_rhythm_audit.py --workload workload.json
    --baseline-source baseline.py --baseline-engine baseline-method-audit.py --out samples.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import random
import statistics
import sys
import threading
import time


TABLES = ("traces", "results", "components", "spectra", "evidence", "normalization_records")


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def engine(module):
    import pymicroglia.figure_tables as package
    key = "pymicroglia.figure_tables.method_audit"
    original = sys.modules[key]
    sys.modules[key] = package.method_audit = module
    try:
        yield
    finally:
        sys.modules[key] = package.method_audit = original


def measure(operation):
    """Optional sampled working set; no extra dependency for the application."""
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None
    samples = []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            samples.append(process.memory_info().rss)
            stop.wait(.05)
    before = process.memory_info().rss if process else None
    thread = threading.Thread(target=sample, daemon=True) if process else None
    if thread:
        thread.start()
    start = time.perf_counter()
    try:
        result = operation()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        if thread:
            thread.join()
    return result, {"seconds": elapsed, "rss_before": before,
                    "rss_peak_sampled": max(samples) if samples else None,
                    "rss_after": process.memory_info().rss if process else None}


def interval(ratios, seed=104):
    rng = random.Random(seed)
    means = sorted(statistics.geometric_mean(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    return [means[125], means[4874]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--backend", action="store_true", help="Measure the native audit engine without UI persistence")
    args = parser.parse_args()
    if not 2 <= args.pairs <= 10:
        parser.error("Use 2 to 10 paired observations.")
    workload = json.loads(args.workload.read_text())
    run = Path(workload["recording"]).resolve()
    stem = workload["stem"]
    input_path = run / "measure" / stem / "cell_frame.csv"
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == workload["sha256"]
    assert not Path(workload["source"]).resolve().is_relative_to(run), "Use an isolated recording copy"
    from pymicroglia.ui import exploration_audit as candidate
    from pymicroglia.figure_tables import method_audit as candidate_engine
    from pymicroglia import workbench
    baseline = load_source("pymicroglia.ui._audit_benchmark_baseline", args.baseline_source)
    baseline_engine = load_source("pymicroglia.figure_tables._audit_benchmark_baseline", args.baseline_engine)
    options = workload["options"]
    cases = {
        "full_fresh": options,
        "small_fresh": {**options, "detrend_methods": ["linear"], "period_methods": ["lomb"]},
        "normalised_fresh": {**options, "detrend_methods": ["linear", "robust_linear"],
                             "trace_normalization": "minmax"},
    }
    report = {"workload": workload, "platform": platform.platform(), "python": platform.python_version(),
              "environment": workbench.rhythm_environment(), "baseline_sha256": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
              "baseline_engine_sha256": hashlib.sha256(args.baseline_engine.read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
              "candidate_engine_sha256": hashlib.sha256(Path(candidate_engine.__file__).read_bytes()).hexdigest(),
              "profiled": False, "seed": 104, "cases": {}}
    if args.backend:
        cases.pop("normalised_fresh")
        report["boundary"] = "native analysis through complete tables; excludes UI saving"
    else:
        report["boundary"] = "run_audit through completed persisted result; excludes imports and browser drawing"

    def operation(module, native, config):
        if not args.backend:
            return module.run_audit(run, stem, config, reuse=False)
        import pandas as pd
        _, _, _, validated, resolved = candidate.validate_request(run, stem, config)
        frame = pd.read_csv(input_path)
        tables = native.analyse_traces(
            frame, identities=[validated["identity"]], metrics=[validated["metric"]],
            detrends=validated["detrend_methods"], estimators=validated["period_methods"],
            median_window_points=validated["median_window_points"], params=resolved["params"],
            fallback_significance_method=validated["significance_method"],
            correction=resolved["multiple_testing"], alpha=resolved["rhythmic_alpha"],
            min_observations=resolved["min_observations"], min_cycles=resolved["min_cycles"])
        return {**dict(zip(TABLES[:5], map(candidate._clean_frame, tables))), "normalization_records": []}

    rng = random.Random(104)
    for name, config in cases.items():
        samples = []
        first_order = ["baseline", "candidate"]
        rng.shuffle(first_order)
        for pair in range(args.pairs):
            order = first_order if pair % 2 == 0 else list(reversed(first_order))
            values, times = {}, {}
            for variant in order:
                module, native = (baseline, baseline_engine) if variant == "baseline" else (candidate, candidate_engine)
                with engine(native):
                    value, measured = measure(lambda: operation(module, native, config))
                values[variant] = value
                times[variant] = measured
                print(json.dumps({"case": name, "pair": pair, "variant": variant, **measured}), flush=True)
            for table in TABLES:
                assert values["baseline"][table] == values["candidate"][table], (name, pair, table)
            samples.append({"pair": pair, "order": order, **times, "exact_parity": True})
        ratios = [row["baseline"]["seconds"] / row["candidate"]["seconds"] for row in samples]
        report["cases"][name] = {"samples": samples, "speedup": statistics.geometric_mean(ratios),
                                  "speedup_interval_95": interval(ratios), "exact_parity": True}
        args.out.write_text(json.dumps(report, indent=2))
    # An identical rerun is measured separately; it must not be called fresh execution.
    samples = []
    for pair in range(0 if args.backend else args.pairs):
        order = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        times, values = {}, {}
        for variant in order:
            module, native = (baseline, baseline_engine) if variant == "baseline" else (candidate, candidate_engine)
            with engine(native):
                value, measured = measure(lambda: module.run_audit(run, stem, options, reuse=True))
            assert value["reused"]
            times[variant], values[variant] = measured, value
        for table in TABLES:
            assert values["baseline"][table] == values["candidate"][table], ("full_reused", table)
        samples.append({"pair": pair, "order": order, **times, "exact_parity": True})
    if samples:
        ratios = [row["baseline"]["seconds"] / row["candidate"]["seconds"] for row in samples]
        report["cases"]["full_reused"] = {"samples": samples, "speedup": statistics.geometric_mean(ratios),
                                           "speedup_interval_95": interval(ratios), "exact_parity": True}
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: {key: value for key, value in row.items() if key != "samples"}
                      for name, row in report["cases"].items()}), flush=True)


if __name__ == "__main__":
    main()
