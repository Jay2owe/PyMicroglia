"""Prepare and verify a complete synthetic rhythm-discovery example."""
from __future__ import annotations
from pymicroglia._results import read_document

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tests.pipelines.invoke import invoke, source_identity, write_json as _write_json
import sys
from unittest.mock import patch

from pymicroglia.pipelines._screening import file_hash, read_table


def declaration():
    return {"pipeline": "rhythm-discovery", "name": "synthetic-discovery",
        "test_measurements": ["signal", "other"], "comparison_measurements": ["size_example"],
        "biological_samples": {"a": "synthetic-sample-1", "b": "synthetic-sample-2", "c": "synthetic-sample-1"},
        "analysis_options": {"fit_method": "spectrum_resampling", "significance_method": "lomb", "detrend": "none",
            "period_min_hours": 2., "period_max_hours": 48., "min_observations": 24, "min_cycles": 3,
            "rhythmic_alpha": .05, "multiple_testing": "bh", "period_config": {"sr_iterations": 100, "sr_seed": 17}},
        "pairs": {"mode": "all"}}


def prepare(output):
    """Generate original tables and known cases through the public Workbench."""
    import pandas as pd
    import pymicroglia.workbench as circadian
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepare requires a new empty folder; --verify-existing resumes a prepared example")
    output.mkdir(parents=True, exist_ok=True)
    def component(period, phase=0, waveform="cosine"):
        return {"id": "declared-component", "period_hours": period, "amplitude": 1., "waveform": waveform, "phase_hours": phase}
    scenarios = [{"id": name, "components": [component(period, phase, waveform)]} for name, period, phase, waveform in (
        ("eight", 8., 0., "cosine"), ("offset-two", 8., 2., "cosine"), ("half-cycle", 8., 4., "cosine"),
        ("twelve-triangle", 12., 0., "triangle"))]
    scenarios.extend({"id": name, "components": [], "noise": {"kind": "white", "sd": 1.}} for name in ("noise-a", "noise-b"))
    scenarios.append({"id": "insufficient", "components": [component(8.)], "retain_observations": 8})
    design = {"replicates": 1, "scenarios": scenarios, "truth_policy": {
        "period_min_hours": 2., "period_max_hours": 48., "min_observations": 24, "min_cycles": 3,
        "relative_tolerance": .1, "absolute_tolerance_hours": .5, "target": "all", "extra_components": "penalize"}}
    hours = [50. + i / 2 for i in range(320)]
    profiles = [{"id": movie, "hours": hours, "origin_hours": 50.,
        "missing": [movie == "c" and 110 <= hour <= 112 for hour in hours], "metadata": {"synthetic_movie": movie}}
        for movie in ("a", "b", "c")]
    generated = circadian.generate_benchmark_cases(design, profiles, partition="development", seed=20260910)
    _write_json(output / "source_generation.json", generated)
    cases = {case["case_id"]: case for case in generated["cases"]}
    traces = {(cases[trace["case_id"]]["profile"], cases[trace["case_id"]]["scenario"]): trace for trace in generated["traces"]}
    choices = [("a", 1, "eight", "eight", "in-phase"), ("a", 2, "eight", "offset-two", "other-offset"),
        ("b", 1, "eight", "half-cycle", "half-cycle"), ("b", 2, "eight", "twelve-triangle", "incompatible periods"),
        ("a", 3, "eight", "noise-a", "one constructed rhythm"), ("b", 3, "noise-a", "noise-b", "constructed negatives"),
        ("c", 1, "eight", "eight", "recorded gap"), ("c", 2, "insufficient", "insufficient", "insufficient observations")]
    observations, summary, truth = [], [], []
    for index, (movie, identity, first, second, purpose) in enumerate(choices):
        reference, target = traces[movie, first], traces[movie, second]
        for frame, hour in enumerate(hours):
            observations.append({"stem": movie, "identity": identity, "frame_index": frame, "hours": hour,
                                 "signal": reference["values"][frame], "other": target["values"][frame]})
        summary.append({"stem": movie, "identity": identity, "size_example": 10. + index * 3})
        truth.append({"movie": movie, "identity": identity, "reference_scenario": first, "target_scenario": second, "purpose": purpose})
    tables = output / "run/pooled/tables"
    tables.mkdir(parents=True)
    pd.DataFrame(observations).to_csv(tables / "cell_frame.csv", index=False)
    pd.DataFrame(summary).to_csv(tables / "cell_summary.csv", index=False)
    _write_json(output / "run/manifest.json", {"synthetic": True,
        "description": "Controlled Workbench-generated verification traces; not biological observations", "movies": [{"stem": movie, "modules": []} for movie in ("a", "b", "c")]})
    _write_json(output / "truth.json", {"biological_result": False, "cells": truth,
        "interpretation": "Generating cases exercise the workflow. A constructed negative does not guarantee a nonsignificant realization."})
    _write_json(output / "discovery.json", declaration())
    _write_json(output / "presentation.json", {"report": {"title": "Synthetic rhythm-discovery verification"},
        "timing_relationships": {"text": {"subtitle": "Controlled generated traces; no biological findings"}}})
    _write_json(output / "fixture.json", {"files": {str(path.relative_to(output)): file_hash(path)
        for path in sorted(output.rglob("*")) if path.is_file()}})
    return output


def verify(output):
    """Use the real command, then prove reuse with scientific operations disabled."""
    import pandas as pd
    import pymicroglia.workbench as circadian
    import pymicroglia.measure.contrast_statistics as contrasts
    from pymicroglia.pipelines import parse
    from tests.pipelines.audit.demo import _latest, check_links
    from pymicroglia.pipelines.rhythm.discovery import resolve_request, run_request
    from pymicroglia.pipelines.rhythm.index import openable_report
    output = Path(output).resolve()
    fixture = read_document(output / "fixture.json")
    for name, fingerprint in fixture["files"].items():
        if file_hash(output / name) != fingerprint: raise ValueError("Prepared fixture changed: " + name)
    command = [sys.executable, "-m", "analysis", "pipeline", str(output / "run"), "--request", str(output / "discovery.json"),
        "--presentation", str(output / "presentation.json"), "--out", str(output / "pipeline")]
    invoke(command, check=True)
    root = output / "pipeline/synthetic-discovery"
    record, saved = _latest(root)
    screen = read_table(saved["rhythm-screen"].artifact("rhythm_results"))
    assert len(screen) == 16 and len(screen[["source_run", "movie", "identity"]].drop_duplicates()) == 8
    assert set(screen.family_requested) == {16}
    timing = read_table(saved["within-cell-timing"].artifact("pairs"))
    assert len(timing) == 8
    assert timing.loc[timing.movie.eq("b") & timing.identity.eq(2), "status"].iloc[0] == "ineligible"
    assert screen.loc[screen.movie.eq("c") & screen.identity.eq(2), "status"].eq("untestable").all()
    request = parse([read_document(output / "discovery.json")])[0]
    paths = {path.stem: path for path in (output / "run/pooled/tables").glob("*.csv")}
    resolved = resolve_request(request, source_run=source_identity(output / "run"),
        tables={name: pd.read_csv(path) for name, path in paths.items()}, input_hashes={name: file_hash(path) for name, path in paths.items()})
    appearance = read_document(output / "presentation.json")
    with ExitStack() as stack:
        def forbidden(*a, **k): raise AssertionError("Saved rerender attempted scientific calculations")
        for name in ("estimate_one", "adjust_pvalues", "rhythm_pair_timing", "rhythm_timing_summary", "rhythm_detection_agreement", "normalize_trace"):
            stack.enter_context(patch.object(circadian, name, forbidden))
        stack.enter_context(patch.object(contrasts, "evaluate_contrast_rows", forbidden))
        redrawn = run_request(resolved, paths, root, presentation={**appearance, "overview": {"overview_columns": 1}},
            only=("screening-overview",))
        assert redrawn.successful and redrawn.results["rhythm-screen"].outcome.status == "reused"
        assert redrawn.results["screening-overview"].outcome.status in {"completed", "reused"}
        refreshed = run_request(resolved, paths, root, presentation=appearance, only=("linked-results-index",))
    assert refreshed.successful, {key: value.outcome.reason for key, value in refreshed.results.items()}
    for name in ("rhythm-screen", "group-comparisons", "detection-agreement", "within-cell-timing", "timing-across-samples"):
        assert refreshed.results[name].outcome.status == "reused"
    report = refreshed.results["linked-results-index"]
    navigation = read_document(report.artifact("navigation.json"))
    proof = {"biological_result": False, "command": command[2:], "execution": str(record), "reopened_execution": str(refreshed.record_path),
        "changed_display_execution": str(redrawn.record_path), "index": str(openable_report(report)),
        "source_screen": saved["rhythm-screen"].outcome.scientific_id,
        "scientific_calls_disabled_for_reopen": True, "cells": len(navigation["cells"]), "pairs": len(navigation["pairs"]),
        "figures": len(navigation["pages"]), "verified_links": check_links(report.root),
        "states": {name: item.outcome.status for name, item in refreshed.results.items()}}
    _write_json(output / "verification.json", proof)
    print(json.dumps(proof, indent=2))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if not args.verify_existing: prepare(args.output)
    if args.verify or args.verify_existing: verify(args.output)
    else: print("Prepared synthetic rhythm-discovery inputs:", args.output.resolve())


if __name__ == "__main__": main()
