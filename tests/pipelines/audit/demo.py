"""Reproducible synthetic audit integration exercise, never a biological default.

Prepare: python -m analysis.pipelines.audit_demo OUTPUT
Run and verify every stage: append --verify. Saved figures use the registered
plot-that exporters. The intentionally permissive policy exercises reserved
confirmation with few cases; it must not be used to choose biological settings.
"""

from __future__ import annotations
from pymicroglia._results import read_document, output_files

import argparse
from contextlib import ExitStack
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from tests.pipelines.invoke import invoke, source_identity, write_json as _write_json
import sys
import tempfile
from unittest.mock import patch

from pymicroglia.pipelines._screening import file_hash, read_table, read_screen


def declaration():
    return {"pipeline": "method-selection-audit", "name": "synthetic-audit",
        "measurements": ["signal", "other"], "biological_samples": {"a": "synthetic-A", "b": "synthetic-B"},
        "correction_scope": "all", "analysis_options": {"detrend": "none"},
        "candidates": [
            {"label": "Unfiltered Lomb estimate / F test", "analysis_options": {
                "fit_method": "lomb", "significance_method": "f", "detrend": "none"}},
            {"label": "Median, linear detrend, MESA estimate / F test", "filter": {
                "method": "median", "window_hours": 3., "max_gap_hours": 1.5},
             "analysis_options": {"fit_method": "mesa", "significance_method": "f", "detrend": "linear"}}],
        "benchmark_design": {"replicates": 1, "seed": 4815,
            "justification": "Synthetic integration exercise: broad periods, non-sinusoidal and multiple rhythms, gaps, negatives and insufficient data. Policy deliberately permissive to exercise confirmation; not a recommended biological audit.",
            "truth_policy": {"min_observations": 24, "min_cycles": 3, "period_min_hours": 2.,
                "period_max_hours": 48., "relative_tolerance": .1, "absolute_tolerance_hours": .5,
                "target": "all", "extra_components": "penalize"},
            "scenarios": [
                {"id": "triangle-12h", "components": [{"id": "short", "period_hours": 12., "amplitude": 1., "waveform": "triangle"}],
                 "noise": {"kind": "white", "sd": .2}},
                {"id": "square-36h", "components": [{"id": "long", "period_hours": 36., "amplitude": 1., "waveform": "square", "duty": .2}],
                 "noise": {"kind": "exponential", "sd": .2, "correlation_hours": 3.}, "drift": {"linear_per_hour": .005}},
                {"id": "two-components", "components": [
                    {"id": "short", "period_hours": 6., "amplitude": 1., "waveform": "sawtooth"},
                    {"id": "long", "period_hours": 30., "amplitude": .5, "waveform": "cosine"}]},
                {"id": "negative", "components": [], "noise": {"kind": "white", "sd": .5}},
                {"id": "disturbed-negative", "components": [], "noise": {"kind": "white", "sd": .5},
                 "disturbances": [{"kind": "pulse", "start_hours": 30., "duration_hours": 1., "amplitude": 2.}]},
                {"id": "insufficient", "components": [{"id": "short", "period_hours": 12., "amplitude": 1., "waveform": "triangle"}],
                 "retain_observations": 8}]},
        "score_policy": {"false_alarm_limit": 1., "confidence": .5, "min_positive": 1,
            "min_negative": 1, "min_valid_fraction": .1, "recovery_margin": .03,
            "uncertainty": "stratified_independent_realizations"},
        "stability": {"alterations": [{"start_fraction": .45, "end_fraction": .55}]}}


def prepare(output):
    """Create a new original-table fixture with the public Workbench generator."""
    import pymicroglia.workbench as circadian
    import pandas as pd

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepare requires a new empty folder; use --verify-existing to resume")
    output.mkdir(parents=True, exist_ok=True)
    request = declaration()
    design = {key: request["benchmark_design"][key] for key in ("replicates", "truth_policy", "scenarios")}
    design["scenarios"] = design["scenarios"][:2]
    profiles = [{"id": movie, "hours": [float(h) for h in range(160)],
                 "missing": [h in (20, 21) for h in range(160)], "metadata": {"synthetic_movie": movie}}
                for movie in ("a", "b")]
    generated = circadian.generate_benchmark_cases(design, profiles, partition="development", seed=7241)
    _write_json(output / "source_generation.json", generated)
    cases = {case["case_id"]: case for case in generated["cases"]}
    rows = {}
    for trace in generated["traces"]:
        case = cases[trace["case_id"]]
        metric = "signal" if case["scenario"] == "triangle-12h" else "other"
        for frame, (hours, value) in enumerate(zip(trace["hours"], trace["values"])):
            key = (case["profile"], frame)
            rows.setdefault(key, {"stem": case["profile"], "identity": 1, "frame_index": frame, "hours": hours})[metric] = value
    table = output / "run" / "pooled" / "tables" / "cell_frame.csv"
    table.parent.mkdir(parents=True)
    pd.DataFrame(rows.values()).to_csv(table, index=False)
    _write_json(output / "run" / "manifest.json", {"synthetic": True,
        "description": "Integration data generated by Circadian Workbench; not biological observations",
        "movies": [{"stem": movie, "modules": []} for movie in ("a", "b")]})
    _write_json(output / "audit.json", request)
    _write_json(output / "presentation.json", {
        "performance": {"text": {kind: {"title": "Synthetic verification: " + kind,
            "subtitle": "Deliberately permissive test policy; not biological calibration"} for kind in
            ("performance", "periods", "decisions", "disagreement")}},
        "focus": {"audit_examples_per_reason": 1, "text": {"title": "Synthetic verification: saved trace evidence",
            "subtitle": "Deliberately permissive test policy; not biological calibration"}},
        "report": {"title": "Synthetic audit verification — deliberately permissive test policy"}})
    return output


def _cli(*arguments):
    command = [sys.executable, "-m", "analysis", *map(str, arguments)]
    invoke(command, check=True)
    return command[2:]


def _latest(output):
    from pymicroglia.pipelines._contracts import result_from_dict
    from pymicroglia.pipelines._runner import SavedResult

    from pymicroglia.pipelines import _records
    path = _records.path(output)
    key = max(_records.read(output)["invocations"], key=lambda key: _records.read(output)["invocations"][key]["started"])
    record = _records.invocation(output, key)
    if not record["successful"]:
        raise ValueError(f"Incomplete pipeline execution: {path}")
    return path, {row["step"]: SavedResult(output / row["result_directory"], result_from_dict(row["result"])) for row in record["steps"]}


def check_links(folder):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links, self.ids = [], set()
        def handle_starttag(self, tag, attrs):
            self.links.extend(v for k, v in attrs if k in {"href", "src"})
            self.ids.update(v for k, v in attrs if k == "id")
    from urllib.parse import unquote, urlsplit

    parser = Links()
    parser.feed((folder / "index.html").read_text(encoding="utf-8"))
    for link in parser.links:
        target = urlsplit(link)
        if target.scheme or target.netloc:
            raise ValueError("Portable report unexpectedly depends on a remote link")
        if target.path:
            path = (folder / unquote(target.path)).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file():
                raise ValueError(f"Missing or escaping report link: {link}")
        elif target.fragment and unquote(target.fragment) not in parser.ids:
            raise ValueError(f"Missing report anchor: {link}")
    return len(parser.links)


def verify(output):
    """Use the documented commands, then rebuild with scientific calls disabled."""
    import pymicroglia.workbench as circadian
    from pymicroglia.pipelines import parse
    from pymicroglia.pipelines.audit.options import resolve_request
    from pymicroglia.pipelines.audit.workflow import run_request
    from pymicroglia.pipelines.audit.confirmation import read_confirmation
    import pandas as pd

    output = Path(output).resolve()
    commands = [_cli("pipeline", output / "run", "--request", output / "audit.json",
        "--presentation", output / "presentation.json", "--out", output / "pipelines")]
    result_root = output / "pipelines" / "synthetic-audit"
    execution, saved = _latest(result_root)
    confirmation = read_confirmation(saved["independent-confirmation"])
    if not confirmation["confirmation_opened"] or confirmation["fresh_cases"] < 1:
        raise ValueError("Integration fixture did not exercise fresh confirmation")
    real = read_table(saved["real-candidates"].artifact("results"))
    if set(real.measurement) != {"signal", "other"} or set(real.movie) != {"a", "b"}:
        raise ValueError("Real evaluation lost movie or measurement identities")
    summary = read_table(saved["development-scores"].artifact("score_summary"))
    if summary.recovery_lower.notna().sum() == 0 or summary.false_alarm_upper.notna().sum() == 0:
        raise ValueError("Required quantitative scores are unavailable")
    report = saved["audit-index"].root
    links = check_links(report)
    with tempfile.TemporaryDirectory(prefix="motion-audit-portable-") as temporary:
        moved = Path(temporary) / "report"
        shutil.copytree(report, moved)
        if check_links(moved) != links:
            raise ValueError("Moved report links changed")
    selection = read_document(saved["candidate-shortlist"].artifact("frozen_selection"))
    choices = {metric: candidate["candidate_id"] for metric, candidate in zip(("signal", "other"), selection["candidates"])}
    _write_json(output / "choices.json", choices)
    commands.append(_cli("pipeline-export", report, "--choices", output / "choices.json",
                         "--override-reason", "Mixed-recipe integration verification; no biological recommendation or equivalent joint calibration is claimed"))
    exported = [path for path in output_files(report / "exports") if path.suffix == ".json"]
    profiles = [p for p in exported if {metric: recipe["candidate_id"] for metric, recipe in
                read_document(p)["measurement_recipes"].items()} == choices]
    if len(profiles) != 1:
        raise ValueError("Expected exactly one exported profile matching the verification choices")
    profile_path = profiles[0]
    _write_json(output / "discovery.json", {"pipeline": "rhythm-discovery", "name": "synthetic-discovery",
        "test_measurements": ["signal", "other"], "settings_profile": str(profile_path),
        "biological_samples": {"a": "synthetic-A", "b": "synthetic-B"}})
    commands.append(_cli("pipeline", output / "run", "--request", output / "discovery.json", "--out", output / "pipelines", "--step", "rhythm-screen"))
    consumer_execution, consumer = _latest(output / "pipelines" / "synthetic-discovery")
    screen = read_screen(consumer["rhythm-screen"].root)
    if len(screen.results) != 4 or set(screen.results.candidate_id) != set(choices.values()):
        raise ValueError("Shared discovery did not apply both complete recipes")
    for row in screen.results.to_dict("records"):
        if row["candidate_id"] != choices[row["measurement"]] or row["significance_method"] != "f":
            raise ValueError("Shared discovery substituted a chosen recipe")
    paths = {"cell_frame": output / "run" / "pooled" / "tables" / "cell_frame.csv"}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    source_hash = source_identity(output / "run")
    resolved = resolve_request(parse([read_document(output / "audit.json")])[0],
        source_run=source_hash, tables={k: pd.read_csv(p) for k, p in paths.items()}, input_hashes=hashes)
    scientific = {step: [(ref.path, ref.sha256) for ref in result.outcome.artifacts]
                  for step, result in saved.items() if step not in {"performance-figures", "focused-pages", "audit-index"}}
    ledger = {str(p): file_hash(p) for p in (paths["cell_frame"].parent / ".pipeline-confirmation").rglob("*") if p.is_file()}
    presentation = read_document(output / "presentation.json")
    presentation["report"]["title"] += " — saved evidence replay"
    def forbidden(*args, **kwargs):
        raise AssertionError("Presentation replay attempted a scientific operation")
    with ExitStack() as stack:
        for name in ("generate_benchmark_cases", "filter_rhythm_trace", "omit_rhythm_intervals", "estimate_one",
                     "estimate_trace", "estimate_grouped_rhythms", "adjust_pvalues", "benchmark_score_interval"):
            stack.enter_context(patch.object(circadian, name, forbidden))
        replay = run_request(resolved, paths, result_root, presentation=presentation)
    if not replay.successful:
        raise ValueError("Saved-only replay failed: " + str({k: v.outcome.reason for k, v in replay.results.items()}))
    for step, artifacts in scientific.items():
        result = replay.results[step]
        if result.outcome.status != "reused" or [(r.path, r.sha256) for r in result.outcome.artifacts] != artifacts:
            raise ValueError("Presentation replay changed scientific evidence")
    after_ledger = {str(p): file_hash(p) for p in (paths["cell_frame"].parent / ".pipeline-confirmation").rglob("*") if p.is_file()}
    if after_ledger != ledger or any(file_hash(paths[k]) != v for k, v in hashes.items()):
        raise ValueError("Presentation replay changed reservation or original inputs")
    verification = {"synthetic": True, "deliberately_permissive_policy": True,
        "workbench_version": circadian.WORKBENCH_VERSION, "commands": commands, "audit_execution": str(execution),
        "replay_execution": str(replay.record_path), "consumer_execution": str(consumer_execution),
        "report": str(report), "replayed_report": str(replay.results["audit-index"].root), "relative_links": links,
        "fresh_confirmation_cases": confirmation["fresh_cases"], "nonfresh_confirmation_cases": confirmation["nonfresh_cases"],
        "scientific_steps_reused": sorted(scientific), "shared_screen_results": len(screen.results),
        "profile": str(profile_path), "figures": {step: str(saved[step].root) for step in ("performance-figures", "focused-pages")}}
    _write_json(output / "verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Prepare a new fixture, run all stages and verify reuse and handoff")
    mode.add_argument("--verify-existing", action="store_true", help="Resume a prepared fixture without changing its design or inputs")
    args = parser.parse_args()
    if not args.verify_existing:
        prepare(args.output)
    if args.verify or args.verify_existing:
        verify(args.output)
    else:
        print("Prepared synthetic verification inputs:", args.output.resolve())


if __name__ == "__main__":
    main()
