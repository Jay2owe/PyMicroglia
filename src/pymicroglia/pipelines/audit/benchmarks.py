"""Known-truth development cases and an unopened confirmation reservation."""
from pymicroglia._results import read_document

from dataclasses import dataclass
import json

import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table, write_table


@dataclass(frozen=True)
class Benchmarks:
    cases: pd.DataFrame
    traces: pd.DataFrame
    manifest: dict
    reservation: dict


CASE_COLUMNS = ["case_id", "profile_id", "source_profile_id", "scenario_id", "scenario",
    "partition", "replicate", "measurement", "movie", "identity", "family_block", "truth_kind",
    "recovery_eligible", "negative_eligible", "targets", "components", "observations", "span_hours",
    "eligibility_reasons", "weight", "fresh_confirmation_eligible", "observation_sha256", "trace_sha256"]
TRACE_COLUMNS = ["case_id", "hours", "values", "missing", "source_missing", "rejection_reason",
                 "latent_rhythm", "baseline_and_drift", "disturbance", "noise", "component_values"]


def reservation_for(resolved):
    """A manifest of future case keys; never generate or inspect their outcomes."""
    design = resolved.request.benchmark_design
    keys = [{"profile": profile["id"], "scenario": scenario["id"], "replicate": replicate}
            for profile in resolved.profiles for scenario in design["scenarios"]
            for replicate in range(design["replicates"])]
    record = {"schema_version": 1, "partition": "confirmation", "opened": False,
        "design_id": content_id({"design": design, "profiles": resolved.profiles}),
        "seed": design["seed"], "case_keys": keys,
        "access_rule": "Frozen shortlist artifact required before generation and evaluation",
        "freshness_rule": "Independent partition streams; deterministic or duplicate observations excluded",
        "generation_backend": "circadian_workbench.benchmark_cases"}
    return {**record, "reservation_id": content_id(record)}


def generate_development(resolved) -> Benchmarks:
    """Generate the frozen input-only design through the scientific gateway.

    Profile batches limit one backend call without changing its content-keyed
    realizations. Candidate choices and results are never passed to generation.
    """
    import pymicroglia.workbench as circadian

    declaration = resolved.request.benchmark_design.as_dict()
    design = {key: declaration[key] for key in ("replicates", "truth_policy", "scenarios")}
    cases, traces, batches = [], [], []
    for profile in resolved.profiles:
        generated = circadian.generate_benchmark_cases(design, [profile.as_dict()],
            partition="development", seed=declaration["seed"])
        metadata = profile["metadata"]
        for case in generated["cases"]:
            cases.append({**case, "source_profile_id": profile["id"],
                "measurement": metadata["measurement"], "movie": metadata["cell"]["movie"],
                "identity": metadata["cell"]["identity"],
                "family_block": content_id({"partition": "development", "scenario": case["scenario_id"],
                                             "replicate": case["replicate"]})})
        traces.extend(generated["traces"])
        batches.append({"profile": profile["id"], "design_id": generated["design_id"],
            "resolved_design": generated["design"], "resolved_profiles": generated["profiles"],
            "provenance": generated["provenance"], "assumptions": generated["assumptions"],
            "workbench_run_record": generated["workbench_run_record"]})
    frame = pd.DataFrame(cases) if cases else pd.DataFrame(columns=CASE_COLUMNS)
    traces = pd.DataFrame(traces, columns=TRACE_COLUMNS)
    manifest = {"schema_version": 1, "partition": "development", "case_count": len(frame),
        "input_design": declaration, "profiles": [p.as_dict() for p in resolved.profiles],
        "batches": batches, "case_ids": sorted(frame.case_id.to_list()),
        "truth_counts": {str(k): int(v) for k, v in frame.truth_kind.value_counts().items()},
        "real_negative_controls": False, "confirmation_opened": False,
        "scope_note": "Scenario amplitudes/noise are declared assumptions in each measurement's units; no calibration from fitted residuals.",
        "empty_reason": "No selected recording profiles" if frame.empty else ""}
    manifest["manifest_id"] = content_id(manifest)
    return Benchmarks(frame, traces, manifest, reservation_for(resolved))


def produce_development(context):
    from pymicroglia.pipelines._screening import read_verified_tables

    read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    result = generate_development(context.request)
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in (("benchmark_cases", result.cases), ("benchmark_traces", result.traces)):
        path = context.output / f"{name}.json"
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id,
                                columns=tuple(frame.columns)))
    for name, value in (("benchmark_manifest", result.manifest), ("confirmation_reservation", result.reservation)):
        path = context.output / f"{name}.json"
        _write_json(path, value)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved known-truth development cases and unopened confirmation reservation", tuple(refs),
        provenance=Settings({"case_count": len(result.cases), "partition": "development",
                             "confirmation_opened": False}))


def read_benchmarks(saved) -> Benchmarks:
    """Validate saved case/trace identities without scientific generation."""
    cases = read_table(saved.artifact("benchmark_cases"))
    traces = read_table(saved.artifact("benchmark_traces"))
    manifest = read_document(saved.artifact("benchmark_manifest"))
    reservation = read_document(saved.artifact("confirmation_reservation"))
    if manifest.get("schema_version") != 1 or reservation.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark manifest schema")
    if cases.case_id.duplicated().any() or traces.case_id.duplicated().any():
        raise ValueError("Saved benchmark case ids are duplicated")
    if sorted(cases.case_id) != sorted(traces.case_id) or sorted(cases.case_id) != manifest["case_ids"]:
        raise ValueError("Saved benchmark inventory and trace ids disagree")
    if manifest["partition"] != "development" or not cases.partition.eq("development").all():
        raise ValueError("Development evidence contains another partition")
    if manifest["confirmation_opened"] or reservation["opened"]:
        raise ValueError("Development producer must not open confirmation")
    if manifest["case_count"] != len(cases):
        raise ValueError("Saved benchmark count disagrees with its inventory")
    if reservation["reservation_id"] != content_id({k: v for k, v in reservation.items() if k != "reservation_id"}):
        raise ValueError("Saved confirmation reservation identity disagrees")
    if manifest["manifest_id"] != content_id({k: v for k, v in manifest.items() if k != "manifest_id"}):
        raise ValueError("Saved benchmark manifest identity disagrees")
    return Benchmarks(cases, traces, manifest, reservation)


def evaluation_inputs(benchmarks):
    """Translate saved benchmark rows into the same evaluator input as real traces."""
    traces = {row["case_id"]: row for row in benchmarks.traces.to_dict("records")}
    inputs = []
    for case in benchmarks.cases.to_dict("records"):
        trace = traces[case["case_id"]]
        inputs.append(Settings({key: case[key] for key in (
            "case_id", "profile_id", "measurement", "movie", "identity", "family_block")} |
            {"hours": trace["hours"], "values": trace["values"], "metadata": {"truth": case}}))
    return tuple(inputs)
