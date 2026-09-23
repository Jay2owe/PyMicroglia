"""One reserved check of an already saved choice; never a second selection round."""
from pymicroglia._results import read_document

from datetime import datetime, timezone
from pathlib import Path
import json
import os

import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.audit.benchmarks import Benchmarks, TRACE_COLUMNS, read_benchmarks, evaluation_inputs
from pymicroglia.pipelines.audit.options import CONFIRMATION_POPULATION
from pymicroglia.pipelines.audit.evaluator import evaluate_candidate, save_evaluations
from pymicroglia.pipelines.audit.scoring import SCORE_COLUMNS, _finite, score_cases, summarize
from pymicroglia.pipelines.audit.selection import assess_requirements, confirmation_checks, read_selection
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


def guard_selection(resolved, selection, development, assessments):
    """Validate the complete frozen instruction before any reserved access."""
    if selection["audit_id"] != resolved.scientific_id or selection["policy"] != resolved.request.score_policy.as_dict():
        raise ValueError("Confirmation settings or policy changed after the choice was frozen")
    current = {c.candidate_id: {k: v for k, v in c.as_dict().items() if k != "labels"} for c in resolved.candidates}
    frozen = {c["candidate_id"]: {k: v for k, v in c.items() if k != "labels"} for c in selection["candidates"]}
    if frozen != current or selection["stability_policy"] != resolved.request.stability.as_dict():
        raise ValueError("Confirmation requires the exact frozen candidate recipes and stability policy")
    if (selection["case_manifest_id"] != development.manifest["manifest_id"]
            or selection["reservation_id"] != development.reservation["reservation_id"]):
        raise ValueError("Frozen choice refers to a different benchmark or reservation")
    chosen = sorted({candidate for d in selection["decisions"] for candidate in d["candidate_ids"]})
    if chosen != selection["candidate_ids_to_confirm"] or set(chosen) - current.keys():
        raise ValueError("Confirmation candidate set differs from the frozen choices")
    checks = confirmation_checks(selection["decisions"], assessments, selection["policy"])
    if checks != selection["confirmation_checks"]:
        raise ValueError("Frozen confirmation thresholds differ from their development evidence")
    if selection.get("confirmation_population") != CONFIRMATION_POPULATION:
        raise ValueError("Confirmation requires the predeclared fresh-evidence population")
    return chosen


def claim_reservation(root, selection, reservation):
    """Claim all reserved streams before generating any of them, allowing exact resume."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (reservation["reservation_id"] + ".json")
    immutable = {"schema_version": 1, "reservation_id": reservation["reservation_id"],
        "selection_id": selection["selection_id"], "audit_id": selection["audit_id"],
        "candidate_ids": selection["candidate_ids_to_confirm"],
        "rule": "All reserved streams are consumed by this frozen choice; only an exact resume may reopen them"}
    try:
        with path.open("x", encoding="utf-8") as handle:
            record = {**immutable, "opened_utc": datetime.now(timezone.utc).isoformat()}
            json.dump(record, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        try:
            record = read_document(path)
        except (OSError, ValueError) as error:
            raise ValueError("Reserved-case consumption history is unreadable; it cannot be treated as unused") from error
        if any(record.get(key) != value for key, value in immutable.items()):
            raise ValueError("These reserved cases were already opened for another frozen choice; declare a new development seed/design")
    return record


def consumption_root(context):
    """Share history across audit output folders beside their original input tables."""
    parents = [str(Path(path).resolve().parent) for path in context.table_paths.values()]
    if not parents:
        raise ValueError("Confirmation requires original table locations for its consumption history")
    root = Path(os.path.commonpath(parents))
    if root == Path(root.anchor):
        raise ValueError("Original tables need a common run folder for reserved-case history")
    return root / ".pipeline-confirmation"


def generate_reserved(resolved, development):
    """Generate the same frozen design using its independent reserved streams."""
    import pymicroglia.workbench as circadian

    declaration = resolved.request.benchmark_design.as_dict()
    design = {key: declaration[key] for key in ("replicates", "truth_policy", "scenarios")}
    cases, traces, batches = [], [], []
    for profile in resolved.profiles:
        generated = circadian.generate_benchmark_cases(design, [profile.as_dict()],
            partition="confirmation", seed=declaration["seed"])
        metadata = profile["metadata"]
        for case in generated["cases"]:
            cases.append({**case, "source_profile_id": profile["id"],
                "measurement": metadata["measurement"], "movie": metadata["cell"]["movie"],
                "identity": metadata["cell"]["identity"],
                "family_block": content_id({"partition": "confirmation", "scenario": case["scenario_id"], "replicate": case["replicate"]})})
        traces.extend(generated["traces"])
        batches.append({"profile": profile["id"], "design_id": generated["design_id"],
            "resolved_design": generated["design"], "resolved_profiles": generated["profiles"],
            "provenance": generated["provenance"], "assumptions": generated["assumptions"],
            "workbench_run_record": generated["workbench_run_record"]})
    frame = pd.DataFrame(cases) if cases else development.cases.iloc[:0].copy()
    manifest = {"schema_version": 1, "partition": "confirmation", "input_design": declaration,
        "case_count": len(frame), "case_ids": sorted(frame.case_id), "batches": batches,
        "reservation_id": development.reservation["reservation_id"], "confirmation_opened": True,
        "scope_note": "Fresh simulation evidence under the declared design; original recordings remain exploratory."}
    manifest["manifest_id"] = content_id(manifest)
    reserved = Benchmarks(frame, pd.DataFrame(traces, columns=TRACE_COLUMNS), manifest, development.reservation)
    validate_separation(reserved, development)
    return reserved


def validate_separation(reserved, development):
    """Verify distinct ids, random streams and actual fresh observation content."""
    cases = reserved.cases
    if not cases.partition.eq("confirmation").all() or cases.case_id.duplicated().any():
        raise ValueError("Reserved cases need unique confirmation identities")
    key = lambda row: (row["source_profile_id"], row["scenario"], row["replicate"])
    old = {key(row): row for row in development.cases.to_dict("records")}
    new = {key(row): row for row in cases.to_dict("records")}
    expected = {(k["profile"], k["scenario"], k["replicate"]) for k in development.reservation["case_keys"]}
    if set(new) != expected or len(new) != len(cases) or set(old) != expected:
        raise ValueError("Reserved case keys differ from the frozen design")
    if set(cases.case_id) & set(development.cases.case_id):
        raise ValueError("Confirmation reuses development case identities")
    original_streams = {tuple(row["seed_entropy"]) for row in old.values()}
    original_content = set(development.cases.observation_sha256)
    fresh_content = set()
    for case_key, case in new.items():
        previous = old[case_key]
        if tuple(case["seed_entropy"]) in original_streams:
            raise ValueError("Confirmation reuses a development random stream")
        for field in ("truth_policy_id", "targets", "components", "recovery_eligible", "negative_eligible", "weight"):
            if case[field] != previous[field]:
                raise ValueError("Confirmation changed frozen truth, eligibility or weights")
        if case["fresh_confirmation_eligible"]:
            digest = case["observation_sha256"]
            if digest in original_content or digest in fresh_content:
                raise ValueError("A purported fresh confirmation case repeats observed data")
            fresh_content.add(digest)


def assess_confirmation(selection, summary):
    """Check frozen thresholds without promoting a runner-up or resolving a tie."""
    checks, decisions = [], []
    for frozen in selection["confirmation_checks"]:
        chosen = summary[summary.candidate_id.eq(frozen["candidate_id"]) & summary.scope.eq(frozen["scope"]) & summary.facet.eq("overall")]
        if frozen["measurement"] is not None:
            chosen = chosen[chosen.measurement.eq(frozen["measurement"])]
        if len(chosen) > 1:
            raise ValueError("Confirmation has conflicting scope summaries")
        row = _json_value(chosen.iloc[0].to_dict()) if len(chosen) else {"candidate_id": frozen["candidate_id"]}
        assessed = assess_requirements(row, selection["policy"])
        lower, upper = _finite(row.get("recovery_lower")), _finite(row.get("recovery_upper"))
        required = frozen["minimum_recovery"]
        state = "insufficient" if lower is None or upper is None else "pass" if lower >= required else "fail" if upper < required else "insufficient"
        gates = assessed["gates"] + [{"requirement": "frozen_recovery", "state": state,
            "observed": {"lower": lower, "upper": upper}, "required": required,
            "reason": frozen["rule"]}]
        status = "failed" if any(g["state"] == "fail" for g in gates) else (
            "insufficient" if any(g["state"] == "insufficient" for g in gates) else "supported")
        checks.append({**frozen, "status": status, "gates": gates, "score": row})
    for decision in selection["decisions"]:
        rows = [c for c in checks if c["decision_id"] == decision["decision_id"]]
        status = "skipped" if not rows else "failed" if any(c["status"] == "failed" for c in rows) else (
            "insufficient" if any(c["status"] == "insufficient" for c in rows) else "supported")
        decisions.append({**decision, "confirmation_status": status,
            "final_state": "tie" if decision["state"] == "tie" else "confirmed_choice" if status == "supported" else decision["state"],
            "promotion_allowed": status == "supported" and decision["state"] == "provisional_choice",
            "confirmation_checks": rows, "no_replacement": True})
    return checks, decisions


def read_confirmation(saved, *, expected_selection=None):
    """Read the completed check, verifying its content and linked evidence only."""
    record = read_document(saved.artifact("confirmation_record"))
    if record.get("schema_version") != 1 or record.get("confirmation_id") != content_id(
        {k: v for k, v in record.items() if k != "confirmation_id"}
    ):
        raise ValueError("Saved confirmation identity changed")
    if expected_selection is not None and record["selection_id"] != expected_selection:
        raise ValueError("Saved confirmation checked a different frozen selection")
    actual = [ref.as_dict() for ref in saved.outcome.artifacts if ref.name != "confirmation_record"]
    if record["artifacts"] != actual:
        raise ValueError("Confirmation evidence references changed")
    for ref in record["artifacts"]:
        saved.artifact(ref["name"])
    return record


def produce_confirmation(context):
    saved = context.saved("candidate-shortlist")
    selection = read_selection(saved, expected_audit=context.request.scientific_id)
    development = read_benchmarks(context.saved("development-cases"))
    assessments = _json_value(read_table(saved.artifact("candidate_assessments")).to_dict("records"))
    chosen = guard_selection(context.request, selection, development, assessments)
    evaluations, frames, opened, reserved = [], [], None, None
    if chosen:
        opened = claim_reservation(consumption_root(context), selection, development.reservation)
        reserved = generate_reserved(context.request, development)
        inputs = evaluation_inputs(reserved)
        for candidate in context.request.candidates:
            if candidate.candidate_id in chosen:
                evaluation = evaluate_candidate(candidate, inputs)
                evaluations.append(evaluation)
                frames.append(score_cases(candidate, evaluation, reserved, context.request.request.benchmark_design["truth_policy"]))
    scored = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCORE_COLUMNS)
    scored["confirmation_eligible"] = scored.fresh_confirmation_eligible.eq(True)
    scored["confirmation_exclusion"] = scored.confirmation_eligible.map({True: "", False: "No fresh stochastic observed realization"})
    # Complete families were evaluated above. This predeclared reporting population
    # excludes deterministic cases from fresh evidence without changing correction.
    summary = summarize(scored[scored.confirmation_eligible], confidence=selection["policy"]["confidence"])
    checks, final = assess_confirmation(selection, summary)
    result = save_evaluations(context, evaluations)
    refs = list(result.artifacts)
    tables = [("case_scores", scored), ("score_summary", summary),
              ("confirmation_checks", pd.DataFrame(checks)), ("final_decisions", pd.DataFrame(final))]
    if reserved is not None:
        tables += [("benchmark_cases", reserved.cases), ("benchmark_traces", reserved.traces)]
    for name, frame in tables:
        path = context.output / f"{name}.json"
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    record = {"schema_version": 1, "selection_id": selection["selection_id"], "audit_id": selection["audit_id"],
        "reservation_id": selection["reservation_id"], "consumption": opened,
        "confirmation_opened": opened is not None, "candidate_ids": chosen,
        "benchmark_manifest": reserved.manifest if reserved else None, "final_decisions": final,
        "fresh_cases": int(reserved.cases.fresh_confirmation_eligible.sum()) if reserved else 0,
        "nonfresh_cases": int((~reserved.cases.fresh_confirmation_eligible).sum()) if reserved else 0,
        "family_compatibility": selection["family_compatibility"],
        "population": CONFIRMATION_POPULATION,
        "scope_note": "Conditional synthetic confirmation; real recordings used in selection remain exploratory",
        "no_retuning": True, "artifacts": [ref.as_dict() for ref in refs]}
    record = _json_value(record)
    record["confirmation_id"] = content_id(record)
    path = context.output / "confirmation_record.json"
    _write_json(path, record)
    refs.append(ArtifactRef("confirmation_record", path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, "completed", "Saved the single reserved check and unchanged choice lineage",
        tuple(refs), provenance=Settings({"confirmation_id": record["confirmation_id"], "selection_id": selection["selection_id"],
                                          "confirmation_opened": opened is not None, "no_retuning": True}))
