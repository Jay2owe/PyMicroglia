"""Frozen development-only admissibility and paired recovery decisions."""
from pymicroglia._results import read_document

import json

import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.audit.benchmarks import read_benchmarks
from pymicroglia.pipelines.audit.scoring import _finite, rate_units
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


def unique_rows(frame, keys):
    """Repeated display copies are harmless; conflicting scientific rows are not."""
    result = {}
    for raw in frame.to_dict("records"):
        row = _json_value(raw)
        key = tuple(row[name] for name in keys)
        if key in result and result[key] != row:
            raise ValueError("Conflicting saved evidence for " + str(key))
        result[key] = row
    return list(result.values())


def assess_requirements(row, policy):
    """Keep failure, insufficient support and acceptance separate for each rule."""
    gates = []
    def add(name, state, observed, required, reason):
        gates.append({"requirement": name, "state": state, "observed": observed,
                      "required": required, "reason": reason})
    for kind, minimum in (("recovery", policy["min_positive"]), ("false_alarm", policy["min_negative"])):
        support = row.get(kind + "_independent_units", 0)
        denominator = row.get(kind + "_denominator", 0)
        add(kind + "_support", "pass" if support >= minimum and denominator >= minimum else "insufficient",
            {"independent_units": support, "eligible_cases": denominator}, minimum,
            "Requires both eligible cases and independent complete-family realizations")
        coverage = _finite(row.get(kind + "_weighted_valid_fraction"))
        add(kind + "_coverage", "insufficient" if coverage is None else "pass" if coverage >= policy["min_valid_fraction"] else "fail",
            coverage, policy["min_valid_fraction"], "Weighted eligible-case fraction with a usable selected significance test")
    upper, lower = _finite(row.get("false_alarm_upper")), _finite(row.get("false_alarm_lower"))
    limit = policy["false_alarm_limit"]
    state = "insufficient" if upper is None or lower is None else "pass" if upper <= limit else "fail" if lower > limit else "insufficient"
    add("false_alarm_control", state, {"lower": lower, "upper": upper}, limit,
        "Upper bound must be at or below the limit; an interval crossing it is inconclusive")
    overall = "inadmissible" if any(g["state"] == "fail" for g in gates) else (
        "insufficient" if any(g["state"] == "insufficient" for g in gates) else "admissible")
    return {"candidate_id": row["candidate_id"], "state": overall, "gates": gates,
            "score": row, "score_id": content_id(_json_value(row))}


def paired_recovery(left, right, policy):
    """Compare paired complete-family outcomes through the public uncertainty API."""
    import pymicroglia.workbench as circadian

    first = {u["id"]: u for u in left["recovery_units"]}
    second = {u["id"]: u for u in right["recovery_units"]}
    if len(first) != len(left["recovery_units"]) or len(second) != len(right["recovery_units"]) or first.keys() != second.keys():
        raise ValueError("Candidate comparison requires the same unique family realizations")
    units = []
    for key in sorted(first):
        a, b = first[key], second[key]
        if a["dependency_id"] != b["dependency_id"] or a["weight"] != b["weight"]:
            raise ValueError("Paired recovery requires matching source dependencies and fixed weights")
        units.append({"id": key, "dependency_id": a["dependency_id"],
                      "value": a["value"] - b["value"], "weight": a["weight"]})
    uncertainty = circadian.benchmark_score_interval(units, confidence=policy["confidence"], bounds=[-1., 1.])
    margin = policy["recovery_margin"]
    # The frozen overlap rule remains conservative even when pairing is precise.
    clear = (uncertainty["lower"] is not None and uncertainty["lower"] > margin
             and left["recovery_lower"] > right["recovery_upper"]
             and left["recovery_rate"] - right["recovery_rate"] > margin)
    reverse = (uncertainty["upper"] is not None and uncertainty["upper"] < -margin
             and right["recovery_lower"] > left["recovery_upper"]
             and right["recovery_rate"] - left["recovery_rate"] > margin)
    return {"left": left["candidate_id"], "right": right["candidate_id"], "units": units,
        "uncertainty": uncertainty, "clear_winner": left["candidate_id"] if clear else right["candidate_id"] if reverse else None,
        "rule": "nonoverlapping recovery intervals and a paired difference lower bound beyond the declared margin",
        "coverage_scope": "Each interval has its declared confidence; no simultaneous winner-probability claim"}


def decide_scope(candidates, summary, policy, *, scope, measurement):
    rows = unique_rows(summary, ["candidate_id"])
    by_id = {r["candidate_id"]: r for r in rows}
    if set(by_id) - {c.candidate_id for c in candidates}:
        raise ValueError("Saved summary includes an undeclared candidate")
    assessments = [assess_requirements(by_id.get(c.candidate_id, {"candidate_id": c.candidate_id}), policy)
                   for c in sorted(candidates, key=lambda c: c.candidate_id)]
    allowed = [a for a in assessments if a["state"] == "admissible"]
    comparisons, dominated = [], set()
    for i, left in enumerate(allowed):
        for right in allowed[i + 1:]:
            comparison = paired_recovery(left["score"], right["score"], policy)
            comparisons.append(comparison)
            if comparison["clear_winner"]:
                dominated.add(right["candidate_id"] if comparison["clear_winner"] == left["candidate_id"] else left["candidate_id"])
    survivors = sorted(a["candidate_id"] for a in allowed if a["candidate_id"] not in dominated)
    if not allowed:
        state = "no_acceptable_candidate" if assessments and all(a["state"] == "inadmissible" for a in assessments) else "insufficient_evidence"
        reason = "Every candidate fails at least one required gate" if state == "no_acceptable_candidate" else "No candidate has enough evidence to pass every gate"
    elif len(survivors) == 1:
        state, reason = "provisional_choice", "One admissible recipe remains under the frozen recovery comparison rule"
    else:
        state, reason = "tie", "Recovery evidence does not establish one better admissible recipe"
    decision = {"scope": scope, "measurement": measurement, "state": state, "reason": reason,
        "candidate_ids": survivors, "admissible_ids": sorted(a["candidate_id"] for a in allowed),
        "excluded_ids": sorted(a["candidate_id"] for a in assessments if a["state"] != "admissible"),
        "score_ids": {a["candidate_id"]: a["score_id"] for a in assessments},
        "stability_role": "separate sensitivity diagnostic; never an arbitrary weighted correctness score"}
    decision["decision_id"] = content_id(decision)
    for row in assessments + comparisons:
        row.update(scope=scope, measurement=measurement, decision_id=decision["decision_id"])
    return decision, assessments, comparisons


def select_candidates(resolved, summary, stability):
    """One scope-wide decision per measurement, or one explicitly requested dataset."""
    summaries = summary[summary.facet.eq("overall")]
    scopes = [("dataset", None)] if resolved.request.recommendation_scope == "dataset" else [
        ("measurement", measurement.column) for measurement in resolved.source.test_measurements]
    decisions, assessments, comparisons = [], [], []
    for scope, metric in scopes:
        chosen = summaries[summaries.scope.eq(scope)]
        if metric is not None:
            chosen = chosen[chosen.measurement.eq(metric)]
        decision, gates, paired = decide_scope(resolved.candidates, chosen, resolved.request.score_policy, scope=scope, measurement=metric)
        diagnostics = stability[stability.scope.eq("measurement")]
        if metric is not None:
            diagnostics = diagnostics[diagnostics.measurement.eq(metric)]
        decision["stability"] = unique_rows(diagnostics, ["candidate_id", "measurement"])
        decision["decision_id"] = content_id({k: v for k, v in decision.items() if k != "decision_id"})
        for row in gates + paired:
            row["decision_id"] = decision["decision_id"]
        decisions.append(decision)
        assessments.extend(gates)
        comparisons.extend(paired)
    return decisions, assessments, comparisons


def family_compatibility(resolved, decisions):
    if not all(d["state"] == "provisional_choice" for d in decisions):
        return {"state": "unresolved_choices", "equivalent_calibration": False}
    selected = {d["candidate_ids"][0] for d in decisions}
    joint = any(c.correction_scope == "all" for c in resolved.candidates if c.candidate_id in selected)
    matched = len(selected) == 1 or not joint
    return {"state": "matches_evaluated_families" if matched else "mixed_joint_profile_not_evaluated",
        "equivalent_calibration": matched, "candidate_ids": sorted(selected),
        "reason": "Joint correction across measurements requires the exact combined recipe mixture to be checked" if not matched else "Selected recipes retain their evaluated correction families"}


def validate_development(resolved, benchmarks, scores, summary, stability):
    expected = {(candidate.candidate_id, case) for candidate in resolved.candidates for case in benchmarks.cases.case_id}
    if scores.duplicated(["candidate_id", "case_id"]).any() or set(zip(scores.candidate_id, scores.case_id)) != expected:
        raise ValueError("Selection needs the complete declared candidate/development-case inventory")
    if not scores.partition.eq("development").all() or benchmarks.reservation["opened"]:
        raise ValueError("Selection consumes only development evidence")
    candidate_ids = {c.candidate_id for c in resolved.candidates}
    if set(stability.candidate_id) != candidate_ids or set(summary.candidate_id) != candidate_ids:
        raise ValueError("Development scores and stability do not cover the declared candidates")
    for row in unique_rows(summary[summary.facet.eq("overall")], ["scope", "measurement", "candidate_id"]):
        population = scores[scores.candidate_id.eq(row["candidate_id"])]
        if row["scope"] == "measurement":
            population = population[population.measurement.eq(row["measurement"])]
        for name, eligible, outcome in (("recovery", "eligible_positive", "recovered"), ("false_alarm", "eligible_negative", "false_alarm")):
            if row[name + "_units"] != rate_units(population, eligible, outcome):
                raise ValueError("Saved score summary has a different case-family population")
    policies = unique_rows(stability, ["candidate_id", "measurement", "scope", "profile_id"])
    if any(row["policy"] != resolved.request.stability.as_dict() for row in policies):
        raise ValueError("Saved stability policy differs from the frozen audit")


def read_selection(saved, *, expected_audit=None):
    record = read_document(saved.artifact("frozen_selection"))
    if record.get("schema_version") != 1 or record.get("confirmation_opened") is not False:
        raise ValueError("A frozen, unopened selection manifest is required")
    if content_id({k: v for k, v in record.items() if k != "selection_id"}) != record.get("selection_id"):
        raise ValueError("Frozen selection identity changed")
    if expected_audit is not None and record["audit_id"] != expected_audit:
        raise ValueError("Frozen selection belongs to a different audit")
    return record


def confirmation_checks(decisions, assessments, policy):
    """Declare the acceptable recovery loss before opening reserved outcomes."""
    scores = {(a["scope"], a["measurement"], a["candidate_id"]): a for a in assessments}
    checks = []
    for decision in decisions:
        for candidate in decision["candidate_ids"]:
            assessment = scores[(decision["scope"], decision["measurement"], candidate)]
            lower = _finite(assessment["score"].get("recovery_lower"))
            if assessment["state"] != "admissible" or lower is None:
                raise ValueError("Only an admissible supported candidate can enter confirmation")
            checks.append({"decision_id": decision["decision_id"], "scope": decision["scope"],
                "measurement": decision["measurement"], "candidate_id": candidate,
                "development_score_id": assessment["score_id"], "development_recovery_lower": lower,
                "minimum_recovery": max(0., lower - policy["recovery_margin"]),
                "rule": "lower confirmation bound >= max(0, development lower bound - recovery_margin)"})
    return checks


def produce_selection(context):
    benchmarks = read_benchmarks(context.saved("development-cases"))
    score_result, stability_result = context.saved("development-scores"), context.saved("real-stability")
    scores = read_table(score_result.artifact("case_scores"))
    summary = read_table(score_result.artifact("score_summary"))
    stability = read_table(stability_result.artifact("stability_summary"))
    validate_development(context.request, benchmarks, scores, summary, stability)
    decisions, assessments, comparisons = select_candidates(context.request, summary, stability)
    record = {"schema_version": 1, "audit_id": context.request.scientific_id,
        "case_manifest_id": benchmarks.manifest["manifest_id"], "reservation_id": benchmarks.reservation["reservation_id"],
        "candidates": [c.as_dict() for c in context.request.candidates],
        "candidate_ids_to_confirm": sorted({candidate for d in decisions for candidate in d["candidate_ids"]}),
        "decisions": decisions, "policy": context.request.request.score_policy,
        "confirmation_checks": confirmation_checks(decisions, assessments, context.request.request.score_policy),
        "confirmation_population": context.request.request.score_policy["confirmation_population"],
        "stability_policy": context.request.request.stability, "confirmation_opened": False,
        "family_compatibility": family_compatibility(context.request, decisions),
        "confirmation_policy": "Check this exact shortlist once; ties remain ties and failed choices are never replaced",
        "input_results": {name: {"scientific_id": saved.outcome.scientific_id,
            "artifacts": [a.as_dict() for a in saved.outcome.artifacts]} for name, saved in context.dependencies.items()},
        "scope_note": "Conditional synthetic evidence. Original recordings helped select recipes; later discoveries on them remain exploratory."}
    record = _json_value(record)
    record["selection_id"] = content_id(record)
    context.output.mkdir(parents=True)
    refs = []
    for name, rows, columns in (("candidate_assessments", assessments, ["candidate_id", "state", "gates", "score", "score_id", "scope", "measurement", "decision_id"]),
        ("paired_recovery", comparisons, ["left", "right", "units", "uncertainty", "clear_winner", "rule", "coverage_scope", "scope", "measurement", "decision_id"]),
        ("provisional_decisions", decisions, None)):
        frame = pd.DataFrame(rows, columns=columns)
        path = context.output / f"{name}.json"
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    path = context.output / "frozen_selection.json"
    _write_json(path, record)
    refs.append(ArtifactRef("frozen_selection", path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, "completed", "Saved development gates and frozen provisional choices",
        tuple(refs), provenance=Settings({"selection_id": record["selection_id"], "confirmation_opened": False}))
