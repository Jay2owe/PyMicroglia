"""Known-outcome accounting; numerical uncertainty stays in Circadian Workbench."""

import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import file_hash, write_table
from pymicroglia.pipelines.audit.benchmarks import evaluation_inputs, read_benchmarks
from pymicroglia.pipelines.audit.evaluator import evaluate_candidate, save_evaluations


SCORING_DEFINITION = {
    "version": 1, "recovery": "eligible positive, detected, supported estimate, all declared targets recovered",
    "matching": "maximum one-to-one within tolerance, then minimum normalized absolute error",
    "dominant": "the selected main period must match the unique declared dominant target",
    "extra_components": "unmatched returned components follow the frozen ignore/penalize rule",
    "denominators": "all predeclared eligible cases, including unavailable/failed answers",
    "uncertainty": "weighted Hoeffding over complete correction-family replicates; shared sources grouped",
    "biological_validation": False,
}

SCORE_COLUMNS = ["candidate_id", "case_id", "profile_id", "source_profile_id", "scenario_id", "scenario",
    "partition", "replicate", "measurement", "movie", "identity", "family_block", "truth_kind", "weight",
    "span_hours", "observations", "fresh_confirmation_eligible", "observation_sha256", "dependency_id", "family_id",
    "eligible_positive", "eligible_negative", "valid_test", "detected", "period_available", "period_supported",
    "correct_period", "component_policy_pass", "recovered", "false_alarm", "period_error_hours", "period_alias",
    "estimated_period_hours", "component_matching", "component_significance", "test_status", "estimate_status",
    "insufficient_response", "periods", "waveforms", "noise", "p_value", "q_value"]


def _finite(value):
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def match_components(truth, predictions, policy):
    """Assign recorded periods to declared truth, never estimate a new rhythm."""
    count = len(truth)
    cost = np.full((count, len(predictions) + count), float(count + 1))
    tolerances = [max(policy["absolute_tolerance_hours"], c["period_hours"] * policy["relative_tolerance"])
                  for c in truth]
    for i, component in enumerate(truth):
        for j, prediction in enumerate(predictions):
            error = abs(prediction - component["period_hours"])
            cost[i, j] = (error / tolerances[i] if tolerances[i] else 0.) if error <= tolerances[i] else count + 2.
    rows, columns = linear_sum_assignment(cost)
    matches = [{"truth_index": int(i), "predicted_index": int(j), "truth_period_hours": truth[i]["period_hours"],
        "estimated_period_hours": predictions[j], "error_hours": predictions[j] - truth[i]["period_hours"],
        "tolerance_hours": tolerances[i]} for i, j in zip(rows, columns) if j < len(predictions) and cost[i, j] <= 1.]
    matched_truth = {row["truth_index"] for row in matches}
    matched_predictions = {row["predicted_index"] for row in matches}
    return {"matches": matches, "missed": [i for i in range(count) if i not in matched_truth],
            "spurious": [j for j in range(len(predictions)) if j not in matched_predictions]}


def _dependencies(cases):
    """Shared corrected evidence is clustered before any uncertainty calculation."""
    result = {}
    for block, group in cases.groupby("family_block", sort=True):
        sources = [{"profile": row["source_profile_id"], "observations": row["observation_sha256"],
                    "truth_policy": row["truth_policy_id"], "targets": row["targets"],
                    "components": row["components"]} for row in group.sort_values("source_profile_id").to_dict("records")]
        result[block] = content_id(sources)
    return result


def score_cases(candidate, evaluation, benchmarks, truth_policy):
    """One row for every requested case, with fixed truth and failure denominators."""
    if evaluation.results.case_id.duplicated().any():
        raise ValueError("Duplicate candidate/case evaluation rows")
    known = set(benchmarks.cases.case_id)
    if set(evaluation.results.case_id) - known:
        raise ValueError("Evaluation contains cases outside the frozen benchmark")
    outcomes = {row["case_id"]: row for row in evaluation.results.to_dict("records")}
    dependencies = _dependencies(benchmarks.cases)
    scenarios = {s["id"]: s for s in benchmarks.manifest["input_design"]["scenarios"]}
    scores = []
    for case in benchmarks.cases.to_dict("records"):
        row = outcomes.get(case["case_id"], {})
        period = _finite(row.get("period_hours"))
        p_value, q_value = _finite(row.get("p_value")), _finite(row.get("q_value"))
        valid_test = row.get("test_status") == "ok" and p_value is not None and q_value is not None and 0 <= p_value <= 1 and 0 <= q_value <= 1
        detected = valid_test and q_value < candidate.analysis_options["rhythmic_alpha"]
        available = bool(row.get("period_available", False)) and period is not None and period > 0
        supported = available and not bool(row.get("period_underdetermined", True))
        returned = row.get("components") if available else []
        returned = returned if isinstance(returned, list) else []
        predictions = [p for c in returned if (p := _finite(c.get("period_hours"))) is not None and p > 0]
        if available and not predictions:
            predictions = [period]
        matching = match_components(case["components"], predictions, truth_policy)
        matched_ids = {case["components"][m["truth_index"]]["id"] for m in matching["matches"]}
        targets = case["targets"]
        correct = bool(targets) and all(t["id"] in matched_ids for t in targets)
        error, alias = None, "unavailable" if not available else "multiple-targets"
        if len(targets) == 1:
            target = targets[0]
            error = period - target["period_hours"] if available else None
            main_correct = available and abs(error) <= target["tolerance_hours"]
            if truth_policy["target"] == "dominant":
                correct = main_correct
            alias = "within-tolerance" if main_correct else "other" if available else "unavailable"
            if available and not main_correct:
                for factor, label in ((.5, "half-period"), (2., "double-period")):
                    alternative = target["period_hours"] * factor
                    tolerance = max(truth_policy["absolute_tolerance_hours"], alternative * truth_policy["relative_tolerance"])
                    if abs(period - alternative) <= tolerance:
                        alias = label
                        break
        component_policy_pass = truth_policy["extra_components"] == "ignore" or not matching["spurious"]
        positive, negative = bool(case["recovery_eligible"]), bool(case["negative_eligible"])
        scenario = scenarios[case["scenario"]]
        scores.append({"candidate_id": candidate.candidate_id, **{key: case[key] for key in (
            "case_id", "profile_id", "source_profile_id", "scenario_id", "scenario", "partition", "replicate",
            "measurement", "movie", "identity", "family_block", "truth_kind", "weight", "span_hours",
            "observations", "fresh_confirmation_eligible", "observation_sha256")},
            "dependency_id": dependencies[case["family_block"]], "family_id": row.get("family_id"),
            "eligible_positive": positive, "eligible_negative": negative, "valid_test": valid_test,
            "detected": detected, "period_available": available, "period_supported": supported,
            "correct_period": correct, "component_policy_pass": component_policy_pass,
            "recovered": positive and detected and supported and correct and component_policy_pass,
            "false_alarm": negative and detected, "period_error_hours": error, "period_alias": alias,
            "estimated_period_hours": period, "component_matching": matching,
            "component_significance": "not separately tested", "test_status": row.get("test_status", "missing_evaluation"),
            "estimate_status": row.get("estimate_status", "missing_evaluation"),
            "insufficient_response": ("unsupported_detection" if detected else
                "period_reported_without_detection" if available else "declined") if not positive and not negative else None,
            "periods": [t["period_hours"] for t in targets],
            "waveforms": sorted({c["waveform"] for c in case["components"]}),
            "noise": scenario.get("noise", {"kind": "none"}),
            "p_value": _finite(row.get("p_value")), "q_value": _finite(row.get("q_value"))})
    return pd.DataFrame(scores, columns=SCORE_COLUMNS)


def rate_units(scores, eligible, outcome):
    """Aggregate dependent corrected calls; failures remain in the denominator."""
    chosen = scores[scores[eligible].eq(True)]
    units = []
    for block, rows in chosen.groupby("family_block", sort=True):
        denominator = float(rows.weight.sum())
        units.append({"id": str(block), "dependency_id": rows.dependency_id.iloc[0],
            "value": float(rows.loc[rows[outcome].eq(True), "weight"].sum()) / denominator,
            "weight": denominator})
    return units


def _summary(rows, scope, measurement, facet, value, confidence):
    import pymicroglia.workbench as circadian

    output = {"candidate_id": rows.candidate_id.iloc[0], "scope": scope, "measurement": measurement,
              "facet": facet, "value": value, "cases": len(rows), "scoring_definition": SCORING_DEFINITION}
    for name, eligible, outcome in (("recovery", "eligible_positive", "recovered"),
                                     ("false_alarm", "eligible_negative", "false_alarm")):
        selected = rows[rows[eligible].eq(True)]
        units = rate_units(rows, eligible, outcome)
        interval = circadian.benchmark_score_interval(units, confidence=confidence, bounds=[0., 1.])
        output.update({name + "_numerator": int(selected[outcome].sum()), name + "_denominator": len(selected),
            name + "_valid_tests": int(selected.valid_test.sum()),
            name + "_valid_fraction": float(selected.valid_test.mean()) if len(selected) else None,
            name + "_weighted_valid_fraction": float(selected.loc[selected.valid_test, "weight"].sum() / selected.weight.sum()) if len(selected) else None,
            name + "_weighted_numerator": float(selected.loc[selected[outcome], "weight"].sum()),
            name + "_weighted_denominator": float(selected.weight.sum()),
            name + "_rate": interval["mean"], name + "_lower": interval["lower"], name + "_upper": interval["upper"],
            name + "_independent_units": interval["independent_units"], name + "_units": units,
            name + "_uncertainty": interval})
    output.update(unavailable_periods=int((~rows.period_available).sum()),
                  insufficient_cases=int((~rows.eligible_positive & ~rows.eligible_negative).sum()),
                  unsupported_estimates=int((rows.period_available & ~rows.period_supported).sum()),
                  missing_tests=int((~rows.valid_test).sum()))
    return output


def summarize(scores, *, confidence):
    """Retain overall and explicit scenario/period/waveform/noise/profile breakdowns."""
    summaries = []
    if scores.empty:
        return pd.DataFrame(columns=["candidate_id", "scope", "measurement", "facet", "value"])
    for candidate, candidate_rows in scores.groupby("candidate_id", sort=True):
        scopes = [("dataset", None, candidate_rows)]
        scopes.extend(("measurement", str(name), rows) for name, rows in candidate_rows.groupby("measurement", sort=True))
        for scope, metric, rows in scopes:
            summaries.append(_summary(rows, scope, metric, "overall", "all", confidence))
            if scope == "dataset":
                continue
            for facet in ("scenario", "periods", "waveforms", "noise", "span_hours", "source_profile_id"):
                labels = rows[facet].map(lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
                for value in sorted(labels.unique()):
                    summaries.append(_summary(rows[labels.eq(value)], scope, metric, facet, value, confidence))
    return pd.DataFrame(summaries)


def score_evaluations(resolved, benchmarks, evaluations):
    if len(evaluations) != len(resolved.candidates):
        raise ValueError("Every declared candidate needs its complete evaluation inventory")
    frames = [score_cases(candidate, evaluation, benchmarks, resolved.request.benchmark_design["truth_policy"])
              for candidate, evaluation in zip(resolved.candidates, evaluations)]
    scored = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return scored, summarize(scored, confidence=resolved.request.score_policy["confidence"])


def produce_scores(context):
    benchmarks = read_benchmarks(context.saved("development-cases"))
    inputs = evaluation_inputs(benchmarks)
    evaluations = [evaluate_candidate(candidate, inputs) for candidate in context.request.candidates]
    scored, summary = score_evaluations(context.request, benchmarks, evaluations)
    outcome = save_evaluations(context, evaluations)
    refs = list(outcome.artifacts)
    for name, frame in (("case_scores", scored), ("score_summary", summary)):
        path = context.output / f"{name}.json"
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id,
                               columns=tuple(frame.columns)))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved development recovery, false alarms, failures and uncertainty", tuple(refs),
        provenance=Settings({"partition": "development", "confirmation_opened": False,
                             "scoring_definition": SCORING_DEFINITION}))
