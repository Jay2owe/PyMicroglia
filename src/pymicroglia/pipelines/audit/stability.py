"""Paired observation sensitivity from altered raw recordings, with honest support."""

import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.audit.evaluator import evaluate_candidate, real_cases, read_evaluation, save_evaluations
from pymicroglia.pipelines.audit.scoring import _finite
from pymicroglia.pipelines._screening import file_hash, write_table


PAIR_COLUMNS = ["candidate_id", "profile_id", "measurement", "movie", "identity", "alteration_id",
    "baseline_case_id", "altered_case_id", "baseline_family_id", "altered_family_id", "sample_assignment",
    "removed_observations", "input_changed", "baseline_observations", "altered_observations",
    "baseline_span_hours", "altered_span_hours", "baseline_period_hours", "altered_period_hours",
    "period_shift_hours", "relative_period_shift", "period_tolerance_hours", "period_comparison",
    "detection_comparison", "component_switch", "baseline_test_status", "altered_test_status",
    "baseline_estimate_status", "altered_estimate_status", "baseline_components", "altered_components"]


def altered_cases(cases, policy):
    """All candidates receive the same declared masks of original raw observations."""
    import pymicroglia.workbench as circadian

    alterations = {content_id(spec): spec for spec in policy["alterations"]}
    output, manifest = [], []
    for alteration_id, spec in sorted(alterations.items()):
        for original in cases:
            hours = original["hours"]
            span = hours[-1] - hours[0] if hours else 0.
            interval = [[hours[0] + spec["start_fraction"] * span,
                         hours[0] + spec["end_fraction"] * span]] if hours else []
            masked = circadian.omit_rhythm_intervals(hours, original["values"], interval)
            changed = {**original.as_dict(), "case_id": content_id({"baseline": original["case_id"],
                "alteration": alteration_id, "values": masked["values"]}),
                "family_block": alteration_id, "values": masked["values"],
                "metadata": {**original["metadata"], "baseline_case_id": original["case_id"],
                             "alteration_id": alteration_id}}
            output.append(Settings(changed))
            removed = sum(v is not None and masked["omitted"][i] for i, v in enumerate(original["values"]))
            surviving = [h for h, v in zip(hours, masked["values"]) if v is not None]
            baseline = [h for h, v in zip(hours, original["values"]) if v is not None]
            manifest.append({"alteration_id": alteration_id, "profile_id": original["profile_id"],
                "baseline_case_id": original["case_id"], "altered_case_id": changed["case_id"],
                "parameters": spec, "intervals_hours": interval, "removed_observations": removed,
                "input_changed": removed > 0, "input_id": content_id({"hours": hours, "values": masked["values"]}),
                "baseline_span_hours": baseline[-1] - baseline[0] if len(baseline) > 1 else 0.,
                "altered_span_hours": surviving[-1] - surviving[0] if len(surviving) > 1 else 0.,
                "operation": masked})
    return tuple(output), pd.DataFrame(manifest, columns=["alteration_id", "profile_id", "baseline_case_id",
        "altered_case_id", "parameters", "intervals_hours", "removed_observations", "input_changed", "input_id",
        "baseline_span_hours", "altered_span_hours", "operation"])


def _lookup(frame):
    if frame.duplicated(["candidate_id", "case_id"]).any():
        raise ValueError("Stability requires unique candidate/case evidence")
    return {(r["candidate_id"], r["case_id"]): r for r in frame.to_dict("records")}


def compare_pairs(candidate, original, altered, manifest, baseline, changed, policy):
    before, after = _lookup(baseline.results), _lookup(changed.results)
    originals = {case["case_id"]: case for case in original}
    variants = {case["case_id"]: case for case in altered}
    pairs = []
    for item in manifest.to_dict("records"):
        raw = originals[item["baseline_case_id"]]
        variant = variants[item["altered_case_id"]]
        left = before.get((candidate.candidate_id, raw["case_id"]), {})
        right = after.get((candidate.candidate_id, variant["case_id"]), {})
        def period(row):
            p = _finite(row.get("period_hours"))
            return p if p is not None and p > 0 and row.get("period_available") and not row.get("period_underdetermined", True) else None
        def detected(row):
            p, q = _finite(row.get("p_value")), _finite(row.get("q_value"))
            if row.get("test_status") != "ok" or p is None or q is None or not 0 <= p <= 1 or not 0 <= q <= 1:
                return None
            return q < candidate.analysis_options["rhythmic_alpha"]
        first, second = period(left), period(right)
        first_call, second_call = detected(left), detected(right)
        tolerance = max(policy["absolute_tolerance_hours"], first * policy["relative_tolerance"]) if first else None
        shift = second - first if first is not None and second is not None else None
        supported = shift is not None and item["input_changed"]
        period_state = ("stable_supported" if abs(shift) <= tolerance else "changed_supported") if supported else (
            "unchanged_input" if not item["input_changed"] else "unevaluable")
        detection = "unevaluable"
        if not item["input_changed"]:
            detection = "unchanged_input"
        elif first_call is not None and second_call is not None:
            detection = ("retained_detection" if first_call else "retained_nondetection") if first_call == second_call else (
                "lost_detection" if first_call else "new_detection")
        components = left.get("components") if isinstance(left.get("components"), list) else []
        switched = False
        if period_state == "changed_supported":
            switched = any(p is not None and abs(p - first) > tolerance and
                abs(second - p) <= max(policy["absolute_tolerance_hours"], p * policy["relative_tolerance"])
                for c in components if (p := _finite(c.get("period_hours"))) is not None)
        pairs.append({"candidate_id": candidate.candidate_id, **{k: raw[k] for k in ("profile_id", "measurement", "movie", "identity")},
            **{k: item[k] for k in ("alteration_id", "baseline_case_id", "altered_case_id", "removed_observations",
                                  "input_changed", "baseline_span_hours", "altered_span_hours")},
            "baseline_family_id": left.get("family_id"), "altered_family_id": right.get("family_id"),
            "sample_assignment": raw["metadata"].get("sample_assignment", {}),
            "baseline_observations": left.get("observations"), "altered_observations": right.get("observations"),
            "baseline_period_hours": _finite(left.get("period_hours")), "altered_period_hours": _finite(right.get("period_hours")),
            "period_shift_hours": shift if supported else None, "relative_period_shift": shift / first if supported else None,
            "period_tolerance_hours": tolerance, "period_comparison": period_state, "detection_comparison": detection,
            "component_switch": switched, "baseline_components": components, "altered_components": right.get("components", []),
            **{f"{prefix}_{field}": row.get(field, "missing_evaluation")
               for prefix, row in (("baseline", left), ("altered", right)) for field in ("test_status", "estimate_status")}})
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def summarize_pairs(resolved, pairs):
    """Descriptive complete-pair counts; no resample is labelled a biological repeat."""
    summaries = []
    for candidate in resolved.candidates:
        for measurement in resolved.source.test_measurements:
            metric = measurement.column
            rows = pairs[pairs.candidate_id.eq(candidate.candidate_id) & pairs.measurement.eq(metric)]
            groups = [("measurement", None, rows)]
            groups.extend(("profile", profile, group) for profile, group in rows.groupby("profile_id", sort=True))
            for scope, profile, group in groups:
                valid_period = group.period_comparison.isin(["stable_supported", "changed_supported"])
                valid_test = group.detection_comparison.isin(["retained_detection", "retained_nondetection", "new_detection", "lost_detection"])
                stable = int(group.period_comparison.eq("stable_supported").sum())
                assignments = group.sample_assignment.to_list()
                summaries.append({"candidate_id": candidate.candidate_id, "measurement": metric, "scope": scope,
                    "profile_id": profile, "status": "evaluated" if len(group) else (
                        "no_profiles" if resolved.request.stability["alterations"] else "not_requested"),
                    "pairs": len(group), "profiles": group.profile_id.nunique(),
                    "alterations": group.alteration_id.nunique(), "changed_inputs": int(group.input_changed.sum()),
                    "valid_period_pairs": int(valid_period.sum()), "stable_supported": stable,
                    "changed_supported": int(group.period_comparison.eq("changed_supported").sum()),
                    "stable_fraction_of_all_pairs": stable / len(group) if len(group) else None,
                    "stable_fraction_of_supported_pairs": stable / valid_period.sum() if valid_period.any() else None,
                    "valid_test_pairs": int(valid_test.sum()), "new_detections": int(group.detection_comparison.eq("new_detection").sum()),
                    "lost_detections": int(group.detection_comparison.eq("lost_detection").sum()),
                    "unevaluable_period_pairs": int(group.period_comparison.eq("unevaluable").sum()),
                    "unevaluable_test_pairs": int(group.detection_comparison.eq("unevaluable").sum()),
                    "unchanged_inputs": int((~group.input_changed.astype(bool)).sum()),
                    "component_switches": int(group.component_switch.sum()),
                    "confirmed_sample_pairs": sum(a.get("confirmed", False) for a in assignments),
                    "sample_assignments": assignments, "weighting": resolved.request.stability["weighting"],
                    "uncertainty": "not computed; repeated omissions are paired descriptive sensitivity, not biological replication",
                    "policy": resolved.request.stability.as_dict()})
    return pd.DataFrame(summaries)


def produce_stability(context):
    baseline = read_evaluation(context.saved("real-candidates"))
    original = real_cases(context.request, context.table_paths)
    altered, manifest = altered_cases(original, context.request.request.stability)
    evaluations, frames = [], []
    for candidate in context.request.candidates:
        changed = evaluate_candidate(candidate, altered)
        evaluations.append(changed)
        frames.append(compare_pairs(candidate, original, altered, manifest, baseline, changed,
                                    context.request.request.stability))
    pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLUMNS)
    summary = summarize_pairs(context.request, pairs)
    result = save_evaluations(context, evaluations)
    refs = list(result.artifacts)
    for name, frame in (("alterations", manifest), ("stability_pairs", pairs), ("stability_summary", summary)):
        path = context.output / f"{name}.json"
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    return StepResult(context.step.name, context.scientific_id, "completed", "Saved paired sensitivity and all unavailable comparisons",
        tuple(refs), provenance=Settings({"partition": "real", "known_correctness": False,
            "policy": context.request.request.stability, "baseline": context.saved("real-candidates").outcome.scientific_id}))
