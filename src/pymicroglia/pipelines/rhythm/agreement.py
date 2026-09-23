"""Detection co-occurrence from complete saved screens, through Workbench."""
from pymicroglia._results import read_document

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_screen, write_table

DEFAULTS = {"confidence": .95, "interval_correction": "bonferroni", "sample_method": "none",
            "permutations": 9999, "seed": 20260910, "correction": "bh", "alpha": .05}
CATEGORIES = ("both", "first_only", "second_only", "neither", "first_missing", "second_missing", "both_missing")
KEYS = ("source_run", "movie", "identity")


def validate_options(declared):
    if not isinstance(declared, dict) or set(declared) - set(DEFAULTS):
        raise ValueError("detection_agreement accepts confidence, interval_correction, sample_method, permutations, seed, correction and alpha")
    result = {**DEFAULTS, **declared}
    for key in ("confidence", "alpha"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(f"detection_agreement.{key} must be a finite fraction strictly between zero and one")
    if result["interval_correction"] not in {"none", "bonferroni"}:
        raise ValueError("interval_correction must be none or bonferroni")
    if result["sample_method"] not in {"none", "spearman_permutation"}:
        raise ValueError("sample_method must be none or spearman_permutation")
    if result["correction"] not in {"none", "bh", "bonferroni", "sidak"}:
        raise ValueError("detection_agreement.correction must be none, bh, bonferroni or sidak")
    for key, minimum in (("permutations", 99), ("seed", 0)):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < minimum:
            raise ValueError(f"detection_agreement.{key} must be an integer >= {minimum}")
    return result


def version():
    import pymicroglia.workbench as circadian
    return content_id({"producer": file_hash(__file__), "gateway": file_hash(circadian.__file__),
                       "workbench": circadian.rhythm_environment()})


def identity(context):
    return content_id({"producer": version(), "screen": context.saved("rhythm-screen").outcome.scientific_id,
        "pairs": context.request.request.pairs, "settings": context.request.request.detection_agreement})


def independence(screen, pair, assignments):
    """Global corrected calls can couple otherwise separate biological samples."""
    reasons = []
    if any(not assignment.confirmed for assignment in assignments.values()):
        reasons.append("Some requested movies have no confirmed biological sample mapping")
    family_ids = set(screen.results.loc[screen.results.measurement.isin([pair.reference, pair.target]), "family_id"])
    dependencies = []
    for row in screen.families.to_dict("records"):
        if row["family_id"] not in family_ids or row["correction"] == "none": continue
        units = set()
        for member in row["members"]:
            movie = member["cell"]["movie"]
            assignment = assignments.get(movie)
            units.add(("sample", assignment.sample) if assignment and assignment.confirmed else ("movie", movie))
        if len(units) > 1:
            dependencies.append({"family_id": row["family_id"], "correction": row["correction"],
                                 "units": [list(unit) for unit in sorted(units)]})
    if dependencies:
        reasons.append("Saved multiple-testing correction families span biological samples; corrected decisions are not certified independent")
    return bool(assignments) and not reasons, reasons, dependencies


def _category(first, second):
    if first is None or second is None:
        return "both_missing" if first is None and second is None else "first_missing" if first is None else "second_missing"
    return "both" if first and second else "first_only" if first else "second_only" if second else "neither"


def produce(context):
    import pymicroglia.workbench as circadian
    resolved = context.request
    settings = validate_options(resolved.request.detection_agreement.as_dict())
    saved = context.saved("rhythm-screen")
    screen = read_screen(saved.root)
    assignments = {row.movie: row for row in resolved.inputs.samples}
    lookup = {tuple(row[k] for k in (*KEYS, "measurement")): row for row in screen.results.to_dict("records")}
    pairs = resolved.request.pairs
    confidence = 1 - (1 - settings["confidence"]) / max(1, len(pairs)) if settings["interval_correction"] == "bonferroni" else settings["confidence"]
    ledger, summaries, units, details, selections = [], [], [], [], []
    evidence_fields = ("status", "reason", "test_status", "method", "significance_method", "p_value", "q_value",
        "family_id", "period_hours", "period_available", "period_underdetermined", "estimate_status", "span_hours", "observations", "input_rows", "invalid_observations")
    for pair in pairs:
        pair_id = content_id(pair)
        pair_info = {"pair_id": pair_id, "reference": pair.reference, "target": pair.target}
        observations, recording_observations, memberships, unit_info = [], [], {key: [] for key in CATEGORIES}, {}
        for cell in resolved.inputs.cells:
            key = tuple(getattr(cell, k) for k in KEYS)
            first, second = [lookup.get((*key, metric), {"status": "untestable", "reason": "No saved partner result"}) for metric in (pair.reference, pair.target)]
            first_call, second_call = [True if row["status"] == "significant" else False if row["status"] == "not-significant" else None for row in (first, second)]
            assignment = assignments[cell.movie]
            token = {"sample": assignment.sample} if assignment.confirmed else {"movie": cell.movie}
            unit = content_id(token)
            unit_info[unit] = {"unit_label": assignment.sample if assignment.confirmed else cell.movie,
                               "unit_type": "biological sample" if assignment.confirmed else "recording", "sample_confirmed": assignment.confirmed}
            observation = {"id": content_id(cell), "unit": unit, "first": first_call, "second": second_call}
            observations.append(observation)
            recording_observations.append({**observation, "unit": cell.movie})
            category = _category(first_call, second_call)
            memberships[category].append(cell)
            ledger.append({**pair_info, **cell.as_dict(), "sample": assignment.sample, "sample_confirmed": assignment.confirmed,
                "outcome": category, **{"first_" + k: first.get(k) for k in evidence_fields}, **{"second_" + k: second.get(k) for k in evidence_fields}})
        independent, reasons, dependencies = independence(screen, pair, assignments)
        result = circadian.rhythm_detection_agreement(observations, independent_units=independent, confidence=confidence,
            sample_method=settings["sample_method"], permutations=settings["permutations"], seed=settings["seed"])
        recording_result = circadian.rhythm_detection_agreement(recording_observations, independent_units=False, confidence=confidence,
            sample_method="none", permutations=settings["permutations"], seed=settings["seed"])
        for scope, completed in (("sample_or_recording", result), ("recording", recording_result)):
            for row in completed["units"]:
                annotation = unit_info[row["unit"]] if scope == "sample_or_recording" else {
                    "unit_label": row["unit"], "unit_type": "recording", "sample_confirmed": assignments[row["unit"]].confirmed}
                units.append({**pair_info, "scope": scope, **annotation, **row})
        summaries.append({**pair_info, **result["pooled"],
            **{"unit_" + k: v for k, v in result["unit_agreement"].items() if k != "assumptions"},
            **{"test_" + k: v for k, v in result["sample_association"].items()},
            "mapping_confirmed": bool(assignments) and all(a.confirmed for a in assignments.values()), "independent_units": independent,
            "independence_reason": "; ".join(reasons), "q_value": np.nan, "significant": False,
            "correction": settings["correction"], "alpha": settings["alpha"], "family": "sample-fraction-associations"})
        if reasons:
            for name in ("unit", "test"):
                if summaries[-1][name + "_status"] == "unavailable": summaries[-1][name + "_reason"] = "; ".join(reasons)
        details.append({**pair_info, "result": result, "recording_result": recording_result, "decision_dependencies": dependencies})
        selections.extend(SelectionRecord(f"{category}:{pair_id}", context.scientific_id,
            Settings({"pair": pair.as_dict(), "outcome": category, "screen_id": saved.outcome.scientific_id}), tuple(members))
            for category, members in memberships.items())
    summary = pd.DataFrame(summaries)
    valid_tests = 0
    if not summary.empty:
        raw = pd.to_numeric(summary.test_p_value, errors="coerce").to_numpy(float)
        valid = np.isfinite(raw) & summary.test_status.eq("available").to_numpy(bool)
        valid_tests = int(valid.sum())
        if settings["sample_method"] != "none":
            adjusted = circadian.adjust_pvalues(np.where(valid, raw, 1.), settings["correction"])
            summary["q_value"] = np.where(valid, adjusted, np.nan)
            summary["significant"] = valid & (adjusted < settings["alpha"])
    ledger_columns = ["pair_id", "reference", "target", *KEYS, "sample", "sample_confirmed", "outcome",
                      *[prefix + name for prefix in ("first_", "second_") for name in evidence_fields]]
    context.output.mkdir(parents=True, exist_ok=True)
    frames = {"pairs": pd.DataFrame(ledger, columns=ledger_columns), "summary": summary,
              "units": pd.DataFrame(units), "families": pd.DataFrame([{
        "family": "sample-fraction-associations", "pairs": [pair.as_dict() for pair in pairs],
        "requested": len(pairs) if settings["sample_method"] != "none" else 0, "valid_tests": valid_tests,
        "correction": settings["correction"], "alpha": settings["alpha"],
        "interval_family": "equal-unit-mean-kappa", "intervals_requested": len(pairs),
        "interval_correction": settings["interval_correction"], "requested_confidence": settings["confidence"],
        "effective_confidence": confidence}])}
    refs = []
    for name, frame in frames.items():
        path = context.output / (name + ".json")
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    _write_json(context.output / "engine_details.json", details)
    _write_json(context.output / "provenance.json", {"schema_version": 1, "screen_id": saved.outcome.scientific_id,
        "source_screen_provenance": read_document(saved.artifact("provenance")),
        "measurements": [m.as_dict() for m in resolved.test_measurements], "pairs": [p.as_dict() for p in pairs],
        "settings": settings, "effective_confidence": confidence, "workbench_version": circadian.WORKBENCH_VERSION,
        "method": "Cohen chance-corrected agreement; equal-unit mean, with conditional independent-sample Hoeffding uncertainty",
        "timing_claim": "Detection co-occurrence does not establish comparable periods, phase or synchrony",
        "correction_policy": "All declared sample-fraction questions enter correction; unavailable tests contribute p=1 only to its denominator and keep missing probabilities",
        "dependency_policy": "Cross-sample screen correction families are conservatively treated as dependent; original calls are never re-corrected",
        "producer": version(), "rhythm_analysis_recomputed": False})
    refs.extend(ArtifactRef(name, filename, file_hash(context.output / filename), context.scientific_id)
                for name, filename in (("engine_details", "engine_details.json"), ("provenance", "provenance.json")))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved every requested detection pair, its denominators and supported unit-level agreement", tuple(refs), tuple(selections))
