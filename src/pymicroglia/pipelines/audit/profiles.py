"""Explicit complete audit choices and validation for the shared discovery consumer."""
from pymicroglia._results import read_document

import json
from pathlib import Path

from pymicroglia.pipelines._contracts import Settings, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table


def settings_id(candidate):
    return content_id({"filter": candidate["filtering"], "analysis": candidate["analysis_options"],
                       "correction_scope": candidate["correction_scope"]})


def scientific_environment(environment):
    """Compare recorded algorithm identity without making plotting libraries scientific settings."""
    versions = environment.get("versions", {})
    code = environment.get("code_sha256", {})
    return {"versions": {k: versions.get(k) for k in ("python", "circadian-workbench", "numpy", "pandas", "scipy", "statsmodels", "PyWavelets", "analysis-kit")},
            "code_sha256": {k: code.get(k) for k in ("circadian_workbench", "analysis_kit")}}


def _read_report(root):
    report = read_document(root / "audit_report.json")
    if report.get("schema_version") != 1 or content_id({k: v for k, v in report.items() if k != "report_id"}) != report.get("report_id"):
        raise ValueError("Audit report identity changed; restore or regenerate its saved index")
    def artifact(step, name, *, table=False):
        ref = report["inputs"][step]["artifacts"][name]
        path = (root / ref["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != ref["sha256"]:
            raise ValueError("Audit evidence is missing or changed: " + step + "/" + name)
        return read_table(path) if table else read_document(path)
    return report, artifact


def export_profile(report_folder, choices, *, output=None, override_reason=None, scope="measurement"):
    """Export exactly an explicit measurement mapping or one dataset candidate id.

    A manual choice outside one supported confirmed winner needs its reason in
    the exported record. That reason is provenance, not a request for permission.
    """
    root = Path(report_folder).resolve()
    report, artifact = _read_report(root)
    design = artifact("audit-design", "audit_design")
    selection = artifact("candidate-shortlist", "frozen_selection")
    confirmation = artifact("independent-confirmation", "confirmation_record")
    decisions = artifact("independent-confirmation", "final_decisions", table=True).to_dict("records")
    candidates = {c["candidate_id"]: c for c in selection["candidates"]}
    known = {m["column"] for m in design["request"]["source"]["test_measurements"]}
    if scope == "dataset":
        if not isinstance(choices, str):
            raise ValueError("Dataset export takes one complete candidate id")
        mapping = {metric: choices for metric in sorted(known)}
    elif scope == "measurement":
        if not isinstance(choices, dict) or not choices:
            raise ValueError("Measurement export needs an explicit measurement-to-candidate mapping")
        mapping = dict(choices)
    else:
        raise ValueError("Export scope must be measurement or dataset")
    if set(mapping) - known or any(not isinstance(cid, str) or cid not in candidates for cid in mapping.values()):
        raise ValueError("Choice names a measurement or complete candidate absent from this audit")
    if confirmation["selection_id"] != selection["selection_id"] or selection["selection_id"] != report["selection_id"]:
        raise ValueError("Profile export sources refer to different frozen choices")
    recipes, overrides, evidence = {}, [], []
    for metric, candidate_id in sorted(mapping.items()):
        candidate = candidates[candidate_id]
        if settings_id(candidate) != candidate_id:
            raise ValueError("Complete candidate settings do not match their saved identity")
        applicable = [d for d in decisions if d["measurement"] in (None, metric)]
        supported = any(d.get("promotion_allowed") and d.get("candidate_ids") == [candidate_id] for d in applicable)
        if not supported:
            overrides.append(metric)
        recipes[metric] = candidate
        evidence.append({"measurement": metric, "candidate_id": candidate_id, "saved_decisions": applicable,
                         "explicit_manual_choice": not supported})
    if overrides and (not isinstance(override_reason, str) or not override_reason.strip()):
        raise ValueError("This explicit choice is tied, failed or unconfirmed; record override_reason for: " + ", ".join(overrides))
    scopes = {r["correction_scope"] for r in recipes.values()}
    if len(scopes) != 1:
        raise ValueError("Selected recipes use different production family scopes; choose a consistent declared family")
    family_scope = scopes.pop()
    correction = {metric: {"correction": r["analysis_options"]["multiple_testing"], "alpha": r["analysis_options"]["rhythmic_alpha"]}
                  for metric, r in recipes.items()}
    if family_scope == "all" and len({content_id(v) for v in correction.values()}) != 1:
        raise ValueError("A joint family requires the same alpha and correction across its measurement recipes")
    full_population = set(mapping) == known
    mixed = len(set(mapping.values())) > 1
    matched = full_population and (family_scope != "all" or not mixed)
    environment = None
    real = artifact("real-candidates", "results", table=True)
    environments = {}
    for row in real[real.candidate_id.isin(mapping.values())].to_dict("records"):
        for branch in ("estimate_result", "significance_result"):
            record = row.get(branch, {}).get("workbench_run_record_json")
            if record:
                record = json.loads(record) if isinstance(record, str) else record
                if record.get("environment"):
                    env = scientific_environment(record["environment"])
                    environments[content_id(env)] = env
    if len(environments) > 1:
        raise ValueError("Selected audit results used different scientific environments")
    if environments:
        environment = next(iter(environments.values()))
    profile = {"schema_version": 1, "kind": "motion-rhythm-settings", "scope": scope,
        "audit_id": report["audit_id"], "report_id": report["report_id"], "selection_id": report["selection_id"],
        "confirmation_id": report["confirmation_id"], "measurement_recipes": recipes,
        "family": {"scope": family_scope, "per_measurement": correction},
        "evaluation_compatibility": {"state": "matching_recipe_family_template" if matched else "different_mixture_or_measurement_family",
            "equivalent_calibration": matched and not overrides and environment is not None,
            "limitation": "Benchmark calibration remains conditional on the declared simulation design and population; not biological validation"},
        "manual_override": {"measurements": overrides, "reason": override_reason} if overrides else None,
        "workbench_version": design["request"]["source"]["workbench_version"], "scientific_environment": environment,
        "evidence": evidence, "input_selection": design["request"]["request"]["population"],
        "original_profiles": design["request"]["profiles"], "source_evidence": report["inputs"],
        "automatic_discovery": False}
    profile["profile_id"] = content_id(profile)
    target = Path(output).resolve() if output else root / "exports" / (profile["profile_id"] + ".json")
    from pymicroglia._results import document
    target = document(target)
    if target.exists():
        if read_document(target) != profile:
            raise FileExistsError("Profile output already contains different settings")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target = _write_json(target, profile)
    return target


def load_profile(path, measurements):
    """Validate complete settings before any original trace can be processed."""
    import pymicroglia.workbench as circadian
    from pymicroglia.pipelines.audit.options import _filter
    profile = read_document(Path(path))
    if profile.get("schema_version") != 1 or profile.get("kind") != "motion-rhythm-settings" or profile.get("profile_id") != content_id({k: v for k, v in profile.items() if k != "profile_id"}):
        raise ValueError("Unsupported or changed rhythm settings profile")
    if set(profile["measurement_recipes"]) != set(measurements):
        raise ValueError("Profile measurements must exactly match test_measurements; export the intended mapping explicitly")
    if profile["workbench_version"] != circadian.WORKBENCH_VERSION:
        raise ValueError("Workbench version changed since the audit; rerun the audit before applying this profile")
    if profile.get("scientific_environment") is not None and profile["scientific_environment"] != scientific_environment(circadian.rhythm_environment()):
        raise ValueError("Scientific engine or numerical dependencies changed since this audit; rerun it before applying the profile")
    for metric, candidate in profile["measurement_recipes"].items():
        if settings_id(candidate) != candidate["candidate_id"]:
            raise ValueError("Saved complete recipe changed for " + metric)
        options = candidate["analysis_options"]
        numeric_filter = {k: v for k, v in candidate["filtering"].items() if k in {"method", "window_hours", "max_gap_hours", "min_observations"}}
        if _filter(numeric_filter).as_dict() != candidate["filtering"]:
            raise ValueError("Incomplete or unsupported saved filtering policy for " + metric)
        if candidate["correction_scope"] != profile["family"]["scope"] or profile["family"]["per_measurement"].get(metric) != {
                "correction": options["multiple_testing"], "alpha": options["rhythmic_alpha"]}:
            raise ValueError("Production family does not preserve the selected recipe settings")
        if set(options) != set(circadian.CIRCADIAN_ANALYSIS_OPTIONS):
            raise ValueError("Profile does not contain the complete shared analysis option contract")
        if options["fit_method"] not in circadian.PERIOD_METHODS or options["significance_method"] not in circadian.PERIOD_METHODS:
            raise ValueError("A selected profile method is unavailable in the live Workbench catalogue")
        resolved = circadian.resolve_analysis_options(candidate["rhythm_params"], lambda key: options[key])
        if resolved["method"] != options["fit_method"] or resolved["significance_method"] != options["significance_method"]:
            raise ValueError("Workbench could not preserve the selected estimator and test")
        expected = circadian.workbench_config(candidate["rhythm_params"], [0., 1.])
        for key, value in candidate["engine_settings"].items():
            if expected.get(key) != value:
                raise ValueError("Stale or unsupported engine setting for " + metric + ": " + key)
    return Settings(profile)


def production_compatibility(profile, tests, cells, tables):
    """Check declared sampling/population equality; never infer biological validation."""
    import numpy as np
    def signature(movie, cell, metric, hours, missing):
        return content_id({"movie": movie, "identity": int(cell), "measurement": metric, "hours": hours, "missing": missing})
    audited = []
    for row in profile["original_profiles"]:
        meta = row["metadata"]
        if meta["measurement"] in {m.column for m in tests}:
            audited.append(signature(meta["cell"]["movie"], meta["cell"]["identity"], meta["measurement"], row["hours"], row["missing"]))
    production = []
    for cell in cells:
        for metric in tests:
            frame = tables[metric.table]
            row = frame[frame.stem.eq(cell.movie) & frame.identity.eq(cell.identity)].sort_values("hours")
            production.append(signature(cell.movie, cell.identity, metric.column, row.hours.tolist(),
                                        (~np.isfinite(row[metric.column].to_numpy(float))).tolist()))
    same = sorted(audited) == sorted(production)
    return {"state": "matching_recording_profiles" if same else "different_recording_population_or_sampling",
            "audited_profiles": len(audited), "production_profiles": len(production),
            "equivalent_calibration": same and profile["evaluation_compatibility"]["equivalent_calibration"],
            "limitation": "Any calibration is conditional on the audit's declared synthetic design, never a biological validation claim"}


def command(args):
    choices = read_document(Path(args.choices))
    path = export_profile(args.report, choices, output=args.out, override_reason=args.override_reason, scope=args.scope)
    print("Saved complete settings profile:", path)
    return 0
