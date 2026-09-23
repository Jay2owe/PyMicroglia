"""Freeze candidate recipes and an input-only benchmark design before fitting."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real

from pymicroglia.pipelines._contracts import Record, Settings, content_id, plain
from pymicroglia.pipelines.rhythm.discovery import RhythmDiscoveryRequest, ResolvedRhythmRequest, _known_keys, _object, resolve_request as resolve_rhythm_request


CONFIRMATION_POPULATION = "Every frozen family is evaluated; only declared fresh stochastic cases contribute confirmation scores"


def _number(value, name, *, minimum=0, maximum=None, integer=False, strict=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < minimum
            or (strict and value == minimum) or (maximum is not None and value > maximum)
            or (integer and value != int(value))):
        raise ValueError(f"{name}: invalid {'integer' if integer else 'number'} {value!r}")
    return int(value) if integer else float(value)


def _filter(value):
    value = _object(value, "filter")
    method = value.get("method", "none")
    if method == "none":
        _known_keys(value, {"method"}, "filter")
        return Settings({"method": "none"})
    if method != "median":
        raise ValueError("filter.method must be none or median")
    _known_keys(value, {"method", "window_hours", "max_gap_hours", "min_observations"}, "filter")
    return Settings({"method": "median",
        "window_hours": _number(value.get("window_hours"), "filter.window_hours", strict=True),
        "max_gap_hours": _number(value.get("max_gap_hours"), "filter.max_gap_hours", strict=True),
        "min_observations": _number(value.get("min_observations", 1),
                                    "filter.min_observations", minimum=1, integer=True),
        "window": "centred, inclusive endpoints",
        "gaps": "split at missing values or consecutive times farther apart than max_gap_hours",
        "missing": "preserve original unavailable positions"})


def _policy(value):
    value = _object(value, "score_policy")
    required = {"false_alarm_limit", "confidence", "min_positive", "min_negative",
                "min_valid_fraction", "recovery_margin", "uncertainty"}
    _known_keys(value, required, "score_policy")
    if required - value.keys():
        raise ValueError(f"score_policy requires {', '.join(sorted(required - value.keys()))}")
    result = dict(value)
    for key in ("false_alarm_limit", "min_valid_fraction", "recovery_margin"):
        result[key] = _number(value[key], f"score_policy.{key}", maximum=1)
    result["confidence"] = _number(value["confidence"], "score_policy.confidence", maximum=1, strict=True)
    if result["confidence"] == 1:
        raise ValueError("score_policy.confidence must be less than one")
    for key in ("min_positive", "min_negative"):
        result[key] = _number(value[key], f"score_policy.{key}", minimum=1, integer=True)
    if value["uncertainty"] != "stratified_independent_realizations":
        raise ValueError("score_policy.uncertainty must be stratified_independent_realizations")
    result.update(false_alarm_acceptance="upper confidence bound <= false_alarm_limit",
                  tie_rule="recovery uncertainty overlaps or improvement <= recovery_margin",
                  failures="eligible cases remain in denominators",
                  confirmation="one frozen shortlist; no replacement or retuning",
                  confirmation_recovery="lower confirmation bound >= max(0, development lower bound - recovery_margin)",
                  confirmation_population=CONFIRMATION_POPULATION,
                  repeated_profiles="fixed simulation strata, not biological replicates")
    return Settings(result)


def _design(value):
    value = _object(value, "benchmark_design")
    _known_keys(value, {"replicates", "truth_policy", "scenarios", "seed", "justification"},
                "benchmark_design")
    if not isinstance(value.get("justification"), str) or not value["justification"].strip():
        raise ValueError("benchmark_design.justification must explain the chosen scenarios and policy")
    result = dict(value)
    result["replicates"] = _number(value.get("replicates"), "benchmark_design.replicates", minimum=1, integer=True)
    result["seed"] = _number(value.get("seed"), "benchmark_design.seed", integer=True)
    truth = _object(value.get("truth_policy"), "benchmark_design.truth_policy")
    required = {"min_observations", "min_cycles", "period_min_hours", "period_max_hours",
                "relative_tolerance", "absolute_tolerance_hours", "target", "extra_components"}
    _known_keys(truth, required, "benchmark_design.truth_policy")
    if required - truth.keys():
        raise ValueError("benchmark_design.truth_policy must declare every truth/recovery setting")
    if truth["target"] not in {"all", "dominant"} or truth["extra_components"] not in {"penalize", "ignore"}:
        raise ValueError("truth_policy requires target all/dominant and extra_components penalize/ignore")
    for key in ("min_cycles", "period_min_hours", "period_max_hours"):
        _number(truth[key], f"truth_policy.{key}", strict=True)
    _number(truth["min_observations"], "truth_policy.min_observations", minimum=2, integer=True)
    _number(truth["relative_tolerance"], "truth_policy.relative_tolerance", maximum=1)
    _number(truth["absolute_tolerance_hours"], "truth_policy.absolute_tolerance_hours")
    if truth["relative_tolerance"] >= 1 or truth["period_max_hours"] <= truth["period_min_hours"]:
        raise ValueError("truth_policy requires ordered period bounds and relative tolerance < 1")
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("benchmark_design.scenarios must be a non-empty list")
    seen = set()
    for scenario in scenarios:
        scenario = _object(scenario, "benchmark scenario")
        _known_keys(scenario, {"id", "components", "baseline", "drift", "noise", "disturbances",
                               "retain_observations", "weight"}, "benchmark scenario")
        name = scenario.get("id")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("benchmark scenarios require unique non-empty ids")
        seen.add(name)
        if not isinstance(scenario.get("components"), list):
            raise ValueError("benchmark scenario.components must explicitly list truth components")
        _number(scenario.get("weight", 1), "scenario.weight", strict=True)
    # Numerical generation and complete waveform/noise validation stay in Workbench.
    return Settings(result)


@dataclass(frozen=True)
class AuditRequest(Record):
    name: str
    declaration: Settings
    population_request: RhythmDiscoveryRequest
    candidates: tuple[Settings, ...]
    population: Settings
    benchmark_design: Settings
    score_policy: Settings
    stability: Settings
    recommendation_scope: str = "measurement"
    pipeline: str = "method-selection-audit"

    @classmethod
    def from_dict(cls, block, groups, *, where="pipeline"):
        _known_keys(block, {"pipeline", "name", "measurements", "analysis_options", "candidates",
            "population", "biological_samples", "correction_scope", "benchmark_design",
            "score_policy", "stability", "recommendation_scope"}, where)
        population = _object(block.get("population", {"mode": "all"}), "population")
        mode = population.get("mode", "all")
        allowed = {"all": {"mode"}, "explicit": {"mode", "cells"},
                   "representative": {"mode", "per_stratum", "seed", "duration_edges_hours", "missing_edges"}}
        if mode not in allowed:
            raise ValueError("population.mode must be all, explicit or representative")
        _known_keys(population, allowed[mode], "population")
        population = {**population, "mode": mode}
        if mode == "explicit" and not isinstance(population.get("cells"), list):
            raise ValueError("population.cells is required for explicit selection")
        if mode == "representative":
            for key, minimum in (("per_stratum", 1), ("seed", 0)):
                population[key] = _number(population.get(key), f"population.{key}", minimum=minimum, integer=True)
            for key in ("duration_edges_hours", "missing_edges"):
                edges = population.get(key)
                if not isinstance(edges, list):
                    raise ValueError(f"population.{key} must be explicit increasing bin edges")
                edges = [_number(e, f"population.{key}", maximum=1 if key == "missing_edges" else None)
                         for e in edges]
                if edges != sorted(set(edges)):
                    raise ValueError(f"population.{key} must be strictly increasing")
                population[key] = edges
        # Reuse the shared table/population resolver without selecting a significant subset.
        base = RhythmDiscoveryRequest.from_dict({
            "pipeline": "rhythm-discovery", "name": block.get("name", "method-selection-audit"),
            "test_measurements": block.get("measurements"), "pairs": {"mode": "explicit", "pairs": []},
            "analysis_options": block.get("analysis_options", {}),
            "biological_samples": block.get("biological_samples", {}),
            "correction_scope": block.get("correction_scope", "all"),
            **({"cells": population["cells"]} if mode == "explicit" else {})}, groups, where=where)
        candidates = block.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must contain complete recipe choices (an empty object inherits settings)")
        frozen = []
        for entry in candidates:
            entry = _object(entry, "candidate")
            _known_keys(entry, {"label", "filter", "analysis_options"}, "candidate")
            options = _object(entry.get("analysis_options", {}), "candidate.analysis_options")
            if "period_config" in options:
                _object(options["period_config"], "candidate.analysis_options.period_config")
            if "label" in entry and (not isinstance(entry["label"], str) or not entry["label"].strip()):
                raise ValueError("candidate.label must be a non-empty string")
            _filter(entry.get("filter", {}))
            frozen.append(Settings(entry))
        scope = block.get("recommendation_scope", "measurement")
        if scope not in {"measurement", "dataset"}:
            raise ValueError("recommendation_scope must be measurement or dataset")
        stability = _object(block.get("stability", {"alterations": []}), "stability")
        _known_keys(stability, {"alterations", "relative_tolerance", "absolute_tolerance_hours"}, "stability")
        if not isinstance(stability.get("alterations"), list):
            raise ValueError("stability.alterations must be a list")
        for alteration in stability["alterations"]:
            alteration = _object(alteration, "stability alteration")
            _known_keys(alteration, {"start_fraction", "end_fraction"}, "stability alteration")
            start = _number(alteration.get("start_fraction"), "start_fraction", maximum=1)
            end = _number(alteration.get("end_fraction"), "end_fraction", maximum=1)
            if not start < end:
                raise ValueError("stability omission fractions require start < end")
        design = _design(block.get("benchmark_design"))
        stability = {**stability,
            "relative_tolerance": _number(stability.get("relative_tolerance", design["truth_policy"]["relative_tolerance"]),
                                          "stability.relative_tolerance", maximum=1),
            "absolute_tolerance_hours": _number(stability.get("absolute_tolerance_hours", design["truth_policy"]["absolute_tolerance_hours"]),
                                                "stability.absolute_tolerance_hours"),
            "weighting": "equal original profiles and equal distinct alterations within profile",
            "period_distance": "altered selected period minus baseline selected period; no component rematching",
            "interval_semantics": "closed interval in original elapsed time; unavailable positions retained"}
        return cls(base.name, Settings(block), base, tuple(frozen), Settings(population),
                   design, _policy(block.get("score_policy")),
                   Settings(stability), scope)


@dataclass(frozen=True)
class Candidate(Record):
    candidate_id: str
    labels: tuple[str, ...]
    filtering: Settings
    analysis_options: Settings
    rhythm_params: Settings
    engine_settings: Settings
    correction_scope: str


@dataclass(frozen=True)
class ResolvedAuditRequest(Record):
    request: AuditRequest
    source: ResolvedRhythmRequest
    candidates: tuple[Candidate, ...]
    profiles: tuple[Settings, ...]
    population_inventory: tuple[Settings, ...]
    evaluation_design: Settings

    @property
    def inputs(self):
        return self.source.inputs

    @property
    def scientific_id(self):
        return content_id({"inputs": self.inputs, "measurements": self.source.test_measurements,
            "candidates": [{k: v for k, v in c.as_dict().items() if k != "labels"} for c in self.candidates],
            "profiles": self.profiles, "design": self.evaluation_design,
            "workbench_version": self.source.workbench_version,
            "period_methods": self.source.period_methods})


def resolve_request(request: AuditRequest, **kwargs) -> ResolvedAuditRequest:
    """Resolve against original tables; no fitting, filtering or synthetic generation."""
    import numpy as np
    import pymicroglia.workbench as circadian

    source = resolve_rhythm_request(request.population_request, **kwargs)
    engine_arguments = circadian.argument_group()
    candidates = {}
    for declared in request.candidates:
        base = request.population_request
        options = {**base.analysis_options.as_dict(), **declared.get("analysis_options", {})}
        # Method-specific dictionaries inherit without dropping other explicit settings.
        options["period_config"] = {**base.analysis_options.get("period_config", {}),
                                     **declared.get("analysis_options", {}).get("period_config", {})}
        resolved = resolve_rhythm_request(replace(base, analysis_options=Settings(options)), **kwargs)
        params = resolved.rhythm_params.as_dict()
        # These legacy gateway parameters otherwise overwrite their public config equivalents.
        for key in ("jtk_periods", "jtk_seed", "ejtk_permutations", "jtk_correction"):
            if key in params["workbench_config"]:
                params[key] = params["workbench_config"][key]
        effective = circadian.workbench_config(params, [0., 1.])
        for key, requested in params["workbench_config"].items():
            if key not in engine_arguments:
                raise ValueError(f"period_config.{key} is not a declared period-analysis setting")
            if key == "bin_minutes":
                raise ValueError("period_config.bin_minutes is derived from each recording's actual sampling")
            aliases = engine_arguments[key].get("aliases", {})
            expected = aliases.get(requested, requested) if isinstance(requested, str) else requested
            if effective.get(key) != expected:
                raise ValueError(f"period_config.{key} conflicts with the resolved shared analysis options; "
                                 "set the corresponding analysis option explicitly")
        applied = resolved.analysis_options.as_dict()
        applied["detrend"] = effective["period_detrend"]
        applied["detrend_polynomial_degree"] = effective["period_detrend_polynomial_degree"]
        params["detrend"] = applied["detrend"]
        params["detrend_polynomial_degree"] = applied["detrend_polynomial_degree"]
        # Store live defaults as well as explicit options. Sampling interval is
        # derived separately for every original profile by the scientific gateway.
        engine = {key: value for key, value in effective.items() if key in engine_arguments}
        engine.pop("bin_minutes", None)
        params["workbench_config"] = engine
        applied["period_config"] = engine
        filtering = _filter(declared.get("filter", {}))
        identity = content_id({"filter": filtering, "analysis": applied,
                               "correction_scope": base.correction_scope})
        label = declared.get("label", f"{applied['detrend']} / {applied['fit_method']} / {applied['significance_method']}")
        if identity in candidates:
            old = candidates[identity]
            candidates[identity] = replace(old, labels=tuple(dict.fromkeys((*old.labels, label))))
        else:
            candidates[identity] = Candidate(identity, (label,), filtering, Settings(applied),
                                              Settings(params), Settings(engine), base.correction_scope)
    samples = {s.movie: s for s in source.inputs.samples}
    profiles = []
    for pair in source.expected_pairs:
        measurement = next(m for m in source.test_measurements if m.column == pair.measurement)
        table = kwargs["tables"][measurement.table]
        frame = table[table.stem.eq(pair.cell.movie) & table.identity.eq(pair.cell.identity)].sort_values("hours")
        hours = frame.hours.to_numpy(dtype=float)
        if not np.isfinite(hours).all() or (len(hours) > 1 and np.any(np.diff(hours) <= 0)):
            raise ValueError(f"recording profile {pair.cell.movie}/{pair.cell.identity}/{pair.measurement}: invalid or repeated hours")
        missing = ~np.isfinite(frame[measurement.column].to_numpy(dtype=float))
        span = float(hours[-1] - hours[0]) if len(hours) else 0.
        missing_fraction = float(missing.mean()) if len(hours) else 1.
        metadata = {**plain(pair), "table": measurement.table, "unit": measurement.unit,
                    "sample_assignment": samples[pair.cell.movie].as_dict()}
        profile = {"hours": hours.tolist(), "missing": missing.tolist(),
                   "origin_hours": float(hours[0]) if len(hours) else 0., "metadata": metadata}
        profile["id"] = content_id(profile)
        strata = {"movie": pair.cell.movie, "measurement": pair.measurement,
            "duration_bin": int(np.searchsorted(request.population.get("duration_edges_hours", []), span, side="right")),
            "missing_bin": int(np.searchsorted(request.population.get("missing_edges", []), missing_fraction, side="right"))}
        profiles.append((profile, {"profile_id": profile["id"], "key": plain(pair), "duration_hours": span,
            "missing_fraction": missing_fraction, "observations": int((~missing).sum()), "stratum": strata}))
    chosen = set()
    if request.population["mode"] == "representative":
        buckets = {}
        for profile, inventory in profiles:
            buckets.setdefault(content_id(inventory["stratum"]), []).append(profile["id"])
        for ids in buckets.values():
            ids.sort(key=lambda key: content_id({"seed": request.population["seed"], "profile": key}))
            chosen.update(ids[:request.population["per_stratum"]])
    else:
        chosen = {p["id"] for p, _ in profiles}
    design = Settings({"schema_version": 1, "benchmark": request.benchmark_design,
        "score_policy": request.score_policy, "population": request.population,
        "stability": request.stability, "recommendation_scope": request.recommendation_scope,
        "family_template": {"scope": request.population_request.correction_scope,
            "members": sorted(chosen), "partition_rule": "one complete scenario/replicate across selected profiles",
            "candidate_count_in_family": False},
        "partitions": ["development", "confirmation"],
        "confirmation_rule": "not generated or evaluated before the choice is frozen"})
    return ResolvedAuditRequest(request, source, tuple(candidates.values()),
        tuple(Settings(p) for p, _ in profiles if p["id"] in chosen),
        tuple(Settings({**inventory, "selected": p["id"] in chosen,
                        "selection_basis": "input coverage only"}) for p, inventory in profiles), design)
