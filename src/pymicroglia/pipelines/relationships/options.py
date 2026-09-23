"""Resolve independent relationship questions without evaluating their evidence.

No scientific model is selected implicitly. Enabled questions name their statistic
and evidence method; the owning producer validates method-specific settings before
evaluation. Measurements and preprocessing remain independent of rhythm discovery.
"""
from __future__ import annotations
from pymicroglia._sources import source_file

from dataclasses import dataclass, replace
import math
from numbers import Real
from pathlib import Path
import re

from pymicroglia.pipelines._contracts import ArtifactRef, InputIdentity, Measurement, MeasurementPair, PipelineRecipe, Record, Settings, StepResult, StepSpec, cell_number, content_id, text_key
from pymicroglia.pipelines.rhythm.discovery import MeasurementChoice, _catalogue, _choices, _known_keys, _object, _pairs, _population, _resolve_measurement, _shape_problem


LAG_CONVENTION = "Negative lag means reference leads target; target is matched at reference time minus lag"
SUMMARIES = ("mean", "median", "min", "max")


def _number(value, where, *, minimum=None, positive=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or (minimum is not None and value < minimum) or (positive and value <= 0)
            or (integer and value != int(value))):
        raise ValueError(f"{where}: invalid {'integer' if integer else 'number'} {value!r}")
    return int(value) if integer else float(value)


def _range(value, where):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{where} must be [start_hours, end_hours]")
    bounds = [_number(item, where) for item in value]
    if bounds[0] >= bounds[1]:
        raise ValueError(f"{where} must have start < end")
    return bounds


def _question(value, name):
    block = _object(value, name)
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"{name}.enabled must be true or false")
    if not enabled:
        _known_keys(block, {"enabled"}, name)
        return Settings({"enabled": False})
    allowed = {"enabled", "statistic", "evidence"}
    if name == "lag":
        allowed |= {"range_hours", "resolution_hours", "peak_resolution"}
    if name == "between_cells":
        allowed |= {"experimental_unit", "aggregation"}
    _known_keys(block, allowed, name)
    result = {**block, "enabled": True}
    result["statistic"] = text_key(block.get("statistic"), f"{name}.statistic")
    evidence = _object(block.get("evidence"), f"{name}.evidence")
    text_key(evidence.get("method"), f"{name}.evidence.method")
    # Preserve the explicit model and every setting. Validation against a
    # supported method belongs to its producer, never to legacy coupling defaults.
    if evidence["method"] == "none" and set(evidence) != {"method"}:
        raise ValueError(f"{name}.evidence: none accepts no additional settings")
    result["evidence"] = Settings(evidence).as_dict()
    if name == "lag":
        low, high = _range(block.get("range_hours"), "lag.range_hours")
        step = _number(block.get("resolution_hours"), "lag.resolution_hours", positive=True)
        count = (high - low) / step
        if not math.isclose(count, round(count), abs_tol=1e-8, rel_tol=0):
            raise ValueError("lag.resolution_hours must divide the declared range exactly")
        if count > 10000:
            raise ValueError("lag search exceeds 10001 points; increase the physical resolution")
        result.update(range_hours=[low, high], resolution_hours=step,
                      peak_resolution=Settings(_object(block.get("peak_resolution"), "lag.peak_resolution")).as_dict())
    if name == "between_cells":
        if block.get("experimental_unit") not in {"cell", "biological_sample"}:
            raise ValueError("between_cells.experimental_unit must be cell or biological_sample")
        aggregation = block.get("aggregation")
        if block["experimental_unit"] == "biological_sample":
            if aggregation not in {"mean", "median"}:
                raise ValueError("between_cells.aggregation must explicitly be mean or median for biological samples")
        elif aggregation is not None:
            raise ValueError("between_cells.aggregation applies only to biological_sample units")
    return Settings(result)


def _support(value):
    block = _object(value, "support")
    required = {"min_observations", "min_span_hours", "max_gap_hours", "matching", "matching_tolerance_hours"}
    _known_keys(block, required, "support")
    if required - block.keys():
        raise ValueError("support requires " + ", ".join(sorted(required - block.keys())))
    if block["matching"] not in {"exact", "nearest_unique"}:
        raise ValueError("support.matching must be exact or nearest_unique")
    result = {**block,
        "min_observations": _number(block["min_observations"], "support.min_observations", minimum=2, integer=True),
        "min_span_hours": _number(block["min_span_hours"], "support.min_span_hours", minimum=0),
        "max_gap_hours": _number(block["max_gap_hours"], "support.max_gap_hours", positive=True),
        "matching_tolerance_hours": _number(block["matching_tolerance_hours"], "support.matching_tolerance_hours", minimum=0)}
    if result["matching"] == "exact" and result["matching_tolerance_hours"] != 0:
        raise ValueError("exact matching requires matching_tolerance_hours=0")
    return Settings(result)


@dataclass(frozen=True)
class RelationshipRequest(Record):
    name: str
    declaration: Settings
    measurements: tuple[MeasurementChoice, ...]
    pairs: tuple[MeasurementPair, ...]
    within_cell: Settings
    lag: Settings
    between_cells: Settings
    support: Settings
    time_range_hours: tuple[float, float] | None
    representation: str
    detrending: Settings
    inference: Settings
    sample_summary: Settings
    biological_samples: Settings
    table_grains: Settings
    cells: tuple[tuple[str, int], ...] | None = None
    pipeline: str = "measurement-relationships"

    @classmethod
    def from_dict(cls, block, groups, *, where="pipeline"):
        _known_keys(block, {"pipeline", "name", "measurements", "pairs", "within_cell", "lag", "between_cells",
            "support", "time_range_hours", "representation", "detrending", "inference", "sample_summary",
            "biological_samples", "cells", "table_grains"}, where)
        if block.get("pipeline") != "measurement-relationships":
            raise ValueError(f"{where}: expected pipeline 'measurement-relationships'")
        name = text_key(block.get("name", "measurement-relationships"), f"{where}.name")
        if re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None:
            raise ValueError(f"{where}.name must be a lower-case identifier")
        measurements = _choices(block.get("measurements"), groups, f"{where}.measurements")
        if len(measurements) < 2:
            raise ValueError("measurements must contain at least two distinct columns")
        questions = [_question(block.get(key, {}), key) for key in ("within_cell", "lag", "between_cells")]
        if not any(q["enabled"] for q in questions):
            raise ValueError("Enable at least one of within_cell, lag or between_cells")
        pairs = _pairs(block.get("pairs"), tuple(m.column for m in measurements))
        pair_block = block.get("pairs", {}) or {}
        if pair_block.get("mode") == "explicit":
            seen = {}
            for reference, target in pair_block["pairs"]:
                key = tuple(sorted((reference, target)))
                if key in seen and seen[key] != (reference, target):
                    raise ValueError("pairs contains conflicting reversed directions; choose one declared direction")
                seen[key] = (reference, target)
        bounds = block.get("time_range_hours")
        bounds = None if bounds is None else tuple(_range(bounds, "time_range_hours"))
        representation = block.get("representation")
        if representation not in {"raw", "detrended"}:
            raise ValueError("representation must explicitly be raw or detrended")
        detrending = _object(block.get("detrending", {}), "detrending")
        if representation == "raw" and detrending:
            raise ValueError("raw representation cannot also request detrending")
        if representation == "detrended":
            text_key(detrending.get("detrend"), "detrending.detrend")
        between = questions[2]["enabled"]
        for m in measurements:
            if m.summary is not None and (not between or m.summary not in SUMMARIES):
                raise ValueError(f"{m.column}: summary requires between_cells and one of {', '.join(SUMMARIES)}")
        inference = _object(block.get("inference", {}), "inference")
        _known_keys(inference, {"alpha", "multiple_testing", "correction_scope"}, "inference")
        inferential = any(q["enabled"] and q["evidence"]["method"] != "none" for q in questions)
        sample_summary = _object(block.get("sample_summary", {}), "sample_summary")
        _known_keys(sample_summary, {"enabled", "aggregation", "evidence"}, "sample_summary")
        enabled = sample_summary.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("sample_summary.enabled must be true or false")
        if enabled:
            if sample_summary.get("aggregation") not in {"mean", "median"}:
                raise ValueError("sample_summary.aggregation must explicitly be mean or median")
            evidence = _object(sample_summary.get("evidence"), "sample_summary.evidence")
            text_key(evidence.get("method"), "sample_summary.evidence.method")
        elif set(sample_summary) - {"enabled"}:
            raise ValueError("Disabled sample_summary accepts only enabled")
        inferential = inferential or (enabled and sample_summary["evidence"]["method"] != "none")
        if inferential:
            alpha = _number(inference.get("alpha"), "inference.alpha", positive=True)
            if alpha >= 1:
                raise ValueError("inference.alpha must be less than one")
            if inference.get("multiple_testing") not in {"none", "bh", "bonferroni", "sidak"}:
                raise ValueError("inference.multiple_testing must be none, bh, bonferroni or sidak")
            if inference.get("correction_scope") not in {"all", "pair", "movie_pair"}:
                raise ValueError("inference.correction_scope must be all, pair or movie_pair")
        elif inference:
            raise ValueError("inference settings require an enabled evidence method")
        samples = _object(block.get("biological_samples", {}), "biological_samples")
        for movie, sample in samples.items():
            text_key(movie, "biological_samples movie")
            text_key(sample, f"biological_samples.{movie}")
        selected = None
        if block.get("cells") is not None:
            if not isinstance(block["cells"], list):
                raise ValueError("cells must be a list of movie/identity objects")
            selected = []
            for cell in block["cells"]:
                cell = _object(cell, "cells")
                _known_keys(cell, {"movie", "identity"}, "cells")
                key = (text_key(cell.get("movie"), "cells.movie"), cell_number(cell.get("identity")))
                if key not in selected:
                    selected.append(key)
        return cls(name, Settings(block), measurements, pairs, *questions, _support(block.get("support")),
            bounds, representation, Settings(detrending), Settings(inference),
            Settings({**sample_summary, "enabled": enabled}), Settings(samples),
            Settings(_object(block.get("table_grains", {}), "table_grains")),
            None if selected is None else tuple(selected))


@dataclass(frozen=True)
class ResolvedRelationshipRequest(Record):
    request: RelationshipRequest
    inputs: InputIdentity
    measurements: tuple[Measurement, ...]
    detrending: Settings
    processing_version: str | None

    @property
    def scientific_id(self):
        # Names and raw spelling cannot redefine a scientific question.
        request = self.request.as_dict()
        request.pop("name")
        request.pop("declaration")
        return content_id({"request": request, "inputs": self.inputs, "measurements": self.measurements,
                           "detrending": self.detrending, "processing_version": self.processing_version})

    @property
    def expected_pairs(self):
        return tuple(Settings({"cell": cell.as_dict(), "pair": pair.as_dict()})
                     for cell in self.inputs.cells for pair in self.request.pairs)


def resolve_request(request, *, source_run, tables, input_hashes, rhythm_params=None, table_grains=None):
    """Resolve measured table grain and identities; do not fit or pair observations."""
    import pandas as pd
    text_key(source_run, "source_run")
    if any(not isinstance(name, str) or not isinstance(frame, pd.DataFrame) for name, frame in tables.items()):
        raise ValueError("tables must map names to measured DataFrames")
    declared_grains = request.table_grains.as_dict()
    for name, grain in (table_grains or {}).items():
        if name in declared_grains and tuple(grain) != tuple(declared_grains[name]):
            raise ValueError(f"Conflicting table_grains for {name}")
        declared_grains[name] = grain
    declarations, grains = _catalogue(declared_grains)
    needs_traces = request.within_cell["enabled"] or request.lag["enabled"]
    measurements = []
    for choice in request.measurements:
        measurement = _resolve_measurement(replace(choice, summary=None) if needs_traces else choice,
            tables, declarations, grains, testing=needs_traces)
        measurement = replace(measurement, summary=choice.summary)
        trace = "hours" in measurement.grain or "frame_index" in measurement.grain
        if request.between_cells["enabled"] and trace and choice.summary is None:
            raise ValueError(f"{choice.column}: between_cells requires an explicit trace summary")
        if not trace and request.time_range_hours is not None:
            raise ValueError(f"{choice.column}: a saved whole-recording scalar cannot represent a new time range")
        if not trace and request.representation == "detrended":
            raise ValueError(f"{choice.column}: a scalar cannot be detrended")
        measurements.append(measurement)
    names = {m.table for m in measurements}
    if "cell_summary" in tables:
        problem = _shape_problem(tables["cell_summary"], grains["cell_summary"], summary=None, testing=False)
        if problem:
            raise ValueError("cell_summary population: " + problem)
        names.add("cell_summary")
    cells, samples = _population(source_run, tables, sorted(names), request)
    fingerprints = {}
    for name in sorted(names):
        fingerprint = input_hashes.get(name)
        if not isinstance(fingerprint, str) or re.fullmatch(r"[a-fA-F0-9]{64}", fingerprint) is None:
            raise ValueError(f"input_hashes.{name}: verified SHA-256 is required")
        fingerprints[name] = fingerprint.lower()
    detrending, processing_version = Settings(), None
    if request.representation == "detrended":
        import pymicroglia.workbench as circadian
        _known_keys(request.detrending, set(circadian.DETREND_DEFAULTS), "detrending")
        detrending = Settings(circadian.detrend_settings(request.detrending.as_dict()))
        processing_version = circadian.WORKBENCH_VERSION
    return ResolvedRelationshipRequest(request, InputIdentity(source_run, Settings(fingerprints), cells, samples),
                                       tuple(measurements), detrending, processing_version)


ANALYSIS_STEPS = (
    StepSpec("paired-inputs", "paired-inputs", ("relationship-design",), inputs=("measured-tables",)),
    StepSpec("within-cell-association", "within-cell-association", ("paired-inputs",),
             inputs=("paired-inputs:inventory", "paired-inputs:same_time_pairs", "paired-inputs:traces")),
    # Delays never depend on a same-time significant selection.
    StepSpec("lag-association", "lag-association", ("paired-inputs",),
             inputs=("paired-inputs:inventory", "paired-inputs:traces", "paired-inputs:lag_support")),
    StepSpec("between-cell-association", "between-cell-association", ("paired-inputs",),
             inputs=("paired-inputs:inventory", "paired-inputs:traces", "paired-inputs:scalars", "paired-inputs:trace_inventory")),
    StepSpec("sample-consistency", "sample-consistency",
             ("within-cell-association", "lag-association", "between-cell-association"),
             inputs=("within-cell-association:results", "lag-association:results", "lag-association:profiles")),
)
RECIPE = PipelineRecipe("measurement-relationships", 1, (
    StepSpec("relationship-design", "relationship-design", inputs=("measured-tables",)), *ANALYSIS_STEPS,
    StepSpec("relationship-overview", "relationship-overview",
        ("paired-inputs", "within-cell-association", "lag-association", "between-cell-association", "sample-consistency"),
        inputs=("within-cell-association:results", "between-cell-association:results", "sample-consistency:summaries"), kind="render"),
    StepSpec("relationship-report-selection", "relationship-report-selection", ("within-cell-association", "lag-association"),
        inputs=("within-cell-association:results", "lag-association:results")),
    StepSpec("relationship-reports", "relationship-reports", ("relationship-report-selection", "paired-inputs", "within-cell-association", "lag-association"),
        inputs=("relationship-report-selection:report_members", "paired-inputs:traces", "paired-inputs:same_time_pairs"),
        selection="relationship-report-selection:supported-relationships", requires_selected_rows=True, kind="render"),
    StepSpec("relationship-lag-profiles", "relationship-lag-profiles", ("lag-association",),
        inputs=("lag-association:results", "lag-association:profiles"), kind="render"),
    StepSpec("relationship-populations", "relationship-populations", ("paired-inputs", "between-cell-association", "sample-consistency"),
        inputs=("sample-consistency:members", "sample-consistency:summaries", "between-cell-association:paired_scalars"), kind="render"),
    StepSpec("linked-results-index", "linked-results-index", ("relationship-design", "paired-inputs", "within-cell-association", "lag-association",
        "between-cell-association", "sample-consistency", "relationship-report-selection", "relationship-overview", "relationship-reports",
        "relationship-lag-profiles", "relationship-populations"), kind="render")))


def _identity(context):
    from pymicroglia.pipelines._screening import file_hash, read_verified_tables
    read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    return content_id({"request": context.request.scientific_id, "code": {
        name: file_hash(source_file(name))
        for name in ("relationship_options.py", "rhythm_discovery.py", "contracts.py")}})


def freeze_design(context):
    from pymicroglia.pipelines._screening import _write_json, file_hash
    context.output.mkdir(parents=True)
    path = context.output / "relationship_design.json"
    _write_json(path, {"schema_version": 1, "scientific_id": context.scientific_id,
        "request": context.request.as_dict(), "lag_convention": LAG_CONVENTION,
        "requested_cell_pairs": len(context.request.expected_pairs),
        "family_population": "All declared cell/pair hypotheses per question; selections cannot change families",
        "scientific_evaluation": False})
    return StepResult(context.step.name, context.scientific_id, "completed", "Frozen relationship inputs and explicit questions",
        (ArtifactRef("relationship_design", path.name, file_hash(path), context.scientific_id),),
        provenance=Settings({"lag_convention": LAG_CONVENTION}))


def _pending(context):
    from pymicroglia.pipelines._runner import Unavailable
    raise Unavailable(f"Required {context.step.name} producer is awaiting its implementation stage")


def producers():
    from pymicroglia.pipelines._runner import Producer
    import pymicroglia.pipelines.relationships.inputs as relationship_inputs
    import pymicroglia.pipelines.relationships.association as relationship_association
    import pymicroglia.pipelines.relationships.lag as relationship_lag
    import pymicroglia.pipelines.relationships.between as relationship_between
    import pymicroglia.pipelines.relationships.consistency as relationship_consistency
    import pymicroglia.pipelines.relationships.figures as relationship_figures
    import pymicroglia.pipelines.relationships.reports as relationship_reports
    import pymicroglia.pipelines.relationships.report_figures as relationship_report_figures
    import pymicroglia.pipelines.relationships.lag_figures as relationship_lag_figures
    import pymicroglia.pipelines.relationships.population_figures as relationship_population_figures
    import pymicroglia.pipelines.relationships.index as relationship_index
    return {**{step.producer: Producer(_pending) for step in ANALYSIS_STEPS},
            "relationship-design": Producer(freeze_design, _identity),
            "paired-inputs": Producer(relationship_inputs.produce, relationship_inputs.identity),
            "within-cell-association": Producer(relationship_association.produce, version=relationship_association.implementation_version()),
            "lag-association": Producer(relationship_lag.produce, version=relationship_lag.implementation_version()),
            "between-cell-association": Producer(relationship_between.produce, version=relationship_between.implementation_version()),
            "sample-consistency": Producer(relationship_consistency.produce, version=relationship_consistency.implementation_version()),
            "relationship-overview": Producer(relationship_figures.produce_overview, version=relationship_figures.version()),
            "relationship-report-selection": Producer(relationship_reports.select_reports, version=relationship_reports.selection_version()),
            "relationship-reports": Producer(relationship_report_figures.produce, version=relationship_report_figures.version()),
            "relationship-lag-profiles": Producer(relationship_lag_figures.produce, version=relationship_lag_figures.version()),
            "relationship-populations": Producer(relationship_population_figures.produce, version=relationship_population_figures.version()),
            "linked-results-index": Producer(relationship_index.produce, version=relationship_index.version(), accepts_unavailable_dependencies=True)}


def run_request(resolved, table_paths, output, *, presentation=None, only=None):
    from pymicroglia.pipelines._runner import run_pipeline
    return run_pipeline(RECIPE, producers(), request=resolved,
        scientific_settings={"relationships": resolved.scientific_id}, table_paths=table_paths,
        output=output, presentation=presentation, only=only)
