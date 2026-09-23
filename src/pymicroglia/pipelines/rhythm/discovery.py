"""Free measurement choices and the saved-input contract for rhythm discovery.

Configuration parsing is structural and does not import the scientific engine.
``resolve_request`` checks actual table grain, numeric columns and input identity,
then resolves all scientific options through ``analysis.circadian``. Neither
operation fits a trace, summarises a comparison measurement or expands plots.
"""

from __future__ import annotations
from pymicroglia._results import read_document

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import TYPE_CHECKING

from pymicroglia.pipelines._contracts import CellKey, CellMeasurementKey, InputIdentity, Measurement, MeasurementPair, PipelineRecipe, Record, SampleAssignment, Settings, StepSpec, cell_number, text_key

if TYPE_CHECKING:
    import pandas as pd


def _screen_id(context):
    import pymicroglia.workbench as circadian
    from pymicroglia.pipelines._screening import read_verified_tables, screen_identity

    read_verified_tables(context.table_paths, context.request.screen_inputs.table_hashes)
    if (context.request.workbench_version != circadian.WORKBENCH_VERSION or
            context.request.period_methods.as_dict() != circadian.PERIOD_METHODS):
        raise ValueError("Workbench methods changed after request resolution")
    return screen_identity(context.request)


def _produce_screen(context):
    from dataclasses import replace
    from pymicroglia.pipelines._screening import screen

    return replace(screen(context.request, context.table_paths, context.output).outcome,
                   step=context.step.name)


def _validate_screen(path, scientific_id):
    from pymicroglia.pipelines._screening import read_screen

    return read_screen(path, expected_id=scientific_id)


def producers():
    """Concrete registered steps; later saved-result analyses extend this map."""
    from pymicroglia.pipelines._runner import Producer
    from pymicroglia.pipelines.rhythm.figures import produce_overview, implementation_version
    import pymicroglia.pipelines.rhythm.reports as rhythm_reports
    import pymicroglia.pipelines.rhythm.time_matrices as time_matrices
    import pymicroglia.pipelines.rhythm.groups as rhythm_groups
    import pymicroglia.pipelines.rhythm.group_figures as rhythm_group_figures
    import pymicroglia.pipelines.rhythm.agreement as rhythm_agreement
    import pymicroglia.pipelines.rhythm.agreement_figures as rhythm_agreement_figures
    import pymicroglia.pipelines.rhythm.timing as rhythm_timing
    import pymicroglia.pipelines.rhythm.timing_samples as timing_samples
    import pymicroglia.pipelines.rhythm.relationship_figures as rhythm_relationship_figures
    import pymicroglia.pipelines.rhythm.index as rhythm_index

    return {"rhythm-screen": Producer(_produce_screen, _screen_id, _validate_screen),
            "screening-overview": Producer(produce_overview, version=implementation_version()),
            "selected-cell-evidence": Producer(rhythm_reports.produce, version=rhythm_reports.implementation_version()),
            "time-matrix-values": Producer(time_matrices.produce_values, version=time_matrices.implementation_version()),
            "time-matrices": Producer(time_matrices.produce_figures, version=time_matrices.figure_version()),
            "group-comparisons": Producer(rhythm_groups.produce, rhythm_groups.identity, version=rhythm_groups.version()),
            "group-comparison-figures": Producer(rhythm_group_figures.produce, version=rhythm_group_figures.version()),
            "detection-agreement": Producer(rhythm_agreement.produce, rhythm_agreement.identity, version=rhythm_agreement.version()),
            "detection-agreement-figures": Producer(rhythm_agreement_figures.produce, version=rhythm_agreement_figures.version()),
            "within-cell-timing": Producer(rhythm_timing.produce, rhythm_timing.identity, version=rhythm_timing.version()),
            "timing-across-samples": Producer(timing_samples.produce, timing_samples.identity, version=timing_samples.version()),
            "timing-relationship-figures": Producer(rhythm_relationship_figures.produce, version=rhythm_relationship_figures.version()),
            "linked-results-index": Producer(rhythm_index.produce, version=rhythm_index.version(), accepts_unavailable_dependencies=True)}


def run_request(resolved, table_paths, output, *, presentation=None, only=None,
                recipe=None, registry=None):
    from pymicroglia.pipelines._runner import run_pipeline
    from pymicroglia.pipelines._screening import screen_identity

    return run_pipeline(recipe or RECIPE, registry or producers(), request=resolved,
                        scientific_settings={"screen": screen_identity(resolved)},
                        table_paths=table_paths, output=output, presentation=presentation,
                        only=only)


def command(args):
    """Resolve declared requests against one existing measurement run."""
    import json
    from pathlib import Path
    from pymicroglia.measure.spec import MeasureConfig
    from pymicroglia.measure.pool import movie_folders, SOURCE_FOLDERS
    from pymicroglia.pipelines import parse
    from pymicroglia.pipelines._screening import file_hash, read_verified_tables

    run = Path(args.run).resolve()
    from pymicroglia.pipelines._requests import inputs
    manifest, source_identity, paths, hashes, tables, inherited = inputs(run)
    loaded_config = MeasureConfig.load(args.config) if args.config else None
    if loaded_config is not None:
        requests = loaded_config.pipelines
    else:
        declared = read_document(Path(args.request))
        requests = parse(declared if isinstance(declared, list) else
                         declared.get("pipelines", [declared]))
    if args.name:
        known = {request.name for request in requests}
        if set(args.name) - known:
            raise ValueError(f"unknown pipeline requests: {sorted(set(args.name) - known)}")
        requests = [request for request in requests if request.name in args.name]
    if not requests:
        raise ValueError("No pipeline requests were declared")
    appearance = (read_document(Path(args.presentation))
                  if args.presentation else {})
    output = Path(args.out).resolve() if args.out else run / "pipelines"
    successful = True
    for request in requests:
        if (request.pipeline in {"rhythm-discovery", "method-selection-audit"}
                and inherited and any(value != inherited[0] for value in inherited[1:])):
            raise ValueError("Movie rhythm settings differ; a shared inherited recipe is ambiguous")
        if request.pipeline == "method-selection-audit":
            from pymicroglia.pipelines.audit.workflow import resolve_request as resolver, run_request as execute
        elif request.pipeline == "measurement-relationships":
            from pymicroglia.pipelines.relationships.options import resolve_request as resolver, run_request as execute
        elif request.pipeline == "cell-behaviour-states":
            from pymicroglia.pipelines.behaviour.options import resolve_request as resolver, run_request as execute
        elif request.pipeline == "spatial-coordination":
            from pymicroglia.pipelines.coordination.options import resolve_request as resolver, run_request as execute
        elif request.pipeline == "intervention-response":
            from pymicroglia.pipelines.intervention.options import resolve_request as resolver, run_request as execute
        else:
            resolver, execute = resolve_request, run_request
        extra = {}
        if request.pipeline == "intervention-response":
            if request.rhythms['enabled'] and inherited and any(value != inherited[0] for value in inherited[1:]):
                raise ValueError('Movie rhythm settings differ; intervention rhythm comparisons cannot inherit an ambiguous recipe')
            if loaded_config is not None:
                from dataclasses import asdict
                extra['recording_windows'] = {movie.stem: [
                    {key: value for key, value in asdict(window).items() if value is not None}
                    for window in movie.windows] for movie in loaded_config.movies if movie.windows}
                extra['default_windows'] = [{key: value for key, value in asdict(window).items() if value is not None} for window in loaded_config.windows]
        resolved = resolver(request, source_run=source_identity, tables=tables,
                                   input_hashes=hashes, rhythm_params=inherited[0] if inherited else {}, **extra)
        execution = execute(resolved, paths, output / request.name,
                                presentation=appearance, only=tuple(args.step) if args.step else None)
        for name, saved in execution.results.items():
            print(f"{request.name} / {name}: {saved.outcome.status} — {saved.outcome.reason}")
        print(f"Execution record: {execution.record_path}")
        report = execution.results.get("linked-results-index")
        if report is not None and report.outcome.status in {"completed", "reused"}:
            from pymicroglia.pipelines.rhythm.index import openable_report
            print(f"Results index: {openable_report(report)}")
        successful = successful and execution.successful
    return 0 if successful else 1


@dataclass(frozen=True)
class MeasurementChoice(Record):
    column: str
    table: str | None = None
    summary: str | None = None


def _object(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    return value


def _known_keys(block: dict, keys: set[str], where: str) -> None:
    unknown = set(block) - keys
    if unknown:
        raise ValueError(f"{where}: unknown setting(s) {', '.join(sorted(unknown))}")


def _choices(values: object, groups: dict, where: str) -> tuple[MeasurementChoice, ...]:
    from pymicroglia.measure.metric_groups import resolve_metrics

    if not isinstance(values, list):
        raise ValueError(f"{where}: expected a list of measurements")
    found: dict[str, MeasurementChoice] = {}
    for entry in values:
        if isinstance(entry, str):
            text_key(entry, where)
            expanded = [MeasurementChoice(column) for column in
                        resolve_metrics([entry], groups, where=where)]
        elif isinstance(entry, dict):
            _known_keys(entry, {"column", "table", "summary"}, where)
            column = text_key(entry.get("column"), f"{where}.column")
            if column.startswith("@"):
                raise ValueError(f"{where}: put a group directly in the list; "
                                 "a table-qualified choice names one column")
            table = entry.get("table")
            summary = entry.get("summary")
            if table is not None:
                text_key(table, f"{where}.table")
            if summary is not None:
                text_key(summary, f"{where}.summary")
            expanded = [MeasurementChoice(column, table, summary)]
        else:
            raise ValueError(f"{where}: expected a column name, group or choice object")
        for choice in expanded:
            previous = found.get(choice.column)
            if previous is not None and previous != choice:
                raise ValueError(f"{where}: {choice.column!r} has conflicting table/summary choices")
            found.setdefault(choice.column, choice)
    return tuple(found.values())


def _pairs(block: object, measurements: tuple[str, ...]) -> tuple[MeasurementPair, ...]:
    if block is None:
        block = {"mode": "all"}
    block = _object(block, "pairs")
    _known_keys(block, {"mode", "pairs", "reference"}, "pairs")
    mode = block.get("mode", "all")
    if mode == "all":
        if set(block) - {"mode"}:
            raise ValueError("pairs: all mode accepts only mode")
        chosen = list(combinations(measurements, 2))
    elif mode == "reference":
        if "pairs" in block:
            raise ValueError("pairs: reference mode does not take explicit pairs")
        reference = block.get("reference")
        if reference not in measurements:
            raise ValueError("pairs.reference must name a test measurement")
        chosen = [(reference, name) for name in measurements if name != reference]
    elif mode == "explicit":
        if "reference" in block:
            raise ValueError("pairs: explicit mode takes direction from each pair")
        chosen = block.get("pairs")
        if not isinstance(chosen, list):
            raise ValueError("pairs.pairs must be a list of [reference, target] pairs")
    else:
        raise ValueError("pairs.mode must be all, explicit or reference")
    resolved: dict[tuple[str, str], MeasurementPair] = {}
    for pair in chosen:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each pair must be [reference, target]")
        if any(not isinstance(name, str) or name not in measurements for name in pair):
            raise ValueError("each pair must name two selected test measurements")
        reference, target = pair
        if reference == target:
            raise ValueError("self-pairs are not allowed")
        result = MeasurementPair(reference, target)
        # Reversed duplicates describe one detection pair. First direction wins
        # and is retained explicitly for later timing calculations.
        resolved.setdefault(result.detection_key, result)
    return tuple(resolved.values())


@dataclass(frozen=True)
class RhythmDiscoveryRequest(Record):
    name: str
    declaration: Settings
    test_measurements: tuple[MeasurementChoice, ...]
    comparison_measurements: tuple[MeasurementChoice, ...]
    pairs: tuple[MeasurementPair, ...]
    analysis_options: Settings
    biological_samples: Settings
    cells: tuple[tuple[str, int], ...] | None = None
    correction_scope: str = "all"
    pipeline: str = "rhythm-discovery"
    settings_profile: str | None = None
    group_comparisons: Settings = field(default_factory=Settings)
    detection_agreement: Settings = field(default_factory=Settings)
    timing: Settings = field(default_factory=Settings)
    timing_summary: Settings = field(default_factory=Settings)

    @classmethod
    def from_dict(cls, block: dict, groups: dict, *, where: str = "pipeline"):
        _known_keys(block, {"pipeline", "name", "test_measurements",
                           "comparison_measurements", "pairs", "analysis_options",
                           "biological_samples", "cells", "correction_scope", "settings_profile", "group_comparisons", "detection_agreement", "timing", "timing_summary"}, where)
        if block.get("pipeline") != "rhythm-discovery":
            raise ValueError(f"{where}: expected pipeline 'rhythm-discovery'")
        name = text_key(block.get("name", "rhythm-discovery"), f"{where}.name")
        if re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None:
            raise ValueError(f"{where}.name must be a lower-case identifier")
        tested = _choices(block.get("test_measurements"), groups,
                          f"{where}.test_measurements")
        if not tested:
            raise ValueError(f"{where}: test_measurements must not be empty")
        if any(choice.summary is not None for choice in tested):
            raise ValueError(f"{where}: test_measurements need time series, not summaries")
        compared = _choices(block.get("comparison_measurements", []), groups,
                            f"{where}.comparison_measurements")
        options = _object(block.get("analysis_options", {}), f"{where}.analysis_options")
        samples = _object(block.get("biological_samples", {}), f"{where}.biological_samples")
        for movie, sample in samples.items():
            text_key(movie, "biological_samples movie")
            text_key(sample, f"biological_samples.{movie}")
        cells = block.get("cells")
        selected = None
        if cells is not None:
            if not isinstance(cells, list):
                raise ValueError(f"{where}.cells must be a list of movie/identity objects")
            selected = []
            for entry in cells:
                entry = _object(entry, f"{where}.cells")
                _known_keys(entry, {"movie", "identity"}, f"{where}.cells")
                key = (text_key(entry.get("movie"), "cells.movie"),
                       cell_number(entry.get("identity")))
                if key not in selected:
                    selected.append(key)
        scope = block.get("correction_scope", "all")
        if scope not in ("all", "measurement", "movie_measurement"):
            raise ValueError("correction_scope must be all, measurement or movie_measurement")
        profile = block.get("settings_profile")
        if profile is not None:
            text_key(profile, "settings_profile")
            if options:
                raise ValueError("A complete settings_profile cannot be combined with analysis_options overrides")
        from pymicroglia.pipelines.rhythm.groups import validate_options
        comparisons = validate_options(block.get("group_comparisons", {}),
                                       [m.column for m in tested], [m.column for m in compared])
        from pymicroglia.pipelines.rhythm.agreement import validate_options as agreement_options
        agreement = agreement_options(block.get("detection_agreement", {}))
        from pymicroglia.pipelines.rhythm.timing import validate_options as timing_options
        timing = timing_options(block.get("timing", {}), [m.column for m in tested])
        from pymicroglia.pipelines.rhythm.timing_samples import validate_options as summary_options
        timing_summary = summary_options(block.get("timing_summary", {}))
        return cls(name, Settings(block), tested, compared,
                   _pairs(block.get("pairs"), tuple(c.column for c in tested)),
                   Settings(options), Settings(samples),
                   None if selected is None else tuple(selected), scope, settings_profile=profile,
                   group_comparisons=Settings(comparisons), detection_agreement=Settings(agreement), timing=Settings(timing), timing_summary=Settings(timing_summary))


@dataclass(frozen=True)
class ResolvedRhythmRequest(Record):
    request: RhythmDiscoveryRequest
    inputs: InputIdentity
    test_measurements: tuple[Measurement, ...]
    comparison_measurements: tuple[Measurement, ...]
    analysis_options: Settings
    rhythm_params: Settings
    workbench_version: str
    period_methods: Settings
    measurement_recipes: Settings = field(default_factory=Settings)
    profile_provenance: Settings = field(default_factory=Settings)

    @property
    def screen_inputs(self):
        """Comparison-only tables cannot invalidate an unchanged rhythm screen."""
        names = {m.table for m in self.test_measurements} | {"cell_summary"}
        return replace(self.inputs, table_hashes=Settings({k: v for k, v in self.inputs.table_hashes.items() if k in names}))

    @property
    def expected_pairs(self) -> tuple[CellMeasurementKey, ...]:
        """Complete screening population, including cells without usable traces."""
        return tuple(CellMeasurementKey(cell, metric.column)
                     for cell in self.inputs.cells for metric in self.test_measurements)


# The overview is unconditional after a successful screen, including empty
# significant selections. Subsequent reports use the saved selection records.
RECIPE = PipelineRecipe("rhythm-discovery", 1, (
    StepSpec("rhythm-screen", "rhythm-screen", inputs=("measured-tables",)),
    StepSpec("screening-overview", "screening-overview", ("rhythm-screen",),
             inputs=("rhythm-screen:rhythm_results", "rhythm-screen:correction_families"), kind="render"),
    StepSpec("selected-cell-evidence", "selected-cell-evidence", ("rhythm-screen",),
             inputs=("rhythm-screen:trace_inputs", "rhythm-screen:display_inputs"),
             selection="rhythm-screen:any-significant", kind="render"),
    StepSpec("time-matrix-values", "time-matrix-values", ("rhythm-screen",),
             inputs=("rhythm-screen:trace_inputs", "rhythm-screen:display_inputs"), kind="render"),
    StepSpec("time-matrices", "time-matrices", ("rhythm-screen", "time-matrix-values"),
             inputs=("time-matrix-values:values", "time-matrix-values:status"), kind="render"),
    StepSpec("group-comparisons", "group-comparisons", ("rhythm-screen",),
             inputs=("rhythm-screen:rhythm_results", "measured-tables")),
    StepSpec("group-comparison-figures", "group-comparison-figures", ("group-comparisons",),
             inputs=("group-comparisons:cells", "group-comparisons:statistics"), kind="render"),
    StepSpec("detection-agreement", "detection-agreement", ("rhythm-screen",),
             inputs=("rhythm-screen:rhythm_results", "rhythm-screen:correction_families")),
    StepSpec("detection-agreement-figures", "detection-agreement-figures", ("detection-agreement",),
             inputs=("detection-agreement:summary", "detection-agreement:units"), kind="render"),
    StepSpec("within-cell-timing", "within-cell-timing", ("rhythm-screen",),
             inputs=("rhythm-screen:rhythm_results", "rhythm-screen:trace_inputs")),
    StepSpec("timing-across-samples", "timing-across-samples", ("rhythm-screen", "within-cell-timing"),
             inputs=("within-cell-timing:pairs", "within-cell-timing:engine_details", "rhythm-screen:correction_families")),
    StepSpec("timing-relationship-figures", "timing-relationship-figures",
             ("rhythm-screen", "detection-agreement", "within-cell-timing", "timing-across-samples"),
             inputs=("timing-across-samples:summary", "within-cell-timing:timecourse"), kind="render"),
    StepSpec("linked-results-index", "linked-results-index", (
        "rhythm-screen", "screening-overview", "selected-cell-evidence", "time-matrix-values", "time-matrices",
        "group-comparisons", "group-comparison-figures", "detection-agreement", "detection-agreement-figures",
        "within-cell-timing", "timing-across-samples", "timing-relationship-figures"), kind="render"),
))


def _catalogue(table_grains: Mapping | None):
    import pymicroglia.measure.modules
    pymicroglia.measure.modules.load()
    from pymicroglia.measure.declare import declared_columns, declared_tables
    from pymicroglia.measure.summarise import ROLLUPS

    grains = {name: output.grain for name, output in declared_tables().items()}
    grains.update({output.name: output.grain for output in ROLLUPS})
    for name, grain in (table_grains or {}).items():
        if not isinstance(grain, (tuple, list)) or not grain:
            raise ValueError(f"table_grains.{name} must name a non-empty row key")
        if any(not isinstance(key, str) or not key for key in grain):
            raise ValueError(f"table_grains.{name} must contain column names")
        if name in grains and set(grains[name]) != set(grain):
            raise ValueError(f"table_grains.{name} conflicts with its existing declaration")
        grains[name] = tuple(grain)
    return declared_columns(), grains


def _shape_problem(frame, grain: tuple[str, ...] | None, *, summary: str | None,
                   testing: bool) -> str | None:
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

    if not isinstance(frame, pd.DataFrame):
        return "input is not a measured table"
    if not frame.columns.is_unique:
        return "duplicate column names"
    if grain is None:
        return "table grain is undeclared; provide its row-key metadata"
    cell_grain = set(grain) - {"stem"}
    trace = cell_grain in ({"identity", "frame_index"}, {"identity", "hours"})
    scalar = cell_grain == {"identity"}
    if testing and not trace:
        return "wrong grain: a test needs one row per cell and observation"
    if not testing and not (scalar or trace):
        return "wrong grain: a comparison needs a per-cell value or declared trace summary"
    if not testing and trace and summary is None:
        return "time-varying comparison requires an explicit summary operation"
    if scalar and summary is not None:
        return "per-cell values are already summaries; remove the summary operation"
    required = set(grain) | {"stem", "identity"} | ({"hours"} if trace else set())
    missing = required - set(frame.columns)
    if missing:
        return "missing identity/time columns: " + ", ".join(sorted(missing))
    keys = list(dict.fromkeys(("stem", *grain)))
    if frame[keys].isna().any().any():
        return "missing values in the declared row key"
    if frame.duplicated(keys).any():
        return "duplicate rows at the declared grain; select the intended table/slice"
    if trace and not frame.empty and (not is_numeric_dtype(frame["hours"].dtype)
                                     or is_bool_dtype(frame["hours"].dtype)
                                     or is_complex_dtype(frame["hours"].dtype)):
        return "observation hours must be real numeric coordinates"
    return None


def _resolve_measurement(choice, tables, declarations, grains, *, testing):
    from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

    if choice.table is not None and choice.table not in tables:
        raise ValueError(f"{choice.column!r}: table {choice.table!r} is unavailable")
    names = [choice.table] if choice.table is not None else list(tables)
    present = [name for name in names if choice.column in tables[name].columns]
    if not present:
        detail = "declared but unavailable in these inputs" if choice.column in declarations \
            else "unavailable in the actual input tables"
        raise ValueError(f"measurement {choice.column!r} is {detail}")
    eligible = []
    problems = []
    for name in present:
        frame = tables[name]
        problem = _shape_problem(frame, grains.get(name), summary=choice.summary, testing=testing)
        if problem is None:
            values = frame[choice.column]
            dtype = values.dtype
            # A header-only CSV has object columns. With no observed values,
            # retain its expected cells for the producer's untestable records.
            missing_only = values.isna().all()
            if ((not is_numeric_dtype(dtype) and not missing_only)
                    or is_bool_dtype(dtype) or is_complex_dtype(dtype)):
                problem = "measurement is not a real numeric column"
        if problem is None and choice.column in set(grains[name]) | {"stem", "hours"}:
            problem = "identity/time coordinates are not measured outcomes"
        if problem:
            problems.append(f"{name}: {problem}")
        else:
            eligible.append(name)
    if not eligible:
        raise ValueError(f"measurement {choice.column!r} is unsuitable: " + "; ".join(problems))
    if len(eligible) > 1:
        raise ValueError(f"measurement {choice.column!r} is ambiguous across "
                         f"{', '.join(eligible)}; select its table explicitly")
    table = eligible[0]
    column = declarations.get(choice.column)
    return Measurement(choice.column, table, tuple(grains[table]),
                       column.label if column else choice.column,
                       column.unit if column else "", column is not None, choice.summary)


def _population(source_run, tables, names, request):
    cells: dict[CellKey, None] = {}
    subjects: dict[str, set[str]] = {}
    for name in names:
        frame = tables[name]
        for movie, identity in frame[["stem", "identity"]].drop_duplicates().itertuples(index=False, name=None):
            cells.setdefault(CellKey(source_run, movie, identity), None)
        if "subject" in frame:
            for movie, subject in frame[["stem", "subject"]].dropna().itertuples(index=False, name=None):
                subjects.setdefault(movie, set()).add(str(subject))
    if request.cells is not None:
        selected = tuple(CellKey(source_run, movie, identity) for movie, identity in request.cells)
        missing = [cell for cell in selected if cell not in cells]
        if missing:
            raise ValueError(f"requested cell is unavailable: {missing[0].movie}/{missing[0].identity}")
    else:
        selected = tuple(sorted(cells))
    movies = sorted({cell.movie for cell in selected})
    known_movies = {cell.movie for cell in cells}
    unknown = set(request.biological_samples) - known_movies
    if unknown:
        raise ValueError("biological_samples names unavailable movies: " + ", ".join(sorted(unknown)))
    samples = []
    for movie in movies:
        observed = subjects.get(movie, set())
        if len(observed) > 1:
            raise ValueError(f"movie {movie!r} has conflicting subject labels in its input tables")
        sample = request.biological_samples.get(movie)
        samples.append(SampleAssignment(movie, sample, sample is not None,
                                        next(iter(observed), None)))
    return selected, tuple(samples)


def resolve_request(request: RhythmDiscoveryRequest, *, source_run: str,
                    tables: Mapping[str, "pd.DataFrame"], input_hashes: Mapping[str, str],
                    rhythm_params: Mapping | None = None,
                    table_grains: Mapping | None = None) -> ResolvedRhythmRequest:
    """Validate a request against measured tables without changing or fitting them.

    ``input_hashes`` contains the verified file/content SHA-256 for each used
    table, obtained by the caller loading the run. Pooled tables retain ``stem``.
    Use ``table_grains`` for additional tables with explicit row-key metadata;
    existing declared grains cannot be overridden. A summary names an operation
    to be implemented/validated by the comparison producer, never an implicit
    reduction performed here.
    """
    import pandas as pd
    import pymicroglia.workbench as circadian
    from pymicroglia.measure.declare import get_derived

    text_key(source_run, "source_run")
    if any(not isinstance(name, str) or not isinstance(frame, pd.DataFrame)
           for name, frame in tables.items()):
        raise ValueError("tables must map table names to measured DataFrames")
    declarations, grains = _catalogue(table_grains)
    tests = tuple(_resolve_measurement(c, tables, declarations, grains, testing=True)
                  for c in request.test_measurements)
    comparisons = tuple(_resolve_measurement(c, tables, declarations, grains, testing=False)
                        for c in request.comparison_measurements)
    population_tables = sorted({m.table for m in tests})
    if "cell_summary" in tables:
        problem = _shape_problem(tables["cell_summary"], grains["cell_summary"],
                                 summary=None, testing=False)
        if problem:
            raise ValueError(f"cell_summary population: {problem}")
        population_tables = sorted(set(population_tables) | {"cell_summary"})
    cells, samples = _population(source_run, tables, population_tables, request)
    used = sorted(set(population_tables) | {m.table for m in comparisons})
    fingerprints = {}
    for name in used:
        fingerprint = input_hashes.get(name)
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is None:
            raise ValueError(f"input_hashes.{name}: verified SHA-256 is required for result identity")
        fingerprints[name] = fingerprint.lower()

    options = request.analysis_options.as_dict()
    _known_keys(options, set(circadian.CIRCADIAN_ANALYSIS_OPTIONS), "analysis_options")
    inherited = {**get_derived("rhythms").defaults, **dict(rhythm_params or {})}
    profile = None
    if request.settings_profile is not None:
        from pymicroglia.pipelines.audit.profiles import load_profile, production_compatibility
        profile = load_profile(request.settings_profile, [m.column for m in tests])
        profile = Settings({**profile.as_dict(), "production_compatibility": production_compatibility(profile, tests, cells, tables)})
        scope = profile["family"]["scope"]
        if "correction_scope" in request.declaration and request.correction_scope != scope:
            raise ValueError("correction_scope conflicts with the complete exported production family")
        request = replace(request, correction_scope=scope)
        first = next(iter(profile["measurement_recipes"].values()))
        options, inherited = first["analysis_options"], first["rhythm_params"]
    # Resolve an unset option from the run, including an explicitly chosen test
    # even when the estimator itself happens to offer significance.
    resolved = circadian.resolve_analysis_options(
        inherited, lambda name: options.get(name, {} if name == "period_config" else None))
    applied = {name: resolved.get(name) for name in circadian.CIRCADIAN_ANALYSIS_OPTIONS}
    applied.update(fit_method=resolved["method"],
                   significance_method=resolved["significance_method"],
                   period_config=resolved["params"]["workbench_config"])
    return ResolvedRhythmRequest(
        request, InputIdentity(source_run, Settings(fingerprints), cells, samples),
        tests, comparisons, Settings(applied), Settings(resolved["params"]),
        circadian.WORKBENCH_VERSION, Settings(circadian.PERIOD_METHODS),
        Settings(profile["measurement_recipes"] if profile else {}),
        Settings({k: v for k, v in profile.items() if k != "measurement_recipes"} if profile else {}))
