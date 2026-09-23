"""Method-selection audit recipe using the common saved-result runner."""
from pymicroglia._sources import source_file

from pathlib import Path

from pymicroglia.pipelines._contracts import ArtifactRef, PipelineRecipe, Settings, StepResult, StepSpec, content_id
from pymicroglia.pipelines.audit.options import resolve_request


RECIPE = PipelineRecipe("method-selection-audit", 1, (
    StepSpec("audit-design", "audit-design", inputs=("measured-tables",)),
    StepSpec("real-candidates", "real-candidates", ("audit-design",),
             inputs=("audit-design:audit_design", "measured-tables")),
    StepSpec("development-cases", "development-cases", ("audit-design",),
             inputs=("audit-design:audit_design", "measured-tables")),
    StepSpec("development-scores", "development-scores", ("development-cases",),
             inputs=("development-cases:benchmark_cases", "development-cases:benchmark_traces")),
    StepSpec("real-stability", "real-stability", ("real-candidates",),
             inputs=("real-candidates:results", "measured-tables")),
    StepSpec("candidate-shortlist", "candidate-shortlist", ("development-cases", "development-scores", "real-stability"),
             inputs=("development-scores:score_summary", "real-stability:stability_summary")),
    StepSpec("independent-confirmation", "independent-confirmation", ("candidate-shortlist", "development-cases"),
             inputs=("candidate-shortlist:frozen_selection", "development-cases:confirmation_reservation")),
    StepSpec("performance-figures", "performance-figures",
             ("audit-design", "real-candidates", "development-scores", "real-stability", "candidate-shortlist", "independent-confirmation"),
             inputs=("independent-confirmation:final_decisions", "development-scores:score_summary"), kind="render"),
    StepSpec("focused-pages", "focused-pages", ("real-candidates", "real-stability", "candidate-shortlist", "independent-confirmation"),
             inputs=("real-candidates:results", "real-candidates:traces"), kind="render"),
    StepSpec("audit-index", "audit-index", ("audit-design", "real-candidates", "development-cases", "development-scores",
             "real-stability", "candidate-shortlist", "independent-confirmation", "performance-figures", "focused-pages"),
             inputs=("independent-confirmation:final_decisions", "performance-figures:figure_manifest.json", "focused-pages:focused_manifest.json"), kind="render"),
))


def identity(context):
    import pymicroglia.workbench as circadian
    from pymicroglia.pipelines._screening import file_hash, producer_identity, read_verified_tables

    read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    if (context.request.source.workbench_version != circadian.WORKBENCH_VERSION
            or context.request.source.period_methods.as_dict() != circadian.PERIOD_METHODS):
        raise ValueError("Workbench catalogue changed after audit resolution")
    return content_id({"request": context.request.scientific_id, "engine": producer_identity(),
        "audit_code": {name: file_hash(source_file(name))
                       for name in ("audit_options.py", "method_audit.py")}})


def freeze_design(context):
    from pymicroglia.pipelines._screening import _write_json, file_hash

    context.output.mkdir(parents=True)
    path = context.output / "audit_design.json"
    _write_json(path, {"schema_version": 1, "scientific_id": context.scientific_id,
        "request": context.request.as_dict(), "confirmation_opened": False,
        "candidate_count": len(context.request.candidates),
        "profile_count": len(context.request.profiles)})
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Frozen input population, complete candidate recipes and evaluation policy",
        (ArtifactRef("audit_design", path.name, file_hash(path), context.scientific_id),),
        provenance=Settings({"workbench_version": context.request.source.workbench_version,
                             "selection_basis": "input coverage only"}))


def producers():
    from pymicroglia.pipelines._runner import Producer
    from pymicroglia.pipelines.audit.evaluator import evaluate_real
    from pymicroglia.pipelines.audit.benchmarks import produce_development
    from pymicroglia.pipelines.audit.scoring import produce_scores
    from pymicroglia.pipelines.audit.stability import produce_stability
    from pymicroglia.pipelines.audit.selection import produce_selection
    from pymicroglia.pipelines.audit.confirmation import produce_confirmation
    from pymicroglia.pipelines.audit.figures import produce_performance_figures, implementation_version
    from pymicroglia.pipelines.audit.focus import produce_focused_pages, implementation_version as focus_version
    from pymicroglia.pipelines.audit.index import produce_index

    return {"audit-design": Producer(freeze_design, identity),
            "real-candidates": Producer(evaluate_real),
            "development-cases": Producer(produce_development),
            "development-scores": Producer(produce_scores),
            "real-stability": Producer(produce_stability),
            "candidate-shortlist": Producer(produce_selection),
            "independent-confirmation": Producer(produce_confirmation),
            "performance-figures": Producer(produce_performance_figures, version=implementation_version()),
            "focused-pages": Producer(produce_focused_pages, version=focus_version()),
            "audit-index": Producer(produce_index)}


def run_request(resolved, table_paths, output, *, presentation=None, only=None):
    from pymicroglia.pipelines._runner import run_pipeline

    return run_pipeline(RECIPE, producers(), request=resolved,
        scientific_settings={"audit": resolved.scientific_id}, table_paths=table_paths,
        output=output, presentation=presentation, only=only)
