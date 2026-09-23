"""Execute finite dependent recipes against immutable, verified saved results.

Only registered Python producers run. Scientific and presentation caches are
separate; every invocation leaves an execution record, including failed branches.
"""

from __future__ import annotations

import inspect
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pymicroglia.pipelines._contracts import ArtifactRef, PipelineRecipe, SelectionRecord, Settings, StepResult, StepSpec, content_id, plain, result_from_dict, result_to_dict
from pymicroglia.pipelines._screening import file_hash
from . import _records


class Unavailable(RuntimeError):
    """A required scientific capability/input is absent, rather than a crash."""


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.partial")
    temporary.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    # Sync clients can briefly hold the existing manifest on Windows. Keep the
    # replacement atomic and retry only sharing/permission failures; never
    # truncate the previous durable execution record to work around a lock.
    import time
    try:
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(.05 * 2 ** attempt, 2.))
    finally:
        if temporary.exists():
            temporary.unlink()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _name(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value) is None:
        raise ValueError(f"unsafe pipeline step/producer name: {value!r}")


@dataclass(frozen=True)
class SavedResult:
    root: Path
    outcome: StepResult

    def artifact(self, name: str) -> Path:
        matches = [ref for ref in self.outcome.artifacts if ref.name == name]
        if not matches and name.endswith('.json'):
            # Logical table names survive the move to CSV. A metadata document
            # must never be substituted: require the recorded table contract.
            from ._tables import schema
            alternative = name[:-5] + '.csv'
            candidates = [ref for ref in self.outcome.artifacts if ref.name == alternative]
            matches = [ref for ref in candidates
                       if (self.root / ref.path).resolve().is_relative_to(self.root.resolve())
                       and schema(self.root / ref.path) is not None]
        if len(matches) != 1:
            raise Unavailable(f"{self.outcome.step} has no unique saved artifact {name!r}")
        ref = matches[0]
        path = (self.root / ref.path).resolve()
        if path.suffix=='.json':
            from .._results import document
            path=document(path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("saved artifact escapes its result directory")
        if not path.is_file() or file_hash(path) != ref.sha256:
            raise ValueError(f"saved artifact is missing or changed: {name}")
        return path


@dataclass(frozen=True)
class ExecutionContext:
    step: StepSpec
    request: Any
    scientific_settings: Settings
    table_paths: Mapping[str, Path]
    dependencies: Mapping[str, SavedResult]
    selection: SelectionRecord | None
    output: Path
    scientific_id: str
    presentation: Settings = field(default_factory=Settings)
    presentation_id: str | None = None

    def saved(self, step: str) -> SavedResult:
        if step not in self.dependencies:
            raise ValueError(f"{self.step.name} did not declare prerequisite {step!r}")
        return self.dependencies[step]


@dataclass(frozen=True)
class Producer:
    run: Callable[[ExecutionContext], StepResult]
    # Optional identity/validator for an existing scientific producer's contract.
    identity: Callable[[ExecutionContext], str] | None = None
    validate: Callable[[Path, str], Any] | None = None
    version: str = "1"
    # An index can describe failed/empty branches; scientific consumers cannot.
    accepts_unavailable_dependencies: bool = False

    def implementation(self) -> dict:
        files = {}
        for operation in (self.run, self.identity, self.validate):
            if operation is None:
                continue
            source = inspect.getsourcefile(operation)
            if source is None or not Path(source).is_file():
                raise ValueError("registered producers must have a fingerprintable source file")
            files[Path(source).name] = file_hash(Path(source))
        return {"version": self.version, "files": files,
                "runner": file_hash(Path(__file__))}


def dependency_order(recipe: PipelineRecipe, producers: Mapping[str, Producer]) -> tuple[StepSpec, ...]:
    """Validate the entire recipe before creating an output directory."""
    _name(recipe.name)
    by_name = {}
    for step in recipe.steps:
        _name(step.name)
        _name(step.producer)
        if step.name in by_name:
            raise ValueError(f"duplicate step: {step.name}")
        if step.producer not in producers:
            raise ValueError(f"unregistered producer: {step.producer}")
        if step.kind not in {"science", "render"}:
            raise ValueError(f"unknown step kind: {step.kind}")
        if producers[step.producer].accepts_unavailable_dependencies and step.kind != "render":
            raise ValueError("only a render producer may inspect unavailable prerequisites")
        if step.requires_selected_rows and step.selection is None:
            raise ValueError(f"{step.name} requires rows but declares no selection")
        by_name[step.name] = step
    for step in recipe.steps:
        for prerequisite in step.prerequisites:
            if prerequisite not in by_name:
                raise ValueError(f"missing prerequisite {prerequisite!r} for {step.name}")
            if step.kind == "science" and by_name[prerequisite].kind == "render":
                raise ValueError("scientific results cannot depend on rendered presentation")
        for reference in (*step.inputs, *((step.selection,) if step.selection else ())):
            if ":" in reference and reference.split(":", 1)[0] not in step.prerequisites:
                raise ValueError(f"{step.name} input/selection {reference!r} lacks a prerequisite")
    ordered, visiting, done = [], set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError(f"pipeline dependency cycle at {name}")
        if name in done:
            return
        visiting.add(name)
        for prerequisite in by_name[name].prerequisites:
            visit(prerequisite)
        visiting.remove(name)
        done.add(name)
        ordered.append(by_name[name])

    for name in by_name:
        visit(name)
    return tuple(ordered)


def _selection(step: StepSpec, dependencies: Mapping[str, SavedResult]) -> SelectionRecord | None:
    if step.selection is None:
        return None
    source, separator, name = step.selection.partition(":")
    name = name if separator else source
    sources = [dependencies[source]] if separator else dependencies.values()
    matches = [selection for saved in sources for selection in saved.outcome.selections
               if selection.name == name]
    if len(matches) != 1:
        raise Unavailable(f"selection {step.selection!r} is absent or ambiguous")
    return matches[0]


def _validate(saved: SavedResult) -> None:
    names = [ref.name for ref in saved.outcome.artifacts]
    if len(names) != len(set(names)):
        raise ValueError("duplicate saved artifact names")
    for ref in saved.outcome.artifacts:
        if ref.scientific_id != saved.outcome.scientific_id:
            raise ValueError("artifact belongs to a different scientific result")
        saved.artifact(ref.name)
    for selection in saved.outcome.selections:
        if selection.scientific_id != saved.outcome.scientific_id:
            raise ValueError("selection belongs to a different scientific result")


def _read_receipt(root: Path, cache_id: str, scientific_id: str,
                  producer: Producer) -> SavedResult:
    _, _, receipt = _records.locate(root)
    if receipt["schema_version"] != 1 or receipt["cache_id"] != cache_id:
        raise ValueError("saved producer identity does not match")
    if content_id(receipt["result"]) != receipt["result_sha256"]:
        raise ValueError("saved outcome or selections were changed")
    result = result_from_dict(receipt["result"])
    if result.scientific_id != scientific_id or result.status != "completed":
        raise ValueError("saved outcome is not a compatible completion")
    saved = SavedResult(root, result)
    _validate(saved)
    if producer.validate is not None:
        producer.validate(root, scientific_id)
    return saved


@dataclass(frozen=True)
class PipelineExecution:
    recipe: str
    record_path: Path
    results: Mapping[str, SavedResult]
    invocation: str = ""

    def record(self):
        from auto_organotypic.layout import owning_folder
        return _records.invocation(owning_folder(self.record_path),self.invocation)

    @property
    def successful(self) -> bool:
        return all(saved.outcome.status in {"completed", "reused", "skipped-empty"}
                   for saved in self.results.values())


def figure_binding(results: Mapping[str, SavedResult], *,
                   inputs: Mapping[str, tuple[str, str]],
                   selections: Mapping[str, tuple[str, str]] | None = None) -> dict:
    """Declare saved table aliases and selection references for one figure item.

    This records the producer's existing provenance and never replaces it with
    current figure defaults. Builders receive only aliases declared by the item.
    """
    saved_sources = {}
    for alias in inputs:
        _name(alias)
        if alias.startswith("pipeline_"):
            raise ValueError("pipeline_ aliases are reserved for scientific provenance sources")
    for name, saved in results.items():
        _name(name)
        if saved.outcome.status not in {"completed", "reused"}:
            raise Unavailable(f"figure source {name} is not completed")
        _validate(saved)
        manifest, entry, stored = _records.locate(saved.root)
        encoded = stored["result"]
        if content_id(encoded) != stored["result_sha256"]:
            raise ValueError("saved figure outcome or selections were changed")
        original = result_from_dict(encoded)
        if original.status != "completed" or replace(saved.outcome, status=original.status, reason=original.reason) != original:
            raise ValueError("figure input differs from its original completed result")
        saved_sources[name] = {"root": str(saved.root.resolve()),
                              "ledger": str(manifest), "entry": entry,
                              "sha256": content_id(stored)}
    for source, artifact in inputs.values():
        results[source].artifact(artifact)
    for source, selection in (selections or {}).values():
        if sum(item.name == selection for item in results[source].outcome.selections) != 1:
            raise ValueError(f"figure selection {source}:{selection} is absent or ambiguous")
    return plain({"schema_version": 1, "results": saved_sources,
                  "inputs": inputs, "selections": selections or {}})


def register_figure_plan(run: str | Path, items: list[dict]) -> Path:
    """Save materialized items after analysis; leave the static plot plan intact.

    Names are immutable. A different presentation must use a new item name, which
    prevents selectively regenerated pages from changing an earlier page's inputs.
    """
    from auto_organotypic.store.ledger import _folder_lock
    from pymicroglia._results import figure_plans, write_document
    run = Path(run).resolve()
    with _folder_lock(run):
        known = figure_plans(run)
        names = set()
        for item in items:
            _name(item["name"])
            if item["name"] in names:
                raise ValueError(f"duplicate generated item: {item['name']}")
            names.add(item["name"])
            if "pipeline" not in item:
                raise ValueError("generated pipeline item must declare saved scientific inputs")
            if item["name"] in known and item != known[item["name"]]:
                raise ValueError(f"figure item name already belongs to a different plan: {item['name']}")
            known[item["name"]] = item
        return write_document(run / "figure-plans.json", {"schema_version": 1, "figure_plans": known})


def run_pipeline(recipe: PipelineRecipe, producers: Mapping[str, Producer], *,
                 request: Any, scientific_settings: Mapping, output: str | Path,
                 table_paths: Mapping[str, Path] | None = None,
                 presentation: Mapping | None = None,
                 only: tuple[str, ...] | None = None) -> PipelineExecution:
    """Run selected steps and their prerequisites; retain all earlier invocations.

    A producer writes inside ``context.output`` and returns references relative to
    that directory. Render producers receive saved results and selections only
    after prerequisites are validated. Each attempt has its own directory, so a
    failed or interrupted attempt cannot overwrite a completed result.
    """
    order = dependency_order(recipe, producers)
    requested = set(only) if only is not None else {step.name for step in order}
    unknown = requested - {step.name for step in order}
    if unknown:
        raise ValueError(f"unknown selected steps: {sorted(unknown)}")
    for step in reversed(order):
        if step.name in requested:
            requested.update(step.prerequisites)
    order = tuple(step for step in order if step.name in requested)
    settings = Settings(scientific_settings)
    appearance = Settings(presentation)
    implementations = {step.producer: producers[step.producer].implementation() for step in order}
    output = Path(output).resolve()
    invocation = uuid4().hex
    record_path = _records.path(output)
    results, entries = {}, []
    execution = {"schema_version": 1, "recipe": recipe.as_dict(), "invocation": invocation,
                 "started": _now(), "finished": None, "steps": entries,
                 "requested_steps": sorted(requested), "presentation": appearance.as_dict()}
    record_path = _records.update(output, "invocations", invocation, execution)
    for step in order:
        dependencies = {name: results[name] for name in step.prerequisites}
        producer = producers[step.producer]
        base = {"recipe": recipe.name, "version": recipe.version, "step": step,
                "settings": settings, "dependencies": {
                    name: {"scientific_id": saved.outcome.scientific_id,
                           "artifacts": [ref.as_dict() for ref in saved.outcome.artifacts],
                           "selections": [selection.record_id for selection in saved.outcome.selections]}
                    for name, saved in dependencies.items()}}
        if producer.accepts_unavailable_dependencies:
            # Reuse certifies the same completed scientific artifact. Its
            # invocation-specific status/reason must not invalidate displays,
            # while an unavailable/failed outcome and its reason still must.
            base["dependency_outcomes"] = {
                name: ({"status": "completed"} if saved.outcome.status in {"completed", "reused"}
                       else {"status": saved.outcome.status, "reason": saved.outcome.reason})
                for name, saved in dependencies.items()}
        scientific_id = content_id(base)
        selected = None
        presentation_id = None
        root = output / "attempts" / invocation / step.name
        context = ExecutionContext(step, request, settings, table_paths or {}, dependencies,
                                   None, root, scientific_id)
        try:
            blocked = [name for name, saved in dependencies.items()
                       if saved.outcome.status not in {"completed", "reused"}]
            if blocked and not producer.accepts_unavailable_dependencies:
                # A scientific empty selection can gate several dependent stages.
                # Preserve that legitimate skip through the chain, while a failed
                # or unavailable prerequisite still blocks the dependent work.
                if (step.requires_selected_rows
                        and all(dependencies[name].outcome.status == "skipped-empty" for name in blocked)):
                    selected = _selection(step, dependencies)
                if selected is None or selected.members:
                    raise Unavailable("prerequisite unavailable: " + "; ".join(
                        f"{name} ({dependencies[name].outcome.status}: {dependencies[name].outcome.reason})"
                        for name in blocked))
            if selected is None:
                selected = _selection(step, dependencies)
            for reference in step.inputs:
                if ":" in reference:
                    source, artifact = reference.split(":", 1)
                    if (dependencies[source].outcome.status == "skipped-empty" and selected is not None
                            and not selected.members and step.requires_selected_rows):
                        continue
                    dependencies[source].artifact(artifact)
                elif reference != "measured-tables" and reference not in (table_paths or {}):
                    raise Unavailable(f"required input is absent: {reference}")
            base["selection"] = selected.record_id if selected else None
            context = replace(context, selection=selected)
            if step.kind == "science":
                scientific_id = (producer.identity(context) if producer.identity else
                                 content_id({**base, "producer": implementations[step.producer]}))
            else:
                scientific_id = content_id(base)
                presentation_id = content_id({"science": scientific_id, "presentation": appearance,
                                               "producer": implementations[step.producer]})
            cache_id = presentation_id or scientific_id
            cache_root = output / ("renders" if step.kind == "render" else "science") / step.name / cache_id
            root = cache_root / invocation
            context = replace(context, output=root, scientific_id=scientific_id,
                              presentation=appearance if step.kind == "render" else Settings(),
                              presentation_id=presentation_id)
            if selected is not None and not selected.members and step.requires_selected_rows:
                result = StepResult(step.name, scientific_id, "skipped-empty",
                                    f"Saved selection {selected.name!r} contains no eligible rows")
            else:
                reused = None
                invalid = []
                for candidate in _records.candidates(output, cache_root):
                    try:
                        reused = _read_receipt(candidate, cache_id, scientific_id, producer)
                        break
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        invalid.append(str(error))
                if reused is not None:
                    root = reused.root
                    result = replace(reused.outcome, step=step.name, status="reused",
                                     reason="Validated matching saved artifacts and selections")
                else:
                    result = producer.run(context)
                    if result.step != step.name or result.scientific_id != scientific_id:
                        raise ValueError("producer returned a different step/scientific identity")
                    _validate(SavedResult(root, result))
                    if result.status == "reused":
                        raise ValueError("only the runner may certify reuse")
                    if result.status == "completed":
                        if producer.validate is not None:
                            producer.validate(root, scientific_id)
                        encoded = result_to_dict(result)
                        _records.completion(output, root, {
                            "schema_version": 1, "cache_id": cache_id,
                            "scientific_id": scientific_id, "presentation_id": presentation_id,
                            "result": encoded, "result_sha256": content_id(encoded),
                            "implementation": implementations[step.producer],
                            "rejected_cache_reasons": invalid})
        except Unavailable as error:
            result = StepResult(step.name, scientific_id, "unavailable", str(error))
        except Exception as error:
            result = StepResult(step.name, scientific_id, "failed", f"{type(error).__name__}: {error}")
        results[step.name] = SavedResult(root, result)
        if result.status in {'completed','reused'}:
            outcome_record={'completion':root.relative_to(output).as_posix(),'outcome':{'status':result.status,'reason':result.reason}}
        else:outcome_record={'result':result_to_dict(result)}
        entries.append({"step": step.name, **outcome_record,
                        "result_directory": str(root.relative_to(output)),
                        "presentation_id": presentation_id,
                        "selection": selected.record_id if selected else None,
                        "selection_source": step.selection, "finished": _now()})
        record_path = _records.update(output, "invocations", invocation, execution)
    execution["finished"] = _now()
    execution["successful"] = all(saved.outcome.status in {"completed", "reused", "skipped-empty"}
                                  for saved in results.values())
    record_path = _records.update(output, "invocations", invocation, execution)
    return PipelineExecution(recipe.name, record_path, results, invocation)
