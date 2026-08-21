"""One execution, recorded — or not, without anybody noticing.

Named ``recording`` rather than ``capture`` so the module and the context
manager it exports do not share a name: ``from .capture import capture`` in the
package __init__ would rebind ``pymicroglia.capture`` from the module to the
function, and then which one you got depended on how you imported it.

Every action runs inside :func:`capture`. With ``analysis_kit`` installed it
times the run, snapshots what was written, asks the artefact store what it
gained, builds the record and writes it twice: a full record beside the outputs,
one line in the global index. Without the kit it yields a stand-in that does
nothing at all.

That asymmetry is the whole point. Losing a run record is an inconvenience;
losing a result because the audit layer was missing would be a disaster. So the
recording is wrapped and the analysis is not.

The record answers three questions, and the third is the one worth the effort:

    what happened      action, resolved parameters, duration, versions
    what came out      the files written, and the store artefacts behind them
    how to do it again the equivalent script — real code, not a description

A record that describes a run is a note. A record that reproduces it is a
result.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from . import catalogue
from ._optional import kit

__all__ = [
    "capture",
    "NullRun",
    "entry_path",
    "ENTRY_PATHS",
    "equivalent_script",
    "ArtefactWatch",
    "append_to_notebook",
    "notebook_for",
]

#: The ways a run can be started. ``imagej`` is reserved for the bridge in
#: stage 13 and is written by that front door, not inferred here.
ENTRY_PATHS: tuple[str, ...] = ("python", "cli", "imagej")

#: Where the reproducibility notebook lives under a results folder. The kit's
#: default, named here so the CLI and the tests agree on one answer.
NOTEBOOK_FOLDER = "Notebooks"

#: Wrap a call across lines once one line would run past this. Matches the
#: kit's own threshold so scripts from both look the same in a diff.
SCRIPT_WIDTH = 88


#: Deliberately plain ASCII. This text is written to a .py file that somebody
#: will redirect through a console, and a console on this machine is cp1252.
SCRIPT_HEADER = '''#!/usr/bin/env python
"""Reproduce one pymicroglia run: {action}.

Written by the run itself, from the API rather than the command line. The
command line may change; the API is what the record is promising.

Every argument below is the *resolved* value that was in force: the ones the
caller passed and every default that applied. So this keeps producing the same
numbers after somebody changes a default.
"""

'''


def entry_path(entry: str = "") -> str:
    """How this run was started: ``python``, ``cli``, or later ``imagej``.

    Reserved from the first record and filled by each front door, so the index
    never has to be migrated when the Fiji bridge lands: a row written today and
    a row written then have the same shape.

    A front door says so by passing its own name. ``PYMICROGLIA_ENTRY`` is for a
    front door that cannot — a bridge that starts the interpreter and hands over
    — and is read only when nothing was passed. Setting it from inside the
    process would be worse than useless: it would outlive the one call that
    meant it, and every later run from the API would claim to be that door.
    """
    return str(entry) or os.environ.get("PYMICROGLIA_ENTRY", "python")


class NullRun:
    """What you get when there is no audit layer. Assignable, and inert."""

    def __init__(self, action: str, params: Mapping[str, Any]):
        self.action = action
        self.params = dict(params)
        self.result: Any = None
        self.record: dict[str, Any] = {}
        self.recorded = False
        self.notebook: str | None = None

    @property
    def outputs(self) -> list[str]:
        return []


def _signature_defaults(action: str) -> dict[str, Any]:
    """Every default the function itself would apply, as it would apply it."""
    function = _callable_for(action)
    if function is None:
        return {}
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        return {}
    return {name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
            and parameter.kind not in (parameter.VAR_KEYWORD,
                                       parameter.VAR_POSITIONAL)}


def _resolved(action: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """The defaults that were in force, overridden by what the caller passed.

    Records carry resolved values rather than the caller's three arguments, so a
    stored run still replays the same way after somebody changes a default.

    The defaults come from the **function**, not from the catalogue. The two are
    not always the same: the catalogue records what the protocol script this was
    copied from declared, and in a handful of places the package deliberately
    settled somewhere else — ``run_controls`` searches 16-32 h where the engine
    searched 15-40. A record has to say what ran, so it asks what ran. The
    catalogue is the fallback for an action whose target cannot be resolved,
    where something is better than nothing.
    """
    defaults = _signature_defaults(action)
    if not defaults:
        defaults = dict((catalogue.action(action) or {}).get("defaults") or {})
    resolved = dict(defaults)
    resolved.update(params)
    return resolved


# ------------------------------------------------------- the equivalent script
def _pipeline_method(action: str) -> str:
    """``pipelines.<name>.run`` for a pipeline that is not a registered action.

    Two of the four pipelines are registered and two compose existing actions,
    so they never reached the catalogue. They still record runs, and a record
    without a script would be the one kind of record this stage exists to stop.
    """
    try:  # local: pipelines import this module, so a top-level import cycles
        from . import pipelines
    except ImportError:  # pragma: no cover - the package is always importable
        return ""
    return f"pipelines.{action}.run" if action in pipelines.PIPELINE_NAMES else ""


def _method_for(action: str) -> str:
    """The dotted target this action binds to, catalogue first."""
    method = str((catalogue.action(action) or {}).get("method") or "")
    return method or _pipeline_method(action)


def _callable_for(action: str) -> Any:
    """The function an action actually calls, or ``None`` while it is pending."""
    method = _method_for(action)
    if not method:
        return None
    from .registry import MODULES, resolve  # local: see the note in capture()

    found = resolve(method, MODULES)
    if found is not None:
        return found
    # The two pipelines that compose existing actions are never registered, so
    # the registry has not imported them. Reach them directly.
    module_path, _, attribute = method.rpartition(".")
    try:
        module = importlib.import_module(f"pymicroglia.{module_path}")
    except ImportError:
        return None
    return getattr(module, attribute, None)


def _method_version(action: str) -> str:
    """Which version of the method ran, catalogue first and module second.

    The catalogue carries one only where the protocol script it was copied from
    declared one, which is five actions out of twenty-six. The rest declare it
    on the module that does the work — ``segmentation.METHOD_VERSION``,
    ``pipelines.dluc_single_cell.METHOD_VERSION`` — and a record of a pipeline
    run that cannot say which version produced it is missing the field that
    decides whether two runs are comparable at all.
    """
    declared = str((catalogue.action(action) or {}).get("method_version") or "")
    if declared:
        return declared
    module_path = _method_for(action).rpartition(".")[0]
    if not module_path:
        return ""
    try:
        module = importlib.import_module(f"pymicroglia.{module_path}")
    except ImportError:
        return ""
    return str(getattr(module, "METHOD_VERSION", "") or "")


def _script_target(action: str) -> tuple[str, str, str] | None:
    """Which public function reproduces this action: package, module, call.

    Read from the catalogue's own binding, so the script calls exactly what the
    run called. If a binding moves the script moves with it, and there is
    nothing here to keep in step by hand.
    """
    module_path, _, function = _method_for(action).rpartition(".")
    if not module_path or not function:
        return None
    package, _, leaf = f"pymicroglia.{module_path}".rpartition(".")
    return package, leaf, f"{leaf}.{function}"


def _text_literal(text: str) -> str:
    r"""A string literal that survives the round trip, backslashes and all.

    Every path in this project contains spaces, and most contain a folder whose
    first letter Python reads as an escape: ``\Users`` is fine only by luck,
    ``\temp`` is a tab and ``\Numbers`` is a broken unicode escape. A raw
    literal keeps the text exactly as it was typed.

    A raw literal cannot end in a backslash and cannot hold the quote that
    delimits it, so those two cases fall back to ``repr`` — which escapes
    correctly, if less readably.
    """
    if "\\" not in text:
        return repr(text)
    if text.endswith("\\") or '"' in text or "\n" in text or "\r" in text:
        return repr(text)
    return f'r"{text}"'


def _literal(value: Any) -> str:
    """One resolved parameter as source code. Walks into dicts and lists.

    A path can arrive nested — a region of interest is ``{"path": ...}`` — and a
    nested path escapes exactly as badly as a top-level one.
    """
    if isinstance(value, Path):
        return _text_literal(str(value))
    if isinstance(value, str):
        return _text_literal(value)
    if isinstance(value, Mapping):
        inner = ", ".join(f"{_literal(key)}: {_literal(item)}"
                          for key, item in value.items())
        return "{" + inner + "}"
    if isinstance(value, tuple):
        inner = ", ".join(_literal(item) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, (set, frozenset)):
        # As a set, not a list. Replaying a set argument as a list would be a
        # different call, and sorted only so two runs emit the same text.
        inner = ", ".join(_literal(item) for item in sorted(value, key=str))
        body = "{" + inner + "}" if value else "set()"
        return f"frozenset({body})" if isinstance(value, frozenset) else body
    if isinstance(value, list):
        return "[" + ", ".join(_literal(item) for item in value) + "]"

    source = repr(value)
    try:
        ast.literal_eval(source)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        # Refused rather than emitted. An object with no source form would make
        # a script that does not parse, and a record carrying one of those is
        # worse than a record that says plainly it could not build one — which
        # is what the kit writes when this raises.
        raise TypeError(
            f"a {type(value).__name__} has no source form, so no script can "
            "reproduce this run. Pass a path or a plain value instead."
        ) from None
    return source


def _call(name: str, params: Mapping[str, Any], *, first: str = "") -> str:
    """The call itself, source first and then the rest in a settled order.

    ``source`` leads because it is the one argument a reader looks for. The rest
    are alphabetical so two runs of the same action produce scripts that diff
    cleanly against each other.
    """
    ordered: list[tuple[str, Any]] = []
    if "source" in params:
        ordered.append(("source", params["source"]))
    ordered += sorted(((key, value) for key, value in params.items()
                       if key != "source"), key=lambda pair: pair[0])

    arguments = ([first] if first else [])
    arguments += [f"{key}={_literal(value)}" for key, value in ordered]
    one_line = f"{name}({', '.join(arguments)})"
    if len(one_line) <= SCRIPT_WIDTH or not arguments:
        return one_line
    return f"{name}(\n    " + ",\n    ".join(arguments) + ",\n)"


def equivalent_script(action: str, params: Mapping[str, Any] | None = None) -> str:
    """The shortest pymicroglia code that does this run again.

    It composes public API calls and nothing else. If writing one ever needed a
    private helper, the public API would be wrong and that would be the thing to
    fix — a second implementation living in the audit layer is how a record
    starts quietly disagreeing with the thing it claims to reproduce.
    """
    resolved = dict(params or {})
    target = _script_target(action)
    if target is None:
        header = "from pymicroglia import run_action"
        body = _call("run_action", resolved, first=repr(str(action)))
    else:
        package, leaf, name = target
        header = f"from {package} import {leaf}"
        body = _call(name, resolved)
    return SCRIPT_HEADER.format(action=action) + header + "\n\n" + body + "\n"


def _equivalent_script(run: Any) -> str:
    """Adapter for the kit's ``script_builder`` hook."""
    return equivalent_script(str(getattr(run, "action", "")),
                             getattr(run, "params", {}) or {})


# ------------------------------------------------------- what the store gained
def _tier_a_entries() -> dict[str, dict[str, Any]]:
    """The tier-A index, keyed as it is on disk. Empty on any trouble."""
    try:
        from . import store

        return dict(store.manifest.load().get("artefacts") or {})
    except Exception:
        return {}


def _tier_b_entries() -> dict[str, dict[str, Any]]:
    """Every materialised pixel array the local cache holds, by stage/digest."""
    try:
        from . import store

        return {f"{entry['stage']}/{entry['digest']}": entry
                for entry in store.tier_b.entries()}
    except Exception:
        return {}


def _decision_stage() -> str:
    try:
        from . import store

        return str(store.DECISION_STAGE)
    except Exception:
        return "_decision"


class ArtefactWatch:
    """What the artefact store gained while a run was in flight.

    A run's ``outputs`` say which files appeared. This says which of them the
    store will hand back on the next lookup and under which digest — the
    difference between "a CSV was written" and "measurement can re-run from it
    with no pixels present".

    Two indexes, because they promise different things. **Tier A** is the
    analysis: permanent, kilobytes, written beside the results. **Tier B** is
    materialised pixels: local, capped and evicted, so it is reported as what
    the cache happens to hold rather than as a result. **Decisions** are split
    off from tier A because a person answered those, and a run that settled one
    did something a re-run will not have to do again.

    Nothing here raises. A missing artefact list is a thinner record; an
    exception raised while assembling one would be a lost run.
    """

    def __init__(self, roots: Iterable[Any] = ()) -> None:
        self.roots = [os.path.normcase(os.path.abspath(str(root)))
                      for root in roots or () if root]
        self.artefacts: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self._before_a: dict[str, str] = {}
        self._before_b: set[str] = set()
        self._before_mtime: dict[str, float] = {}

    def arm(self) -> "ArtefactWatch":
        entries = _tier_a_entries()
        self._before_a = {slot: str(entry.get("indexed") or "")
                          for slot, entry in entries.items()}
        self._before_mtime = {slot: mtime for slot, mtime in
                              ((slot, self._mtime_of(entry))
                               for slot, entry in entries.items())
                              if mtime is not None}
        self._before_b = set(_tier_b_entries())
        return self

    __enter__ = arm

    def _under_a_root(self, path: str) -> bool:
        folded = os.path.normcase(os.path.abspath(path))
        return any(folded == root or folded.startswith(root + os.sep)
                   for root in self.roots)

    def _mtime_of(self, entry: Mapping[str, Any]) -> float | None:
        """When the artefact file was last written, if it is one of ours.

        Only files under this run's own output folders are stated, so the cost
        is bounded by what the run writes rather than by everything the store
        has ever held.
        """
        path = str(entry.get("path") or "")
        if not path or not self._under_a_root(path):
            return None
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def _rewritten(self, slot: str, entry: Mapping[str, Any]) -> bool:
        """An artefact replaced in place inside this run's own results folder.

        The index stamp is written to the second, so a cold re-run that
        overwrites an artefact within a second of its last indexing looks
        untouched. The file's own timestamp settles it.

        Compared against the timestamp taken when this watch was armed, never
        against the wall clock. A file's mtime is finer-grained than
        ``time.time()`` on Windows, so a file written moments *before* the run
        began can read as later than the clock did — which would credit this run
        with an artefact an earlier one wrote.
        """
        before = self._before_mtime.get(slot)
        if before is None:
            return False
        now = self._mtime_of(entry)
        return now is not None and now != before

    def disarm(self) -> "ArtefactWatch":
        """Diff both indexes. Runs whether or not the analysis raised."""
        decision_stage = _decision_stage()
        fresh = [entry for slot, entry in _tier_a_entries().items()
                 if self._before_a.get(slot) != str(entry.get("indexed") or "")
                 or self._rewritten(slot, entry)]
        fresh.sort(key=lambda entry: (str(entry.get("created") or ""),
                                      str(entry.get("path") or "")))

        self.artefacts = [_tier_a_row(entry) for entry in fresh
                          if entry.get("stage") != decision_stage]
        self.decisions = [_decision_row(entry) for entry in fresh
                          if entry.get("stage") == decision_stage]

        current = _tier_b_entries()
        gained = [entry for slot, entry in current.items()
                  if slot not in self._before_b]
        gained.sort(key=lambda entry: (str(entry.get("stage") or ""),
                                       str(entry.get("digest") or "")))
        self.artefacts += [_tier_b_row(entry) for entry in gained]
        self._before_a, self._before_b = {}, set()
        self._before_mtime = {}
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.disarm()
        return False


def _tier_a_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": str(entry.get("stage") or ""),
        "digest": str(entry.get("digest") or ""),
        "tier": "A",
        "kind": str(entry.get("kind") or ""),
        "display_only": bool(entry.get("display_only")),
        "method_version": str(entry.get("method_version") or ""),
        "path": str(entry.get("path") or ""),
        "bytes": int(entry.get("bytes") or 0),
    }


def _tier_b_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": str(entry.get("stage") or ""),
        "digest": str(entry.get("digest") or ""),
        "tier": "B",
        "kind": "array",
        "shape": list(entry.get("shape") or []),
        "dtype": str(entry.get("dtype") or ""),
        "path": str(entry.get("path") or ""),
        "bytes": int(entry.get("bytes") or 0),
    }


def _decision_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    params = entry.get("params") or {}
    return {
        "decision": str(params.get("decision") or ""),
        "digest": str(entry.get("digest") or ""),
        "path": str(entry.get("path") or ""),
    }


# -------------------------------------------------------------- the notebook
def notebook_for(root: Any) -> Path:
    """Where this project's reproducibility notebook lives under a folder."""
    ak = kit()
    if ak is None:
        return Path(root) / NOTEBOOK_FOLDER / "pymicroglia.ipynb"
    return Path(ak.capture.notebook_path(root, "pymicroglia",
                                         folder=NOTEBOOK_FOLDER))


def _notebook_lines(record: Mapping[str, Any]) -> list[str]:
    """The pymicroglia facts worth reading above the code, and no others."""
    artefacts = list(record.get("artefacts") or ())
    tier_a = [item for item in artefacts if item.get("tier") == "A"]
    lines = [
        f"- Started from: `{record.get('entry_path') or 'python'}`",
        f"- Method version: `{record.get('method_version') or '(none declared)'}`",
        f"- pymicroglia: `{record.get('package_version') or ''}`",
        f"- Claim: {str(record.get('claim') or '').strip() or '(none given)'}",
    ]
    if tier_a:
        listed = ", ".join(f"`{item['stage']}` `{item['digest'][:12]}`"
                           for item in tier_a[:6])
        more = f" and {len(tier_a) - 6} more" if len(tier_a) > 6 else ""
        lines.append(f"- Artefacts kept: {listed}{more}")
    if record.get("error"):
        lines.append(f"- **Failed:** `{record['error']}`")
    return lines


def append_to_notebook(record: Mapping[str, Any], *, root: Any,
                       request: str = "", previews: Iterable[Any] = ()) -> Path:
    """Add one run's section to the project's reproducibility notebook.

    Off unless asked for. A notebook is one JSON document rewritten in full on
    every append, so a cell per run would make it the largest file in the
    results folder inside a month — and most runs are a step on the way to one
    worth narrating, not the one worth narrating.

    ``request`` is what the person actually asked for, in their words. It is the
    only thing in the whole record that cannot be reconstructed afterwards.
    """
    ak = kit()
    if ak is None:
        raise RuntimeError(
            "the notebook is written by analysis_kit, which is not importable. "
            "Run 'pymicroglia doctor' to see which interpreter is in use.")
    return Path(ak.capture.append_run(
        notebook_for(root), record, request=request, previews=previews,
        extra_lines=_notebook_lines(record)))


# ------------------------------------------------------------------- capturing
@contextmanager
def capture(
    action: str,
    params: Mapping[str, Any] | None = None,
    *,
    claim: str = "",
    output_roots: Iterable[Any] = (),
    kind: str = "analysis",
    store_root: Any = None,
    notebook: bool = False,
    request: str = "",
    entry: str = "",
) -> Iterator[Any]:
    """Run something and record it. Never raises on account of the recording."""
    from . import __version__
    # Local: the registry imports every science module, and the pipelines import
    # this one, so a top-level import here would close the loop and leave the
    # pipelines looking permanently unimplemented.
    from .registry import claim_for

    params = dict(params or {})
    resolved = _resolved(action, params)
    written_claim, owed = claim_for(action, claim, resolved)
    ak = kit()

    if ak is None:
        yield NullRun(action, resolved)
        return

    roots = [Path(root) for root in output_roots]
    watch = ArtefactWatch(roots).arm()
    run: Any = None
    try:
        with ak.capture.RunCapture(
            project="pymicroglia",
            action=action,
            params=resolved,
            kind=kind,
            output_roots=roots,
            entry_path=entry_path(entry),
            method_version=_method_version(action),
            package_version=__version__,
            claim=written_claim,
            claim_required=owed,
            script_builder=_equivalent_script,
        ) as run:
            run.recorded = False
            run.notebook = None
            try:
                yield run
            finally:
                # In a finally, so a run that raised still reports the artefacts
                # it managed to write. The record is built by __exit__, which
                # reads run.extra after this block.
                #
                # Guarded because this sits directly in the analysis's own
                # unwind path: anything raised here would replace the
                # exception the analysis raised, or invent one where the
                # analysis had succeeded.
                try:
                    watch.disarm()
                except Exception:
                    pass
                run.extra["artefacts"] = watch.artefacts
                run.extra["decisions"] = watch.decisions
    finally:
        if run is not None:
            target = (Path(store_root) if store_root is not None
                      else (roots[0] if roots else Path.cwd()))
            _write_record(ak, run, target, written_claim, notebook, request)


def _write_record(ak: Any, run: Any, target: Path, claim: str,
                  notebook: bool, request: str) -> None:
    """Write the record twice, and never let that cost the run.

    Called from a ``finally``, so a failed analysis is recorded with its error
    and whatever outputs did appear — a run that broke is the one you most want
    to be able to look up.
    """
    try:
        ak.audit.record_run(run.record, store_root=target, claim=claim)
        run.recorded = True
    except Exception:
        # An audit failure is a missing record, never a failed analysis. This is
        # the one bare except in the package and it is deliberate: do not
        # narrow it, and do not let it re-raise.
        run.recorded = False
    if not notebook:
        return
    try:
        run.notebook = str(append_to_notebook(run.record, root=target,
                                              request=request))
    except Exception:
        run.notebook = None
