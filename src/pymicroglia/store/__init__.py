"""The artefact store: keep the recipe, not the cooked dish.

The question this package was built around is how to store image series so that
re-analysis is cheap. The answer is that you mostly should not. Registration is
the expensive step and its entire output is a 298 KB table of per-frame shifts
against a 10.8 GB input — 0.003% — so what gets kept is what a step *derived*,
keyed tightly enough that changing a parameter misses on its own.

Three classes of stored thing, and they are not equivalent:

**Tier A** — derived artefacts: transforms, masks, labels, traces. Permanent,
kilobytes to a few megabytes, written beside the results in Dropbox. This is the
analysis: measurement re-runs from tier A alone, with no pixels present.

**Tier B** — materialised pixel arrays. Capped, evicted, always rebuildable
from tier A plus the original file, and kept in the project's ``PixelStore``
folder so the hours that produced them are not paid again on the next machine.
Deleting all of it loses hours, not findings.

**Decisions** — the channel assignment, the usable time window, the region of
interest, whether a faint object counts. A person answered these, so they are
keyed on the source alone and survive every parameter change and version bump.
A new engine version is not a reason to ask somebody the same question again.

    from pymicroglia import store

    source = store.fingerprint("VID52_timestack.tif")
    store.put("registration", source, params, kind="table", value=shifts,
              name="registration_shifts_and_qc", output_dir=run_folder,
              method_version=METHOD_VERSION)

    hit = store.get("registration", source, params,
                    method_version=METHOD_VERSION)
    if hit is None:
        print(store.explain("registration", source, params,
                            method_version=METHOD_VERSION))
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .. import __version__ as _package_version
from . import budget, keys, manifest, tier_a, tier_b
from .keys import Key, SourceId

__all__ = [
    "Key",
    "SourceId",
    "Stored",
    "ArtefactMissing",
    "AmbiguousArtefact",
    "fingerprint",
    "identify",
    "key_for",
    "put",
    "get",
    "load",
    "explain",
    "resolve",
    "materialise",
    "decision",
    "decisions_for",
    "scan",
    "verify_source",
    "status",
    "budget",
    "keys",
    "manifest",
    "tier_a",
    "tier_b",
    "DECISION_STAGE",
]

#: Decisions ride the tier-A machinery under a reserved stage name, so they get
#: the same sidecars, the same index and the same five-year-readable JSON.
DECISION_STAGE = "_decision"

_MISSING = object()


class ArtefactMissing(LookupError):
    """Nothing stored matches, and the caller said it needed one."""


class AmbiguousArtefact(LookupError):
    """Several stored artefacts match and nothing chooses between them."""


@dataclass(frozen=True)
class Stored:
    """One artefact on disk, with the key it was written under."""

    path: Path
    record: dict[str, Any]

    @property
    def stage(self) -> str:
        return str(self.record.get("stage", ""))

    @property
    def digest(self) -> str:
        return str(self.record.get("digest", ""))

    @property
    def kind(self) -> str:
        return str(self.record.get("kind", ""))

    @property
    def display_only(self) -> bool:
        return bool(self.record.get("display_only"))

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.record.get("params", {}) or {})

    def load(self) -> Any:
        """Read the artefact back in the type it was written as."""
        return tier_a.read(self.path, self.record.get("kind"))


# ----------------------------------------------------------------- the source
def identify(source) -> SourceId:
    """Accept a path or an already-computed identity; return an identity."""
    if isinstance(source, SourceId):
        return source
    if isinstance(source, Mapping):
        return SourceId.from_dict(source)
    return manifest.fingerprint(source)


def fingerprint(source, *, use_cache: bool = True) -> SourceId:
    """Identify a source file, re-reading it only when it may have changed."""
    if isinstance(source, SourceId):
        return source
    return manifest.fingerprint(source, use_cache=use_cache)


def verify_source(source) -> SourceId:
    """Compute and record the full SHA-256 of a source. Opt-in, and slow.

    A whole-file hash of one stack is a full 10.8 GB read. Nothing calls this
    on the normal path; it exists so that when somebody wants certainty about a
    file they can have it, once, and the answer is kept.
    """
    target = Path(source)
    sid = manifest.fingerprint(target)
    full = keys.full_sha256(target)
    sid = SourceId(size=sid.size, sample=sid.sample, path=str(target),
                   mtime_ns=sid.mtime_ns, full_sha256=full)
    index = manifest.load()
    index["fingerprints"][os.path.normcase(os.path.abspath(str(target)))] = \
        sid.as_dict()
    manifest.save(index)
    return sid


def key_for(stage: str, source, params: Mapping[str, Any] | None = None, *,
            method_version: str = "", upstream: Iterable[str] = ()) -> Key:
    """The key a run with these inputs would be stored under."""
    return Key(stage=str(stage), source=identify(source),
               params=dict(params or {}), method_version=str(method_version),
               upstream=tuple(str(u) for u in upstream))


# ------------------------------------------------------------------- writing
def put(stage: str, source, params: Mapping[str, Any] | None = None, *,
        kind: str, value: Any, name: str, output_dir,
        method_version: str = "", upstream: Iterable[str] = (),
        display_only: bool = False,
        extra: Mapping[str, Any] | None = None) -> Stored:
    """Write a derived artefact into tier A and index it.

    ``output_dir`` is the run's results folder, so the artefact lands beside the
    figures and tables it belongs with rather than in a private cache nobody
    can find from a file browser.
    """
    key = key_for(stage, source, params, method_version=method_version,
                  upstream=upstream)
    target = tier_a.artefact_path(output_dir, name, kind,
                                  display_only=display_only)
    written = tier_a.write(target, kind, value, key, display_only=display_only,
                           extra=extra, package_version=_package_version)
    record = tier_a.read_sidecar(written) or key.as_dict()
    manifest.add(record)
    return Stored(path=written, record=record)


# ------------------------------------------------------------------- reading
def get(stage: str, source, params: Mapping[str, Any] | None = None, *,
        method_version: str = "", upstream: Iterable[str] = (),
        display_only: bool = False) -> Stored | None:
    """The artefact stored under exactly this key, or ``None``.

    Exact: same source content, same parameters, same ``METHOD_VERSION``, same
    upstream artefacts. Anything else is a miss, and ``explain`` says which.
    """
    key = key_for(stage, source, params, method_version=method_version,
                  upstream=upstream)
    found = manifest.find(digest=key.digest(), display_only=display_only)
    if not found:
        return None
    return Stored(path=Path(found[0]["path"]), record=found[0])


def load(stage: str, source, params: Mapping[str, Any] | None = None, **kwargs) -> Any:
    """The stored value under this key, or ``None`` on a miss."""
    hit = get(stage, source, params, **kwargs)
    return None if hit is None else hit.load()


def explain(stage: str, source, params: Mapping[str, Any] | None = None, *,
            method_version: str = "", upstream: Iterable[str] = ()) -> str:
    """Why a lookup missed, in the terms a person would ask it in.

    The reason the key is not just an opaque hash. "Cache miss" tells somebody
    to wait six hours; "differs in threshold_sigma, stored 12, requested 8"
    tells them whether they meant to.
    """
    key = key_for(stage, source, params, method_version=method_version,
                  upstream=upstream)
    candidates = manifest.find(stage=stage, source=key.source,
                               display_only=True)
    return keys.explain_miss(key, candidates)


def resolve(stage: str, source, *, params: Mapping[str, Any] | None = None,
            method_version: str | None = None,
            upstream: Iterable[str] | None = None,
            explicit=None, required: bool = True,
            search: Sequence[Any] = (), display_only: bool = False) -> Stored | None:
    """Find the one stored artefact this stage should build on. Never guess.

    An explicit path always wins. Resolution only fills in when no path was
    given *and* exactly one stored artefact matches; zero or several is an
    error naming what was looked for and every candidate found.

    A deliberate divergence from a script this package was copied from.
    ``microglia_raw_registered_stack_export.py`` says of ``--metrics`` and
    ``--shifts``: "they name a specific audited registration run and must never
    acquire a default." That rule was right, because using the wrong
    registration silently corrupts an export and nothing downstream notices.
    What made it necessary was that a person had to remember which dated
    ``AI_Exports`` folder held the right run. Here the key carries the source,
    the parameters and the ``METHOD_VERSION``, so the protection moves from "no
    default exists" to "only an exact match resolves, and ambiguity is a hard
    failure". That script is not changed and keeps its rule; this is a different
    one, chosen, not overlooked.

    Because the protection is now the key, the ambiguity error must stay an
    error. A warning a script can run past would give back exactly the silent
    corruption the old rule prevented.
    """
    for folder in search:
        manifest.scan(folder)

    if explicit is not None:
        target = Path(explicit)
        if not target.exists():
            raise ArtefactMissing(
                f"{target} was named explicitly for stage {stage!r} and does "
                f"not exist.")
        record = tier_a.read_sidecar(target) or {"stage": stage,
                                                 "path": str(target)}
        return Stored(path=target, record=record)

    sid = identify(source)
    found = manifest.find(stage=stage, source=sid, params=params,
                          method_version=method_version, upstream=upstream,
                          display_only=display_only)
    if len(found) == 1:
        return Stored(path=Path(found[0]["path"]), record=found[0])
    if len(found) > 1:
        raise AmbiguousArtefact(keys.explain_ambiguity(stage, sid, found))
    if not required:
        return None
    raise ArtefactMissing(
        explain(stage, sid, params or {},
                method_version=method_version or "")
        + f"\n  Scanned {len(search)} folder(s) this call; "
          "run 'pymicroglia scan <results folder>' to index one more.")


# ---------------------------------------------------------------- tier B work
def materialise(stage: str, source, params: Mapping[str, Any] | None = None, *,
                shape, dtype, fill: Callable[[Any], None],
                method_version: str = "", upstream: Iterable[str] = ()):
    """A memory-mapped array for this key, built once and reused after.

    ``fill`` receives a writable memory map and fills it in place; it is called
    only on a miss. Room is made under the cap before the fill starts, so a long
    write does not discover the cap half way through.
    """
    key = key_for(stage, source, params, method_version=method_version,
                  upstream=upstream)
    digest = key.digest()
    hit = tier_b.read(stage, digest, shape, dtype)
    if hit is not None:
        return hit
    import numpy as np

    count = 1
    for length in shape:
        count *= int(length)
    budget.room_for(count * int(np.dtype(dtype).itemsize))
    with tier_b.open_write(stage, digest, shape, dtype, key.as_dict()) as array:
        fill(array)
    return tier_b.read(stage, digest, shape, dtype)


# ----------------------------------------------------------------- decisions
def decisions_root(source) -> Path:
    """Where a source's human decisions live.

    Beside the results, in Dropbox, because losing one costs a person's time
    rather than a machine's — and because the next person to open that folder
    should be able to see what was decided. ``PYMICROGLIA_DECISIONS`` moves them
    somewhere else for a test or a scratch run.
    """
    override = os.environ.get("PYMICROGLIA_DECISIONS")
    sid = identify(source)
    base = Path(override) if override else Path(sid.path).parent / "AI_Exports"
    return base / "_decisions" / f"{Path(sid.path).stem}_{sid.sample[:8]}"


def decision(name: str, source, value: Any = _MISSING, *, root=None,
             note: str = "") -> Any:
    """Read or write a human judgement call about a source.

        store.decision("channel_assignment", src,
                       value={"dluc": 2, "structural": 1})
        store.decision("channel_assignment", src)   # -> {"dluc": 2, ...}

    Keyed on the source alone. Not invalidated by a parameter change, a
    ``METHOD_VERSION`` bump or a full tier-B eviction: a person answered a
    question once, and a version bump is not a reason to ask them again.
    Reading a decision that was never made returns ``None``.
    """
    sid = identify(source)
    key = Key(stage=DECISION_STAGE, source=sid, params={"decision": str(name)})

    if value is _MISSING:
        found = manifest.find(stage=DECISION_STAGE, source=sid,
                              params={"decision": str(name)})
        if not found:
            found = [record for record
                     in tier_a.sidecars_in(root or decisions_root(sid))
                     if record.get("stage") == DECISION_STAGE
                     and (record.get("params") or {}).get("decision") == str(name)
                     and Path(record.get("path", "")).exists()]
            for record in found:
                manifest.add(record)
        if not found:
            return None
        stored = tier_a.read(found[-1]["path"], "scalars")
        return stored.get("value") if isinstance(stored, Mapping) else stored

    folder = Path(root) if root is not None else decisions_root(sid)
    target = tier_a.artefact_path(folder, str(name), "scalars")
    written = tier_a.write(target, "scalars",
                           {"decision": str(name), "value": value,
                            "note": note, "source": sid.as_dict()},
                           key, package_version=_package_version)
    record = tier_a.read_sidecar(written) or key.as_dict()
    manifest.add(record)
    return value


def decisions_for(source, *, root=None) -> dict[str, Any]:
    """Every decision recorded about one source."""
    sid = identify(source)
    out: dict[str, Any] = {}
    records = manifest.find(stage=DECISION_STAGE, source=sid)
    seen = {record.get("path") for record in records}
    for record in tier_a.sidecars_in(root or decisions_root(sid)):
        if record.get("stage") == DECISION_STAGE and record.get("path") not in seen:
            records.append(record)
    for record in records:
        if not Path(record.get("path", "")).exists():
            continue
        stored = tier_a.read(record["path"], "scalars")
        if isinstance(stored, Mapping) and "decision" in stored:
            out[str(stored["decision"])] = stored.get("value")
    return out


# ------------------------------------------------------------------- indexing
def scan(folder) -> dict[str, Any]:
    """Index one results folder's artefacts. Reads nothing outside it."""
    return manifest.scan(folder)


def status() -> dict[str, Any]:
    """What both tiers are holding, for ``doctor`` and for a person."""
    return {"tier_a": manifest.summary(), "tier_b": budget.report()}
