"""What makes one stored artefact different from another.

A key is the whole point of the store. If two runs share a key they must be
interchangeable, and if they differ anywhere that could move a number they must
not share one. Everything else in this package trusts that promise.

    key = fingerprint(source) + canonical(parameters)
          + METHOD_VERSION + upstream keys

Two things about the source are recorded and deliberately **not** hashed:

**The path.** Dropbox moves and copies files. An artefact has to survive its
source being reorganised into a differently named folder, so identity is the
content, not where the content sits.

**The modification time.** A Dropbox re-sync rewrites a file locally without
changing a byte of it. Hashing mtime would force a full re-registration of a
10.8 GB stack every time a folder resynced. It is recorded instead, and a
changed mtime only triggers a re-fingerprint — 24 MB — after which a genuine
content change misses and a re-sync does not.

**Sampling, not a full hash.** The fingerprint reads the first, middle and last
8 MB plus the TIFF description, which carries the OME-XML and so covers the
acquisition settings without touching pixels. A full SHA-256 of one stack is a
full 10.8 GB read; ``full_sha256`` computes it on request and is recorded when
computed, but nothing computes it by default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "WINDOW_BYTES",
    "DIGEST_CHARS",
    "SourceId",
    "Key",
    "canonical",
    "fingerprint",
    "full_sha256",
    "differences",
    "explain_miss",
    "explain_ambiguity",
]

#: How much of the file each sampled window covers. Three of these plus the
#: description is 24 MB, against a source of 3.4-10.8 GB.
WINDOW_BYTES = 8 * 1024 * 1024

#: Length of a digest in hex characters. Sixteen is the same order as PyFLASH's
#: six-character run slug, with room for far more artefacts than one project.
DIGEST_CHARS = 16

#: Bumping this invalidates every stored fingerprint on purpose. It belongs in
#: the hash so a change to *how* a source is sampled cannot be mistaken for the
#: source having stayed the same.
FINGERPRINT_VERSION = "s1"

_CHUNK = 1 << 20


# ------------------------------------------------------------------ the source
@dataclass(frozen=True)
class SourceId:
    """Which file this artefact was derived from.

    ``size`` and ``sample`` are the identity. ``path`` and ``mtime_ns`` are
    carried for messages and for the cheap re-fingerprint check, and neither
    reaches the digest.
    """

    size: int
    sample: str
    path: str = ""
    mtime_ns: int = 0
    full_sha256: str | None = None

    @property
    def identity(self) -> str:
        """The part that is hashed. Everything else is bookkeeping."""
        return f"{self.size}:{self.sample}"

    @property
    def name(self) -> str:
        return Path(self.path).name if self.path else self.sample[:12]

    def as_dict(self) -> dict[str, Any]:
        data = {
            "size": self.size,
            "sample": self.sample,
            "path": self.path,
            "mtime_ns": self.mtime_ns,
        }
        if self.full_sha256:
            data["full_sha256"] = self.full_sha256
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceId":
        return cls(
            size=int(data.get("size", 0)),
            sample=str(data.get("sample", "")),
            path=str(data.get("path", "")),
            mtime_ns=int(data.get("mtime_ns", 0)),
            full_sha256=data.get("full_sha256"),
        )


def _description_bytes(path: Path) -> bytes:
    """The TIFF ImageDescription of the first page, or nothing.

    For an OME-TIFF this is the OME-XML: channel names, pixel size, per-plane
    timestamps. Including it means two acquisitions that happen to share their
    first, middle and last windows still fingerprint apart. ``tifffile`` is a
    hard dependency of this package, so whether this contributes is a property
    of the file, not of the environment.
    """
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return b""
    try:
        import tifffile

        with tifffile.TiffFile(path) as handle:
            description = handle.pages[0].description or ""
    except Exception:
        return b""
    return str(description).encode("utf-8", "replace")


def _windows(handle, size: int, window: int) -> Iterable[bytes]:
    """First, middle and last ``window`` bytes, in that order."""
    if size <= 3 * window:
        handle.seek(0)
        while True:
            block = handle.read(_CHUNK)
            if not block:
                return
            yield block
        return
    for offset in (0, (size - window) // 2, size - window):
        handle.seek(offset)
        remaining = window
        while remaining > 0:
            block = handle.read(min(_CHUNK, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block


def fingerprint(path, *, window: int = WINDOW_BYTES) -> SourceId:
    """Identify a source file by size plus a sampled SHA-256.

    Reads 24 MB of a multi-gigabyte stack. The sources sit in a folder with
    online-only placeholders, so this hydrates a placeholder; that is the
    accepted cost, and hashing the whole file — which would hydrate all of it —
    is what this exists to avoid.
    """
    target = Path(path)
    stat = target.stat()
    digest = hashlib.sha256()
    digest.update(FINGERPRINT_VERSION.encode("ascii"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(_description_bytes(target))
    with open(target, "rb") as handle:
        for block in _windows(handle, stat.st_size, window):
            digest.update(block)
    return SourceId(
        size=int(stat.st_size),
        sample=digest.hexdigest(),
        path=str(target),
        mtime_ns=int(stat.st_mtime_ns),
    )


def full_sha256(path, *, chunk: int = _CHUNK) -> str:
    """The whole-file hash, on request only.

    Opt-in verification. Nothing in the normal path calls this: one stack is a
    10.8 GB read, and ``microglia_cosmic_ray_removal.py`` already pays that cost
    on every run purely to fill in a provenance field. Here it is a deliberate
    act, and the result is recorded so it need not be paid twice.
    """
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# -------------------------------------------------------------- the parameters
def canonical(value: Any) -> Any:
    """Order-insensitive form of a parameter value, for hashing.

    Copied from ``PyFLASH/pipeline_io.py`` ``_canon`` — twelve lines, and the
    rule is worth keeping identical across the lab's packages rather than
    importing a sibling analysis package.

    Dictionaries sort by key and sets sort by string, because their order is an
    accident. Lists keep theirs, because a channel order or a time window is
    meaningful. Anything else falls through to ``str`` at ``json.dumps`` time.
    """
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v) for v in value), key=str)
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _stable(payload: Any) -> str:
    return json.dumps(canonical(payload), sort_keys=True, default=str)


# --------------------------------------------------------------------- the key
@dataclass(frozen=True)
class Key:
    """Everything that must match for a stored artefact to be reusable."""

    stage: str
    source: SourceId
    params: Mapping[str, Any]
    method_version: str = ""
    upstream: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "source": self.source.identity,
            "params": canonical(dict(self.params)),
            "method_version": self.method_version,
            "upstream": sorted(str(u) for u in self.upstream),
        }

    def digest(self) -> str:
        raw = _stable(self.payload()).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:DIGEST_CHARS]

    def as_dict(self) -> dict[str, Any]:
        """The form written into a sidecar and the manifest."""
        return {
            "stage": self.stage,
            "digest": self.digest(),
            "method_version": self.method_version,
            "source": self.source.as_dict(),
            "params": canonical(dict(self.params)),
            "upstream": list(self.upstream),
        }


# ------------------------------------------------------------ explaining a miss
def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if value is None:
        return "-"
    return str(value)


def differences(stored: Mapping[str, Any], key: Key) -> list[tuple[str, Any, Any]]:
    """Every field in which a stored record and a requested key disagree.

    Returns ``(field, stored_value, requested_value)`` triples. A parameter
    present on one side only reports ``None`` for the other, which is the honest
    answer: it was not asked for, or it was not recorded.
    """
    out: list[tuple[str, Any, Any]] = []
    stored_source = SourceId.from_dict(stored.get("source", {}) or {})
    if stored_source.identity != key.source.identity:
        out.append(("source", stored_source.name, key.source.name))
    if str(stored.get("method_version", "")) != str(key.method_version):
        out.append(("METHOD_VERSION", stored.get("method_version", ""),
                    key.method_version))
    stored_up = sorted(str(u) for u in stored.get("upstream", []) or [])
    wanted_up = sorted(str(u) for u in key.upstream)
    if stored_up != wanted_up:
        out.append(("upstream", ", ".join(stored_up) or "-",
                    ", ".join(wanted_up) or "-"))

    stored_params = canonical(dict(stored.get("params", {}) or {}))
    wanted_params = canonical(dict(key.params))
    for name in sorted(set(stored_params) | set(wanted_params)):
        was, now = stored_params.get(name), wanted_params.get(name)
        if was != now:
            out.append((name, was, now))
    return out


def _nearest(key: Key, candidates: Sequence[Mapping[str, Any]]):
    scored = [(len(differences(c, key)), i, c) for i, c in enumerate(candidates)]
    scored.sort()
    return scored[0][2] if scored else None


def explain_miss(key: Key, candidates: Sequence[Mapping[str, Any]] = ()) -> str:
    """Why this key found nothing, in the terms a person would ask it in.

    The reason a digest alone is not enough. "Cache miss" tells you to wait six
    hours; "differs in threshold_sigma, stored 12, requested 8" tells you
    whether you meant to.
    """
    head = (f"Cache miss for stage {key.stage!r} on {key.source.name}.")
    near = _nearest(key, candidates)
    if near is None:
        return (f"{head}\n  Nothing is stored for this stage and source. "
                f"Looked for digest {key.digest()}.")
    delta = differences(near, key)
    if not delta:
        return (f"{head}\n  A record matches on every field, so the artefact "
                f"file itself is missing from {near.get('path', 'its folder')}.")
    lines = [f"{head}", "Nearest stored artefact differs in:"]
    width = max(len(name) for name, _, _ in delta)
    for name, was, now in delta:
        lines.append(f"  {name:<{width}}   stored {_format(was)}   "
                     f"requested {_format(now)}")
    moved = {name for name, _, _ in delta}
    same = [f"the {w}" for w in ("source", "METHOD_VERSION") if w not in moved]
    if same:
        lines.append(f"Everything else matches, including "
                     f"{' and '.join(same)}.")
    lines.append(f"  stored at {near.get('path', '?')}")
    return "\n".join(lines)


def explain_ambiguity(stage: str, source: SourceId,
                      candidates: Sequence[Mapping[str, Any]]) -> str:
    """Several artefacts match and nothing chooses between them.

    Names every candidate and the parameters that separate them, because the
    only safe resolution is a person saying which one they meant.
    """
    varying: set[str] = set()
    seen: list[Mapping[str, Any]] = [c.get("params", {}) or {} for c in candidates]
    for name in {n for params in seen for n in params}:
        values = {_stable(params.get(name)) for params in seen}
        if len(values) > 1:
            varying.add(name)

    lines = [f"{len(candidates)} stored artefacts match stage {stage!r} on "
             f"{source.name}, and nothing chooses between them.",
             "Name one explicitly, or give the parameters that separate them."]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"  {index}  {candidate.get('path', '?')}")
        params = candidate.get("params", {}) or {}
        detail = "   ".join(f"{name} {_format(params.get(name))}"
                            for name in sorted(varying)) or "identical parameters"
        lines.append(f"       {detail}")
    if varying:
        lines.append(f"They differ in: {', '.join(sorted(varying))}.")
    else:
        lines.append("They differ in nothing recorded, which means one is a "
                     "stale copy. Delete the one you do not want.")
    return "\n".join(lines)
