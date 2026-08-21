"""A local index of what tier A holds, so lookup never walks Dropbox.

Tier A lives beside the results, in a synced folder full of online-only
placeholders. Searching it directly would hydrate files nobody asked for and
take minutes, so the store keeps its own index on the local disk and reads only
folders it has been pointed at:

    pymicroglia scan "...\\AI_Exports\\registered_2026-07-23"

The index is a cache of claims, not a source of truth. Every record names the
sidecar it came from, so it is rebuildable by scanning again, and a record whose
artefact has disappeared is dropped rather than returned.

It also caches source fingerprints. A fingerprint costs a 24 MB read, and the
answer only changes when the file's content does — so it is stored against the
file's size and modification time. A Dropbox re-sync bumps the mtime without
changing a byte; that costs one 24 MB re-read, and then the sampled digest is
the same and everything downstream still hits.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import config
from . import keys as _keys
from . import tier_a

__all__ = [
    "path",
    "load",
    "save",
    "add",
    "remove",
    "scan",
    "find",
    "fingerprint",
    "forget_missing",
    "summary",
]

VERSION = 1


def path() -> Path:
    """Where the index file lives. Local, whatever the pixels do.

    Deliberately ``index_root`` and not ``store_root``: since 2026-08-20 the
    pixels live in the synced project folder, and this file — small, rewritten
    after every run, written by both machines — is exactly what a sync client
    turns into a pair of conflicted copies.
    """
    return config.index_root() / "manifest.json"


def _empty() -> dict[str, Any]:
    return {"version": VERSION, "artefacts": {}, "fingerprints": {}}


def load() -> dict[str, Any]:
    target = path()
    if not target.exists():
        return _empty()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty()
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return _empty()
    data.setdefault("artefacts", {})
    data.setdefault("fingerprints", {})
    return data


def save(data: Mapping[str, Any]) -> Path:
    """Write the index atomically, in its own folder.

    ``os.replace`` is only atomic within one volume, so the temporary file goes
    beside the target rather than in a system temp folder.
    """
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.json")
    partial.write_text(json.dumps(data, indent=2, default=str) + "\n",
                       encoding="utf-8")
    os.replace(partial, target)
    return target


def _slot(value) -> str:
    """One index key per file, case-folded because Windows paths are."""
    return os.path.normcase(os.path.abspath(str(value)))


# --------------------------------------------------------------- the artefacts
def add(record: Mapping[str, Any], data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Register one sidecar's claim. Returns the index, saved unless deferred."""
    standalone = data is None
    index = load() if standalone else data
    side = record.get("sidecar") or (str(tier_a.sidecar_path(record["path"]))
                                     if record.get("path") else None)
    if side is None:
        raise ValueError("an artefact record must name its file or its sidecar")
    entry = dict(record)
    entry["sidecar"] = str(side)
    entry["indexed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index["artefacts"][_slot(side)] = entry
    if standalone:
        save(index)
    return index


def remove(target, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Forget one artefact, by artefact path or sidecar path."""
    standalone = data is None
    index = load() if standalone else data
    for candidate in (target, tier_a.sidecar_path(target)):
        index["artefacts"].pop(_slot(candidate), None)
    if standalone:
        save(index)
    return index


def scan(folder, *, prune: bool = True) -> dict[str, Any]:
    """Index every claimed artefact in **one** folder.

    Not recursive, by rule. A recursive search over a Dropbox path hydrates
    online-only files by the gigabyte, and two hooks in this project block one
    for exactly that reason. A results folder is the unit somebody means anyway:
    one registration run, one export, one set of traces.
    """
    target = Path(folder)
    if not target.is_dir():
        return {"ok": False, "folder": str(target), "error": "not_a_folder",
                "message": f"{target} is not a folder that can be scanned."}

    index = load()
    found = tier_a.sidecars_in(target)
    for record in found:
        add(record, index)

    dropped = 0
    if prune:
        here = {_slot(record["sidecar"]) for record in found}
        stale = [slot for slot, entry in index["artefacts"].items()
                 if _slot(Path(entry["sidecar"]).parent) == _slot(target)
                 and slot not in here]
        for slot in stale:
            index["artefacts"].pop(slot, None)
        dropped = len(stale)

    save(index)
    return {
        "ok": True,
        "folder": str(target),
        "indexed": len(found),
        "dropped": dropped,
        "total": len(index["artefacts"]),
        "stages": sorted({record.get("stage", "") for record in found}),
    }


def _matches(entry: Mapping[str, Any], *, stage: str | None,
             source: "_keys.SourceId | None", params: Mapping[str, Any] | None,
             method_version: str | None, upstream: Iterable[str] | None,
             display_only: bool) -> bool:
    if stage is not None and entry.get("stage") != stage:
        return False
    if bool(entry.get("display_only")) and not display_only:
        return False
    if source is not None:
        stored = _keys.SourceId.from_dict(entry.get("source", {}) or {})
        if stored.identity != source.identity:
            return False
    if method_version is not None and str(entry.get("method_version", "")) != str(method_version):
        return False
    if upstream is not None:
        want = sorted(str(u) for u in upstream)
        if sorted(str(u) for u in entry.get("upstream", []) or []) != want:
            return False
    if params:
        stored_params = _keys.canonical(dict(entry.get("params", {}) or {}))
        for name, value in _keys.canonical(dict(params)).items():
            if stored_params.get(name) != value:
                return False
    return True


def find(*, stage: str | None = None, source=None, params=None,
         method_version: str | None = None, upstream=None,
         digest: str | None = None, display_only: bool = False,
         require_file: bool = True) -> list[dict[str, Any]]:
    """Every indexed artefact matching a whole or partial key.

    ``params`` is a subset match on purpose: the caller resolving a registration
    usually knows the stage and the source and not much else, and asking for
    more than it knows would turn every lookup into a miss.
    """
    index = load()
    out = []
    for entry in index["artefacts"].values():
        if digest is not None and entry.get("digest") != digest:
            continue
        if not _matches(entry, stage=stage, source=source, params=params,
                        method_version=method_version, upstream=upstream,
                        display_only=display_only):
            continue
        if require_file and not Path(entry.get("path", "")).exists():
            continue
        out.append(entry)
    out.sort(key=lambda e: (e.get("created", ""), e.get("path", "")))
    return out


def forget_missing() -> dict[str, Any]:
    """Drop records whose artefact file is gone. Touches no folder to find out."""
    index = load()
    gone = [slot for slot, entry in index["artefacts"].items()
            if not Path(entry.get("path", "")).exists()]
    for slot in gone:
        index["artefacts"].pop(slot, None)
    save(index)
    return {"dropped": len(gone), "total": len(index["artefacts"])}


# ------------------------------------------------------------ the fingerprints
def fingerprint(source, *, use_cache: bool = True) -> "_keys.SourceId":
    """A source's identity, re-read only when the file might have changed.

    Cheap when nothing moved: the cached sample is reused whenever the size and
    modification time still agree with the file on disk. When the mtime changed
    — which a Dropbox re-sync does without altering a byte — this costs one
    24 MB re-read and then usually returns the identical sample, so the artefact
    still hits.
    """
    target = Path(source)
    stat = target.stat()
    slot = _slot(target)
    index = load() if use_cache else _empty()
    cached = index["fingerprints"].get(slot) if use_cache else None

    if (cached and int(cached.get("size", -1)) == int(stat.st_size)
            and int(cached.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            and cached.get("sample")):
        return _keys.SourceId(size=int(cached["size"]), sample=cached["sample"],
                              path=str(target), mtime_ns=int(cached["mtime_ns"]),
                              full_sha256=cached.get("full_sha256"))

    fresh = _keys.fingerprint(target)
    if cached and cached.get("sample") == fresh.sample:
        fresh = _keys.SourceId(size=fresh.size, sample=fresh.sample,
                               path=fresh.path, mtime_ns=fresh.mtime_ns,
                               full_sha256=cached.get("full_sha256"))
    if use_cache:
        index["fingerprints"][slot] = fresh.as_dict()
        save(index)
    return fresh


def summary() -> dict[str, Any]:
    index = load()
    stages: dict[str, int] = {}
    for entry in index["artefacts"].values():
        stages[entry.get("stage", "?")] = stages.get(entry.get("stage", "?"), 0) + 1
    return {
        "path": str(path()),
        "artefacts": len(index["artefacts"]),
        "sources": len(index["fingerprints"]),
        "stages": dict(sorted(stages.items())),
    }
