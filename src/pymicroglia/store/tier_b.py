"""Tier B — the materialised arrays. Rebuildable, capped, disposable.

Registered pixels are big: one float32 copy of a 10.8 GB three-channel uint16
stack is roughly 21 GB. Nothing scientific is lost by deleting it, because tier
A plus the original TIFF rebuilds it exactly — but rebuilding costs hours, so
it lives in the project folder where a second machine finds it rather than
re-deriving it. The cap still applies, and eviction still takes the least
recently used first.

**A dehydrated array is the failure mode of a synced store.** Dropbox frees
space by leaving a file that looks present until something reads it, and
memory-mapping one faults 21 GB back down the network a page at a time. Mark
the store folder "Make available offline"; ``doctor`` says so when it finds one.

The write pattern is the one already proven in ``dluc_pipeline.py``: fill a
memory-mapped ``.partial.npy`` and rename it into place. A crash mid-write
leaves a partial file that nothing will ever accept as a hit, so a truncated
array is never readable as a whole one.
"""

from __future__ import annotations

import gc
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .. import config

__all__ = [
    "root",
    "array_path",
    "sidecar_path",
    "partial_path",
    "valid",
    "read",
    "open_write",
    "materialise",
    "release",
    "held",
    "entries",
    "forget",
    "HOLD_STALE_SECONDS",
]

#: A hold marker left by a process that died is stale after this long. Windows
#: has no portable way to ask whether a pid is alive that does not risk killing
#: it, so eviction leans on two safer signals instead: this age, and the fact
#: that Windows refuses to delete a file that is currently mapped. A stale hold
#: only ever costs one skipped eviction, retried on the next call.
HOLD_STALE_SECONDS = 12 * 3600

#: Paths this process has mapped and not released. Authoritative for this
#: process, and the reason eviction can promise never to pull an array out from
#: under a live reader.
_HELD: dict[str, int] = {}


def root() -> Path:
    """The tier-B root: the project's own ``PixelStore``, or wherever it is set.

    No longer refuses a synced path. The store lives in the project folder on
    purpose, so the hours that produced these arrays are not paid again on the
    next machine. What a synced store has to be watched for is dehydration, and
    that is :func:`config.is_placeholder`'s job rather than this one's.
    """
    return config.store_root()


def _folder(stage: str) -> Path:
    return root() / str(stage)


def array_path(stage: str, digest: str) -> Path:
    return _folder(stage) / f"{digest}.npy"


def sidecar_path(stage: str, digest: str) -> Path:
    return _folder(stage) / f"{digest}.json"


def partial_path(stage: str, digest: str) -> Path:
    """Beside its target, never in a system temp folder.

    ``os.replace`` is only atomic within one volume, so a partial written to
    ``%TEMP%`` on a different drive would be a copy, and a copy can be
    interrupted half way.
    """
    return _folder(stage) / f"{digest}.partial.npy"


def _hold_path(stage: str, digest: str, pid: int) -> Path:
    return _folder(stage) / f"{digest}.hold-{pid}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n",
                    encoding="utf-8")


def _unmap(array) -> None:
    """Release the mapping, whatever else still holds a reference to it.

    Windows will not rename a mapped file, and the caller of ``open_write``
    still has the array bound to its ``with`` target while the commit runs. So
    the mapping is closed explicitly rather than left to reference counting.
    Touching the array afterwards raises, which is correct: once it is
    committed, the way to read it is ``read()``.
    """
    handle = getattr(array, "_mmap", None)
    if handle is None:
        return
    try:
        handle.close()
    except (BufferError, ValueError):
        pass


def _expected_bytes(shape, dtype) -> int:
    import numpy as np

    count = 1
    for length in shape:
        count *= int(length)
    return count * int(np.dtype(dtype).itemsize)


def valid(stage: str, digest: str, shape=None, dtype=None) -> bool:
    """Is the stored array trustworthy, and is it the one being asked for?

    A cache hit has to be indistinguishable from a cold run, so the declared
    shape and dtype are checked against the sidecar *and* against the file size.
    A file that is the right shape on paper and the wrong size on disk is a
    partial write that got renamed by something other than this module.
    """
    path = array_path(stage, digest)
    if not path.exists():
        return False
    side = _read_json(sidecar_path(stage, digest))
    if not side:
        return False
    if shape is not None and list(side.get("shape", [])) != [int(n) for n in shape]:
        return False
    if dtype is not None and str(side.get("dtype")) != str(dtype):
        return False
    try:
        import numpy as np

        header_free = path.stat().st_size
        expected = _expected_bytes(side.get("shape", []), side.get("dtype", "u1"))
        if header_free < expected:
            return False
        array = np.load(path, mmap_mode="r")
    except Exception:
        return False
    if list(array.shape) != list(side.get("shape", [])):
        return False
    return str(array.dtype) == str(side.get("dtype"))


def read(stage: str, digest: str, shape=None, dtype=None):
    """A read-only memory map of the stored array, or ``None`` on a miss.

    Touches the artefact's last-read timestamp, which is what eviction sorts on.
    Windows access times are unreliable and updated lazily, so the store keeps
    its own rather than asking the filesystem.
    """
    if not valid(stage, digest, shape, dtype):
        return None
    import numpy as np

    path = array_path(stage, digest)
    array = np.load(path, mmap_mode="r")
    side = _read_json(sidecar_path(stage, digest))
    side["last_read"] = time.time()
    side["last_read_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(sidecar_path(stage, digest), side)
    _hold(stage, digest)
    return array


def _hold(stage: str, digest: str) -> None:
    key = str(array_path(stage, digest))
    _HELD[key] = _HELD.get(key, 0) + 1
    marker = _hold_path(stage, digest, os.getpid())
    try:
        marker.write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


def release(stage: str, digest: str) -> None:
    """Give up one reference to a mapped array so eviction may take it.

    Not required for correctness — eviction skips anything the filesystem
    refuses to delete — but a long-running process that never releases pins the
    cache above its cap.
    """
    key = str(array_path(stage, digest))
    remaining = _HELD.get(key, 0) - 1
    if remaining > 0:
        _HELD[key] = remaining
        return
    _HELD.pop(key, None)
    try:
        _hold_path(stage, digest, os.getpid()).unlink()
    except OSError:
        pass


def held() -> set[str]:
    """Arrays that must not be evicted: mapped here, or claimed recently."""
    out = {path for path, count in _HELD.items() if count > 0}
    base = config.store_root()
    if not base.exists():
        return out
    cutoff = time.time() - HOLD_STALE_SECONDS
    for stage_dir in base.iterdir():
        if not stage_dir.is_dir():
            continue
        for marker in stage_dir.glob("*.hold-*"):
            digest = marker.name.split(".hold-")[0]
            try:
                claimed = float(marker.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                claimed = 0.0
            if claimed >= cutoff:
                out.add(str(stage_dir / f"{digest}.npy"))
            else:
                try:
                    marker.unlink()
                except OSError:
                    pass
    return out


@contextmanager
def open_write(stage: str, digest: str, shape, dtype,
               key: Mapping[str, Any] | None = None) -> Iterator[Any]:
    """A writable memory map that only becomes a cache entry if it completes.

        with open_write("registered", digest, (n, h, w), "float32") as arr:
            for i in range(n):
                arr[i] = frame(i)

    On a clean exit the array is flushed and renamed into place atomically. On
    an exception the partial file is removed, so an interrupted run leaves no
    half-written array for the next one to trust.
    """
    import numpy as np

    folder = _folder(stage)
    folder.mkdir(parents=True, exist_ok=True)
    partial = partial_path(stage, digest)
    target = array_path(stage, digest)
    array = np.lib.format.open_memmap(partial, mode="w+",
                                      dtype=np.dtype(dtype),
                                      shape=tuple(int(n) for n in shape))
    try:
        yield array
    except BaseException:
        _unmap(array)
        del array
        gc.collect()
        try:
            partial.unlink()
        except OSError:
            pass
        raise
    array.flush()
    _unmap(array)
    del array
    gc.collect()
    os.replace(partial, target)
    side = dict(key or {})
    side.update({
        "stage": stage,
        "digest": digest,
        "shape": [int(n) for n in shape],
        "dtype": str(np.dtype(dtype)),
        "bytes": target.stat().st_size,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_read": time.time(),
    })
    _write_json(sidecar_path(stage, digest), side)


def materialise(stage: str, digest: str, shape, dtype, fill,
                key: Mapping[str, Any] | None = None):
    """Return the stored array, building it with ``fill`` if it is not there.

    ``fill`` is handed a writable memory map and fills it in place. It is called
    only on a miss, which is the whole arrangement: the caller writes the
    expensive loop once and stops caring whether it runs.
    """
    hit = read(stage, digest, shape, dtype)
    if hit is not None:
        return hit
    with open_write(stage, digest, shape, dtype, key) as array:
        fill(array)
    return read(stage, digest, shape, dtype)


def entries(stage: str | None = None) -> list[dict[str, Any]]:
    """Every stored array, with the fields eviction needs."""
    base = config.store_root()
    if not base.exists():
        return []
    out: list[dict[str, Any]] = []
    folders = [base / stage] if stage else [d for d in base.iterdir() if d.is_dir()]
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.npy")):
            if path.name.endswith(".partial.npy"):
                continue
            side = _read_json(folder / (path.stem + ".json"))
            try:
                size = path.stat().st_size
            except OSError:
                continue
            out.append({
                "stage": folder.name,
                "digest": path.stem,
                "path": str(path),
                "bytes": size,
                "last_read": float(side.get("last_read", 0.0)),
                "shape": side.get("shape", []),
                "dtype": side.get("dtype", ""),
            })
    return out


def forget(stage: str, digest: str) -> bool:
    """Delete one stored array and its sidecar. False if it could not be taken.

    Never raises on a locked file. A mapped array cannot be deleted on Windows,
    and that refusal is a feature: eviction skips it and tries again later
    rather than pulling it out from under a live reader.
    """
    path = array_path(stage, digest)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return False
    for companion in (sidecar_path(stage, digest),):
        try:
            companion.unlink()
        except OSError:
            pass
    return True
