"""Tier A — the permanent derived artefacts, beside the outputs.

This is the analysis. Registration's whole output is a 298 KB table of per-frame
shifts against a 10.8 GB input, so keeping what a step *derived* costs almost
nothing and means a measurement can be re-run with no pixels present at all.
Tier A lives in Dropbox with the results it belongs to, and is never evicted.

Every artefact gets a sidecar carrying its key:

    registration_shifts_and_qc.csv
    registration_shifts_and_qc.artefact.json

**Never pickle.** A stored artefact has to be readable in five years by
something that is not this package, so a table is a CSV, an array is a
compressed ``.npz`` and a scalar is JSON. Reading one back needs numpy and the
standard library, and nothing else.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import keys as _keys

__all__ = [
    "KINDS",
    "EXTENSION",
    "SIDECAR_SUFFIX",
    "sidecar_path",
    "artefact_path",
    "write",
    "read",
    "read_sidecar",
    "sidecars_in",
    "DISPLAY_ONLY_MARK",
]

#: What a stage may store, and how each is written. The list is short on
#: purpose: a new kind is a decision about a file format somebody has to read in
#: five years, not a convenience.
KINDS = ("table", "array", "mask", "labels", "scalars")

EXTENSION = {
    "table": ".csv",       # what the engines already write
    "array": ".npz",       # compressed, dtype preserved
    "mask": ".npz",        # boolean, packed to bits
    "labels": ".npz",      # integer label image
    "scalars": ".json",
}

SIDECAR_SUFFIX = ".artefact.json"

#: The house rule, enforced in the filename. A display-only artefact says so in
#: its own name so a measurement stage can refuse it without opening it.
DISPLAY_ONLY_MARK = "_DISPLAY_ONLY"


def sidecar_path(path) -> Path:
    """``x.csv`` -> ``x.artefact.json``, beside it."""
    target = Path(path)
    return target.with_name(target.stem + SIDECAR_SUFFIX)


def artefact_path(directory, name: str, kind: str, *,
                  display_only: bool = False) -> Path:
    """Where an artefact of this kind and name goes."""
    if kind not in EXTENSION:
        raise ValueError(f"unknown artefact kind {kind!r}; expected one of "
                         f"{', '.join(KINDS)}")
    stem = str(name)
    if display_only and DISPLAY_ONLY_MARK not in stem.upper():
        stem += DISPLAY_ONLY_MARK
    return Path(directory) / f"{stem}{EXTENSION[kind]}"


# ------------------------------------------------------------------- tables
def _normalise_table(value: Any) -> tuple[list[str], list[list[Any]]]:
    """Accept columns-of-values or a list of rows; store columns-of-values."""
    if isinstance(value, Mapping):
        columns = [str(name) for name in value]
        data = [list(_as_list(value[name])) for name in value]
        lengths = {len(column) for column in data}
        if len(lengths) > 1:
            raise ValueError(f"table columns have different lengths: "
                             f"{dict(zip(columns, (len(c) for c in data)))}")
        return columns, data
    rows = list(value)
    if not rows:
        return [], []
    if not isinstance(rows[0], Mapping):
        raise TypeError("a table is a mapping of column -> values, or a "
                        "sequence of row mappings")
    columns = list(dict.fromkeys(name for row in rows for name in row))
    data = [[row.get(name) for row in rows] for name in columns]
    return columns, data


def _as_list(column: Any) -> list[Any]:
    tolist = getattr(column, "tolist", None)
    return list(tolist()) if callable(tolist) else list(column)


def _column_type(values: Sequence[Any]) -> str:
    """One dtype name per column, so a read gives back what was written.

    A CSV is text. Without this, a float column comes back as strings and a
    cache hit stops being indistinguishable from a cold run.
    """
    seen = {type(v).__name__ for v in values if v is not None}
    seen.discard("bool_")
    if not seen:
        return "str"
    if seen <= {"bool"}:
        return "bool"
    if seen <= {"int", "int32", "int64"}:
        return "int"
    if seen <= {"int", "int32", "int64", "float", "float32", "float64"}:
        return "float"
    return "str"


_READERS = {
    "bool": lambda text: text.strip().lower() in {"1", "true", "yes"},
    "int": int,
    "float": float,
    "str": str,
}


def _write_table(path: Path, value: Any) -> dict[str, Any]:
    columns, data = _normalise_table(value)
    types = {name: _column_type(column) for name, column in zip(columns, data)}
    rows = zip(*data) if data else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])
    return {"columns": columns, "dtypes": types,
            "rows": len(data[0]) if data else 0}


def _read_table(path: Path, side: Mapping[str, Any]) -> dict[str, list[Any]]:
    types = (side.get("table", {}) or {}).get("dtypes", {})
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        out: dict[str, list[Any]] = {name: [] for name in header}
        for row in reader:
            for name, cell in zip(header, row):
                convert = _READERS.get(types.get(name, "str"), str)
                out[name].append(None if cell == "" else convert(cell))
    return out


# ------------------------------------------------------------------- arrays
def _write_array(path: Path, kind: str, value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    if kind == "mask":
        if array.dtype != bool:
            raise TypeError(f"a mask is boolean; got {array.dtype}. Store an "
                            f"integer image as kind 'labels'.")
        np.savez_compressed(path, packed=np.packbits(array),
                            shape=np.asarray(array.shape, dtype=np.int64))
    elif kind == "labels":
        if array.dtype.kind not in "ui":
            raise TypeError(f"labels are integers; got {array.dtype}")
        np.savez_compressed(path, data=array)
    else:
        np.savez_compressed(path, data=array)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _read_array(path: Path, kind: str):
    import numpy as np

    with np.load(path) as bundle:
        if kind == "mask":
            shape = tuple(int(n) for n in bundle["shape"])
            count = int(np.prod(shape)) if shape else 0
            flat = np.unpackbits(bundle["packed"])[:count]
            return flat.astype(bool).reshape(shape)
        return bundle["data"]


# -------------------------------------------------------------- write and read
def write(path, kind: str, value: Any, key: "_keys.Key", *,
          display_only: bool = False, extra: Mapping[str, Any] | None = None,
          package_version: str = "") -> Path:
    """Write one artefact and its sidecar. Returns the artefact path.

    The sidecar is written after the artefact, so a crash between the two leaves
    an unclaimed file rather than a claim with nothing behind it. Resolution
    looks for sidecars, so an unclaimed file is invisible and harmless.
    """
    if kind not in EXTENSION:
        raise ValueError(f"unknown artefact kind {kind!r}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if kind == "table":
        detail = _write_table(target, value)
    elif kind == "scalars":
        target.write_text(json.dumps(value, indent=2, sort_keys=True,
                                     default=str) + "\n", encoding="utf-8")
        detail = {"fields": sorted(value)} if isinstance(value, Mapping) else {}
    else:
        detail = _write_array(target, kind, value)

    side = dict(key.as_dict())
    side.update({
        "kind": kind,
        "display_only": bool(display_only),
        "pymicroglia": package_version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path": str(target),
        "bytes": target.stat().st_size,
    })
    if kind == "table":
        side["table"] = detail
    else:
        side.update(detail)
    if extra:
        side["extra"] = dict(extra)

    sidecar_path(target).write_text(
        json.dumps(side, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8")
    return target


def read(path, kind: str | None = None) -> Any:
    """Read an artefact back in the type it was written as."""
    target = Path(path)
    side = read_sidecar(target) or {}
    kind = kind or side.get("kind") or _kind_from_suffix(target)
    if kind == "table":
        return _read_table(target, side)
    if kind == "scalars":
        return json.loads(target.read_text(encoding="utf-8"))
    return _read_array(target, kind)


def _kind_from_suffix(path: Path) -> str:
    return {".csv": "table", ".json": "scalars"}.get(path.suffix.lower(), "array")


def read_sidecar(path) -> dict[str, Any] | None:
    """The key an artefact was written under, or ``None`` if it has no claim."""
    target = Path(path)
    side = target if target.name.endswith(SIDECAR_SUFFIX) else sidecar_path(target)
    if not side.exists():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("path", str(side.with_name(side.name[:-len(SIDECAR_SUFFIX)])))
    data["sidecar"] = str(side)
    return data


def sidecars_in(directory) -> list[dict[str, Any]]:
    """Every claimed artefact in **one** folder. Never recursive.

    Deliberately flat. The sources sit in Dropbox with online-only
    placeholders, and walking a tree there hydrates files nobody asked for; a
    results folder is the unit somebody actually means.
    """
    folder = Path(directory)
    if not folder.is_dir():
        return []
    out = []
    for side in sorted(folder.glob("*" + SIDECAR_SUFFIX)):
        data = read_sidecar(side)
        if data is not None:
            out.append(data)
    return out
