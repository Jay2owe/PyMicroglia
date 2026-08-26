"""The provenance bundle a figure arrives in, written rather than remembered.

The house rule is that every chart saved to a file goes through the
``plot-that`` skill, which produces a fixed layout: the figure, the exact
plotted table as a CSV, copies of the source files with their hashes, and a
README saying what the figure shows and where its numbers came from. This
module implements that layout from inside the package, so a figure produced by
a registered action *is* an audited bundle instead of needing one built around
it afterwards.

    <stem>_bundle/
      README.md                 what it shows, where the data came from
      data/sources.csv          path, copy, size, modified, SHA256, per source
      data/sources.md           the same thing to read
      data/src/                 the source files, copied
      data/der/figure_data.csv  the exact plotted table
      fig/<slug>.svg            the vector figure
      fig/preview.png           a raster preview for a quick look

``data/der/figure_data.csv`` is at that exact path because the skill's
``register.py`` requires it there; so does ``data/sources.csv``. Getting either
wrong makes the bundle unregisterable, which is the check that the figure is
finished.

One deliberate limit: a source above :data:`SOURCE_COPY_MAX_BYTES` is listed
and hashed but **not** copied. The sources here are sometimes 10 GB TIFFs, and
a bundle that duplicates one beside every figure drawn from it fills a Dropbox
rather than documenting anything. The row says so, and so does the README.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BUNDLE_SUFFIX",
    "SOURCE_COPY_MAX_BYTES",
    "FLOAT_FORMAT",
    "describe_sources",
    "write_table",
    "table_bytes",
    "write_provenance",
    "figure_targets",
    "write_bundle",
]

BUNDLE_SUFFIX = "_bundle"

#: Above this, a source is hashed and listed but not copied into the bundle.
SOURCE_COPY_MAX_BYTES = 32 * 1024 * 1024

#: The engines write ``%.6f``. Ports produce the same characters, not merely
#: the same value, so a stored table can be compared with ``diff``.
FLOAT_FORMAT = "%.6f"

_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_sources(sources: Iterable[Any]) -> list[dict[str, Any]]:
    """Every source file, with the facts needed to find it again.

    Accepts a path, or a mapping carrying ``path`` and optionally ``name`` and
    ``role``. Anything already carrying a ``sha256`` is trusted — a caller that
    read the file once should not pay to read it twice.
    """
    described: list[dict[str, Any]] = []
    for index, entry in enumerate(sources or ()):
        row = dict(entry) if isinstance(entry, Mapping) else {"path": entry}
        path = Path(str(row.get("path", ""))).expanduser()
        name = str(row.get("name") or path.stem or f"s{index + 1}")
        stat = path.stat() if path.is_file() else None
        described.append({
            "name": name,
            "role": str(row.get("role", "source")),
            "path": str(path),
            "file_name": path.name,
            "exists": stat is not None,
            "bytes": int(stat.st_size) if stat else None,
            "modified": (datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                         .isoformat() if stat else None),
            "sha256": str(row["sha256"]) if row.get("sha256") else (
                _sha256(path) if stat else None),
            "rows": row.get("rows"),
        })
    return described


# -------------------------------------------------------------------- table
def _cell(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return FLOAT_FORMAT % float(value)
    except (TypeError, ValueError):
        return str(value)


def table_bytes(table: Mapping[str, Sequence[Any]]) -> bytes:
    """Serialize the exact plotted table once for sidecars and embedded records."""

    columns = [str(name) for name in table]
    data = [list(values) for values in table.values()]
    height = max((len(column) for column in data), default=0)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in range(height):
        writer.writerow([
            _cell(column[row]) if row < len(column) else "nan"
            for column in data])
    return stream.getvalue().encode("utf-8")


def write_table(path: Path, table: Mapping[str, Sequence[Any]]) -> Path:
    """The exact plotted values, one column per drawn series.

    Columns of different length are padded with ``nan`` rather than truncated:
    merged traces keep their own time vectors, and a short recording stopping
    early is a fact about the data, not a reason to drop the long one's tail.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(table_bytes(table))
    return path


# --------------------------------------------------------------- provenance
def _artefact_rows(artefacts: Iterable[Any]) -> list[dict[str, Any]]:
    """Stored artefacts a figure was drawn from, named so they can be found.

    Accepts a ``store.Stored``, a mapping, or a plain path. The point of the
    list is that a reader can go from the figure back to the keyed artefact and
    from there to the parameters that produced it.
    """
    rows = []
    for item in artefacts or ():
        record = getattr(item, "record", None)
        if record is not None:
            rows.append({"path": str(getattr(item, "path", "")),
                         "stage": record.get("stage"),
                         "digest": record.get("digest"),
                         "method_version": record.get("method_version")})
        elif isinstance(item, Mapping):
            rows.append({key: item[key] for key in item})
        else:
            rows.append({"path": str(item)})
    return rows


def write_provenance(path: Path, *, sources: Sequence[Mapping[str, Any]],
                     figures: Sequence[Path], table: Path,
                     settings: Mapping[str, Any], artefacts: Iterable[Any],
                     claim: str, notes: Sequence[str] = (),
                     drawn: Any = None) -> Path:
    """Where the figure came from, in the shape the engines already wrote it.

    ``drawn`` is the per-series detail: for a trace panel, each trace's colour,
    detrend, shaded width, normalisation denominator and spread. The plotted
    table says what was drawn; this says what each of those numbers *is*, which
    is the part somebody reading the figure a year later cannot reconstruct.
    """
    document = {
        "package": "pymicroglia",
        "claim": claim,
        "figure_files": [str(p) for p in figures],
        "plotted_table": str(table),
        "generated_from": list(sources),
        "artefacts_drawn": _artefact_rows(artefacts),
        "drawn": drawn,
        "settings": {key: value for key, value in settings.items()
                     if not callable(value)},
        "notes": list(notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, default=str) + "\n",
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------- bundle
_SOURCE_COLUMNS = ("name", "role", "source_path", "copied_path", "file_name",
                   "modified", "bytes", "sha256", "copied")


def _copy_sources(root: Path, sources: Sequence[Mapping[str, Any]]
                  ) -> list[dict[str, Any]]:
    folder = root / "data" / "src"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in sources:
        origin = Path(entry["path"])
        size = entry.get("bytes") or 0
        copied = ""
        if entry.get("exists") and size <= SOURCE_COPY_MAX_BYTES:
            destination = folder / f"{entry['name']}_{origin.name}"
            shutil.copy2(origin, destination)
            copied = str(destination.relative_to(root).as_posix())
        rows.append({
            "name": entry["name"], "role": entry.get("role", "source"),
            "source_path": str(origin), "copied_path": copied,
            "file_name": origin.name, "modified": entry.get("modified") or "",
            "bytes": size, "sha256": entry.get("sha256") or "",
            "copied": "yes" if copied else "no",
        })
    return rows


def _write_sources(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(root / "data" / "sources.csv", "w", encoding="utf-8",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_SOURCE_COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Sources", ""]
    for row in rows:
        lines.append(f"## {row['name']} ({row['role']})")
        lines.append("")
        lines.append(f"- original: `{row['source_path']}`")
        lines.append(f"- copied: "
                     + (f"`{row['copied_path']}`" if row["copied_path"] else
                        f"not copied — {row['bytes']} bytes is over the "
                        f"{SOURCE_COPY_MAX_BYTES} byte copy limit"))
        lines.append(f"- modified: {row['modified'] or 'unknown'}")
        lines.append(f"- sha256: `{row['sha256'] or 'unavailable'}`")
        lines.append("")
    (root / "data" / "sources.md").write_text("\n".join(lines),
                                              encoding="utf-8")


def _readme(root: Path, *, slug: str, claim: str,
            rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any],
            artefacts: Iterable[Any], notes: Sequence[str]) -> None:
    uncopied = [row for row in rows if row["copied"] == "no"]
    lines = [
        f"# {slug}", "",
        claim or "No claim was stated for this figure.", "",
        "## What is here", "",
        f"- `fig/{slug}.svg` — the figure",
        "- `fig/preview.png` — a raster preview",
        "- `data/der/figure_data.csv` — every value actually drawn",
        "- `data/sources.csv` — every source file, with its SHA256",
        "- `data/src/` — copies of those sources",
        "",
        "## How it was drawn", "",
        "Drawn by PyMicroglia through `visualisation.panels.save`, which is "
        "the only place this package writes a figure. The plotted table is "
        "not a summary of the figure — it is the values the figure was drawn "
        "from, so the two cannot disagree.",
        "",
    ]
    if artefacts:
        lines += ["## Artefacts drawn", ""]
        lines += [f"- `{row.get('stage') or row.get('path')}`"
                  for row in _artefact_rows(artefacts)]
        lines += [""]
    if uncopied:
        lines += ["## Caveats", ""]
        lines += [f"- `{row['file_name']}` was hashed and listed but not "
                  f"copied: {row['bytes']} bytes is over the "
                  f"{SOURCE_COPY_MAX_BYTES} byte limit. Copying a 10 GB "
                  f"acquisition beside every figure drawn from it documents "
                  f"nothing and fills a disk."
                  for row in uncopied]
        lines += [""]
    if notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in notes] + [""]
    if settings:
        lines += ["## Settings", "", "```json",
                  json.dumps({k: v for k, v in settings.items()
                              if not callable(v)}, indent=2, default=str),
                  "```", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def figure_targets(root: Path, slug: str) -> dict[str, Path]:
    """Where the bundle's two rendered figures go.

    Returned rather than written, because rendering a figure is
    ``panels.save``'s job and only ``panels.save``'s job — there is exactly one
    ReproFig render call in this package and it is not in here. The SVG is the primary
    one: the bundle is what somebody opens in two years to change a label, and
    a raster figure cannot be edited. ``preview.png`` exists so they do not
    have to open the SVG to see what it is.
    """
    root = Path(root)
    (root / "fig").mkdir(parents=True, exist_ok=True)
    return {"svg": root / "fig" / f"{slug}.svg",
            "preview": root / "fig" / "preview.png"}


def write_bundle(root: Path, *, slug: str,
                 table: Mapping[str, Sequence[Any]],
                 sources: Sequence[Mapping[str, Any]], claim: str,
                 artefacts: Iterable[Any], settings: Mapping[str, Any],
                 notes: Sequence[str] = (), table_csv: bytes | None = None) -> Path:
    """Everything in the layout except the two rendered figures."""
    root = Path(root)
    (root / "data" / "der").mkdir(parents=True, exist_ok=True)

    table_path = root / "data" / "der" / "figure_data.csv"
    if table_csv is None:
        write_table(table_path, table)
    else:
        table_path.write_bytes(table_csv)
    rows = _copy_sources(root, sources)
    _write_sources(root, rows)
    _readme(root, slug=slug, claim=claim, rows=rows, settings=settings,
            artefacts=artefacts, notes=notes)
    return root
