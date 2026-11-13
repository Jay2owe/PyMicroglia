"""Give final display outputs gap-free cell numbers, 1 to N.

Motion's identity numbers have gaps once identities merge or are discarded.
This module maps the complete final identity set, in ascending order, to
1..N and writes renumbered *copies* of label stacks and an added
``cell_number`` column for final tables. Motion's own files, and every
calculation that joins on native identities, are never changed here.
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import tifffile

from .tracking.contract import sha256_of

__all__ = ["MAP_COLUMNS", "numbering", "save_map", "renumber_labels",
           "add_cell_numbers"]

MAP_COLUMNS = ("motion_identity", "cell_number")


def _frames(labels) -> tuple[tuple[int, ...], np.dtype, Iterator[np.ndarray]]:
    """Shape, dtype and a frame-by-frame reader for a label array or TIFF."""
    if not isinstance(labels, (str, os.PathLike)):
        array = np.asarray(labels)
        frames = (array,) if array.ndim <= 2 else array.reshape(
            (-1,) + array.shape[-2:])
        return array.shape, array.dtype, iter(frames)
    path = Path(labels)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape, dtype, pages = tuple(series.shape), np.dtype(series.dtype), len(series.pages)
    if len(shape) <= 2 or pages != int(np.prod(shape[:-2])):
        array = np.asarray(tifffile.imread(path))
        frames = (array,) if array.ndim <= 2 else array.reshape(
            (-1,) + array.shape[-2:])
        return array.shape, array.dtype, iter(frames)

    def read() -> Iterator[np.ndarray]:
        with tifffile.TiffFile(path) as tif:
            for index in range(pages):
                yield np.asarray(tif.asarray(key=index, series=0))
    return shape, dtype, read()


def numbering(labels) -> dict[int, int]:
    """Ascending non-zero identities of a complete label stack -> 1..N."""
    _shape, _dtype, frames = _frames(labels)
    present: set[int] = set()
    for frame in frames:
        present.update(int(value) for value in np.unique(frame) if value != 0)
    return {identity: number
            for number, identity in enumerate(sorted(present), start=1)}


def save_map(mapping: Mapping[int, int], path) -> Path:
    """Write the map as ``motion_identity,cell_number``, one row per cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MAP_COLUMNS)
        for identity, number in sorted(mapping.items()):
            writer.writerow((int(identity), int(number)))
    return path


def _lookup(mapping: Mapping[int, int]) -> np.ndarray:
    top = max((int(key) for key in mapping), default=0)
    table = np.full(top + 1, -1, dtype=np.int64)
    table[0] = 0
    for identity, number in mapping.items():
        if int(identity) <= 0:
            raise ValueError("motion identities in the map must be positive")
        table[int(identity)] = int(number)
    return table


def _renumbered(frame: np.ndarray, table: np.ndarray, dtype: np.dtype,
                source: Path) -> np.ndarray:
    values = np.asarray(frame)
    if values.size and (values.min() < 0 or values.max() >= len(table)):
        bad = sorted(set(np.unique(values[(values < 0) | (values >= len(table))])
                         .tolist()))
        raise ValueError(f"{source} holds identities absent from the map: "
                         + ", ".join(map(str, bad[:10])))
    out = table[values]
    if (out < 0).any():
        bad = sorted(set(np.unique(values[out < 0]).tolist()))
        raise ValueError(f"{source} holds identities absent from the map: "
                         + ", ".join(map(str, bad[:10])))
    return out.astype(dtype, copy=False)


def renumber_labels(source, mapping: Mapping[int, int],
                    destination) -> tuple[Path, str]:
    """Write a renumbered copy of a label TIFF; return its path and sha256.

    Background stays 0 and the source dtype is kept. A source identity that
    the map does not contain is refused rather than written as background.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("renumbered labels must be written to a new file")
    table = _lookup(mapping)
    shape, dtype, frames = _frames(source)
    if np.dtype(dtype).kind not in "iu":
        raise ValueError(f"{source} is not an integer label stack")
    if mapping and max(int(v) for v in mapping.values()) > np.iinfo(dtype).max:
        raise ValueError("the cell numbers do not fit the source label dtype")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        if len(shape) <= 2:
            data = _renumbered(next(frames), table, dtype, source)
            tifffile.imwrite(partial, data, compression="zlib",
                             photometric="minisblack")
        else:
            tifffile.imwrite(
                partial, (_renumbered(frame, table, dtype, source)
                          for frame in frames),
                shape=shape, dtype=dtype, compression="zlib",
                photometric="minisblack")
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination, sha256_of(destination)


def _number_for(value: str, mapping: Mapping[int, int]) -> str:
    text = value.strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return ""
    if not np.isfinite(number) or number != int(number) or int(number) == 0:
        return ""
    found = mapping.get(int(number))
    return "" if found is None else str(int(found))


def add_cell_numbers(table_path, mapping: Mapping[int, int], *,
                     column: str = "identity") -> dict[str, Any]:
    """Insert ``cell_number`` right after ``column`` in a final CSV, in place.

    Every existing value is written back as read; rows whose identity is
    empty, 0, not a number or absent from the map get an empty cell number.
    A table without ``column`` is left untouched and reported as such.
    """
    path = Path(table_path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text.split("\n", 1)[0] + "\n" else "\n"
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or column not in rows[0]:
        return {"path": str(path), "added": False,
                "reason": f"no {column} column"}
    if "cell_number" in rows[0]:
        raise ValueError(f"{path} already has a cell_number column")
    where = rows[0].index(column)
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator=newline)
    writer.writerow(rows[0][:where + 1] + ["cell_number"] + rows[0][where + 1:])
    numbered = 0
    for row in rows[1:]:
        if not row:
            writer.writerow(row)
            continue
        value = _number_for(row[where], mapping) if where < len(row) else ""
        numbered += bool(value)
        writer.writerow(row[:where + 1] + [value] + row[where + 1:])
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(out.getvalue().encode("utf-8"))
    os.replace(partial, path)
    return {"path": str(path), "added": True, "rows": len(rows) - 1,
            "numbered_rows": numbered, "sha256": sha256_of(path)}
