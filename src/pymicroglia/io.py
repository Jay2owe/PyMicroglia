"""Reading and writing image files, on this machine's terms.

Three things about this project make plain ``tifffile.imwrite`` insufficient,
and all three are already solved in the loose engines — this is where those
solutions live now:

**Paths are long.** A results file two folders deep inside a synced experiment
tree routinely passes 260 characters, and Windows refuses it without the
extended-length prefix.

**Handles are held.** Dropbox and the antivirus both open a file briefly after
it is written, so a rename that would succeed a second later fails now. Every
replace and delete retries.

**A half-written TIFF is worse than no TIFF.** Writes go to a temporary file
beside the target and are renamed into place, so an interrupted export leaves
the previous good file, or nothing, and never a truncated one.

The compression defaults are the ones five engines already use — zlib at level
4. Changing them would silently alter the size of every file people have come
to recognise, which is not this package's decision to make.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FILE_LOCK_RETRY_SECONDS",
    "DEFAULT_COMPRESSION",
    "DEFAULT_COMPRESSION_LEVEL",
    "extended",
    "makedirs",
    "isfile",
    "isdir",
    "unlink_with_retry",
    "replace_with_retry",
    "open_tiff",
    "read_tiff",
    "memmap_tiff",
    "write_tiff",
    "read_csv",
    "write_csv",
    "partial_path",
]

#: How long a delete or rename keeps retrying a transient lock. Sixty seconds,
#: as in the engines: Dropbox and the antivirus release quickly, and a failure
#: after a minute is a real one.
FILE_LOCK_RETRY_SECONDS = 60.0

#: Deflate, level 4. ``DEFAULT_COMPRESSION_LEVEL = 4`` appears in five of the
#: protocol engines; this is that value, not a new opinion.
DEFAULT_COMPRESSION = "zlib"
DEFAULT_COMPRESSION_LEVEL = 4


# ------------------------------------------------------------------- paths
def extended(path) -> str:
    """A Windows extended-length path; a no-op everywhere else.

    Twelve lines copied from ``PyFLASH/pipeline_io.py``
    ``windows_extended_path``. Copied rather than imported: PyFLASH is a
    sibling analysis package and this one depends on no such thing.
    """
    if os.name != "nt":
        return str(path)
    absolute = os.path.abspath(str(path))
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute.lstrip("\\")
    return "\\\\?\\" + absolute


def makedirs(path) -> None:
    if path:
        os.makedirs(extended(path), exist_ok=True)


def isfile(path) -> bool:
    return os.path.isfile(extended(path))


def isdir(path) -> bool:
    return os.path.isdir(extended(path))


def partial_path(path) -> Path:
    """The temporary name a write goes to first, beside its target.

    Beside, not in a temp folder: ``os.replace`` is only atomic within one
    volume, and the results folder and ``%TEMP%`` are frequently not.
    """
    target = Path(path)
    return target.with_name(f".{target.stem}.partial{target.suffix}")


# ------------------------------------------------------------ locked files
def unlink_with_retry(path, timeout_seconds: float = FILE_LOCK_RETRY_SECONDS) -> None:
    """Remove one file despite transient Dropbox or antivirus locks."""
    target = Path(path)
    deadline = time.monotonic() + timeout_seconds
    delay = 0.1
    while True:
        try:
            os.unlink(extended(target))
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 1.7, 2.0)


def replace_with_retry(source, destination,
                       timeout_seconds: float = FILE_LOCK_RETRY_SECONDS) -> None:
    """Atomically replace a file, retrying transient locks."""
    deadline = time.monotonic() + timeout_seconds
    delay = 0.1
    while True:
        try:
            os.replace(extended(source), extended(destination))
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 1.7, 2.0)


# ------------------------------------------------------------------- TIFFs
def open_tiff(path):
    """A ``TiffFile`` on a possibly very long path.

    Opening reads the page directory, not the pixels. On a source that Dropbox
    is holding as an online-only placeholder even that hydrates the file, so do
    not open a folder's worth of stacks to look at their metadata.
    """
    import tifffile

    return tifffile.TiffFile(extended(path))


def read_tiff(path, *, series: int = 0, key=None):
    """Read a TIFF into memory.

    Deliberately unlike ``PyFLASH/image_io.py``, which squeezes what it reads
    down to a two-dimensional image and, handed a T/C/Y/X stack, returns the
    first frame. That is right for single-field immunofluorescence and would
    silently drop the time axis here. The backend-order idea is worth copying;
    the normalisation is not.
    """
    with open_tiff(path) as handle:
        target = handle.series[series]
        return target.asarray(key=key)


def memmap_tiff(path):
    """A memory map of an uncompressed TIFF, or a memory-mapped read of one
    that is compressed. Same fallback the engines use in ``open_tcyx``."""
    import tifffile

    try:
        return tifffile.memmap(extended(path))
    except ValueError:
        return tifffile.imread(extended(path), out="memmap")


def write_tiff(array, path, *, compression: str | None = DEFAULT_COMPRESSION,
               level: int | None = DEFAULT_COMPRESSION_LEVEL,
               imagej: bool = False, metadata: Mapping[str, Any] | None = None,
               description: str | None = None, overwrite: bool = False,
               photometric: str | None = "minisblack", **kwargs) -> Path:
    """Write a TIFF atomically. Returns the path written.

    The target only ever appears complete. If the write fails, the temporary
    file is removed and whatever was already at the target is untouched.
    """
    import tifffile

    target = Path(path)
    if isfile(target) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {target}. Pass overwrite=True if replacing "
            "it is what you meant: the default protects the figures somebody "
            "may be comparing against.")
    makedirs(target.parent)
    temporary = partial_path(target)
    if isfile(temporary):
        unlink_with_retry(temporary)

    options: dict[str, Any] = dict(kwargs)
    if compression:
        # tifffile changed this signature: ``compression`` names the codec and
        # ``compressionargs`` carries its level. Passing a tuple raises on
        # 2023 and later, which is the same fix
        # ``microglia_static_background_removal.py`` already carries.
        options["compression"] = compression
        if level is not None:
            options["compressionargs"] = {"level": int(level)}
    if imagej:
        options["imagej"] = True
        if metadata is not None:
            options["metadata"] = metadata
    else:
        options["metadata"] = metadata          # None suppresses tifffile's own
        if description is not None:
            options["description"] = description
        if photometric is not None:
            options["photometric"] = photometric

    try:
        tifffile.imwrite(extended(temporary), array, **options)
        replace_with_retry(temporary, target)
    except BaseException:
        if isfile(temporary):
            try:
                unlink_with_retry(temporary)
            except OSError:
                pass
        raise
    return target


# -------------------------------------------------------------------- CSVs
def read_csv(path) -> list[dict[str, str]]:
    """Rows as dictionaries of strings.

    ``utf-8-sig`` because Excel writes a byte-order mark and the first column
    name otherwise arrives with an invisible character glued to it — which the
    engines learned the hard way.
    """
    with open(extended(path), "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows: Sequence[Mapping[str, Any]], *,
              fieldnames: Iterable[str] | None = None) -> Path:
    """Write rows to a CSV atomically, taking the header from the first row."""
    target = Path(path)
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError("cannot write an empty CSV without fieldnames")
    makedirs(target.parent)
    temporary = partial_path(target)
    try:
        with open(extended(temporary), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fieldnames or rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        replace_with_retry(temporary, target)
    except BaseException:
        if isfile(temporary):
            try:
                unlink_with_retry(temporary)
            except OSError:
                pass
        raise
    return target
