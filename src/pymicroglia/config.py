"""Where things live, and how much room they get.

The rebuildable pixel tier lives **in the project folder**, beside the analysis
it belongs to, and it has a cap so it cannot quietly fill the only drive the
machine has.

That is a reversal. Until 2026-08-20 tier B was forced onto the local disk and
a synced path was refused outright, on the grounds that syncing tens of
gigabytes of rebuildable pixels is a waste. The waste is real; so is the other
side of it, which is that the pixels take hours to produce and a machine that
has never seen them pays those hours again. Keeping them beside the project
makes a second machine — and a rebuilt one — a cache hit.

**The thing to check on a synced store is dehydration, not location.** Dropbox
frees space by turning a file into an online-only placeholder that looks
exactly like a file until something reads it. Memory-mapping a 21 GB
placeholder faults it back down the network a page at a time, which is far
slower than rebuilding it. :func:`is_placeholder` is how that is spotted, and
``doctor`` reports it; the fix is to mark the store folder "Make available
offline" in Dropbox.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["store_root", "index_root", "project_root", "cache_cap_bytes",
           "free_bytes", "inside_dropbox", "is_placeholder", "STORE_DIRNAME"]

#: 64 GiB. One registered float32 copy of a 10.8 GB three-channel uint16 stack
#: is roughly 21 GB plus a cleaned copy at about 7 GB, so this admits two or
#: three working stacks. Override with PYMICROGLIA_CACHE_GB.
DEFAULT_CACHE_GB = 64

#: The folder the store gets inside the project. Named for what it holds rather
#: than for what it is, because the person who finds 20 GB in their Dropbox
#: needs to know it is pixels and that deleting it costs time and nothing else.
STORE_DIRNAME = "PixelStore"

#: The project folder this package ships inside. Found by name rather than by a
#: fixed number of ``..`` steps, so moving the checkout one level does not
#: silently relocate the store.
PROJECT_FOLDER_NAME = "Microglia Project"


def project_root() -> Path | None:
    """The project folder this package is installed inside, if it is."""
    for parent in Path(__file__).resolve().parents:
        if parent.name == PROJECT_FOLDER_NAME:
            return parent
    return None


def store_root() -> Path:
    """Where the rebuildable pixel tier lives.

    The project folder, so the expensive arrays travel with the analysis and a
    second machine finds them rather than re-deriving them. ``PYMICROGLIA_STORE``
    overrides it — for a scratch disk, for a machine where the project is not
    synced, or to point a test somewhere disposable.

    Falls back to the local cache directory when the package is installed
    outside the project, which is the only case where there is no project
    folder to put it in.
    """
    override = os.environ.get("PYMICROGLIA_STORE")
    if override:
        return Path(override)
    project = project_root()
    if project is not None:
        return project / STORE_DIRNAME
    return local_cache()


def local_cache() -> Path:
    """This machine's own cache directory. Never the project folder."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "pymicroglia" / "cache"
    return Path.home() / ".cache" / "pymicroglia"


def index_root() -> Path:
    """Where the tier-A lookup index lives, which is **not** where the pixels do.

    The index is a few hundred kilobytes rewritten after every run. That is the
    one shape of file a sync client handles worst: constant churn, and two
    machines writing it produce a "conflicted copy" and two silently divergent
    indexes rather than an error.

    So it follows the store while the store is local — which keeps a test's
    whole world inside its temporary folder — and stays on this machine as soon
    as the store is synced. ``PYMICROGLIA_INDEX`` overrides it. Nothing is lost
    if it is deleted: every entry names the sidecar it came from and ``scan``
    rebuilds it.
    """
    override = os.environ.get("PYMICROGLIA_INDEX")
    if override:
        return Path(override)
    store = store_root()
    return store if not inside_dropbox(store) else local_cache()


def cache_cap_bytes() -> int:
    """How large the rebuildable tier may grow before it evicts."""
    raw = os.environ.get("PYMICROGLIA_CACHE_GB")
    try:
        gigabytes = float(raw) if raw else DEFAULT_CACHE_GB
    except ValueError:
        gigabytes = DEFAULT_CACHE_GB
    return int(max(1.0, gigabytes) * 1024 ** 3)


def free_bytes(path: Path | None = None) -> int:
    """Free space on the volume holding ``path``, without walking anything."""
    target = Path(path) if path is not None else store_root()
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        return int(shutil.disk_usage(target).free)
    except OSError:
        return 0


def inside_dropbox(path: Path) -> bool:
    """Whether a path sits under a synced folder.

    No longer a refusal — the store lives in one on purpose. Kept because
    ``doctor`` says so out loud, and because whether dehydration is possible at
    all depends on the answer. Cheap and textual on purpose: asking the Dropbox
    client would mean depending on it being installed.
    """
    return "dropbox" in str(Path(path).resolve()).replace("\\", "/").lower()


#: Windows file attributes that mean "this file's contents are not here".
#: ``OFFLINE`` is the old tape-archive flag Dropbox and OneDrive reuse;
#: ``RECALL_ON_DATA_ACCESS`` is the modern placeholder bit, set on a file whose
#: bytes are fetched only when something reads them.
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
_NOT_HERE = (FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_OPEN
             | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS)


def is_placeholder(path: Path) -> bool:
    """Whether the file exists in name only, with its bytes still in the cloud.

    A dehydrated array is the one failure mode a synced store has that a local
    one does not: it is a hit by every test the store can make, and reading it
    is a download. Returns ``False`` on anything that is not Windows or cannot
    be interrogated — the check exists to catch a known state, not to refuse
    what it cannot read.
    """
    try:
        attributes = os.stat(path).st_file_attributes  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        return False
    return bool(attributes & _NOT_HERE)
