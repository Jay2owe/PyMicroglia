"""The pipelines people actually run, and the shape they all share.

One module per pipeline, not one module for all of them. PyFLASH puts six
pipelines in a single 7,728-line file and keeps their helpers apart with a
``_corr_``, ``_adj_``, ``_ovw_`` prefix on every private function. Those
prefixes exist because the file boundary does not. Here the file boundary does
the work, so a helper in ``dluc_single_cell.py`` needs no prefix to say which
pipeline it belongs to — its filename already did.

Two of the modules here are not pipelines and are named for what they are.
``registered`` is the front half every bioluminescence run shares: channels,
window, registration, cosmic-ray removal. ``objects`` is the next stretch:
tissue mask, background, masks, admissibility, traces. Both were carved out
when ``dluc_single_cell`` crossed the 800-line cap, which is what that cap is
for — and both turned out to be the pieces the parity tests actually wanted to
call.

What is shared by everything lives in this file: how a run folder is named and
what happens when one already exists, the order the scientific stages must run
in, and the record of which stages a run actually performed.

**The stage order is not ours to change.** ``AGENTS.md`` fixes it::

    VSI conversion -> registration -> cosmic-ray removal -> unsmoothed measurement
                                                       \\-> display-only smoothing

Registration before cosmic-ray removal, always: on an unregistered stack the
neighbouring frames show different tissue and real motion is removed as if it
were a spike. Display smoothing hangs off the measurement branch and never
feeds it. :class:`StageLog` records what a run did in the order it did it, and
:func:`check_stage_order` refuses a run that got it wrong — a runtime check
rather than a comment, so a future edit cannot quietly reorder the calls.

The ``if_exists`` policy is PyFLASH's, adopted whole, because "I re-ran it and
it silently replaced the figures I was comparing against" is a mistake that
only has to happen once.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "PIPELINE_NAMES",
    "IF_EXISTS",
    "CANONICAL_ORDER",
    "MEASUREMENT_STAGE",
    "DISPLAY_BRANCH",
    "RunFolder",
    "StageOrderError",
    "StageLog",
    "PipelineResult",
    "StackView",
    "available",
    "get",
    "describe",
    "slug",
    "run_folder",
    "append_runs_index",
    "check_stage_order",
    "manifest_path",
    "read_manifest",
    "write_manifest",
]

#: Every pipeline this package ships. Four is the whole list; a fifth idea gets
#: written down for later rather than added here. ``registered`` and ``objects``
#: are shared stretches of a pipeline rather than pipelines, so they are absent:
#: listing one would offer an agent a run it cannot start.
PIPELINE_NAMES: tuple[str, ...] = (
    "dluc_single_cell",
    "cry1_dluc_photon",
    "bioluminescence",
    "phase_green_red",
)

#: What to do about a run folder that already exists.
IF_EXISTS: tuple[str, ...] = ("overwrite", "version", "error", "skip")

#: ``AGENTS.md``'s order for microglial bioluminescence, as stage names.
CANONICAL_ORDER: tuple[str, ...] = ("convert", "register", "cosmic_rays",
                                    "measure")
MEASUREMENT_STAGE = "measure"
#: Display smoothing is a *branch* off the measurement, never a step before it.
DISPLAY_BRANCH = "display"

MANIFEST_NAME = "manifest.json"
RUNS_INDEX_NAME = "_runs_index.csv"


# --------------------------------------------------------------- discovery
def available() -> dict[str, Any]:
    """The pipelines that import cleanly, by name.

    A pipeline whose optional dependencies are missing is absent from this
    mapping rather than present and broken, which is the same contract the
    action registry uses for a stage that has not landed.
    """
    found: dict[str, Any] = {}
    for name in PIPELINE_NAMES:
        try:
            found[name] = importlib.import_module(f"pymicroglia.pipelines.{name}")
        except ImportError:
            continue
    return found


def get(name: str):
    """One pipeline module by name."""
    if name not in PIPELINE_NAMES:
        raise KeyError(f"unknown pipeline {name!r}. Known: "
                       f"{', '.join(PIPELINE_NAMES)}")
    return importlib.import_module(f"pymicroglia.pipelines.{name}")


def describe() -> list[dict[str, Any]]:
    """One row per pipeline: what it is for, and the stages it runs."""
    rows = []
    for name in PIPELINE_NAMES:
        try:
            module = get(name)
        except ImportError as exc:
            rows.append({"name": name, "available": False, "why": str(exc)})
            continue
        rows.append({
            "name": name,
            "available": True,
            "summary": (module.__doc__ or "").strip().split("\n")[0],
            "stages": list(getattr(module, "STAGES", ())),
            "method_version": getattr(module, "METHOD_VERSION", ""),
        })
    return rows


# ------------------------------------------------------- arrays as a series
class StackView:
    """Registered arrays already in memory, wearing a :class:`~series.Series`'s face.

    Every measurement function in this package takes something with ``.shape``
    and ``.frame(t, c)``, because the thing it usually gets is a file being read
    a page at a time. A pipeline has already read that file and holds the
    registered result, and handing it back a path would mean writing 1.5 GB out
    and reading it in again to get at pixels it is holding.

    Deliberately not a subclass of ``Series``: it has no file, no metadata and
    no source identity, and inheriting would promise all three. It is the small
    protocol those functions actually use, and nothing else.
    """

    def __init__(self, channels: Sequence[Any], *, source=None, name: str = ""):
        if not len(channels):
            raise ValueError("a StackView needs at least one channel array")
        self.channels = list(channels)
        first = self.channels[0]
        self._shape = (int(first.shape[0]), len(self.channels),
                       int(first.shape[1]), int(first.shape[2]))
        self._source = source
        self.name = name

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self._shape

    @property
    def source(self):
        if self._source is None:
            raise AttributeError(
                "this StackView was built from arrays and has no source "
                "identity. Pass source= if something downstream needs to key "
                "an artefact on the recording these pixels came from.")
        return self._source

    @property
    def display_only(self) -> bool:
        """Never. A registered measurement array is the measurement branch.

        Present because ``guards.require_measurement`` asks, and answering
        honestly is the point: pixels that went through the display filter
        never reach a StackView, because the pipelines put that branch after
        the measurement and never feed it back.
        """
        return False

    def frame(self, t: int, c: int = 0):
        return self.channels[int(c)][int(t)]

    def window(self, t0: int, t1: int, c: int = 0):
        for t in range(int(t0), int(t1)):
            yield self.frame(t, c)

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        frames, channels, height, width = self._shape
        return (f"StackView(T={frames}, C={channels}, {height}x{width}"
                + (f", {self.name}" if self.name else "") + ")")


# ------------------------------------------------------------- stage order
class StageOrderError(RuntimeError):
    """A run performed the scientific stages in an order ``AGENTS.md`` forbids."""


def check_stage_order(names: Sequence[str]) -> None:
    """Refuse an order that puts cosmic-ray removal before registration.

    Only the constraint that matters is enforced, not the full sequence: a
    pipeline may legitimately skip conversion, or measure without cleaning.
    What it may never do is clean before it registers, or measure from the
    display branch.
    """
    order = [str(name) for name in names]
    if "cosmic_rays" in order and "register" in order:
        if order.index("cosmic_rays") < order.index("register"):
            raise StageOrderError(
                "cosmic-ray removal ran before registration. On an "
                "unregistered stack the neighbouring frames show different "
                "tissue, so real motion is removed as if it were a spike. "
                "AGENTS.md fixes this order and it is not ours to change.")
    if DISPLAY_BRANCH in order and MEASUREMENT_STAGE in order:
        if order.index(DISPLAY_BRANCH) < order.index(MEASUREMENT_STAGE):
            raise StageOrderError(
                "display smoothing ran before the measurement. Display "
                "filtering is a branch off the measured data, never a step "
                "on the way to it; a number computed after it is a number "
                "computed from pixels nobody measured.")


@dataclass
class StageLog:
    """What a run did, in the order it did it.

    Every pipeline drives its stages through this, so the order is a fact
    recorded at run time rather than a claim made in a docstring. The test for
    gate 7 reads :attr:`names` from a real run.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [entry["stage"] for entry in self.entries]

    @contextmanager
    def __call__(self, stage: str, **detail: Any):
        import time

        started = time.time()
        entry: dict[str, Any] = {"stage": str(stage), **detail}
        self.entries.append(entry)
        check_stage_order(self.names)
        try:
            yield entry
        finally:
            entry["seconds"] = round(time.time() - started, 3)

    def add(self, stage: str, **detail: Any) -> dict[str, Any]:
        """Record a stage that took no measurable time — a skip, or a cache hit."""
        entry = {"stage": str(stage), "seconds": 0.0, **detail}
        self.entries.append(entry)
        check_stage_order(self.names)
        return entry

    def as_records(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.entries]


# -------------------------------------------------------------- run folders
@dataclass(frozen=True)
class RunFolder:
    """Where one run writes, and whether it should write at all."""

    path: Path
    label: str
    reuse: bool = False        # True only under if_exists="skip"

    def __fspath__(self) -> str:
        return str(self.path)

    def __truediv__(self, other) -> Path:
        return self.path / other


def slug(prefix: str, payload: Mapping[str, Any] | Sequence[Any]) -> str:
    """A deterministic ``"<prefix>_<6 hex>"`` run name from the settings.

    Identical settings hash to the same folder, so a repeat run finds its own
    previous output and the ``if_exists`` policy has something to act on. A
    different configuration hashes elsewhere and never collides. The payload is
    canonicalised first, so a dict written in a different key order is the same
    run rather than a new one.
    """
    digest = hashlib.sha1(
        json.dumps(_canonical(payload), sort_keys=True, default=str)
        .encode("utf-8")).hexdigest()[:6]
    return f"{prefix}_{digest}"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key])
                for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_folder(root, pipeline: str, label: str, if_exists: str = "version"
               ) -> RunFolder:
    """``<root>/<pipeline>/<label>``, resolved against the ``if_exists`` policy.

    ``overwrite`` clears the folder, ``version`` picks the next free ``_vN`` so
    the previous output survives, ``error`` refuses, and ``skip`` hands back the
    existing folder with ``reuse=True`` so the caller can return the stored
    manifest without recomputing.

    The default is ``version`` on purpose. Overwriting is the policy that costs
    a person the comparison they were in the middle of making, and a default
    should not be able to do that.
    """
    policy = str(if_exists).strip().lower()
    if policy not in IF_EXISTS:
        raise ValueError(
            f"if_exists must be one of {', '.join(IF_EXISTS)}; "
            f"got {if_exists!r}")
    base = Path(root) / str(pipeline)
    safe = _folder_name(label)
    target = base / safe

    if not target.is_dir():
        return RunFolder(path=target, label=safe)
    if policy == "skip":
        return RunFolder(path=target, label=safe, reuse=True)
    if policy == "error":
        raise FileExistsError(
            f"{pipeline} run {safe!r} already exists at {target}. Pass "
            f"if_exists='version' to keep both, 'overwrite' to replace it, or "
            f"'skip' to return what is already there.")
    if policy == "overwrite":
        shutil.rmtree(target)
        return RunFolder(path=target, label=safe)

    version = 2
    while True:
        candidate = base / f"{safe}_v{version}"
        if not candidate.is_dir():
            return RunFolder(path=candidate, label=candidate.name)
        version += 1


def _folder_name(label: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._- " else "_"
        for character in str(label)).strip().rstrip(".")
    if not cleaned:
        raise ValueError("run label must resolve to a non-empty folder name")
    return cleaned


# ---------------------------------------------------------------- manifests
def manifest_path(folder) -> Path:
    return Path(getattr(folder, "path", folder)) / MANIFEST_NAME


def write_manifest(folder, payload: Mapping[str, Any]) -> Path:
    target = manifest_path(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, default=str),
                      encoding="utf-8")
    return target


def read_manifest(folder) -> dict[str, Any] | None:
    target = manifest_path(folder)
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def append_runs_index(root, pipeline: str, row: Mapping[str, Any], *,
                      key: str = "run_label") -> Path | None:
    """Append or replace one summary row in ``<pipeline>/_runs_index.csv``.

    Bookkeeping, so it never breaks a run: a failure here is swallowed and the
    analysis stands. PyFLASH takes the same view for the same reason.
    """
    import csv

    base = Path(root) / str(pipeline)
    target = base / RUNS_INDEX_NAME
    try:
        base.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        if target.is_file():
            with open(target, newline="", encoding="utf-8") as handle:
                rows = [item for item in csv.DictReader(handle)
                        if str(item.get(key)) != str(row.get(key))]
        rows.append({k: _flat(v) for k, v in row.items()})
        columns: list[str] = []
        for item in rows:
            for name in item:
                if name not in columns:
                    columns.append(name)
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for item in rows:
                writer.writerow({name: item.get(name, "") for name in columns})
        return target
    except Exception:
        return None


def _flat(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple, set)):
        return json.dumps(_canonical(value), default=str)
    if isinstance(value, Path):
        return str(value)
    return value


# ------------------------------------------------------------- the result
@dataclass
class PipelineResult:
    """What every pipeline returns: where it wrote, what it found, what to check."""

    pipeline: str
    source: str
    folder: Path
    label: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    review: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    reused: bool = False

    @property
    def blocked(self) -> bool:
        return any(item.get("severity") == "blocker" and not item.get("answered")
                   for item in self.review)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "source": str(self.source),
            "folder": str(self.folder),
            "run_label": self.label,
            "stages": list(self.stages),
            "outputs": dict(self.outputs),
            "summary": dict(self.summary),
            "review": list(self.review),
            "open_questions": list(self.open_questions),
            "reused": self.reused,
            "blocked": self.blocked,
        }


def default_output_root(source, *, folder: str = "AI_Exports") -> Path:
    """``AI_Exports`` beside the recording, the convention every wrapper uses.

    A source already inside an ``AI_Exports`` tree keeps that tree rather than
    growing a second one inside it.
    """
    source = Path(source).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name.lower() == folder.lower():
            return parent
    return source.parent / folder


def _iter_paths(values: Iterable[Any]) -> list[str]:
    return [str(value) for value in values if value is not None]
