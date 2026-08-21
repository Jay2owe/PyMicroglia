"""Everything one recording produced, in one line and without opening it.

    from pymicroglia import Recording

    rec = Recording("MCG_04 - 1 - 595_traces")
    rec.traces["object_1"]          # the trace
    rec.cells                       # one row per cell
    rec.rhythm["period_h"]          # what the periodogram found
    rec.cosmic_rays.sum()           # pixel-frames replaced

**No pixels are read.** Every result a run produced is already an ordinary file
beside the outputs with a sidecar naming what produced it, so this reads those
sidecars and loads a value only when you touch it. Pointed at a results folder
it never touches the recording at all — which is the point, because the
recording is ten gigabytes and probably online-only.

**Nothing is saved and nothing can go stale.** This is a view over what is on
disk, rebuilt each time it is opened, which takes a few hundred milliseconds
because the artefacts are kilobytes. There is no object to serialise, no
version to migrate and no chance of holding numbers that were computed under
settings you have since changed — the alternative, an object pickled after a
run, will happily hand you the old numbers afterwards and never mention it.

**A new analysis needs no change here.** Store one under any stage name and it
appears as ``rec.<stage name>`` next time. The friendly names below are only
shorthand for the dozen that already existed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from . import store
from .store import tier_a

__all__ = ["Recording", "Batch", "read_conditions",
           "ALIASES", "EXPORTS", "CONDITION_FILES",
           "RECORDING_COLUMN"]

#: Readable name -> the stage it was stored under. Shorthand only: every stage
#: is always reachable by its own name, and one an alias has never heard of is
#: reachable the same way. ``rec.cells`` and ``rec.segmentation_objects`` are
#: the same table.
ALIASES: dict[str, str] = {
    "shifts": "registration",
    "cosmic_events": "cosmic_rays_events",
    "cosmic_bleed": "cosmic_rays_bleed",
    "cosmic_summary": "cosmic_rays_summary",
    "background": "off_tissue_background",
    "labels": "segmentation",
    "cells": "segmentation_objects",
    "trace_objects": "traces_objects",
    "decoys": "decoy_test",
    "control": "instrumental_control",
}

#: Where the pipelines write, relative to a recording. Used only when a
#: ``Recording`` is built from a source file and has to find its own results.
EXPORTS = "AI_Exports"


def _stage_of(name: str) -> str:
    return ALIASES.get(name, name)


def _kind_of(value) -> str:
    """Which of the five stored kinds this value is, from the value itself.

    A guess, and a narrow one: a table is columns of lists, scalars are plain
    numbers and strings, a boolean array is a mask and an integer one is a
    label image. Anything it gets wrong is corrected by passing ``kind=``, and
    anything it cannot place is stored as an array, which loses nothing.
    """
    if isinstance(value, dict):
        if value and all(isinstance(column, (list, tuple))
                         for column in value.values()):
            return "table"
        return "scalars"
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return "scalars"
    import numpy as np

    if dtype == bool:
        return "mask"
    if np.issubdtype(dtype, np.integer) and int(getattr(value, "ndim", 0)) == 2:
        return "labels"
    return "array"


class Recording:
    """Every stored result for one recording, loaded when you ask for it.

    Build it from a **results folder** — nothing is read but its sidecars, and
    the recording itself is never opened:

        rec = Recording(r"...\\AI_Exports\\MCG_04_traces")

    or from the **recording**, which finds the folders beside it under
    ``AI_Exports`` and indexes them. That costs one 24 MB read of the recording
    to confirm its identity, and nothing after the first time.

        rec = Recording(r"...\\MCG_04 - 1 - 595.tif")

    ``list(rec)`` says what is actually there, ``rec.about(name)`` says under
    what settings it was produced, and anything a new analysis stores turns up
    by its own stage name without this module knowing about it.
    """

    def __init__(self, path, *, scan: bool = True, display_only: bool = False,
                 records=None, name: str = ""):
        paths = ([Path(one) for one in path]
                 if isinstance(path, (list, tuple)) else [Path(path)])
        self._path = paths[0]
        self._paths = paths
        self._name = name
        self._display_only = bool(display_only)
        self._loaded: dict[str, Any] = {}
        self._entries: dict[str, list[dict[str, Any]]] = {}
        self._source = None
        self._folders: list[Path] = []
        self._gather(scan=scan, records=records)

    # ------------------------------------------------------------- building
    def _gather(self, *, scan: bool, records=None) -> None:
        if records is not None:
            self._folders = [one for one in self._paths if one.is_dir()]
        elif all(one.is_dir() for one in self._paths):
            self._folders = list(self._paths)
            records = [record for folder in self._folders
                       for record in tier_a.sidecars_in(folder)]
        else:
            self._folders = self._result_folders()
            if scan:
                for folder in self._folders:
                    store.scan(folder)
            self._source = store.identify(self._path)
            records = store.manifest.find(source=self._source,
                                          display_only=self._display_only)
            self._name = self._name or self._path.name
        for record in records:
            if bool(record.get("display_only")) != self._display_only:
                continue
            if str(record.get("stage", "")).startswith("_"):
                continue        # decisions ride the same rails; see .decisions
            self._entries.setdefault(str(record["stage"]), []).append(record)
        for entries in self._entries.values():
            entries.sort(key=lambda record: str(record.get("created", "")))
        if self._source is None and records:
            raw = records[0].get("source")
            self._source = (store.SourceId.from_dict(raw)
                            if isinstance(raw, dict) else None)
        if not self._name:
            self._name = (self._source.name if self._source is not None
                          else self._path.name)

    def _result_folders(self) -> list[Path]:
        """The folders a run would have written beside this recording.

        One level, never a walk: the sources live in Dropbox behind online-only
        placeholders and a recursive search there hydrates files nobody asked
        for. Every action in this package writes to
        ``<source>/../AI_Exports/<stem>_<something>``, so listing that one
        folder finds them all.
        """
        exports = self._path.parent / EXPORTS
        if not exports.is_dir():
            return []
        stem = self._path.stem
        return sorted(folder for folder in exports.iterdir()
                      if folder.is_dir() and folder.name.startswith(stem))

    # ------------------------------------------------------------- reading
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"{self._describe()} has no {name!r}. It holds "
                f"{sorted(self._entries) or 'nothing yet'}. If the run wrote it "
                f"somewhere else, point Recording at that folder."
            ) from exc

    def __getitem__(self, name: str) -> Any:
        stage = _stage_of(name)
        if stage not in self._entries:
            raise KeyError(name)
        if stage not in self._loaded:
            self._loaded[stage] = self._stored(stage).load()
        return self._loaded[stage]

    def _stored(self, name: str) -> store.Stored:
        """The newest artefact for this stage, as a :class:`store.Stored`."""
        record = self._entries[_stage_of(name)][-1]
        return store.Stored(path=Path(record["path"]), record=dict(record))

    def all(self, name: str) -> list[store.Stored]:
        """Every artefact stored for this stage, oldest first.

        A stage has more than one when the same step ran under different
        settings. ``rec.traces`` gives the most recent; this gives all of them
        with the settings that separate them, which is what a comparison needs.
        """
        return [store.Stored(path=Path(record["path"]), record=dict(record))
                for record in self._entries.get(_stage_of(name), ())]

    def about(self, name: str) -> dict[str, Any]:
        """The key it was written under: settings, method version, when, where.

        The answer to "is this the run I think it is", without loading it.
        """
        stored = self._stored(name)
        record = stored.record
        return {"stage": stored.stage, "kind": stored.kind,
                "params": stored.params,
                "method_version": record.get("method_version", ""),
                "created": record.get("created", ""),
                "path": str(stored.path),
                "digest": stored.digest,
                "versions": len(self._entries[_stage_of(name)])}

    @property
    def decisions(self) -> dict[str, Any]:
        """What a person judged about this recording, keyed on the source alone.

        Kept out of the ordinary listing because a decision is not a result: it
        survives every parameter change and version bump, which is the whole
        reason it is stored separately.
        """
        if self._source is None:
            return {}
        return store.decisions_for(self._source)

    # -------------------------------------------------------------- looking
    @property
    def stages(self) -> list[str]:
        return sorted(self._entries)

    @property
    def source(self):
        return self._source

    @property
    def folders(self) -> list[Path]:
        return list(self._folders)

    def paths(self) -> dict[str, Path]:
        """Stage -> the file on disk, for handing to something that is not this."""
        return {stage: Path(entries[-1]["path"])
                for stage, entries in self._entries.items()}

    def __iter__(self) -> Iterator[str]:
        return iter(self.stages)

    def __contains__(self, name: object) -> bool:
        return _stage_of(str(name)) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __dir__(self) -> list[str]:
        """Stages and aliases, so an editor's completion lists what is there."""
        present = set(self._entries)
        aliases = {name for name, stage in ALIASES.items() if stage in present}
        return sorted({*super().__dir__(), *present, *aliases})

    @property
    def name(self) -> str:
        """What to call this recording in a table: the file's own name."""
        return self._name

    @property
    def identity(self) -> str:
        """What makes two folders the same recording. Content, never a path."""
        return self._source.identity if self._source is not None else ""

    def _describe(self) -> str:
        return f"Recording({self._name or self._path.name!r})"

    def __repr__(self) -> str:
        if not self._entries:
            return (f"<{self._describe()}: nothing stored — "
                    f"looked in {len(self._folders)} folder(s)>")
        return f"<{self._describe()}: {', '.join(self.stages)}>"


    # -------------------------------------------------------------- writing
    #: What ``add`` stamps a result with when the caller names no method. It is
    #: also how ``add`` tells its own results from a run's, which is what lets
    #: it replace one freely and refuse to clobber the other.
    BY_HAND = "by-hand"

    def add(self, name: str, value, *, params=None, method_version: str = "",
            folder=None, kind: str = "", replace: bool = False) -> store.Stored:
        """Store a result against this recording, and have it back next time.

            rec.add("branch_complexity", {"label": [1, 2], "branches": [7, 9]})
            rec.branch_complexity["branches"]        # now, and next session

        The counterpart of reading. Whatever you worked out in a notebook lands
        beside the run's own results with a sidecar naming it, so it is found
        the same way everything else is and by anything else that opens this
        recording.

        ``kind`` is worked out from the value and can be given if that guess is
        wrong. ``method_version`` defaults to ``by-hand``, which is the honest
        answer for something a person computed rather than an action: it says
        no version of any method produced this. Pass one as soon as the working
        becomes a function worth versioning.

        Adding under a name you added before **replaces** it, which is what
        re-running a cell in a notebook should do. Adding under a name a *run*
        produced is refused, because ``rec.add("traces", ...)`` would otherwise
        overwrite the traces the pipeline measured and the record that says how.
        Pass ``replace=True`` if that is genuinely what you mean.
        """
        if self._source is None:
            raise ValueError(
                "this Recording has nothing stored yet, so there is no "
                "recording to attach a result to. Open it from a folder that "
                "holds at least one artefact, or from the recording itself.")
        stage = _stage_of(name)
        existing = self._entries.get(stage, [])
        if existing and not replace:
            was = str(existing[-1].get("method_version", ""))
            if was != self.BY_HAND:
                raise ValueError(
                    f"{stage!r} was produced by a run at {was!r}, and adding "
                    f"over it would replace what the pipeline measured and the "
                    f"record of how. Choose another name, or pass replace=True.")
        target = Path(folder) if folder is not None else self._write_folder()
        stored = store.put(stage, self._source, dict(params or {}),
                           kind=kind or _kind_of(value), value=value,
                           name=stage, output_dir=target,
                           method_version=method_version or self.BY_HAND)
        stage = stored.stage
        self._entries.setdefault(stage, [])
        self._entries[stage] = [record for record in self._entries[stage]
                                if record.get("path") != str(stored.path)]
        self._entries[stage].append(dict(stored.record))
        self._loaded.pop(stage, None)
        if target not in self._folders:
            self._folders.append(target)
        return stored

    def _write_folder(self) -> Path:
        """Where a hand-added result goes: the folder this was opened from."""
        if self._path.is_dir():
            return self._path
        if self._folders:
            return self._folders[0]
        return self._path.parent / EXPORTS / f"{self._path.stem}_added"

# ---------------------------------------------------------------- many of them
#: What a conditions file is called if you do not name one. Either spelling; the
#: second is PyFLASH's, so a folder already set up for that pipeline is right.
CONDITION_FILES = ("conditions.csv", "Condition Labels.csv")

#: The column naming the recording. Matched case-insensitively, and any other
#: column becomes a factor — genotype, treatment, sex, timepoint, whatever the
#: design has.
RECORDING_COLUMN = "recording"


def read_conditions(path) -> dict[str, dict[str, str]]:
    """``recording -> {factor: level}`` from a small CSV you maintain by hand.

    PyFLASH derives the condition from the animal's name by stripping the
    digits off it — ``WT3`` becomes ``WT``. That works on names chosen for it
    and not on a microscope's: ``VID52_B6_phase-green-red_timestack`` strips to
    ``VIDBphasegreenredtimestack``. So the mapping is written down instead, in
    the folder the recordings sit in, where a person can read and correct it.

    The first column names the recording, with or without its extension. Every
    other column is a factor, so a crossed design is two columns and needs no
    syntax of its own.
    """
    import csv

    target = Path(path)
    if not target.is_file():
        return {}
    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    header = [name for name in rows[0] if name is not None]
    key = next((name for name in header
                if str(name).strip().lower() == RECORDING_COLUMN), header[0])
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get(key) or "").strip()
        if not name:
            continue
        factors = {str(column).strip(): str(value or "").strip()
                   for column, value in row.items()
                   if column is not None and column != key}
        out[name] = factors
        out.setdefault(Path(name).stem, factors)
    return out


class Batch:
    """Every recording in one folder, and every result each of them produced.

        from pymicroglia import Batch

        b = Batch(exports_folder)
        b.cells                 # every cell of every recording, one table
        b.summary               # one row per recording
        b.recordings[0].traces  # or drop back to one of them

    Recordings are grouped by **what the file contains**, not by folder name, so
    the three or four folders one recording writes come back as one recording.

    Conditions come from a CSV beside the recordings — one row per recording,
    one column per factor. They are joined onto every table as extra columns, so
    a grouped comparison is a filter rather than a lookup.

    Like :class:`Recording` this is a view. It reads sidecars, loads a table
    only when you touch it, opens no recording, and holds nothing to save.
    """

    def __init__(self, root, *, conditions=None, display_only: bool = False,
                 recordings=None):
        self._root = Path(root) if root is not None else None
        self._display_only = bool(display_only)
        self._cache: dict[str, dict[str, list]] = {}
        self.recordings: list[Recording] = (list(recordings) if recordings
                                            else self._discover())
        self.conditions = self._conditions(conditions)

    # ------------------------------------------------------------- building
    def _discover(self) -> list[Recording]:
        """One :class:`Recording` per source, from one level of subfolders."""
        if self._root is None or not self._root.is_dir():
            return []
        folders = [self._root, *(child for child in sorted(self._root.iterdir())
                                 if child.is_dir())]
        grouped: dict[str, list[dict[str, Any]]] = {}
        homes: dict[str, list[Path]] = {}
        for folder in folders:
            for record in tier_a.sidecars_in(folder):
                raw = record.get("source")
                if not isinstance(raw, dict):
                    continue
                key = f"{raw.get('size')}:{raw.get('sample')}"
                grouped.setdefault(key, []).append(record)
                if folder not in homes.setdefault(key, []):
                    homes[key].append(folder)
        found = [Recording(homes[key], records=records,
                           display_only=self._display_only)
                 for key, records in grouped.items()]
        found.sort(key=lambda one: one.name)
        return found

    def _conditions(self, given) -> dict[str, dict[str, str]]:
        if isinstance(given, dict):
            return dict(given)
        if given is not None:
            return read_conditions(given)
        if self._root is None:
            return {}
        for folder in (self._root, self._root.parent):
            for name in CONDITION_FILES:
                found = read_conditions(folder / name)
                if found:
                    return found
        return {}

    def factors_for(self, name: str) -> dict[str, str]:
        """The factor levels of one recording, by name or by file stem."""
        return dict(self.conditions.get(name)
                    or self.conditions.get(Path(name).stem) or {})

    @property
    def factors(self) -> list[str]:
        """Every factor named in the conditions file, in the order given."""
        for factors in self.conditions.values():
            return list(factors)
        return []

    # -------------------------------------------------------------- reading
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"no recording in this batch stored {name!r}. Between them "
                f"they hold {self.stages or 'nothing yet'}."
            ) from exc

    def __getitem__(self, name: str) -> dict[str, list]:
        stage = _stage_of(name)
        if stage not in self.stages:
            raise KeyError(name)
        if stage not in self._cache:
            self._cache[stage] = self._concat(stage)
        return self._cache[stage]

    def _concat(self, stage: str) -> dict[str, list]:
        """One table from every recording that has this stage, stacked.

        Columns are the union in the order first seen, and a recording missing
        one contributes ``None`` rather than a short column — a short column
        would silently pair the wrong cell with the wrong condition.
        """
        out: dict[str, list] = {name: [] for name in
                                (RECORDING_COLUMN, *self.factors)}
        for recording in self.recordings:
            if stage not in recording:
                continue
            table = recording[stage]
            if not isinstance(table, dict) or not table:
                continue
            rows = max((len(values) for values in table.values()), default=0)
            if not rows:
                continue
            height = len(out[RECORDING_COLUMN])
            factors = self.factors_for(recording.name)
            out[RECORDING_COLUMN].extend([recording.name] * rows)
            for factor in self.factors:
                out[factor].extend([factors.get(factor)] * rows)
            for column, values in table.items():
                if column not in out:
                    out[column] = [None] * height
                out[column].extend(list(values) + [None] * (rows - len(values)))
            full = len(out[RECORDING_COLUMN])
            for values in out.values():
                if len(values) < full:
                    values.extend([None] * (full - len(values)))
        return out

    @property
    def summary(self) -> dict[str, list]:
        """One row per recording: its factors and every number it produced.

        Built from the scalar artefacts — the cosmic-ray summary, the
        instrumental control, the rhythm — with the stage in front of each name,
        because ``period_h`` stops meaning one thing the moment two stages report
        one. Anything that is not a single number is left out; it is still in the
        artefact it came from, which ``about`` names.
        """
        rows: list[dict[str, Any]] = []
        for recording in self.recordings:
            row: dict[str, Any] = {RECORDING_COLUMN: recording.name}
            row.update(self.factors_for(recording.name))
            for stage in recording.stages:
                if recording.about(stage)["kind"] != "scalars":
                    continue
                for key, value in (recording[stage] or {}).items():
                    if value is None or isinstance(value, (str, int, float, bool)):
                        row[f"{stage}.{key}"] = value
            rows.append(row)
        columns: list[str] = []
        for row in rows:
            columns.extend(name for name in row if name not in columns)
        return {name: [row.get(name) for row in rows] for name in columns}

    # -------------------------------------------------------------- looking
    @property
    def stages(self) -> list[str]:
        return sorted({stage for recording in self.recordings
                       for stage in recording.stages})

    @property
    def names(self) -> list[str]:
        return [recording.name for recording in self.recordings]

    def __iter__(self) -> Iterator[Recording]:
        return iter(self.recordings)

    def __len__(self) -> int:
        return len(self.recordings)

    def __contains__(self, name: object) -> bool:
        return _stage_of(str(name)) in self.stages

    def __dir__(self) -> list[str]:
        present = set(self.stages)
        aliases = {name for name, stage in ALIASES.items() if stage in present}
        return sorted({*super().__dir__(), *present, *aliases})

    def __repr__(self) -> str:
        conditions = (f", {len(self.factors)} factor(s)" if self.factors
                      else ", no conditions file")
        return (f"<Batch of {len(self.recordings)} recording(s){conditions}: "
                f"{', '.join(self.stages) or 'nothing stored'}>")
