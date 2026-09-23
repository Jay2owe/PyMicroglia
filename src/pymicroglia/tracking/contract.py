"""What a tracker hands back, by file and never by array.

The frozen Motion tracker ships inside PyMicroglia. Its output is five files
and a folder of decision tables. This module keeps that shape as data so the
measurement step reads files rather than tracker arrays.

Every path here is a path. A tracker that returned arrays would have to be in
the same process as the measurement, and the one thing this seam exists to
allow is that it need not be.

Today's file names are the Motion tracker's, recorded in
:data:`MOTION_OUTPUTS` and :data:`MOTION_DECISIONS_ROOT` from
``Motion/analysis_config.example.json`` and its ``README.md`` "Output"
section, so that :meth:`TrackingResult.from_folder` can read a finished Motion
run. They are data on this contract, not constants in the
measure step: a different tracker declares different names.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

__all__ = ["TrackingResult", "DecisionTables", "ROLES", "MOTION_OUTPUTS",
           "MOTION_DECISIONS_ROOT", "MOTION_DECISION_TABLES",
           "MOTION_LABEL_TABLES", "expected_files", "sha256_of"]

#: The five files, in the words Motion's ``analysis_config.json`` uses. The
#: first two are required; the other three are optional and a movie without
#: them is measured exactly as before, it simply cannot say what rests on a
#: reconstructed attribution or what the tracker's own evidence was.
ROLES: tuple[str, ...] = ("labels", "raw", "unclaimed", "provenance", "evidence")
REQUIRED: tuple[str, ...] = ("labels", "raw")

#: Where the Motion tracker writes each file, relative to its accepted run
#: folder (``m21_stationary_reconciliation/<run>/`` or
#: ``m22_accepted_history/<run>/``). ``raw`` is absent on purpose: the
#: registered stack is the tracker's *input*, pinned in ``motion_inputs.json``
#: and never copied, so it lives where the handoff says it does.
MOTION_OUTPUTS: dict[str, str] = {
    "labels": "out/{stem}.tif",
    "unclaimed": "out/{stem}_unclaimed_original_ids.tif",
    "provenance": "out/{stem}_provenance.tif",
    "evidence": "qc/{stem}_motion_evidence.tif",
}
#: The tracker's decision folder, relative to the same run folder.
MOTION_DECISIONS_ROOT = "mid/accepted_history"

#: The decision CSVs today's Motion tracker emits, as ``(name, relative path,
#: grain)``, relative to :data:`MOTION_DECISIONS_ROOT`. Moved from
#: ``Motion/analysis/modules/history.py`` (``SOURCES``). Chains differ between
#: movies, so any one of these may be absent from a run and a reader skips and
#: records it rather than failing.
#:
#: Every grain is empty for the reason ``history.py`` gives: naming the grain
#: of somebody else's CSV would be this package asserting what one row of it
#: means, and these are copied through precisely so the tracker's account is
#: not paraphrased.
MOTION_DECISION_TABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("gap_evidence", "21_continuity_evidence/out/gap_evidence.csv", ()),
    ("mechanism_summary", "21_continuity_evidence/out/mechanism_summary.csv", ()),
    ("termination_evidence", "21_continuity_evidence/out/termination_evidence.csv", ()),
    ("termination_audit", "19_continuity_census/out/termination_audit.csv", ()),
    ("gap_runs", "19_continuity_census/out/gap_runs.csv", ()),
    ("residency_registry", "18_seat_identity/out/residency_registry.csv", ()),
    ("seat_table", "18_seat_identity/out/seat_table.csv", ()),
    ("partition_actions", "18_seat_identity/out/partition_actions.csv", ()),
    ("merge_split_events", "15_merge_split_profile/out/merge_split_events.csv", ()),
    ("merge_split_actions", "16_merge_split_reconcile/out/merge_split_actions.csv", ()),
    ("quarantine_events", "08_global_quarantine/out/quarantine_events.csv", ()),
    ("ordered_actions", "11_ordered_quarantine/out/ordered_actions.csv", ()),
    ("hierarchy_actions", "13_hierarchy_reconcile/out/hierarchy_actions.csv", ()),
    ("merge_events", "09_general_merge_bridge/out/merge_events.csv", ()),
)

#: Two more tables the Motion tracker writes *beside the labels* (in ``out/``)
#: rather than into the decision folder (``history.py``'s ``LABEL_SOURCES``).
#: The pixel-evidence table is deliberately left out: it is one row per
#: renamed pixel and belongs to the tracking run, not to every analysis run.
MOTION_LABEL_TABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("stationary_takeover_events", "stationary_takeover_events.csv", ()),
    ("stationary_component_runs", "stationary_component_runs.csv", ()),
)

_CHUNK = 1 << 20


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_files(stem: str, raw: str | Path | None = None) -> dict[str, str]:
    """The names a tracker is expected to write for one stem.

    What ``motion_inputs.json`` carries under ``expects`` so the tracker and
    the measure step agree by file rather than by convention. Paths other than
    ``raw`` are relative to the tracker's run folder; ``raw`` is the pinned
    registered stack, already absolute, or the role name alone when no path
    is known yet.
    """
    out = {role: MOTION_OUTPUTS[role].format(stem=stem)
           for role in ROLES if role in MOTION_OUTPUTS}
    out["raw"] = str(raw) if raw is not None else "<the pinned registered_raw>"
    out["decisions"] = MOTION_DECISIONS_ROOT
    return {role: out[role] for role in (*ROLES, "decisions")}


@dataclass(frozen=True)
class DecisionTables:
    """Decision CSVs a tracker emits; each entry is ``(name, relative path, grain)``.

    ``root`` is the folder the relative paths are read against. ``beside`` is
    the smaller set written next to the labels rather than under ``root``,
    with paths relative to the labels' own folder. Both are declared lists,
    not a scan: a reader knows what it is allowed to look for and records what
    it did not find.
    """

    root: Path
    tables: tuple[tuple[str, str, tuple[str, ...]], ...] = MOTION_DECISION_TABLES
    beside: tuple[tuple[str, str, tuple[str, ...]], ...] = ()

    @classmethod
    def motion(cls, root: str | Path) -> "DecisionTables":
        """Today's Motion tracker's declaration, rooted at its decision folder."""
        return cls(Path(root), MOTION_DECISION_TABLES, MOTION_LABEL_TABLES)

    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in (*self.tables, *self.beside))

    def path(self, name: str, labels_folder: str | Path | None = None) -> Path:
        """The absolute path of one declared table, present or not."""
        for table, relative, _ in self.tables:
            if table == name:
                return self.root / relative
        for table, relative, _ in self.beside:
            if table == name:
                base = Path(labels_folder) if labels_folder is not None else self.root
                return base / relative
        raise KeyError(f"{name!r} is not a declared decision table; "
                       f"declared: {', '.join(self.names())}")

    def present(self, labels_folder: str | Path | None = None) -> dict[str, Path]:
        """The declared tables that exist on disk, by name."""
        found = {}
        for name in self.names():
            path = self.path(name, labels_folder)
            if path.is_file():
                found[name] = path
        return found

    def as_dict(self) -> dict[str, Any]:
        return {"root": str(self.root),
                "tables": [list((n, p, list(g))) for n, p, g in self.tables],
                "beside": [list((n, p, list(g))) for n, p, g in self.beside]}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], root: Path | None = None) -> "DecisionTables":
        where = Path(str(raw["root"]))
        if root is not None and not where.is_absolute():
            where = root / where
        tables = tuple((str(n), str(p), tuple(g)) for n, p, g in raw.get("tables", ()))
        beside = tuple((str(n), str(p), tuple(g)) for n, p, g in raw.get("beside", ()))
        return cls(where, tables or MOTION_DECISION_TABLES, beside)


@dataclass(frozen=True)
class TrackingResult:
    """What a tracker hands back for one recording, by file, never by array.

    ``labels`` is the ``uint16 (frames, y, x)`` identity stack and ``raw`` the
    registered signal the labels were built from; both are required. The
    other three stacks and the decision tables are optional. ``sha256`` holds
    one digest per present file, keyed by role, so a downstream step can pin
    what it read the way Motion's own configuration does.
    """

    stem: str
    labels: Path
    raw: Path
    unclaimed: Path | None = None
    provenance: Path | None = None
    evidence: Path | None = None
    decisions: DecisionTables | None = None
    #: How many leading frames of ``raw`` the labels do not cover. Label
    #: frame k is raw frame k + offset. The sidecars are in label space and
    #: need no offset.
    source_frame_offset: int = 0
    sha256: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for role in REQUIRED:
            if getattr(self, role) is None:
                raise ValueError(f"a tracking result needs {role}: it is required")
        for role in ROLES:
            value = getattr(self, role)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, role, Path(str(value)))

    # ------------------------------------------------------------- reading
    def files(self) -> dict[str, Path]:
        """Role -> path for every stack this result names."""
        return {role: getattr(self, role) for role in ROLES
                if getattr(self, role) is not None}

    def hashed(self) -> "TrackingResult":
        """The same result with every named file's SHA-256 filled in."""
        digests = {role: sha256_of(path) for role, path in self.files().items()}
        return replace(self, sha256=digests)

    def verify(self, pinned: Mapping[str, str]) -> dict[str, bool]:
        """Each pinned role against the file on disk; refuses nothing itself."""
        return {role: sha256_of(getattr(self, role)) == digest
                for role, digest in pinned.items()
                if role in ROLES and getattr(self, role) is not None}

    # ------------------------------------------------------------- records
    def as_dict(self) -> dict[str, Any]:
        """A flat record: paths as strings, never contents."""
        out: dict[str, Any] = {"stem": self.stem}
        for role in ROLES:
            value = getattr(self, role)
            out[role] = str(value) if value is not None else None
        out["decisions"] = self.decisions.as_dict() if self.decisions else None
        out["source_frame_offset"] = int(self.source_frame_offset)
        out["sha256"] = dict(self.sha256)
        return out

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, root: str | Path | None = None,
                     hashes: bool = True) -> "TrackingResult":
        """A result from what a tracker returned or a configuration declared.

        The keys are :data:`ROLES` plus ``stem``, ``decisions`` (a folder, or
        a :class:`DecisionTables` record) and ``source_frame_offset``, in the
        words Motion's ``analysis_config.json`` already uses -- so a movie
        entry from that file reads straight in, with ``history`` accepted as
        the older name of ``decisions``. Relative paths resolve against
        ``root``. Motion pins each file under ``sha256`` and refuses a changed
        input; so does this: a pinned digest that does not match what is on
        disk is a ``ValueError``, not a warning.
        """
        base = Path(root) if root is not None else None

        def resolve(value):
            if value in (None, ""):
                return None
            path = Path(str(value))
            return path if path.is_absolute() or base is None else base / path

        paths = {role: resolve(raw.get(role)) for role in ROLES}
        decisions = raw.get("decisions", raw.get("history"))
        if isinstance(decisions, DecisionTables):
            tables = decisions
        elif isinstance(decisions, Mapping):
            tables = DecisionTables.from_mapping(decisions, base)
        elif decisions not in (None, ""):
            tables = DecisionTables.motion(resolve(decisions))
        else:
            tables = None

        result = cls(stem=str(raw.get("stem") or paths["labels"].stem),
                     decisions=tables,
                     source_frame_offset=int(raw.get("source_frame_offset", 0) or 0),
                     **paths)
        if hashes:
            result = result.hashed()
        pinned = raw.get("sha256") or {}
        if hashes and pinned:
            wrong = {role: (pinned[role], result.sha256.get(role))
                     for role in pinned
                     if role in result.sha256 and pinned[role] != result.sha256[role]}
            if wrong:
                raise ValueError(
                    f"{result.stem}: {sorted(wrong)} on disk do not match the "
                    f"pinned SHA-256; refusing to use a changed input: {wrong}")
        return result

    @classmethod
    def from_folder(cls, folder: str | Path, stem: str, *,
                    raw: str | Path | None = None,
                    evidence: str | Path | None = None,
                    source_frame_offset: int = 0,
                    hashes: bool = True) -> "TrackingResult":
        """A finished Motion run folder, mapped onto the contract.

        ``folder`` is the accepted run
        (``m21_stationary_reconciliation/<run>/`` or
        ``m22_accepted_history/<run>/``): the labels, the unclaimed ledger and
        the provenance sidecar are read from its ``out/``, the motion evidence
        from its ``qc/`` when the run holds one, and the decision tables from
        ``mid/accepted_history``. The registered stack is the tracker's input
        rather than its output, so it is not in that folder: pass ``raw``.
        ``evidence`` overrides the in-folder location, because on a real
        dataset it is written by the motion-inputs stage, one run folder over.
        """
        run = Path(folder)
        labels = run / MOTION_OUTPUTS["labels"].format(stem=stem)
        if not labels.is_file():
            raise FileNotFoundError(
                f"no accepted labels at {labels}: from_folder reads a finished "
                f"Motion run folder, whose out/ holds <stem>.tif")
        if raw is None:
            raise ValueError(
                "from_folder needs raw=<the registered stack the labels were "
                "built from>: it is the tracker's input, pinned in "
                "motion_inputs.json, and is not written into the run folder")

        def optional(role: str) -> Path | None:
            path = run / MOTION_OUTPUTS[role].format(stem=stem)
            return path if path.is_file() else None

        found_evidence = Path(evidence) if evidence is not None else optional("evidence")
        decisions_root = run / MOTION_DECISIONS_ROOT
        result = cls(stem=stem, labels=labels, raw=Path(raw),
                     unclaimed=optional("unclaimed"),
                     provenance=optional("provenance"),
                     evidence=found_evidence,
                     decisions=(DecisionTables.motion(decisions_root)
                                if decisions_root.is_dir() else None),
                     source_frame_offset=int(source_frame_offset))
        return result.hashed() if hashes else result
