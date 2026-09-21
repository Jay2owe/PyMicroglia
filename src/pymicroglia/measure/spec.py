"""What a measurement run is told: the movies, their extra inputs, the design.

Deliberately separate from the tracking configuration. Tracking configuration
describes how outlines are produced; this describes how an already accepted
set of outlines is measured. A user who never runs the tracker -- who has
outlines from somewhere else -- still needs this and nothing else.

Ported from Motion's ``analysis/config.py`` on 2026-09-21. The dataclasses are
renamed (``MovieConfig`` -> :class:`MovieSpec`, and so on) because in this
package "config" is the store's settings; the JSON shape they read is
unchanged, so :func:`load_config` reads an existing ``analysis_config.json``
verbatim. The figure, plot and pipeline blocks Motion parsed here belong to
later port stages and are carried through as raw mappings, unread.

No pandas at import time: ``describe`` and ``validate`` read this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .conditions import UNASSIGNED, Assignment, ConditionSet
from .contrasts import (CONTRAST_AGGREGATES, CONTRAST_UNITS, ContrastSpec,
                        parse_contrasts)

__all__ = [
    "ChannelSpec", "ObjectSetSpec", "SideTableSpec", "MovieSpec",
    "WindowSpec", "ContrastSpec", "MeasureConfig", "load_config",
    "CONTRAST_UNITS", "CONTRAST_AGGREGATES",
]


def _declared_condition(value: object) -> str | None:
    text = str(value).strip() if value not in (None, "") else ""
    return None if text in ("", UNASSIGNED) else text


#: A channel name becomes a value in the ``channel`` column of three tables and
#: a series label on any figure that draws them, so it is held to the same
#: shape as a column name rather than accepted as free text.
_CHANNEL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

#: Names the package already uses for a stack of its own. Reusing one would put
#: two different images under one word in the same manifest.
_RESERVED_CHANNEL_NAMES = frozenset(
    {"labels", "raw", "unclaimed", "evidence", "provenance", "history"}
)


def _resolver(root: Path | None):
    def resolve(value):
        if value in (None, ""):
            return None
        path = Path(str(value))
        return path if path.is_absolute() or root is None else (root / path)
    return resolve


def _aligned_fields(raw: Mapping[str, Any], what: str, name: str) -> dict[str, Any]:
    """The alignment settings a channel and an object set share."""
    if not raw.get("path"):
        raise ValueError(f"{what} {name!r} has no path")
    origin = tuple(raw.get("crop_origin", (0, 0)))
    if len(origin) != 2:
        raise ValueError(f"{what} {name!r}: crop_origin must be [y, x]")
    columns = tuple(raw.get("shift_columns", ("shift_y", "shift_x")))
    if len(columns) != 2:
        raise ValueError(
            f"{what} {name!r}: shift_columns must name exactly two columns, "
            "the y one first"
        )
    index = raw.get("channel_index")
    offset = raw.get("frame_offset")
    return {
        "channel_index": None if index is None else int(index),
        "frame_offset": None if offset is None else int(offset),
        "crop_origin": (int(origin[0]), int(origin[1])),
        "shift_columns": (str(columns[0]), str(columns[1])),
        "shift_scale": float(raw.get("shift_scale", 1.0)),
        "expected_sha256": raw.get("sha256"),
        "description": str(raw.get("description", "")),
    }


def _aligned_dict(spec) -> dict[str, Any]:
    """The alignment settings back in the keys ``from_dict`` reads."""
    return {
        "name": spec.name,
        "path": str(spec.path),
        "channel_index": spec.channel_index,
        "frame_offset": spec.frame_offset,
        "crop_origin": list(spec.crop_origin),
        "shifts": None if spec.shifts is None else str(spec.shifts),
        "shift_columns": list(spec.shift_columns),
        "shift_scale": spec.shift_scale,
        "sha256": spec.expected_sha256,
        "description": spec.description,
    }


def _checked_name(raw: Mapping[str, Any], what: str, column: str) -> str:
    name = str(raw.get("name", "")).strip()
    if not _CHANNEL_NAME.match(name):
        raise ValueError(
            f"{what} name {name!r} is not usable: the name becomes a value in "
            f"the `{column}` column and a label on a figure, so it must be "
            "lower case, start with a letter and hold only letters, digits "
            "and underscores"
        )
    if name in _RESERVED_CHANNEL_NAMES:
        raise ValueError(
            f"{what} name {name!r} is already the name of a stack this package "
            f"loads; reserved names: {', '.join(sorted(_RESERVED_CHANNEL_NAMES))}"
        )
    return name


@dataclass
class ChannelSpec:
    """One extra imaging channel, and how to line it up with the outlines.

    The alignment contract is one sentence::

        analysed[i, y, x] == source[frame_offset + i,
                                    crop_origin[0] + y - shift_y[i],
                                    crop_origin[1] + x - shift_x[i]]

    ``shift_y``/``shift_x`` come from ``shifts``, a CSV with one row per source
    frame, read from the columns named in ``shift_columns`` and multiplied by
    ``shift_scale`` before rounding.
    """

    name: str
    path: Path
    channel_index: int | None = None
    frame_offset: int | None = None
    crop_origin: tuple[int, int] = (0, 0)
    shifts: Path | None = None
    shift_columns: tuple[str, str] = ("shift_y", "shift_x")
    shift_scale: float = 1.0
    expected_sha256: str | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, raw_dict: Mapping[str, Any], resolve) -> "ChannelSpec":
        name = _checked_name(raw_dict, "channel", "channel")
        fields = _aligned_fields(raw_dict, "channel", name)
        return cls(name=name, path=resolve(raw_dict["path"]),
                   shifts=resolve(raw_dict.get("shifts")), **fields)

    def as_dict(self) -> dict[str, Any]:
        """The block back in the keys ``from_dict`` reads; paths as text."""
        return _aligned_dict(self)


@dataclass
class ObjectSetSpec:
    """One set of labelled reference shapes, and how to line it up.

    ``static`` says the file holds a single map that applies to every frame.
    ``frame_offset`` and ``shifts`` are refused alongside it rather than
    ignored, since a setting quietly dropped reads exactly like one honoured.
    """

    name: str
    path: Path
    static: bool = False
    channel_index: int | None = None
    frame_offset: int | None = None
    crop_origin: tuple[int, int] = (0, 0)
    shifts: Path | None = None
    shift_columns: tuple[str, str] = ("shift_y", "shift_x")
    shift_scale: float = 1.0
    expected_sha256: str | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, raw_dict: Mapping[str, Any], resolve) -> "ObjectSetSpec":
        name = _checked_name(raw_dict, "object set", "object_set")
        fields = _aligned_fields(raw_dict, "object set", name)
        static = bool(raw_dict.get("static", False))
        if static:
            for setting in ("frame_offset", "shifts"):
                if raw_dict.get(setting) not in (None, ""):
                    raise ValueError(
                        f"object set {name!r} is static and also sets "
                        f"{setting}, which has nothing to act on: a static set "
                        "is one map applied to every frame, drawn in the "
                        "analysed field's own space and held there. Drop "
                        f"{setting}, or drop static and supply one map per frame."
                    )
        return cls(name=name, path=resolve(raw_dict["path"]), static=static,
                   shifts=resolve(raw_dict.get("shifts")), **fields)

    def as_dict(self) -> dict[str, Any]:
        """The block back in the keys ``from_dict`` reads; paths as text."""
        out = _aligned_dict(self)
        out["static"] = bool(self.static)
        if self.static:
            # A static set never sets these; writing them back would be
            # refused on the next read.
            out.pop("frame_offset")
            out.pop("shifts")
        return out


def _named_list(entries: object, what: str, stem: str, parse) -> list:
    """Parse one per-movie block, refusing a repeated name.

    A repeat is refused here rather than left to the loader because the second
    one would silently replace the first in the context dictionary.
    """
    if not entries:
        return []
    if not isinstance(entries, list):
        raise TypeError(
            f"{stem}: {what} must be a list of blocks, not {type(entries).__name__}")
    parsed = [parse(entry) for entry in entries]
    seen: set[str] = set()
    for item in parsed:
        if item.name in seen:
            raise ValueError(f"{stem}: {what} {item.name!r} is declared twice")
        seen.add(item.name)
    return parsed


#: The two keys a side table may be joined on, and nothing else.
_SIDE_KEYS = ("frame_index", "identity")
#: How a frame number is written in the user's file. ``label`` is a 0-based
#: index into the analysed movie; ``source`` is a 1-based ImageJ frame number.
_SIDE_SPACES = ("label", "source")


@dataclass
class SideTableSpec:
    """One spreadsheet of the user's own, and which column of it is the key.

    A side table is joined, never copied. ``keyed_on`` is ``frame_index`` (one
    row per timepoint) or ``identity`` (one row per cell). ``key_space``
    converts a log written in the microscope's own numbering::

        frame_index = key - 1 - movie.source_frame_offset
    """

    name: str
    path: Path
    keyed_on: str = "frame_index"
    key_column: str | None = None
    key_space: str = "label"
    columns: tuple[str, ...] | None = None
    expected_sha256: str | None = None
    description: str = ""

    @property
    def key(self) -> str:
        """The column to look for in the user's file."""
        return self.key_column or self.keyed_on

    def as_dict(self) -> dict[str, Any]:
        """The block back in the keys ``from_dict`` reads; paths as text."""
        out: dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
            "keyed_on": self.keyed_on,
            "key_column": self.key_column,
            "columns": None if self.columns is None else list(self.columns),
            "sha256": self.expected_sha256,
            "description": self.description,
        }
        if self.keyed_on == "frame_index":
            out["key_space"] = self.key_space
        return out

    @classmethod
    def from_dict(cls, raw_dict: Mapping[str, Any], resolve) -> "SideTableSpec":
        name = str(raw_dict.get("name", "")).strip()
        if not _CHANNEL_NAME.match(name):
            raise ValueError(
                f"side table name {name!r} is not usable: the name becomes the "
                "prefix on every column this table contributes, so it must be "
                "lower case, start with a letter and hold only letters, digits "
                "and underscores"
            )
        if name in _RESERVED_CHANNEL_NAMES:
            raise ValueError(
                f"side table name {name!r} is already the name of a stack this "
                f"package loads; reserved names: "
                f"{', '.join(sorted(_RESERVED_CHANNEL_NAMES))}"
            )
        if not raw_dict.get("path"):
            raise ValueError(f"side table {name!r} has no path")
        keyed_on = str(raw_dict.get("keyed_on", "frame_index"))
        if keyed_on not in _SIDE_KEYS:
            raise ValueError(
                f"side table {name!r}: keyed_on={keyed_on!r} is not a key this "
                f"package can join on; it accepts {' or '.join(_SIDE_KEYS)}"
            )
        key_space = str(raw_dict.get("key_space", "label"))
        if key_space not in _SIDE_SPACES:
            raise ValueError(
                f"side table {name!r}: key_space={key_space!r} is not a frame "
                f"numbering this package knows; it accepts "
                f"{' or '.join(_SIDE_SPACES)}"
            )
        if "key_space" in raw_dict and keyed_on != "frame_index":
            raise ValueError(
                f"side table {name!r} is keyed on {keyed_on!r} and also sets "
                "key_space, which only means something for a frame key. A cell "
                "identity is not numbered two ways; remove key_space."
            )
        columns = raw_dict.get("columns")
        if columns is not None:
            if not isinstance(columns, (list, tuple)):
                raise ValueError(
                    f"side table {name!r}: columns must be a list of column "
                    f"names, or null for all of them, not "
                    f"{type(columns).__name__}"
                )
            columns = tuple(str(column) for column in columns)
        key_column = raw_dict.get("key_column")
        return cls(
            name=name,
            path=resolve(raw_dict["path"]),
            keyed_on=keyed_on,
            key_column=None if key_column in (None, "") else str(key_column),
            key_space=key_space,
            columns=columns,
            expected_sha256=raw_dict.get("sha256"),
            description=str(raw_dict.get("description", "")),
        )


#: A window name becomes a value in a ``window`` column that people filter on.
_WINDOW_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class WindowSpec:
    """One named stretch of a recording.

    Half-open on purpose: ``from_hours <= h < to_hours``. Declared in hours
    **or** in frames, never both. Hours are hours since the recording began.
    """

    name: str
    from_hours: float | None = None
    to_hours: float | None = None
    from_frame: int | None = None
    to_frame: int | None = None
    baseline: str | None = None
    description: str = ""

    @property
    def in_frames(self) -> bool:
        return self.from_frame is not None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.in_frames:
            out.update(from_frame=self.from_frame, to_frame=self.to_frame)
        else:
            out.update(from_hours=self.from_hours, to_hours=self.to_hours)
        if self.baseline is not None:
            out["baseline"] = self.baseline
        if self.description:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, raw_dict: Mapping[str, Any]) -> "WindowSpec":
        name = str(raw_dict.get("name", "")).strip()
        if not _WINDOW_NAME.fullmatch(name):
            raise ValueError(
                f"window name {name!r} is not a plain lower-case identifier; "
                "it becomes a value in a `window` column that people filter on, "
                "so it must look like `baseline` or `after_drug`"
            )
        hours = ("from_hours" in raw_dict) or ("to_hours" in raw_dict)
        frames = ("from_frame" in raw_dict) or ("to_frame" in raw_dict)
        if hours and frames:
            raise ValueError(
                f"window {name!r} is declared in hours and in frames at once; "
                "give from_hours/to_hours or from_frame/to_frame, not both, "
                "because the two would have to agree and nothing checks that"
            )
        if not hours and not frames:
            raise ValueError(
                f"window {name!r} declares neither hours nor frames; give "
                "from_hours and to_hours, or from_frame and to_frame"
            )
        if hours:
            if "from_hours" not in raw_dict or "to_hours" not in raw_dict:
                raise ValueError(
                    f"window {name!r} needs both from_hours and to_hours; a "
                    "window with one open end would silently change length when "
                    "a longer recording was analysed"
                )
            start, stop = float(raw_dict["from_hours"]), float(raw_dict["to_hours"])
            if not stop > start:
                raise ValueError(
                    f"window {name!r} ends at {stop} h, which is not after its "
                    f"start at {start} h; a window is half-open, from <= h < to"
                )
            bounds = {"from_hours": start, "to_hours": stop}
        else:
            if "from_frame" not in raw_dict or "to_frame" not in raw_dict:
                raise ValueError(
                    f"window {name!r} needs both from_frame and to_frame; a "
                    "window with one open end would silently change length when "
                    "a longer recording was analysed"
                )
            start, stop = int(raw_dict["from_frame"]), int(raw_dict["to_frame"])
            if not stop > start:
                raise ValueError(
                    f"window {name!r} ends at frame {stop}, which is not after "
                    f"its start at frame {start}; a window is half-open, "
                    "from <= frame < to"
                )
            bounds = {"from_frame": start, "to_frame": stop}

        baseline = raw_dict.get("baseline")
        baseline = None if baseline in (None, "") else str(baseline)
        if baseline == name:
            raise ValueError(
                f"window {name!r} names itself as its own baseline, which would "
                "make every change zero and every ratio one"
            )
        return cls(name=name, baseline=baseline,
                   description=str(raw_dict.get("description", "")), **bounds)


def parse_windows(entries: object, where: str) -> list[WindowSpec]:
    """Parse a ``windows`` block, refusing a repeat and a dangling baseline."""
    if not entries:
        return []
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError(
            f"{where}: windows must be a list of window blocks, not "
            f"{type(entries).__name__}"
        )
    parsed = [entry if isinstance(entry, WindowSpec) else WindowSpec.from_dict(entry)
              for entry in entries]
    seen: set[str] = set()
    for window in parsed:
        if window.name in seen:
            raise ValueError(f"{where}: window {window.name!r} is declared twice")
        seen.add(window.name)
    for window in parsed:
        if window.baseline is not None and window.baseline not in seen:
            raise ValueError(
                f"{where}: window {window.name!r} names {window.baseline!r} as "
                f"its baseline, but no window is called that; declared windows "
                f"are {sorted(seen)}"
            )
    return parsed


@dataclass
class MovieSpec:
    """One movie: its accepted outlines and the signal they were built from.

    The five tracker files -- ``labels``, ``raw``, ``unclaimed``, ``evidence``,
    ``provenance`` -- are exactly the roles :class:`pymicroglia.tracking.
    TrackingResult` names, and :meth:`tracking` hands them over in that form.
    ``history`` is the tracker's decision folder (``decisions`` is accepted as
    the newer spelling).
    """

    stem: str
    labels: Path
    raw: Path
    unclaimed: Path | None = None
    evidence: Path | None = None
    provenance: Path | None = None
    history: Path | None = None
    valid_mask: Path | None = None
    channels: list[ChannelSpec] = field(default_factory=list)
    side_tables: list[SideTableSpec] = field(default_factory=list)
    objects: list[ObjectSetSpec] = field(default_factory=list)
    windows: list[WindowSpec] = field(default_factory=list)
    source_frame_offset: int = 0
    condition: str | None = None
    subject: str | None = None
    expected_sha256: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, raw_dict: Mapping[str, Any], root: Path | None = None) -> "MovieSpec":
        if isinstance(raw_dict, MovieSpec):
            return raw_dict
        resolve = _resolver(root)
        stem = raw_dict.get("stem", "?")
        return cls(
            stem=str(raw_dict["stem"]),
            labels=resolve(raw_dict["labels"]),
            raw=resolve(raw_dict["raw"]),
            unclaimed=resolve(raw_dict.get("unclaimed")),
            evidence=resolve(raw_dict.get("evidence")),
            provenance=resolve(raw_dict.get("provenance")),
            history=resolve(raw_dict.get("history", raw_dict.get("decisions"))),
            valid_mask=resolve(raw_dict.get("valid_mask")),
            channels=_named_list(raw_dict.get("channels"), "channel", stem,
                                 lambda entry: ChannelSpec.from_dict(entry, resolve)),
            side_tables=_named_list(raw_dict.get("side_tables"), "side table", stem,
                                    lambda entry: SideTableSpec.from_dict(entry, resolve)),
            objects=_named_list(raw_dict.get("objects"), "object set", stem,
                                lambda entry: ObjectSetSpec.from_dict(entry, resolve)),
            windows=parse_windows(raw_dict.get("windows"), str(stem)),
            source_frame_offset=int(raw_dict.get("source_frame_offset", 0)),
            condition=_declared_condition(raw_dict.get("condition")),
            subject=raw_dict.get("subject"),
            expected_sha256=dict(raw_dict.get("sha256", {})),
            notes=str(raw_dict.get("notes", "")),
        )

    def tracking_mapping(self) -> dict[str, Any]:
        """The five tracker files in :class:`TrackingResult`'s own words."""
        out: dict[str, Any] = {"stem": self.stem,
                               "source_frame_offset": int(self.source_frame_offset)}
        for role in ("labels", "raw", "unclaimed", "provenance", "evidence"):
            value = getattr(self, role)
            out[role] = str(value) if value is not None else None
        if self.history is not None:
            out["decisions"] = str(self.history)
        return out

    def as_dict(self) -> dict[str, Any]:
        """The movie as the run manifest records it: paths, never contents."""
        def text(value):
            return None if value is None else str(value)
        return {
            "stem": self.stem,
            "labels": text(self.labels), "raw": text(self.raw),
            "unclaimed": text(self.unclaimed), "evidence": text(self.evidence),
            "provenance": text(self.provenance), "history": text(self.history),
            "valid_mask": text(self.valid_mask),
            "channels": [c.as_dict() for c in self.channels],
            "side_tables": [s.as_dict() for s in self.side_tables],
            "objects": [o.as_dict() for o in self.objects],
            "windows": [w.as_dict() for w in self.windows],
            "source_frame_offset": int(self.source_frame_offset),
            "condition": self.condition,
            "subject": self.subject,
            "sha256": dict(self.expected_sha256),
            "notes": self.notes,
        }


def parse_metric_groups(block: object) -> dict:
    """Parse the ``metric_groups`` block into resolved groups.

    Imported inside the function: resolving a group means asking the
    registered modules what they write, and a caller that only wanted to read
    a configuration should not pay for importing every measurement module.
    """
    if not block:
        return {}
    from .metric_groups import build

    return build(block)


@dataclass
class MeasureConfig:
    """Everything one measurement run was told."""

    dataset: str
    frame_interval_min: float
    movies: list[MovieSpec]
    output_root: Path | None = None
    microns_per_pixel: float | None = None
    calibration_tiffs: list[Path] = field(default_factory=list)
    enabled_modules: list[str] = field(default_factory=list)
    module_params: dict = field(default_factory=dict)
    verify_hashes: bool = True
    #: Carried through to the manifest as a section, unread here (stage 07).
    theme: dict = field(default_factory=dict)
    conditions: ConditionSet = field(default_factory=ConditionSet)
    #: Carried through to the manifest as a section, unread here (stage 07).
    figures: dict = field(default_factory=dict)
    windows: list[WindowSpec] = field(default_factory=list)
    contrasts: list[ContrastSpec] = field(default_factory=list)
    metric_groups: dict = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def from_parts(cls, movies: Sequence[Any], *, frame_interval_min: float,
                   root: Path | None = None, dataset: str = "unnamed dataset",
                   microns_per_pixel: float | None = None,
                   calibration_tiffs: Sequence[Any] = (),
                   enabled_modules: Sequence[str] | None = None,
                   module_params: Mapping[str, Any] | None = None,
                   verify_hashes: bool = True, theme: Mapping[str, Any] | None = None,
                   conditions: Any = None, figures: Mapping[str, Any] | None = None,
                   windows: Sequence[Any] = (), contrasts: Sequence[Any] = (),
                   metric_groups: Mapping[str, Any] | None = None,
                   output_root: Path | None = None,
                   source_path: Path | None = None) -> "MeasureConfig":
        """Build one from keyword parts -- what the ``measure`` action takes."""
        if frame_interval_min is None:
            raise ValueError("frame_interval_min is required: the interval between "
                             "frames, in minutes, is what turns a frame index into "
                             "an hour and nothing in a label stack states it")
        # Parsed before contrasts because a contrast may reference a group.
        groups = parse_metric_groups(metric_groups)
        resolve = _resolver(root)
        return cls(
            dataset=str(dataset),
            frame_interval_min=float(frame_interval_min),
            movies=[MovieSpec.from_dict(entry, root) for entry in movies],
            output_root=resolve(output_root),
            microns_per_pixel=(None if microns_per_pixel in (None, "")
                               else float(microns_per_pixel)),
            calibration_tiffs=[resolve(p) for p in calibration_tiffs or ()],
            enabled_modules=list(enabled_modules or []),
            module_params=dict(module_params or {}),
            verify_hashes=bool(verify_hashes),
            theme=dict(theme or {}),
            conditions=(conditions if isinstance(conditions, ConditionSet)
                        else ConditionSet.from_config(conditions)),
            figures=dict(figures or {}),
            windows=parse_windows(windows, "windows"),
            contrasts=parse_contrasts(contrasts, groups),
            metric_groups=groups,
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MeasureConfig":
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(data, root=path.parent, source_path=path)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, root: Path | None = None,
                     source_path: Path | None = None) -> "MeasureConfig":
        """The JSON shape of Motion's ``analysis_config.json``, unchanged."""
        return cls.from_parts(
            data["movies"],
            frame_interval_min=float(data["frame_interval_min"]),
            root=root,
            dataset=data.get("dataset", "unnamed dataset"),
            microns_per_pixel=data.get("microns_per_pixel"),
            calibration_tiffs=data.get("calibration_tiffs", []),
            enabled_modules=data.get("enabled_modules", []),
            module_params=data.get("modules", {}),
            verify_hashes=bool(data.get("verify_hashes", True)),
            theme=data.get("theme", {}),
            conditions=data.get("conditions"),
            figures=data.get("figures", {}),
            windows=data.get("windows"),
            contrasts=data.get("contrasts"),
            metric_groups=data.get("metric_groups"),
            output_root=data.get("output_root", "outputs"),
            source_path=source_path,
        )

    # --------------------------------------------------------------- windows

    def windows_for(self, movie: MovieSpec) -> list[WindowSpec]:
        """The windows that apply to one movie.

        A movie's own block *replaces* the shared one rather than adding to it.
        """
        return list(movie.windows) if movie.windows else list(self.windows)

    # ------------------------------------------------------------ conditions

    def assignment(self, movie: MovieSpec) -> Assignment:
        """Which experimental group one movie belongs to, and how that was decided."""
        return self.conditions.assign(
            self.conditions.text_for(movie),
            declared=movie.condition,
            stem=movie.stem,
        )

    def assignments(self) -> list[Assignment]:
        return [self.assignment(movie) for movie in self.movies]

    def condition_problems(self) -> list[str]:
        """Every reason a run must not start. Empty means the design is readable."""
        return [
            f"{a.stem}: {a.problem}" for a in self.assignments() if a.problem is not None
        ]

    def movie(self, stem: str) -> MovieSpec:
        for entry in self.movies:
            if entry.stem == stem:
                return entry
        known = ", ".join(m.stem for m in self.movies)
        raise KeyError(f"stem {stem!r} is not in the configuration; known stems: {known}")

    def settings(self) -> dict[str, Any]:
        """The run's settings as the manifest records them."""
        return {
            "dataset": self.dataset,
            "frame_interval_min": self.frame_interval_min,
            "microns_per_pixel": self.microns_per_pixel,
            "calibration_tiffs": [str(p) for p in self.calibration_tiffs],
            "enabled_modules": list(self.enabled_modules),
            "modules": dict(self.module_params),
            "verify_hashes": self.verify_hashes,
            "windows": [w.as_dict() for w in self.windows],
            "contrasts": [c.as_dict() for c in self.contrasts],
            "metric_groups": {name: list(group.columns)
                              for name, group in self.metric_groups.items()},
            "metric_groups_declared": {name: group.declared
                                       for name, group in self.metric_groups.items()},
        }


def load_config(path: str | Path) -> MeasureConfig:
    """Read an ``analysis_config.json`` exactly as Motion wrote it."""
    return MeasureConfig.load(path)
