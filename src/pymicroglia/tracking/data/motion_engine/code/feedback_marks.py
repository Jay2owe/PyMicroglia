"""Low-effort tracking feedback: marks Jamie makes on a label stack.

A mark is one sentence of the kind Jamie already says in review ("35 becomes
23 at frame 67", "50 and 103 swap at 94", "there is a cell here with no
outline") stored so that a machine can (a) read it back as that sentence and
(b) test it against any later candidate label stack.

Every claim is anchored to *positions* (a pixel inside the clicked cell, in a
given frame), never only to identity numbers, because candidate runs renumber
cells. The identity numbers are kept for the human sentence.

Frame conventions, stored side by side on every anchor:

* ``review_frame``  one-based index into the accepted label stack; the review
  TIFF header prints it as ``review N``;
* ``imagej_frame``  the source ImageJ slice, ``review_frame + offset`` where
  the offset is the number of source frames removed (two for VID95).

Storage is an append-only JSON-lines file per well,
``issues/NN_<well>/feedback/marks.jsonl``; a mistake is undone by appending a
``retract`` record, never by editing history. ``marks.csv`` beside it is a
regenerated flat view for spreadsheets.

Command line::

    python code/feedback_marks.py wells
    python code/feedback_marks.py digest --well 95_B4
    python code/feedback_marks.py evaluate --well 95_B4 --candidate <labels.tif>
    python code/feedback_marks.py to-cases --well 95_B4 --output review_cases.csv
    python code/feedback_marks.py parse --well 95_B4 "35>23 @67; 50<>103 @94"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from well_workflow import find_motion_root  # noqa: E402

SCHEMA = "motion.feedback-marks"
SCHEMA_VERSION = 1

# kind -> (one-line meaning, review-case type)
KINDS = {
    "becomes": ("the body now labelled B is really the earlier cell A", "failure"),
    "swap": ("two cells exchanged their numbers from this frame", "failure"),
    "same": ("these two outlines are one physical cell", "failure"),
    "different": ("these two outlines are different cells", "failure"),
    "persist": ("this identity is one cell for the whole recording", "failure"),
    "missing": ("a visible cell has no outline here", "failure"),
    "duplicate": ("one identity sits on two separate bodies in this frame", "failure"),
    "split": ("one outline covers two cells", "failure"),
    "join": ("two outlines in one frame are one cell", "failure"),
    "ok": ("this cell is tracked correctly here", "control"),
    "note": ("free text", ""),
}

REVIEW_CASE_COLUMNS = (
    "case_id", "case_type", "mechanism", "review_frame_start",
    "review_frame_end", "source_imagej_frame_start",
    "source_imagej_frame_end", "track_ref", "identities",
    "expected_identity", "notes",
)


# ------------------------------------------------------------------ helpers

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """A pixel inside ``mask`` as close to its centroid as possible (y, x)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("empty mask has no interior point")
    cy, cx = ys.mean(), xs.mean()
    iy, ix = int(round(cy)), int(round(cx))
    if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]:
        return iy, ix
    best = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(ys[best]), int(xs[best])


def label_at(labels: np.ndarray, frame_idx: int, y: int, x: int,
             radius: int = 2) -> int:
    """Label under a point, tolerating a small outline shift (mode of the
    nonzero labels within ``radius``)."""
    plane = labels[frame_idx]
    y = int(min(max(y, 0), plane.shape[0] - 1))
    x = int(min(max(x, 0), plane.shape[1] - 1))
    value = int(plane[y, x])
    if value:
        return value
    window = plane[max(y - radius, 0):y + radius + 1,
                   max(x - radius, 0):x + radius + 1]
    values = window[window > 0]
    if values.size == 0:
        return 0
    return int(Counter(values.tolist()).most_common(1)[0][0])


class Presence:
    """Which review frames (0-based) each identity occupies."""

    def __init__(self, labels: np.ndarray):
        self.frames: dict[int, list[int]] = {}
        for index in range(labels.shape[0]):
            for identity in np.unique(labels[index]):
                if identity:
                    self.frames.setdefault(int(identity), []).append(index)

    def identities(self) -> list[int]:
        return sorted(self.frames)

    def last_before(self, identity: int, frame_idx: int) -> int | None:
        earlier = [f for f in self.frames.get(identity, []) if f < frame_idx]
        return earlier[-1] if earlier else None

    def first_after(self, identity: int, frame_idx: int) -> int | None:
        later = [f for f in self.frames.get(identity, []) if f > frame_idx]
        return later[0] if later else None

    def nearest(self, identity: int, frame_idx: int) -> int | None:
        frames = self.frames.get(identity, [])
        if not frames:
            return None
        if frame_idx in frames:
            return frame_idx
        return min(frames, key=lambda f: (abs(f - frame_idx), f))


# ------------------------------------------------------------------ records

@dataclass
class Anchor:
    review_frame: int              # one-based
    imagej_frame: int
    identity: int                  # 0 for an empty point
    y: int
    x: int
    area: int = 0
    role: str = "body"             # body | reference | point | side | control
    click_y: int | None = None
    click_x: int | None = None

    @property
    def frame_idx(self) -> int:
        return self.review_frame - 1


@dataclass
class Mark:
    mark_id: str
    well: str
    kind: str
    made_at: str
    labels_sha256: str
    source_frame_offset: int
    anchors: list[Anchor] = field(default_factory=list)
    must_match: list[list[int]] = field(default_factory=list)
    must_differ: list[list[int]] = field(default_factory=list)
    must_exist: list[int] = field(default_factory=list)
    zero_ok: list[int] = field(default_factory=list)   # anchors allowed to vanish
    region: dict | None = None      # {"review_frame", "identity", "area", "bbox"}
    line: list[list[int]] | None = None   # [[y, x], [y, x]] drawn cut
    unresolved: list[str] = field(default_factory=list)
    identities: list[int] = field(default_factory=list)
    expected_identity: int | None = None
    note: str = ""
    sentence: str = ""
    origin: str = "ui"              # ui | text
    retracted: bool = False
    retract_reason: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        data["record"] = "mark"
        data["schema"] = SCHEMA
        data["schema_version"] = SCHEMA_VERSION
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Mark":
        data = {k: v for k, v in data.items()
                if k not in ("record", "schema", "schema_version")}
        data["anchors"] = [Anchor(**a) for a in data.get("anchors", [])]
        return cls(**data)

    def frames(self) -> list[int]:
        return sorted({a.review_frame for a in self.anchors})


# ------------------------------------------------------------------ storage

def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_marks(path: Path, include_retracted: bool = False) -> list[Mark]:
    marks: dict[str, Mark] = {}
    for record in read_records(path):
        kind = record.get("record")
        if kind == "mark":
            mark = Mark.from_dict(record)
            marks[mark.mark_id] = mark
        elif kind == "retract":
            target = marks.get(record.get("of", ""))
            if target is not None:
                target.retracted = True
                target.retract_reason = record.get("reason", "")
    result = list(marks.values())
    if not include_retracted:
        result = [m for m in result if not m.retracted]
    return result


def next_mark_id(path: Path, well: str) -> str:
    count = sum(1 for r in read_records(path) if r.get("record") == "mark")
    return f"{well}-M{count + 1:04d}"


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _refresh_csv(path: Path) -> None:
    """Best effort: the CSV is a derived view, the JSON lines file is the record."""
    try:
        write_csv_view(path.with_suffix(".csv"), read_marks(path, include_retracted=True))
    except OSError as error:
        print(f"marks saved to {path.name}; could not refresh {path.with_suffix('.csv').name}: "
              f"{error}", file=sys.stderr)


def append_mark(path: Path, mark: Mark) -> Mark:
    _append(path, mark.to_json())
    _refresh_csv(path)
    return mark


def retract_mark(path: Path, mark_id: str, reason: str = "undo") -> None:
    _append(path, json.dumps({"record": "retract", "of": mark_id,
                              "reason": reason, "at": now_iso()}))
    _refresh_csv(path)


def append_session(path: Path, well: str, labels_sha256: str,
                   frames_viewed: Iterable[int], marks_made: list[str],
                   started: str, ended: str) -> None:
    _append(path, json.dumps({
        "record": "session", "well": well, "labels_sha256": labels_sha256,
        "started": started, "ended": ended,
        "review_frames_viewed": sorted(set(int(f) for f in frames_viewed)),
        "marks_made": list(marks_made)}))


def read_sessions(path: Path) -> list[dict]:
    return [r for r in read_records(path) if r.get("record") == "session"]


CSV_COLUMNS = ("mark_id", "kind", "status", "review_frames", "imagej_frames",
               "identities", "expected_identity", "points_yx", "sentence",
               "note", "made_at", "labels_sha256", "origin")


def write_csv_view(path: Path, marks: list[Mark]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _write_csv_rows(tmp, marks)
    # Dropbox holds a short lock on a freshly synced file; retry the swap.
    import time
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    try:
        _write_csv_rows(path, marks)       # last resort: write in place
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _write_csv_rows(path: Path, marks: list[Mark]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for mark in marks:
            writer.writerow({
                "mark_id": mark.mark_id,
                "kind": mark.kind,
                "status": "retracted" if mark.retracted else "active",
                "review_frames": "|".join(str(f) for f in mark.frames()),
                "imagej_frames": "|".join(str(f + mark.source_frame_offset)
                                          for f in mark.frames()),
                "identities": "|".join(str(i) for i in mark.identities),
                "expected_identity": "" if mark.expected_identity is None
                else mark.expected_identity,
                "points_yx": "|".join(f"{a.review_frame}:{a.y},{a.x}" for a in mark.anchors),
                "sentence": mark.sentence,
                "note": mark.note,
                "made_at": mark.made_at,
                "labels_sha256": mark.labels_sha256[:12],
                "origin": mark.origin,
            })


# ------------------------------------------------------------------ wells

@dataclass
class WellPaths:
    well: str
    well_dir: Path
    labels: Path
    unclaimed: Path | None
    raw: Path
    accepted_parent: str
    source_frame_offset: int
    labels_sha256: str
    labels_source: str = "accepted"     # "accepted" or a review-set name

    @property
    def marks(self) -> Path:
        return self.well_dir / "feedback" / "marks.jsonl"


def list_wells(root: Path) -> list[Path]:
    issues = root / "issues"
    if not issues.is_dir():
        return []
    return sorted(p for p in issues.iterdir()
                  if p.is_dir() and re.fullmatch(r"\d\d_[A-Za-z0-9_-]+", p.name)
                  and (p / "WELL_BASELINE.json").is_file())


REVIEW_SET_SCHEMA = "motion.feedback-review-set"


def review_sets_dir(root: Path) -> Path:
    return root / "issues" / "review_sets"


def list_review_sets(root: Path) -> list[str]:
    folder = review_sets_dir(root)
    return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []


def read_review_set(root: Path, name: str) -> dict:
    path = review_sets_dir(root) / f"{name}.json"
    if not path.is_file():
        available = ", ".join(list_review_sets(root)) or "none"
        raise FileNotFoundError(f"no review set {name!r}; available: {available}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != REVIEW_SET_SCHEMA:
        raise ValueError(f"{path} is not a {REVIEW_SET_SCHEMA} file")
    return data


def build_review_set(root: Path, name: str, runs_dir: Path, description: str,
                     source: str = "", verify_raw: bool = True) -> dict:
    """Scan ``runs_dir`` for per-well runs (``*_<stem>/out/<stem>.tif``) and record
    one label stack per well, with its fingerprint and the run that produced it."""
    import tifffile
    runs_dir = Path(runs_dir)
    runs_dir = runs_dir if runs_dir.is_absolute() else (root / runs_dir)
    runs_dir = runs_dir.resolve()
    root = root.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"no such runs folder: {runs_dir}")
    wells: dict[str, dict] = {}
    for well_dir in list_wells(root):
        baseline = json.loads((well_dir / "WELL_BASELINE.json").read_text(encoding="utf-8"))
        stem = baseline["well"]
        matches = sorted(p for p in runs_dir.glob(f"*_{stem}")
                         if (p / "out" / f"{stem}.tif").is_file())
        if not matches:
            continue
        run = matches[-1]
        labels = run / "out" / f"{stem}.tif"
        entry = {"labels": str(labels.relative_to(root)),
                 "sha256": sha256_of(labels),
                 "run": str(run.relative_to(root)),
                 "source_frame_offset": len(baseline.get("source_frames_removed") or []),
                 "accepted_parent": str(baseline["accepted_parent"])}
        unclaimed = run / "out" / f"{stem}_unclaimed_original_ids.tif"
        if unclaimed.is_file():
            entry["unclaimed"] = str(unclaimed.relative_to(root))
        manifest = run / "run.json"
        if manifest.is_file():
            record = json.loads(manifest.read_text(encoding="utf-8"))
            entry["attempt"] = record.get("attempt", "")
            entry["targeting_mode"] = record.get("targeting_mode", "")
        if verify_raw:
            raw = root / "registered inputs" / f"{stem}.tif"
            shape = tifffile.TiffFile(labels).series[0].shape
            raw_shape = tifffile.TiffFile(raw).series[0].shape
            if tuple(shape) != tuple(raw_shape):
                raise ValueError(f"{stem}: labels {shape} do not match raw {raw_shape}")
            entry["shape"] = list(shape)
        wells[stem] = entry
    if not wells:
        raise ValueError(f"no per-well outputs found under {runs_dir}")
    data = {"schema": REVIEW_SET_SCHEMA, "schema_version": 1, "name": name,
            "description": description, "source": source or str(runs_dir.relative_to(root)),
            "created_at": now_iso(), "wells": wells}
    folder = review_sets_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def well_paths(root: Path, well: str, labels: Path | None = None,
               raw: Path | None = None, verify: bool = True,
               review_set: str | None = None) -> WellPaths:
    """Resolve the accepted stacks for ``well`` (a stem such as ``95_B4`` or a
    well folder name such as ``09_95_B4``)."""
    candidates = [p for p in list_wells(root)
                  if p.name == well or p.name.split("_", 1)[1] == well]
    if not candidates:
        raise FileNotFoundError(f"no well folder with WELL_BASELINE.json for {well!r}")
    well_dir = candidates[0]
    baseline = json.loads((well_dir / "WELL_BASELINE.json").read_text(encoding="utf-8"))
    stem = baseline["well"]
    parent = str(baseline["accepted_parent"]).replace("\\", "/")
    parent_dir = root / parent
    accepted_labels = parent_dir / "out" / f"{stem}.tif"
    unclaimed = parent_dir / "out" / f"{stem}_unclaimed_original_ids.tif"
    source = "accepted"
    if review_set and labels is None:
        data = read_review_set(root, review_set)
        entry = data["wells"].get(stem)
        if entry is None:
            raise FileNotFoundError(f"review set {review_set!r} has no stack for {stem}")
        labels = root / str(entry["labels"]).replace("\\", "/")
        if entry.get("unclaimed"):
            unclaimed = root / str(entry["unclaimed"]).replace("\\", "/")
        digest = sha256_of(labels)
        if entry.get("sha256") and digest != entry["sha256"]:
            raise ValueError(f"{stem}: {labels} fingerprint {digest[:12]} differs from "
                             f"review set {review_set!r} ({entry['sha256'][:12]})")
        source = review_set
        verify = False
    labels = Path(labels) if labels else accepted_labels
    if source == "accepted" and labels != accepted_labels:
        source = "custom"
    raw = Path(raw) if raw else root / "registered inputs" / f"{stem}.tif"
    for name, path in (("labels", labels), ("raw", raw)):
        if not path.is_file():
            raise FileNotFoundError(f"{stem}: {name} stack missing at {path}")
    offset = len(baseline.get("source_frames_removed") or [])
    digest = sha256_of(labels)
    expected = baseline.get("labels_sha256")
    if verify and expected and labels == accepted_labels and digest != expected:
        raise ValueError(f"{stem}: accepted labels fingerprint {digest[:12]} differs "
                         f"from WELL_BASELINE.json {expected[:12]}")
    return WellPaths(stem, well_dir, labels, unclaimed if unclaimed.is_file() else None,
                     raw, parent, offset, digest, source)


# ------------------------------------------------------------- constructors

class MarkBuilder:
    """Builds marks against one label stack, resolving auto anchors."""

    def __init__(self, well: str, labels: np.ndarray, labels_sha256: str,
                 source_frame_offset: int, presence: Presence | None = None,
                 origin: str = "ui"):
        self.well = well
        self.labels = labels
        self.sha = labels_sha256
        self.offset = source_frame_offset
        self.presence = presence or Presence(labels)
        self.origin = origin

    # anchors -------------------------------------------------------------
    def cell_anchor(self, frame_idx: int, identity: int, role: str = "body",
                    click: tuple[int, int] | None = None) -> Anchor | None:
        mask = self.labels[frame_idx] == identity
        if not mask.any():
            return None
        y, x = interior_point(mask)
        return Anchor(frame_idx + 1, frame_idx + 1 + self.offset, int(identity),
                      y, x, int(mask.sum()), role,
                      None if click is None else int(click[0]),
                      None if click is None else int(click[1]))

    def point_anchor(self, frame_idx: int, y: int, x: int, role: str = "point") -> Anchor:
        identity = int(self.labels[frame_idx, y, x])
        return Anchor(frame_idx + 1, frame_idx + 1 + self.offset, identity,
                      int(y), int(x), 0, role, int(y), int(x))

    def reference_anchor(self, identity: int, before_idx: int) -> Anchor | None:
        """Where ``identity`` was last seen before ``before_idx`` (or, failing
        that, nearest to it)."""
        frame = self.presence.last_before(identity, before_idx)
        if frame is None:
            frame = self.presence.nearest(identity, before_idx)
        if frame is None:
            return None
        return self.cell_anchor(frame, identity, role="reference")

    def _new(self, mark_id: str, kind: str, anchors: list[Anchor], **extra) -> Mark:
        identities = []
        for anchor in anchors:
            if anchor.identity and anchor.identity not in identities:
                identities.append(anchor.identity)
        mark = Mark(mark_id, self.well, kind, now_iso(), self.sha, self.offset,
                    anchors=anchors, identities=identities, origin=self.origin, **extra)
        mark.sentence = sentence(mark)
        return mark

    # kinds ---------------------------------------------------------------
    def becomes(self, mark_id: str, body: Anchor, true_identity: int) -> Mark:
        """The body ``body`` (labelled body.identity) is really ``true_identity``."""
        anchors = [body]
        must_match: list[list[int]] = []
        unresolved = []
        reference = self.reference_anchor(true_identity, body.frame_idx)
        if reference is None:
            unresolved.append(f"identity {true_identity} never appears in this stack")
        else:
            anchors.append(reference)
            must_match.append([0, 1])
        return self._new(mark_id, "becomes", anchors, must_match=must_match,
                         expected_identity=int(true_identity), unresolved=unresolved)

    def swap(self, mark_id: str, first: Anchor, second: Anchor) -> Mark:
        """``first`` and ``second`` (same frame) carry each other's number."""
        anchors = [first, second]
        must_match: list[list[int]] = []
        unresolved = []
        frame_idx = max(first.frame_idx, second.frame_idx)
        for body_index, body in enumerate((first, second)):
            other = second if body is first else first
            reference = self.reference_anchor(other.identity, frame_idx)
            if reference is None or reference.frame_idx >= frame_idx:
                unresolved.append(f"identity {other.identity} has no earlier frame "
                                  f"to serve as reference")
                continue
            anchors.append(reference)
            must_match.append([body_index, len(anchors) - 1])
        return self._new(mark_id, "swap", anchors, must_match=must_match,
                         unresolved=unresolved)

    def same(self, mark_id: str, first: Anchor, second: Anchor) -> Mark:
        kind = "join" if first.review_frame == second.review_frame else "same"
        return self._new(mark_id, kind, [first, second], must_match=[[0, 1]])

    def different(self, mark_id: str, first: Anchor, second: Anchor) -> Mark:
        return self._new(mark_id, "different", [first, second], must_differ=[[0, 1]])

    def persist(self, mark_id: str, identity: int, kind: str = "persist",
                checked_frame_idx: int | None = None) -> Mark:
        """The physical body carrying ``identity`` at ``checked_frame_idx`` must
        carry one number in every frame. The body is followed forwards and
        backwards by outline overlap, whatever number it carries there, so the
        claim is not circular on the stack it was made against."""
        frames = self.presence.frames.get(int(identity), [])
        if not frames:
            raise ValueError(f"identity {identity} is not in this stack")
        if checked_frame_idx is None or checked_frame_idx not in frames:
            checked_frame_idx = frames[0]
        chain = follow_body(self.labels, checked_frame_idx, int(identity))
        anchors = []
        for frame_idx, mask in chain:
            y, x = interior_point(mask)
            anchors.append(Anchor(frame_idx + 1, frame_idx + 1 + self.offset,
                                  int(self.labels[frame_idx, y, x]), y, x, int(mask.sum()),
                                  "body" if frame_idx == checked_frame_idx else "control"))
        start = anchors.index(next(a for a in anchors if a.role == "body"))
        return self._new(mark_id, kind, anchors,
                         must_match=[[start, i] for i in range(len(anchors)) if i != start],
                         expected_identity=int(identity))

    def missing(self, mark_id: str, frame_idx: int, y: int, x: int,
                identity: int | None = None) -> Mark:
        point = self.point_anchor(frame_idx, y, x)
        anchors = [point]
        must_match: list[list[int]] = []
        unresolved = []
        expected = None
        if identity:
            expected = int(identity)
            reference = self.reference_anchor(expected, frame_idx)
            if reference is None:
                unresolved.append(f"identity {expected} never appears in this stack")
            else:
                anchors.append(reference)
                must_match.append([0, 1])
        return self._new(mark_id, "missing", anchors, must_exist=[0],
                         must_match=must_match, expected_identity=expected,
                         unresolved=unresolved)

    def duplicate(self, mark_id: str, real: Anchor,
                  impostors: list[Anchor] | None = None) -> Mark:
        """``real`` is the true body of its identity; every other separated
        component with that number in the same frame is a wrong copy."""
        from scipy import ndimage as ndi
        if impostors is None:
            plane = self.labels[real.frame_idx] == real.identity
            components, count = ndi.label(plane)
            own = components[real.y, real.x]
            impostors = []
            for component in range(1, count + 1):
                if component == own:
                    continue
                mask = components == component
                y, x = interior_point(mask)
                impostors.append(Anchor(real.review_frame, real.imagej_frame,
                                        real.identity, y, x, int(mask.sum()), "body"))
        anchors = [real] + list(impostors)
        return self._new(mark_id, "duplicate", anchors,
                         must_differ=[[0, i] for i in range(1, len(anchors))],
                         zero_ok=list(range(1, len(anchors))),
                         expected_identity=int(real.identity),
                         unresolved=[] if impostors else
                         [f"identity {real.identity} is one connected body in this frame"])

    def split(self, mark_id: str, body: Anchor,
              line: tuple[tuple[int, int], tuple[int, int]] | None = None) -> Mark:
        mask = self.labels[body.frame_idx] == body.identity
        ys, xs = np.nonzero(mask)
        region = {"review_frame": body.review_frame, "identity": body.identity,
                  "area": int(mask.sum()),
                  "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]}
        anchors = [body]
        must_differ: list[list[int]] = []
        stored_line = None
        if line is not None:
            (y0, x0), (y1, x1) = line
            stored_line = [[int(y0), int(x0)], [int(y1), int(x1)]]
            sides = _line_sides(mask, (y0, x0), (y1, x1))
            for y, x in sides:
                anchors.append(Anchor(body.review_frame, body.imagej_frame,
                                      body.identity, int(y), int(x), 0, "side"))
            if len(sides) == 2:
                must_differ.append([1, 2])
        return self._new(mark_id, "split", anchors, must_differ=must_differ,
                         must_exist=list(range(1, len(anchors))),
                         region=region, line=stored_line)

    def note(self, mark_id: str, frame_idx: int, text: str,
             about: list[Anchor] | None = None) -> Mark:
        anchors = list(about or [])
        if not anchors:
            anchors = [Anchor(frame_idx + 1, frame_idx + 1 + self.offset, 0, 0, 0, 0, "point")]
        return self._new(mark_id, "note", anchors, note=text)


def follow_body(labels: np.ndarray, frame_idx: int, identity: int,
                min_overlap: float = 0.2, max_gap: int = 2) -> list[tuple[int, np.ndarray]]:
    """Follow the physical body under ``identity`` at ``frame_idx`` through the
    stack by outline overlap, ignoring the numbers it carries. Returns
    ``(frame_idx, mask)`` pairs in frame order. The chain stops after
    ``max_gap`` consecutive frames without an overlapping outline."""
    chain = {frame_idx: labels[frame_idx] == identity}
    if not chain[frame_idx].any():
        return []
    n_frames = labels.shape[0]
    for direction in (1, -1):
        mask = chain[frame_idx]
        frame = frame_idx
        gap = 0
        while True:
            frame += direction
            if not 0 <= frame < n_frames:
                break
            plane = labels[frame]
            values = plane[mask]
            values = values[values > 0]
            if values.size == 0:
                gap += 1
                if gap > max_gap:
                    break
                continue
            best, count = Counter(values.tolist()).most_common(1)[0]
            if count / float(mask.sum()) < min_overlap:
                gap += 1
                if gap > max_gap:
                    break
                continue
            gap = 0
            mask = plane == best
            chain[frame] = mask
    return sorted(chain.items())


def _line_sides(mask: np.ndarray, start: tuple[int, int],
                end: tuple[int, int]) -> list[tuple[int, int]]:
    """Two points inside ``mask`` on opposite sides of the drawn cut line: the
    mask pixels farthest from the line on each side."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    (y0, x0), (y1, x1) = start, end
    dy, dx = y1 - y0, x1 - x0
    norm = float(np.hypot(dy, dx))
    if norm < 1e-6:
        return []
    signed = ((ys - y0) * dx - (xs - x0) * dy) / norm
    sides = []
    top = int(np.argmax(signed))
    if signed[top] > 0.5:
        sides.append((int(ys[top]), int(xs[top])))
    bottom = int(np.argmin(signed))
    if signed[bottom] < -0.5:
        sides.append((int(ys[bottom]), int(xs[bottom])))
    return sides


# ------------------------------------------------------------------ sentences

def _f(anchor: Anchor) -> str:
    return f"review frame {anchor.review_frame} (ImageJ {anchor.imagej_frame})"


def _xy(anchor: Anchor) -> str:
    return f"x={anchor.x}, y={anchor.y}"


def sentence(mark: Mark) -> str:
    a = mark.anchors
    kind = mark.kind
    if kind == "becomes":
        body = a[0]
        text = (f"{mark.expected_identity} becomes {body.identity} at {_f(body)}: "
                f"the body labelled {body.identity} there ({_xy(body)}) is really "
                f"{mark.expected_identity}")
        if len(a) > 1:
            text += f", last seen as itself at {_f(a[1])}"
        return text + "."
    if kind == "swap":
        first, second = a[0], a[1]
        text = (f"{first.identity} and {second.identity} are swapped from "
                f"{_f(first)}: the body labelled {first.identity} ({_xy(first)}) is "
                f"really {second.identity} and the body labelled {second.identity} "
                f"({_xy(second)}) is really {first.identity}")
        refs = [r for r in a[2:] if r.role == "reference"]
        if refs:
            text += "; before that " + " and ".join(
                f"{r.identity} was last correct at review frame {r.review_frame}" for r in refs)
        return text + "."
    if kind == "same":
        return (f"The cell labelled {a[0].identity} at {_f(a[0])} ({_xy(a[0])}) and the "
                f"cell labelled {a[1].identity} at {_f(a[1])} ({_xy(a[1])}) are the same "
                f"cell.")
    if kind == "join":
        return (f"The outlines labelled {a[0].identity} and {a[1].identity} at "
                f"{_f(a[0])} are one cell.")
    if kind == "different":
        return (f"The cell labelled {a[0].identity} at {_f(a[0])} ({_xy(a[0])}) and the "
                f"cell labelled {a[1].identity} at {_f(a[1])} ({_xy(a[1])}) are "
                f"different cells.")
    if kind in ("persist", "ok"):
        frames = mark.frames()
        span = f"review frames {frames[0]}-{frames[-1]}" if frames else "no frames"
        checked = [x for x in a if x.role == "body"]
        where = f" at {_f(checked[0])}" if checked else ""
        others = sorted({x.identity for x in a if x.identity and
                         x.identity != mark.expected_identity})
        carried = (f"; followed by outline overlap it currently carries "
                   f"{', '.join(str(i) for i in others)} in other frames") if others else ""
        if kind == "persist":
            return (f"The cell labelled {mark.expected_identity}{where} is one cell "
                    f"that must keep one number and never disappear (body followed over "
                    f"{len(frames)} frames, {span}{carried}).")
        return (f"The cell labelled {mark.expected_identity}{where} is tracked "
                f"correctly (body followed over {len(frames)} frames, {span}{carried}).")
    if kind == "missing":
        point = a[0]
        text = f"There is a visible cell with no outline at {_xy(point)} in {_f(point)}"
        if mark.expected_identity:
            text += f"; it is identity {mark.expected_identity}"
            if len(a) > 1:
                text += f" (last seen at {_f(a[1])})"
        return text + "."
    if kind == "duplicate":
        real = a[0]
        copies = ", ".join(_xy(c) for c in a[1:]) or "none found"
        return (f"Identity {real.identity} is on {len(a)} separate bodies at "
                f"{_f(real)}: the real one is at {_xy(real)}; wrong copies at {copies}.")
    if kind == "split":
        body = a[0]
        area = mark.region["area"] if mark.region else body.area
        text = (f"The outline labelled {body.identity} at {_f(body)} ({_xy(body)}, "
                f"{area} px) covers two cells")
        if mark.line:
            (y0, x0), (y1, x1) = mark.line
            text += f"; cut along the line from (x={x0}, y={y0}) to (x={x1}, y={y1})"
        return text + "."
    if kind == "note":
        about = ", ".join(str(i) for i in mark.identities)
        where = f"at review frame {a[0].review_frame}" if a else ""
        return f"Note {where}{' about ' + about if about else ''}: {mark.note}"
    return f"{kind}: {mark.note}"


# ------------------------------------------------------------------ evaluation

def evaluate(mark: Mark, candidate: np.ndarray) -> dict:
    """Test one mark against a candidate label stack of the same shape.

    Returns ``status`` satisfied / violated / undecidable / not-testable and a
    ``detail`` string. Undecidable means an anchor no longer sits inside any
    candidate outline, so the claim cannot be checked by position."""
    if mark.kind == "note":
        return _result(mark, "not-testable", "free text", [])
    if candidate.shape[0] < max((a.review_frame for a in mark.anchors), default=0):
        return _result(mark, "not-testable", "candidate has fewer frames than the mark", [])
    found = [label_at(candidate, a.frame_idx, a.y, a.x) for a in mark.anchors]
    problems: list[str] = []
    undecided: list[str] = []
    checks = 0

    for index in mark.must_exist:
        checks += 1
        if found[index] == 0:
            problems.append(f"no outline at anchor {index}")

    if mark.kind in ("persist", "ok"):
        nonzero = [v for v in found if v]
        if not nonzero:
            return _result(mark, "undecidable", "identity absent everywhere in candidate", found)
        modal = Counter(nonzero).most_common(1)[0][0]
        lost = sum(1 for v in found if v == 0)
        relabelled = sum(1 for v in nonzero if v != modal)
        detail = (f"{len(found)} anchor frames: {len(found) - lost - relabelled} keep one "
                  f"number ({modal}), {relabelled} carry another number, {lost} have no outline")
        status = "satisfied" if lost == 0 and relabelled == 0 else "violated"
        return _result(mark, status, detail, found)

    for i, j in mark.must_match:
        checks += 1
        if found[i] == 0 or found[j] == 0:
            undecided.append(f"anchor {i if found[i] == 0 else j} outside candidate foreground")
        elif found[i] != found[j]:
            problems.append(f"anchors {i} and {j} carry {found[i]} and {found[j]}")
    for i, j in mark.must_differ:
        checks += 1
        zero_ok = i in mark.zero_ok or j in mark.zero_ok
        if found[i] == 0 or found[j] == 0:
            if not zero_ok:
                undecided.append(f"anchor {i if found[i] == 0 else j} outside candidate foreground")
        elif found[i] == found[j]:
            problems.append(f"anchors {i} and {j} both carry {found[i]}")

    if mark.kind == "split" and not mark.must_differ and mark.region:
        checks += 1
        distinct = _split_region_distinct(mark, candidate)
        if distinct < 2:
            problems.append("outline region still carries a single number")

    if checks == 0:
        return _result(mark, "undecidable", "; ".join(mark.unresolved) or
                       "no positional check could be built", found)
    if problems:
        return _result(mark, "violated", "; ".join(problems), found)
    if undecided:
        return _result(mark, "undecidable", "; ".join(undecided), found)
    return _result(mark, "satisfied", "all positional checks hold", found)


def _split_region_distinct(mark: Mark, candidate: np.ndarray) -> int:
    region = mark.region
    y0, x0, y1, x1 = region["bbox"]
    window = candidate[region["review_frame"] - 1, y0:y1 + 1, x0:x1 + 1]
    return int(np.unique(window[window > 0]).size)


def _result(mark: Mark, status: str, detail: str, found: list[int]) -> dict:
    return {"mark_id": mark.mark_id, "kind": mark.kind, "status": status,
            "detail": detail, "candidate_labels": found, "sentence": mark.sentence}


def evaluate_all(marks: list[Mark], candidate: np.ndarray) -> list[dict]:
    return [evaluate(m, candidate) for m in marks]


def summarise(results: list[dict]) -> dict[str, int]:
    counts = Counter(r["status"] for r in results)
    return {k: counts.get(k, 0) for k in ("satisfied", "violated", "undecidable",
                                           "not-testable")}


# ------------------------------------------------------------------ exports

def to_review_cases(marks: list[Mark]) -> list[dict]:
    rows = []
    for mark in marks:
        if mark.kind == "note":
            continue
        frames = mark.frames()
        body_frames = [a.review_frame for a in mark.anchors
                       if a.role in ("body", "point", "side")]
        start = min(body_frames or frames)
        end = max(body_frames or frames)
        rows.append({
            "case_id": mark.mark_id,
            "case_type": KINDS[mark.kind][1],
            "mechanism": mark.kind,
            "review_frame_start": start,
            "review_frame_end": end,
            "source_imagej_frame_start": start + mark.source_frame_offset,
            "source_imagej_frame_end": end + mark.source_frame_offset,
            "track_ref": "",
            "identities": "|".join(str(i) for i in mark.identities),
            "expected_identity": "" if mark.expected_identity is None else mark.expected_identity,
            "notes": (mark.sentence + (" " + mark.note if mark.note else "")).strip(),
        })
    return rows


def digest(marks: list[Mark], results: list[dict] | None = None,
           sessions: list[dict] | None = None) -> str:
    by_id = {r["mark_id"]: r for r in results or []}
    lines = []
    if not marks:
        lines.append("No active marks.")
    for kind in KINDS:
        group = [m for m in marks if m.kind == kind]
        if not group:
            continue
        lines.append(f"## {kind}: {KINDS[kind][0]} ({len(group)})")
        for mark in group:
            line = f"- {mark.mark_id}: {mark.sentence}"
            if mark.note and mark.kind != "note":
                line += f" Note: {mark.note}"
            if mark.unresolved:
                line += f" [unresolved: {'; '.join(mark.unresolved)}]"
            if mark.mark_id in by_id:
                r = by_id[mark.mark_id]
                line += f" -> {r['status'].upper()} ({r['detail']})"
            lines.append(line)
        lines.append("")
    stacks = Counter(m.labels_sha256[:12] for m in marks)
    if len(stacks) > 1:
        lines.append("Marks were made against more than one label stack "
                     + ", ".join(f"{sha} ({count})" for sha, count in stacks.most_common())
                     + ". Every check is positional, so a mark still applies to a stack "
                     "that numbers the cells differently.")
    if sessions:
        viewed = set()
        for session in sessions:
            viewed.update(session.get("review_frames_viewed", []))
        lines.append(f"Frames looked at across {len(sessions)} session(s): "
                     f"{_ranges(sorted(viewed))}. Anything on those frames that is not "
                     f"marked was not reported as wrong.")
    return "\n".join(lines).rstrip() + "\n"


def mark_counts(root: Path) -> dict[str, int]:
    """Active marks per well stem, for progress display."""
    counts = {}
    for well_dir in list_wells(root):
        stem = well_dir.name.split("_", 1)[1]
        counts[stem] = len(read_marks(well_dir / "feedback" / "marks.jsonl"))
    return counts


def _ranges(values: list[int]) -> str:
    if not values:
        return "none"
    parts = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = value
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(parts)


# ------------------------------------------------------------------ text form

_FRAME = (r"(?:@|at frame|in frame|from frame|at|frame|from|f|in)\s*"
          r"(?P<ij>ij|imagej|i)?\s*(?P<frame>\d+)")
_PATTERNS = [
    ("swap", re.compile(
        r"^(?P<a>\d+)\s*(?:<>|<->|swaps? with|swapped with|and)\s*(?P<b>\d+)\s*"
        r"(?:swap(?:ped)?|are swapped|exchange)?\s*" + _FRAME + r"$")),
    ("becomes", re.compile(
        r"^(?P<a>\d+)\s*(?:>|->|=>|becomes|became|turns? (?:in)?to|flips? to|swaps? to|"
        r"switche?s? to|goes to|taken over by|is taken over by|to)\s*(?P<b>\d+)\s*"
        + _FRAME + r"$")),
    ("join", re.compile(
        r"^(?P<a>\d+)\s*(?:=|\+|joins?|is|are one cell with|same as)\s*(?P<b>\d+)\s*"
        r"(?:are one cell|same cell)?\s*" + _FRAME + r"$")),
    ("persist", re.compile(
        r"^(?P<a>\d+)\s*(?:persists?|stays?(?: \d+)?|should (?:be|stay) \d+"
        r"(?: for)?(?: the)?(?: whole| entire| full)?(?: video| time| recording)?|"
        r"whole video|entire video)$")),
    ("missing", re.compile(
        r"^(?:missing|undetected|no outline|cell)\s*(?:cell\s*)?(?:at\s*)?"
        r"\(?\s*(?P<x>\d+)\s*[, ]\s*(?P<y>\d+)\s*\)?\s*(?:missing|undetected|no outline)?\s*"
        + _FRAME + r"(?:\s*(?:is|=)\s*(?P<a>\d+))?$")),
    ("split", re.compile(
        r"^(?P<a>\d+)\s*(?:should split|split|is two|is 2|two|contains two|/)\s*"
        + _FRAME + r"$")),
    ("duplicate", re.compile(
        r"^(?P<a>\d+)\s*(?:dup|duplicate|duplicated|is in two places|twice|x2)\s*"
        + _FRAME + r"$")),
]


def parse_text(text: str, builder: MarkBuilder, first_id: int = 1,
               frame_convention: str = "review") -> tuple[list[Mark], list[str]]:
    """Turn shorthand or plain sentences into marks anchored on ``builder``'s
    stack. Frames are review frames unless written ``ij69`` or the convention
    is ``imagej``. Coordinates are ImageJ order ``x,y``.

    Returns the marks and the segments that could not be parsed."""
    marks: list[Mark] = []
    failures: list[str] = []
    counter = first_id
    for segment in re.split(r"[;\n]|\.\s+|,\s*(?=\d+\s*(?:<>|->|>|=|and))", text):
        segment = segment.strip().strip(".").lower()
        if not segment:
            continue
        segment = re.sub(r"\b(?:cell|cells|identity|identities|the)\b", "", segment)
        segment = re.sub(r"\s+", " ", segment).strip()
        parsed = None
        for kind, pattern in _PATTERNS:
            match = pattern.match(segment)
            if match:
                parsed = (kind, match.groupdict())
                break
        if parsed is None:
            failures.append(segment)
            continue
        kind, groups = parsed
        frame_idx = None
        if groups.get("frame") is not None:
            frame = int(groups["frame"])
            if groups.get("ij") or frame_convention == "imagej":
                frame -= builder.offset
            frame_idx = frame - 1
            if not 0 <= frame_idx < builder.labels.shape[0]:
                failures.append(f"{segment} (frame out of range)")
                continue
        mark_id = f"{builder.well}-M{counter:04d}"
        try:
            mark = _text_mark(builder, mark_id, kind, groups, frame_idx)
        except ValueError as error:
            failures.append(f"{segment} ({error})")
            continue
        mark.origin = "text"
        mark.note = mark.note or f"typed: {segment}"
        marks.append(mark)
        counter += 1
    return marks, failures


def _text_mark(builder: MarkBuilder, mark_id: str, kind: str, groups: dict,
               frame_idx: int | None) -> Mark:
    def body(identity: int) -> Anchor:
        anchor = builder.cell_anchor(frame_idx, identity)
        if anchor is None:
            raise ValueError(f"identity {identity} is not in review frame {frame_idx + 1}")
        return anchor

    a = int(groups["a"]) if groups.get("a") else None
    b = int(groups["b"]) if groups.get("b") else None
    if kind == "becomes":
        return builder.becomes(mark_id, body(b), a)
    if kind == "swap":
        return builder.swap(mark_id, body(a), body(b))
    if kind == "join":
        return builder.same(mark_id, body(a), body(b))
    if kind == "persist":
        if a not in builder.presence.frames:
            raise ValueError(f"identity {a} is not in this stack")
        return builder.persist(mark_id, a)
    if kind == "missing":
        x, y = int(groups["x"]), int(groups["y"])
        if not (0 <= y < builder.labels.shape[1] and 0 <= x < builder.labels.shape[2]):
            raise ValueError("point outside the field")
        return builder.missing(mark_id, frame_idx, y, x, a)
    if kind == "split":
        return builder.split(mark_id, body(a))
    if kind == "duplicate":
        return builder.duplicate(mark_id, body(a))
    raise ValueError(f"unknown kind {kind}")


# ------------------------------------------------------------------ CLI

def _load_labels(path: Path) -> np.ndarray:
    import tifffile
    stack = tifffile.imread(path)
    if stack.ndim != 3:
        raise SystemExit(f"{path}: expected a frame stack, got shape {stack.shape}")
    return stack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("wells", help="list wells that can take feedback")
    sub.add_parser("review-sets", help="list saved review sets")
    make = sub.add_parser("make-review-set",
                          help="record one label stack per well as a named review set")
    make.add_argument("--name", required=True)
    make.add_argument("--runs", type=Path, required=True,
                      help="folder holding per-well runs named *_<stem> with out/<stem>.tif")
    make.add_argument("--description", default="")
    for name in ("digest", "evaluate", "to-cases", "parse"):
        p = sub.add_parser(name)
        p.add_argument("--well", required=True)
        p.add_argument("--labels", type=Path, help="label stack the marks refer to "
                       "(default: the well's accepted labels)")
        p.add_argument("--review-set", help="use this named review set's stack for the well")
        if name == "evaluate":
            p.add_argument("--candidate", type=Path, required=True)
            p.add_argument("--json", type=Path, help="write results here")
        if name == "digest":
            p.add_argument("--candidate", type=Path)
        if name == "to-cases":
            p.add_argument("--output", type=Path, required=True)
        if name == "parse":
            p.add_argument("text")
            p.add_argument("--save", action="store_true",
                           help="append the parsed marks to the well's marks file")
            p.add_argument("--imagej-frames", action="store_true",
                           help="frame numbers in the text are ImageJ slices")
    args = parser.parse_args(argv)
    root = find_motion_root(Path.cwd())

    if args.command == "wells":
        counts = mark_counts(root)
        for well_dir in list_wells(root):
            stem = well_dir.name.split("_", 1)[1]
            print(f"{well_dir.name:14} {counts.get(stem, 0):>4} marks")
        return 0
    if args.command == "review-sets":
        names = list_review_sets(root)
        if not names:
            print("no review sets; make one with make-review-set")
        for name in names:
            data = read_review_set(root, name)
            print(f"{name}: {data.get('description') or data.get('source')} "
                  f"({len(data['wells'])} wells, created {data.get('created_at', '?')[:10]})")
            for stem, entry in data["wells"].items():
                print(f"    {stem:8} {entry['sha256'][:12]}  {entry.get('attempt', '')}  "
                      f"{entry['labels']}")
        return 0
    if args.command == "make-review-set":
        data = build_review_set(root, args.name, args.runs, args.description)
        print(f"wrote {review_sets_dir(root) / (args.name + '.json')} "
              f"with {len(data['wells'])} wells")
        for stem, entry in data["wells"].items():
            print(f"    {stem:8} {entry['sha256'][:12]}  {entry['labels']}")
        return 0

    paths = well_paths(root, args.well, labels=args.labels,
                       verify=args.labels is None,
                       review_set=getattr(args, "review_set", None))
    marks = read_marks(paths.marks)

    if args.command == "digest":
        results = None
        if args.candidate:
            results = evaluate_all(marks, _load_labels(args.candidate))
        print(digest(marks, results, read_sessions(paths.marks)))
        return 0
    if args.command == "evaluate":
        results = evaluate_all(marks, _load_labels(args.candidate))
        if args.json:
            args.json.write_text(json.dumps({"candidate": str(args.candidate),
                                             "summary": summarise(results),
                                             "results": results}, indent=2),
                                 encoding="utf-8")
        for r in results:
            print(f"{r['mark_id']:14} {r['status']:12} {r['kind']:10} {r['detail']}")
        print(json.dumps(summarise(results)))
        return 0
    if args.command == "to-cases":
        rows = to_review_cases(marks)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_CASE_COLUMNS,
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} review cases to {args.output}")
        return 0
    if args.command == "parse":
        labels = _load_labels(paths.labels)
        builder = MarkBuilder(paths.well, labels, paths.labels_sha256,
                              paths.source_frame_offset, origin="text")
        start = int(next_mark_id(paths.marks, paths.well).rsplit("M", 1)[1])
        parsed, failures = parse_text(args.text, builder, start,
                                      "imagej" if args.imagej_frames else "review")
        for mark in parsed:
            print(f"{mark.mark_id}: {mark.sentence}")
            if mark.unresolved:
                print(f"    unresolved: {'; '.join(mark.unresolved)}")
            if args.save:
                append_mark(paths.marks, mark)
        for failure in failures:
            print(f"could not parse: {failure}")
        if args.save and parsed:
            print(f"saved {len(parsed)} marks to {paths.marks}")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
