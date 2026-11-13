"""Feedback marks inside manual editing: check them, and turn them into edits.

Marks (``feedback_marks.py``) record Jamie's verdicts by pixel position. This
module reads them against an editing session's current labels and proposes
the manual-editing operations that would satisfy them.

A proposal is tried on an in-memory copy first. An edit enters the plan only
when it satisfies its own mark and leaves every mark that already held still
holding; anything else is reported for a person to resolve in the editor.
Nothing here writes a session: ``apply_plan`` hands the accepted operations to
the editor's own immutable save path.

Frame convention: the editor numbers frames by position in the label stack
(its "ImageJ frame" is ``index + 1``), which is the marks' *review* frame. A
mark's own ``imagej_frame`` adds the source offset and is never used for
edits.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from feedback_marks import (Anchor, Mark, MarkBuilder, Presence, append_mark,
                            evaluate, label_at, list_wells, next_mark_id,
                            parse_text, read_marks, retract_mark, summarise)
from manual_editing import (EditingBundle, EditingHistoryController,
                            _apply_operation_to_state,
                            _normalize_operation_request)
from well_workflow import find_motion_root

SNAP_RADIUS = 3
AUTOMATIC_KINDS = ("swap", "becomes", "same", "join", "persist", "ok")
WORKSPACE_KINDS = {"missing": "add_new_cells", "split": "force_split",
                   "duplicate": "force_split"}
# Same palette as the stand-alone marker (feedback_ui.KIND_COLOUR).
KIND_COLOUR = {
    "becomes": "#ff4d4d", "swap": "#ff9d00", "same": "#3fd6ff", "join": "#3fd6ff",
    "different": "#ff5ce1", "persist": "#5cff8a", "missing": "#ffe100",
    "duplicate": "#c58bff", "split": "#ffffff", "ok": "#18c8b0", "note": "#9a9a9a",
}


# ------------------------------------------------------------------ binding

@dataclass
class MarkSource:
    """Where an editing session's marks live and how its frames map to them."""
    stem: str
    marks_path: Path | None
    source_frame_offset: int
    well_dir: Path | None = None

    @property
    def available(self) -> bool:
        return self.marks_path is not None


def bundle_frame_offset(bundle: EditingBundle) -> int:
    """Source frames dropped before label frame 1, from the raw frame mapping."""
    indices = np.asarray(bundle.registered_raw_indices, dtype=int)
    return int(indices[0]) if indices.size else 0


def mark_source(bundle: EditingBundle, marks_path: Path | None = None,
                root: Path | None = None) -> MarkSource:
    """Find the well feedback folder for ``bundle`` by its stem."""
    stem = str(bundle.manifest.get("stem", ""))
    offset = bundle_frame_offset(bundle)
    if marks_path is not None:
        return MarkSource(stem, Path(marks_path), offset)
    try:
        root = root or find_motion_root(bundle.manifest_path)
    except FileNotFoundError:
        return MarkSource(stem, None, offset)
    for well_dir in list_wells(root):
        if well_dir.name.split("_", 1)[1] == stem:
            return MarkSource(stem, well_dir / "feedback" / "marks.jsonl",
                              offset, well_dir)
    return MarkSource(stem, None, offset)


def labels_fingerprint(labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str((labels.shape, labels.dtype.str)).encode())
    digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()


def load_marks(source: MarkSource, include_retracted: bool = False) -> list[Mark]:
    if not source.available:
        return []
    return read_marks(source.marks_path, include_retracted)


def frame_mismatch(mark: Mark, source: MarkSource, frames: int) -> str:
    """Why ``mark`` cannot be read on this session's frames, or ``""``."""
    if mark.source_frame_offset != source.source_frame_offset:
        return (f"mark counts frames from source offset {mark.source_frame_offset}, "
                f"this session from {source.source_frame_offset}")
    late = [a.review_frame for a in mark.anchors if a.review_frame > frames]
    if late:
        return f"mark refers to frame {max(late)} but this session has {frames}"
    return ""


# ------------------------------------------------------------------ checking

def check_marks(marks: list[Mark], labels: np.ndarray,
                source: MarkSource | None = None) -> dict[str, Any]:
    """Evaluate every mark on ``labels``; the summary counts each status."""
    results = []
    for mark in marks:
        problem = frame_mismatch(mark, source, len(labels)) if source else ""
        if problem:
            results.append({"mark_id": mark.mark_id, "kind": mark.kind,
                            "status": "not-testable", "detail": problem,
                            "candidate_labels": [], "sentence": mark.sentence})
        else:
            results.append(evaluate(mark, labels))
    return {"summary": summarise(results), "results": results}


# ------------------------------------------------------------------ planning

@dataclass
class PlanStep:
    mark_id: str
    kind: str
    sentence: str
    status_before: str
    action: str                 # edit | none | manual
    reason: str
    frame: int                  # editor frame to look at
    identities: list[int] = field(default_factory=list)
    operation: dict[str, Any] | None = None
    confidence: str = ""        # direct | check (edits only)
    workspace: str = ""         # editor tool for manual steps
    status_after: str = ""


def mark_frame(mark: Mark) -> int:
    """The frame a mark is about: its earliest clicked body or point."""
    frames = [a.review_frame for a in mark.anchors
              if a.role in ("body", "point", "side")]
    return min(frames or [a.review_frame for a in mark.anchors] or [1])


def _present(labels: np.ndarray, identity: int) -> np.ndarray:
    """One-based frames in which ``identity`` has any pixel."""
    return np.flatnonzero(np.any(labels == identity, axis=(1, 2))) + 1


def _relabel_body(labels: np.ndarray, frame: int, carried: int, true: int,
                  extend_back: bool) -> tuple[dict[str, Any], str, str]:
    """Operation making the body carrying ``carried`` at ``frame`` carry ``true``."""
    frames = len(labels)
    true_frames = set(_present(labels, true).tolist())
    if frame in true_frames:
        return ({"type": "swap_identities", "identity_a": carried,
                 "identity_b": true, "start_imagej_frame": frame,
                 "end_imagej_frame": None}, "check",
                f"{true} is still on another body at frame {frame}, so the two "
                f"numbers are exchanged from frame {frame} to the end")
    carried_frames = _present(labels, carried)
    later_true = sorted(f for f in true_frames if f > frame)
    end = int(carried_frames.max()) if carried_frames.size else frame
    confidence, reason = "direct", (
        f"{true} is absent from frame {frame} on, so {carried} takes its number")
    if later_true:
        end = min(end, later_true[0] - 1)
        confidence = "check"
        reason = (f"{true} reappears at frame {later_true[0]}, so the relabel "
                  f"stops at frame {end}")
    start = frame
    if extend_back:
        present = set(carried_frames.tolist())
        while start - 1 >= 1 and start - 1 in present and start - 1 not in true_frames:
            start -= 1
    end = max(start, min(end, frames))
    return ({"type": "reassign_identities", "source_identities": [carried],
             "target_identity": true, "start_imagej_frame": start,
             "end_imagej_frame": end}, confidence, reason)


def _runs(frames: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for frame in sorted(frames):
        if runs and frame == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], frame)
        else:
            runs.append((frame, frame))
    return runs


def propose_operations(mark: Mark, labels: np.ndarray
                       ) -> tuple[list[dict[str, Any]], str, str]:
    """Operations for one mark on ``labels``: ``(operations, confidence, reason)``.

    An empty list with a reason means a person has to resolve the mark."""
    found = [label_at(labels, a.frame_idx, a.y, a.x) for a in mark.anchors]
    kind = mark.kind
    if kind == "swap":
        first, second = found[0], found[1]
        if not first or not second or first == second:
            return [], "", "the two marked bodies do not carry two distinct numbers here"
        return ([{"type": "swap_identities", "identity_a": first,
                  "identity_b": second,
                  "start_imagej_frame": mark.anchors[0].review_frame,
                  "end_imagej_frame": None}], "direct",
                f"exchange {first} and {second} from frame "
                f"{mark.anchors[0].review_frame} to the end")
    if kind == "becomes":
        if len(found) < 2 or not found[1]:
            return [], "", "the true identity has no readable earlier position"
        carried, true = found[0], found[1]
        if not carried:
            return [], "", "no outline under the marked body"
        operation, confidence, reason = _relabel_body(
            labels, mark.anchors[0].review_frame, carried, true, extend_back=False)
        return [operation], confidence, reason
    if kind == "same":
        early, late = sorted(range(2), key=lambda i: mark.anchors[i].review_frame)
        true, carried = found[early], found[late]
        if not true or not carried:
            return [], "", "a marked body has no outline here"
        operation, confidence, reason = _relabel_body(
            labels, mark.anchors[late].review_frame, carried, true, extend_back=True)
        return [operation], confidence, reason
    if kind == "join":
        first, second = found[0], found[1]
        if not first or not second or first == second:
            return [], "", "the two outlines do not carry two distinct numbers here"
        longer = max((first, second), key=lambda i: (_present(labels, i).size, -i))
        shorter = second if longer == first else first
        frame = mark.anchors[0].review_frame
        return ([{"type": "reassign_identities", "source_identities": [shorter],
                  "target_identity": longer, "start_imagej_frame": frame,
                  "end_imagej_frame": frame}], "check",
                f"{shorter} joins {longer} in frame {frame} only; neighbouring "
                f"frames may need the same")
    if kind in ("persist", "ok"):
        bodies = [i for i, a in enumerate(mark.anchors) if a.role == "body"]
        nonzero = [v for v in found if v]
        if not nonzero:
            return [], "", "the followed body has no outline anywhere"
        keep = found[bodies[0]] if bodies and found[bodies[0]] else \
            Counter(nonzero).most_common(1)[0][0]
        wrong: dict[int, list[int]] = {}
        for anchor, value in zip(mark.anchors, found):
            if value and value != keep:
                wrong.setdefault(value, []).append(anchor.review_frame)
        lost = sorted(a.review_frame for a, v in zip(mark.anchors, found) if not v)
        operations = []
        confidence = "direct"
        for other, frames in sorted(wrong.items()):
            for start, end in _runs(frames):
                keep_frames = set(_present(labels, keep).tolist())
                if any(f in keep_frames for f in range(start, end + 1)):
                    confidence = "check"
                    operations.append({
                        "type": "swap_identities", "identity_a": keep,
                        "identity_b": other, "start_imagej_frame": start,
                        "end_imagej_frame": end})
                else:
                    operations.append({
                        "type": "reassign_identities", "source_identities": [other],
                        "target_identity": keep, "start_imagej_frame": start,
                        "end_imagej_frame": end})
        reason = f"keep {keep} on the followed body"
        if lost:
            reason += (f"; frames {', '.join(f'{a}-{b}' if a != b else str(a) for a, b in _runs(lost))}"
                       " have no outline and need a new-cell edit")
        if not operations:
            return [], "", reason
        return operations, confidence, reason
    if kind in WORKSPACE_KINDS:
        return [], "", f"needs drawn geometry: use {WORKSPACE_KINDS[kind].replace('_', ' ')}"
    if kind == "different":
        return [], "", "separating two bodies that share a number needs a split or new identity"
    return [], "", "free-text note"


def _simulate(bundle: EditingBundle, labels: np.ndarray,
              operations: list[dict[str, Any]]) -> np.ndarray:
    trial = labels.copy()
    provenance = np.zeros(trial.shape, np.uint8)
    excluded: set[int] = set()
    for operation in operations:
        normalized = _normalize_operation_request(operation, len(trial))
        _apply_operation_to_state(bundle, trial, provenance, excluded, normalized)
    return trial


def plan_from_marks(bundle: EditingBundle, labels: np.ndarray,
                    marks: list[Mark], source: MarkSource,
                    mark_ids: list[str] | None = None) -> dict[str, Any]:
    """Propose operations for the violated marks, earliest frame first.

    Each proposal is tried on the running result; it is kept only when its
    own mark becomes satisfied and no currently satisfied mark breaks."""
    wanted = set(mark_ids) if mark_ids else None
    unknown = sorted(wanted - {m.mark_id for m in marks}) if wanted else []
    if unknown:
        raise ValueError(f"unknown or retracted marks: {', '.join(unknown)}")
    usable = [m for m in marks if not frame_mismatch(m, source, len(labels))]
    before = {r["mark_id"]: r for r in check_marks(marks, labels, source)["results"]}
    state = labels.copy()
    steps: list[PlanStep] = []
    operations: list[dict[str, Any]] = []
    ordered = sorted(marks, key=lambda m: (mark_frame(m), m.mark_id))
    for mark in ordered:
        if wanted is not None and mark.mark_id not in wanted:
            continue
        status = before[mark.mark_id]["status"]
        step = PlanStep(mark.mark_id, mark.kind, mark.sentence, status, "none", "",
                        mark_frame(mark), list(mark.identities))
        steps.append(step)
        mismatch = frame_mismatch(mark, source, len(labels))
        if mismatch:
            step.action, step.reason = "manual", mismatch
            continue
        current = evaluate(mark, state)["status"]
        if current == "satisfied":
            step.reason = "already holds"
            step.status_after = current
            continue
        if mark.kind == "note":
            step.reason = "free-text note"
            continue
        proposed, confidence, reason = propose_operations(mark, state)
        step.reason = reason
        if not proposed:
            step.action = "manual"
            step.workspace = WORKSPACE_KINDS.get(mark.kind, "")
            step.status_after = current
            continue
        try:
            trial = _simulate(bundle, state, proposed)
        except ValueError as error:
            step.action, step.reason = "manual", f"{reason}; editor refused: {error}"
            step.status_after = current
            continue
        after = evaluate(mark, trial)["status"]
        broken = [m.mark_id for m in usable if m is not mark
                  and evaluate(m, state)["status"] == "satisfied"
                  and evaluate(m, trial)["status"] == "violated"]
        if after != "satisfied" or broken:
            step.action = "manual"
            step.status_after = current
            step.reason = reason + (
                f"; not proposed: the mark would still be {after}" if after != "satisfied"
                else f"; not proposed: it would break {', '.join(broken)}")
            step.operation = proposed[0] if len(proposed) == 1 else {"batch": proposed}
            continue
        state = trial
        step.action, step.confidence, step.status_after = "edit", confidence, after
        step.operation = proposed[0] if len(proposed) == 1 else {"batch": proposed}
        for operation in proposed:
            operations.append({**operation, "mark_id": mark.mark_id})
    final = check_marks(marks, state, source)
    after_by_id = {r["mark_id"]: r["status"] for r in final["results"]}
    regressions = [mark_id for mark_id, result in before.items()
                   if result["status"] == "satisfied"
                   and after_by_id.get(mark_id) == "violated"]
    return {
        "schema": "motion.manual-feedback-plan",
        "schema_version": 1,
        "stem": source.stem,
        "marks_path": str(source.marks_path) if source.marks_path else None,
        "labels_fingerprint": labels_fingerprint(labels),
        "operations": operations,
        "steps": [asdict(step) for step in steps],
        "summary": {
            "marks_considered": len(steps),
            "edits": sum(s.action == "edit" for s in steps),
            "manual": sum(s.action == "manual" for s in steps),
            "already_holding": sum(s.reason == "already holds" for s in steps),
            "before": check_marks(marks, labels, source)["summary"],
            "after": final["summary"],
        },
        "regressions": regressions,
    }


def describe_operation(operation: dict[str, Any] | None, frames: int | None = None) -> str:
    """One line of plain words for an editor operation."""
    if not operation:
        return ""
    if "batch" in operation:
        return "; ".join(describe_operation(o, frames) for o in operation["batch"])
    start = operation.get("start_imagej_frame")
    end = operation.get("end_imagej_frame")
    span = (f"frames {start}-{end if end is not None else (frames or 'end')}"
            if start != end else f"frame {start}")
    kind = operation.get("type")
    if kind == "swap_identities":
        return f"swap {operation['identity_a']} and {operation['identity_b']}, {span}"
    if kind == "reassign_identities":
        sources = ", ".join(str(i) for i in operation["source_identities"])
        return f"relabel {sources} as {operation['target_identity']}, {span}"
    return str(kind)


def batch_operations(plan: dict[str, Any],
                     mark_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """The plan's operations as editor requests (without the ``mark_id`` note)."""
    chosen = set(mark_ids) if mark_ids else None
    return [{k: v for k, v in operation.items() if k != "mark_id"}
            for operation in plan.get("operations", [])
            if chosen is None or operation.get("mark_id") in chosen]


def apply_plan(controller: EditingHistoryController, plan: dict[str, Any],
               output_dir: Path, mark_ids: list[str] | None = None,
               user: str | None = None) -> Path:
    """Save the plan's operations as one new editing session.

    Refuses a plan made against different labels than the controller's
    current checkpoint, so a stale plan cannot be replayed onto new edits."""
    labels, _provenance, _excluded = controller.materialize()
    if plan.get("labels_fingerprint") != labels_fingerprint(labels):
        raise ValueError("the plan was made on different labels; make a new plan "
                         "from this checkpoint")
    operations = batch_operations(plan, mark_ids)
    if not operations:
        raise ValueError("the plan has no edits to apply")
    return controller.apply_batch(operations, Path(output_dir), user)


# ------------------------------------------------------------------ recording

def mark_builder(source: MarkSource, labels: np.ndarray, origin: str = "ui",
                 labels_sha256: str | None = None) -> MarkBuilder:
    if not source.available:
        raise ValueError(f"no feedback folder for {source.stem!r}")
    return MarkBuilder(source.stem, labels, labels_sha256 or labels_fingerprint(labels),
                       source.source_frame_offset, Presence(labels), origin)


def next_id(source: MarkSource) -> str:
    return next_mark_id(source.marks_path, source.stem)


def snapped_identity(labels: np.ndarray, frame: int, y: int, x: int) -> int:
    """Identity under an editor click, tolerating a near miss (one-based frame)."""
    return label_at(labels, frame - 1, y, x, radius=SNAP_RADIUS)


def body_anchor(builder: MarkBuilder, frame: int, y: int, x: int) -> Anchor | None:
    identity = label_at(builder.labels, frame - 1, y, x, radius=SNAP_RADIUS)
    if not identity:
        return None
    return builder.cell_anchor(frame - 1, identity, click=(y, x))


def build_mark(builder: MarkBuilder, mark_id: str, kind: str,
               clicks: list[tuple[int, int, int]], true_identity: int | None = None,
               note: str = "") -> Mark:
    """Build one mark from editor clicks ``(frame, y, x)`` (frames one-based)."""
    def body(index: int) -> Anchor:
        frame, y, x = clicks[index]
        anchor = body_anchor(builder, frame, y, x)
        if anchor is None:
            raise ValueError(f"click {index + 1} is not on an outline")
        return anchor

    def need(count: int) -> None:
        if len(clicks) < count:
            raise ValueError(f"{kind} needs {count} click(s)")

    if kind == "swap":
        need(2)
        if clicks[0][0] != clicks[1][0]:
            raise ValueError("both swap clicks must be in the same frame")
        first, second = body(0), body(1)
        if first.identity == second.identity:
            raise ValueError("swap needs two different cells")
        mark = builder.swap(mark_id, first, second)
    elif kind == "becomes":
        need(1)
        if not true_identity:
            raise ValueError("becomes needs the true identity number")
        mark = builder.becomes(mark_id, body(0), int(true_identity))
    elif kind in ("same", "different", "join"):
        need(2)
        if kind == "join" and clicks[0][0] != clicks[1][0]:
            raise ValueError("both join clicks must be in the same frame")
        first, second = body(0), body(1)
        mark = (builder.different if kind == "different" else builder.same)(
            mark_id, first, second)
    elif kind in ("persist", "ok"):
        need(1)
        anchor = body(0)
        mark = builder.persist(mark_id, anchor.identity, kind, anchor.frame_idx)
    elif kind == "missing":
        need(1)
        frame, y, x = clicks[0]
        mark = builder.missing(mark_id, frame - 1, y, x, true_identity)
    elif kind == "duplicate":
        need(1)
        mark = builder.duplicate(mark_id, body(0))
    elif kind == "split":
        need(1)
        line = None
        if len(clicks) >= 3:
            line = ((clicks[1][1], clicks[1][2]), (clicks[2][1], clicks[2][2]))
        mark = builder.split(mark_id, body(0), line)
    elif kind == "note":
        frame = clicks[0][0] if clicks else 1
        about = [a for a in (body_anchor(builder, *c) for c in clicks) if a]
        mark = builder.note(mark_id, frame - 1, note, about)
    else:
        raise ValueError(f"unknown mark kind {kind!r}")
    if note and kind != "note":
        mark.note = note
    return mark


def save_mark(source: MarkSource, mark: Mark) -> Mark:
    return append_mark(source.marks_path, mark)


def retract(source: MarkSource, mark_id: str, reason: str = "retracted in editor") -> None:
    retract_mark(source.marks_path, mark_id, reason)


def parse_feedback(source: MarkSource, labels: np.ndarray, text: str,
                   frame_convention: str = "review") -> tuple[list[Mark], list[str]]:
    """Shorthand or plain sentences to marks on the session's current labels."""
    builder = mark_builder(source, labels, origin="text")
    first = int(next_id(source).rsplit("M", 1)[1])
    return parse_text(text, builder, first, frame_convention)
