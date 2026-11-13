"""Tests for feedback marks: building, storing, reading back and evaluating."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feedback_marks as fm  # noqa: E402


def _stack() -> np.ndarray:
    """Ten frames, 40x40. Cell A (id 5) sits top-left, cell B (id 9) bottom-right.
    From frame 6 the labels are swapped: A's body carries 9 and B's carries 5.
    Cell C (id 3) is a small cell present in frames 0-3 only, then its body is
    unlabelled in frames 4-9 (a missing detection)."""
    labels = np.zeros((10, 40, 40), np.uint16)
    for f in range(10):
        a, b = (5, 9) if f < 6 else (9, 5)
        labels[f, 5:12, 5:12] = a
        labels[f, 25:34, 25:34] = b
        if f < 4:
            labels[f, 30:33, 5:8] = 3
    return labels


def _builder(labels: np.ndarray) -> fm.MarkBuilder:
    return fm.MarkBuilder("T1", labels, "abc123", 2)


def test_becomes_anchors_reference_by_position():
    labels = _stack()
    b = _builder(labels)
    body = b.cell_anchor(6, 9)          # A's body at frame 7 carries 9
    mark = b.becomes("T1-M0001", body, 5)
    assert mark.kind == "becomes"
    assert mark.expected_identity == 5
    assert [a.role for a in mark.anchors] == ["body", "reference"]
    assert mark.anchors[1].review_frame == 6      # last frame where 5 was itself
    assert mark.anchors[0].imagej_frame == 9      # review 7 + offset 2
    assert "5 becomes 9 at review frame 7 (ImageJ 9)" in mark.sentence

    # unfixed candidate violates, fixed candidate satisfies
    assert fm.evaluate(mark, labels)["status"] == "violated"
    fixed = labels.copy()
    fixed[6:][labels[6:] == 9] = 5
    fixed[6:][labels[6:] == 5] = 9
    assert fm.evaluate(mark, fixed)["status"] == "satisfied"
    # a renumbered but correct candidate also satisfies (positions, not numbers)
    renum = fixed.copy()
    renum[fixed == 5] = 77
    assert fm.evaluate(mark, renum)["status"] == "satisfied"


def test_swap_uses_previous_frame_references():
    labels = _stack()
    b = _builder(labels)
    mark = b.swap("T1-M0002", b.cell_anchor(6, 9), b.cell_anchor(6, 5))
    assert mark.kind == "swap"
    assert len(mark.must_match) == 2
    assert not mark.unresolved
    assert fm.evaluate(mark, labels)["status"] == "violated"
    fixed = labels.copy()
    fixed[6:][labels[6:] == 9] = 5
    fixed[6:][labels[6:] == 5] = 9
    assert fm.evaluate(mark, fixed)["status"] == "satisfied"


def test_same_and_join_and_different():
    labels = _stack()
    b = _builder(labels)
    same = b.same("T1-M0003", b.cell_anchor(2, 5), b.cell_anchor(8, 9))
    assert same.kind == "same"
    assert fm.evaluate(same, labels)["status"] == "violated"
    join = b.same("T1-M0004", b.cell_anchor(2, 5), b.cell_anchor(2, 9))
    assert join.kind == "join"
    assert fm.evaluate(join, labels)["status"] == "violated"
    # "the 5 at frame 3 and the 5 at frame 9 are different cells": true, so the
    # stack as marked violates it and the corrected stack satisfies it
    diff = b.different("T1-M0005", b.cell_anchor(2, 5), b.cell_anchor(8, 5))
    assert fm.evaluate(diff, labels)["status"] == "violated"
    fixed = labels.copy()
    fixed[6:][labels[6:] == 9] = 5
    fixed[6:][labels[6:] == 5] = 9
    assert fm.evaluate(diff, fixed)["status"] == "satisfied"


def test_persist_reports_relabelled_frames():
    labels = _stack()
    b = _builder(labels)
    mark = b.persist("T1-M0006", 5, "persist", checked_frame_idx=2)
    assert len(mark.anchors) == 10                  # body A followed through all frames
    assert [a.identity for a in mark.anchors] == [5] * 6 + [9] * 4
    assert all(a.y < 12 for a in mark.anchors)      # stays on body A, not on B
    assert "currently carries 9" in mark.sentence
    result = fm.evaluate(mark, labels)
    assert result["status"] == "violated"
    assert "4 carry another number" in result["detail"]
    fixed = labels.copy()
    fixed[6:][labels[6:] == 9] = 5
    fixed[6:][labels[6:] == 5] = 9
    assert fm.evaluate(mark, fixed)["status"] == "satisfied"


def test_missing_with_identity():
    labels = _stack()
    b = _builder(labels)
    mark = b.missing("T1-M0007", 5, 31, 6, identity=3)
    assert mark.anchors[0].identity == 0
    assert mark.anchors[1].review_frame == 4      # last seen (frame idx 3)
    assert fm.evaluate(mark, labels)["status"] == "violated"
    fixed = labels.copy()
    fixed[4:, 30:33, 5:8] = 3
    assert fm.evaluate(mark, fixed)["status"] == "satisfied"
    wrong_number = labels.copy()
    wrong_number[4:, 30:33, 5:8] = 42
    assert fm.evaluate(mark, wrong_number)["status"] == "violated"


def test_duplicate_finds_other_component():
    labels = _stack()
    labels[2, 20:23, 20:23] = 5                    # a stray copy of 5
    b = _builder(labels)
    mark = b.duplicate("T1-M0008", b.cell_anchor(2, 5))
    assert len(mark.anchors) == 2
    assert fm.evaluate(mark, labels)["status"] == "violated"
    fixed = labels.copy()
    fixed[2, 20:23, 20:23] = 0
    assert fm.evaluate(mark, fixed)["status"] == "satisfied"     # copy may vanish
    fixed[2, 20:23, 20:23] = 8
    assert fm.evaluate(mark, fixed)["status"] == "satisfied"     # or take a new number


def test_split_with_and_without_line():
    labels = _stack()
    b = _builder(labels)
    body = b.cell_anchor(1, 9)
    plain = b.split("T1-M0009", body)
    assert plain.region["identity"] == 9
    assert fm.evaluate(plain, labels)["status"] == "violated"
    fixed = labels.copy()
    fixed[1, 25:34, 29:34] = 11
    assert fm.evaluate(plain, fixed)["status"] == "satisfied"
    lined = b.split("T1-M0010", body, line=((24, 29), (34, 29)))
    assert lined.line == [[24, 29], [34, 29]]
    assert len(lined.anchors) == 3
    assert fm.evaluate(lined, labels)["status"] == "violated"
    assert fm.evaluate(lined, fixed)["status"] == "satisfied"


def test_storage_roundtrip_and_retract(tmp_path):
    labels = _stack()
    b = _builder(labels)
    path = tmp_path / "feedback" / "marks.jsonl"
    first = fm.append_mark(path, b.becomes(fm.next_mark_id(path, "T1"), b.cell_anchor(6, 9), 5))
    second = fm.append_mark(path, b.persist(fm.next_mark_id(path, "T1"), 9))
    assert first.mark_id == "T1-M0001" and second.mark_id == "T1-M0002"
    back = fm.read_marks(path)
    assert [m.mark_id for m in back] == ["T1-M0001", "T1-M0002"]
    assert back[0].anchors[1].role == "reference"
    assert back[0].sentence == first.sentence
    fm.retract_mark(path, "T1-M0001", "misclick")
    assert [m.mark_id for m in fm.read_marks(path)] == ["T1-M0002"]
    assert fm.next_mark_id(path, "T1") == "T1-M0003"        # ids never reused
    with (path.with_suffix(".csv")).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "retracted" and rows[1]["status"] == "active"
    fm.append_session(path, "T1", "abc123", {3, 4, 5}, ["T1-M0002"], "t0", "t1")
    assert fm.read_sessions(path)[0]["review_frames_viewed"] == [3, 4, 5]
    text = fm.digest(fm.read_marks(path), fm.evaluate_all(fm.read_marks(path), labels),
                     fm.read_sessions(path))
    assert "## persist" in text and "VIOLATED" in text and "3-5" in text


def test_review_cases_export():
    labels = _stack()
    b = _builder(labels)
    marks = [b.becomes("T1-M0001", b.cell_anchor(6, 9), 5),
             b.persist("T1-M0002", 9, "ok"),
             b.note("T1-M0003", 2, "just a note")]
    rows = fm.to_review_cases(marks)
    assert [r["case_id"] for r in rows] == ["T1-M0001", "T1-M0002"]
    assert rows[0]["case_type"] == "failure" and rows[1]["case_type"] == "control"
    assert rows[0]["review_frame_start"] == 7 and rows[0]["source_imagej_frame_start"] == 9
    assert rows[0]["expected_identity"] == 5
    assert set(rows[0]) == set(fm.REVIEW_CASE_COLUMNS)


@pytest.mark.parametrize("text, kind, ids", [
    ("5>9 @7", "becomes", [9, 5]),
    ("5 becomes 9 at frame 7", "becomes", [9, 5]),
    ("cell 5 flips to 9 at 7", "becomes", [9, 5]),
    ("5 swaps to 9 at 7", "becomes", [9, 5]),
    ("5<>9 @7", "swap", [5, 9]),
    ("5 and 9 swap at frame 7", "swap", [5, 9]),
    ("5 swapped with 9 at 7", "swap", [5, 9]),
    ("5=9 @2", "join", [5, 9]),
    ("5 persists", "persist", [5, 9]),                 # body A carries 9 later
    ("5 should be 5 the whole video", "persist", [5, 9]),
    ("missing at 6,31 @6 is 3", "missing", [3]),
    ("9 split @2", "split", [9]),
    ("9 is two cells at 2", "split", [9]),
])
def test_parse_text_forms(text, kind, ids):
    labels = _stack()
    b = _builder(labels)
    marks, failures = fm.parse_text(text, b)
    assert not failures, failures
    assert marks[0].kind == kind
    assert marks[0].identities == ids
    assert marks[0].origin == "text"


def test_parse_text_multiple_and_imagej_frames():
    labels = _stack()
    b = _builder(labels)
    marks, failures = fm.parse_text("5>9 @ij9; 5 and 9 swap at 7. nonsense here", b, first_id=4)
    assert [m.mark_id for m in marks] == ["T1-M0004", "T1-M0005"]
    assert marks[0].anchors[0].review_frame == 7        # ImageJ 9 - offset 2
    assert failures == ["nonsense here"]
    marks, failures = fm.parse_text("5 becomes 9 at 3", b)   # 9 is not at frame 3? it is
    assert not failures
    marks, failures = fm.parse_text("5 becomes 44 at 3", b)
    assert failures and "not in review frame" in failures[0]


def _fake_project(tmp_path: Path) -> Path:
    """A minimal Motion tree: one well, an accepted parent and a candidate run."""
    import tifffile
    root = tmp_path / "Motion"
    (root / "WORKFLOW_CONTRACT.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "WORKFLOW_CONTRACT.md").write_text("contract", encoding="utf-8")
    labels = _stack()
    accepted = root / "m22_accepted_history" / "acc_T1" / "out"
    accepted.mkdir(parents=True)
    tifffile.imwrite(accepted / "T1.tif", labels)
    tifffile.imwrite(accepted / "T1_unclaimed_original_ids.tif", np.zeros_like(labels))
    raw = root / "registered inputs"
    raw.mkdir(parents=True)
    tifffile.imwrite(raw / "T1.tif", labels.astype(np.uint16))
    candidate = root / "runs" / "r02_a002_T1" / "out"
    candidate.mkdir(parents=True)
    fixed = labels.copy()
    fixed[6:][labels[6:] == 9] = 5
    fixed[6:][labels[6:] == 5] = 9
    tifffile.imwrite(candidate / "T1.tif", fixed)
    tifffile.imwrite(candidate / "T1_unclaimed_original_ids.tif", np.zeros_like(fixed))
    (candidate.parent / "run.json").write_text(json.dumps(
        {"attempt": "R02-A002", "targeting_mode": "field_wide_discovery"}), encoding="utf-8")
    well = root / "issues" / "01_T1"
    well.mkdir(parents=True)
    (well / "WELL_BASELINE.json").write_text(json.dumps({
        "well": "T1", "accepted_parent": "m22_accepted_history/acc_T1",
        "source_frames_removed": [1, 2],
        "labels_sha256": fm.sha256_of(accepted / "T1.tif")}), encoding="utf-8")
    return root


def test_review_set_build_and_resolve(tmp_path):
    root = _fake_project(tmp_path)
    data = fm.build_review_set(root, "cand", root / "runs", "a candidate set")
    assert list(data["wells"]) == ["T1"]
    entry = data["wells"]["T1"]
    assert entry["attempt"] == "R02-A002"
    assert entry["labels"].replace("\\", "/").endswith("runs/r02_a002_T1/out/T1.tif")
    assert entry["source_frame_offset"] == 2
    assert fm.list_review_sets(root) == ["cand"]

    default = fm.well_paths(root, "T1")
    assert default.labels_source == "accepted"
    chosen = fm.well_paths(root, "T1", review_set="cand")
    assert chosen.labels_source == "cand"
    assert chosen.labels_sha256 == entry["sha256"]
    assert chosen.labels != default.labels
    assert chosen.unclaimed is not None
    # marks still go to the one file per well, whichever stack was reviewed
    assert chosen.marks == default.marks


def test_review_set_rejects_changed_stack(tmp_path):
    import tifffile
    root = _fake_project(tmp_path)
    fm.build_review_set(root, "cand", root / "runs", "")
    tifffile.imwrite(root / "runs" / "r02_a002_T1" / "out" / "T1.tif", _stack())
    with pytest.raises(ValueError, match="fingerprint"):
        fm.well_paths(root, "T1", review_set="cand")
    with pytest.raises(FileNotFoundError, match="no review set"):
        fm.well_paths(root, "T1", review_set="nope")


def test_digest_flags_marks_from_several_stacks():
    labels = _stack()
    b = _builder(labels)
    first = b.becomes("T1-M0001", b.cell_anchor(6, 9), 5)
    other = fm.MarkBuilder("T1", labels, "def456", 2)
    second = other.persist("T1-M0002", 9)
    text = fm.digest([first, second])
    assert "more than one label stack" in text
    assert "abc123" in text and "def456" in text
    assert "more than one label stack" not in fm.digest([first])


def test_label_at_tolerates_small_shift():
    labels = _stack()
    assert fm.label_at(labels, 0, 4, 4) == 5      # just outside the 5:12 block
    assert fm.label_at(labels, 0, 20, 20) == 0
