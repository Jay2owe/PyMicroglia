"""The audit must expose marks aimed at the wrong disconnected body."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from feedback_ground_truth_audit import audit_well
from feedback_marks import Anchor, Mark, sha256_of


def test_click_and_cut_disagreements_are_reported_without_editing_mark(tmp_path: Path):
    root = tmp_path
    well = "95_TEST"
    source = root / "parent" / "out"
    source.mkdir(parents=True)
    labels = np.zeros((1, 30, 30), dtype=np.uint16)
    labels[0, 3:9, 3:9] = 7
    labels[0, 20:27, 20:27] = 7
    labels_path = source / f"{well}.tif"
    tifffile.imwrite(labels_path, labels)
    digest = sha256_of(labels_path)
    well_dir = root / "issues" / "01_95_TEST"
    marks_dir = well_dir / "feedback"
    marks_dir.mkdir(parents=True)
    (well_dir / "WELL_BASELINE.json").write_text(json.dumps({
        "well": well, "accepted_parent": "parent", "labels_sha256": digest,
    }), encoding="utf-8")
    mark = Mark(mark_id="95_TEST-M0001", well=well, kind="split",
                made_at="2026-09-20T00:00:00Z", labels_sha256=digest,
                source_frame_offset=0,
                anchors=[Anchor(1, 1, 7, 5, 5, 36, "body", 23, 23),
                         Anchor(1, 1, 7, 4, 4, 0, "side"),
                         Anchor(1, 1, 7, 22, 22, 0, "side"),
                         Anchor(1, 1, 7, 5, 5, 0, "reference")],
                must_differ=[[1, 2]], must_exist=[1, 2],
                line=[[20, 23], [27, 23]], identities=[7])
    marks_path = marks_dir / "marks.jsonl"
    marks_path.write_text(mark.to_json() + "\n" + json.dumps({
        "record": "session", "review_frames_viewed": [1],
    }) + "\n", encoding="utf-8")
    original = marks_path.read_bytes()

    summary, disagreements, notes, uncertain = audit_well(root, well_dir)

    assert summary["active_marks"] == 1
    assert summary["reviewed_frames"] == 1
    assert {row["disagreement"] for row in disagreements} >= {
        "saved_vs_clicked_component", "saved_vs_cut_component",
        "cut_side_on_other_component",
    }
    assert not notes
    assert len(uncertain) == 1
    assert uncertain[0]["anchor_index"] == 3
    assert marks_path.read_bytes() == original


def test_mismatched_source_hash_is_rejected(tmp_path: Path):
    root = tmp_path
    source = root / "parent" / "out"
    source.mkdir(parents=True)
    tifffile.imwrite(source / "95_TEST.tif", np.zeros((1, 4, 4), dtype=np.uint16))
    well_dir = root / "issues" / "01_95_TEST"
    (well_dir / "feedback").mkdir(parents=True)
    (well_dir / "WELL_BASELINE.json").write_text(json.dumps({
        "well": "95_TEST", "accepted_parent": "parent", "labels_sha256": "wrong",
    }), encoding="utf-8")
    (well_dir / "feedback" / "marks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="accepted label hash"):
        audit_well(root, well_dir)
