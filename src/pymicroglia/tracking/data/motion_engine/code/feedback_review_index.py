"""Build a frame-sorted review index from immutable feedback and candidate labels.

This is an evaluation tool. It does not pass feedback into a label producer.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from feedback_marks import read_marks, sha256_of
from feedback_ground_truth_scoring import evaluate_at_feedback_cues


def _uncertain_automatic_anchors(mark, source: np.ndarray) -> list[str]:
    uncertain = []
    for index, anchor in enumerate(mark.anchors):
        if anchor.identity <= 0 or anchor.role not in ("reference", "control"):
            continue
        _, count = ndimage.label(source[anchor.frame_idx] == anchor.identity,
                                 structure=np.ones((3, 3), dtype=np.uint8))
        if count > 1:
            uncertain.append(f"{index}:{anchor.role}:{anchor.review_frame}")
    return uncertain


def build_index(root: Path, parent_manifest: Path, output: Path) -> list[dict]:
    parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if parent.get("attempt") != "R01-A002":
        raise ValueError("review index requires the accepted R01-A002 parent")
    well_dirs = {json.loads((p / "WELL_BASELINE.json").read_text())["well"]: p
                 for p in (root / "issues").glob("[0-9][0-9]_*")
                 if (p / "WELL_BASELINE.json").is_file()}
    rows = []
    for well, candidate_record in parent["well_outputs"].items():
        well_dir = well_dirs[well]
        baseline = json.loads((well_dir / "WELL_BASELINE.json").read_text())
        source_path = root / baseline["accepted_parent"] / "out" / f"{well}.tif"
        candidate_path = root / candidate_record["labels"]
        if sha256_of(source_path) != baseline["labels_sha256"]:
            raise ValueError(f"{well}: source labels changed")
        if sha256_of(candidate_path) != candidate_record["sha256"]:
            raise ValueError(f"{well}: accepted round-one labels changed")
        source = tifffile.imread(source_path)
        candidate = tifffile.imread(candidate_path)
        for mark in read_marks(well_dir / "feedback" / "marks.jsonl"):
            if mark.labels_sha256 != baseline["labels_sha256"]:
                raise ValueError(f"{mark.mark_id}: feedback source changed")
            before = evaluate_at_feedback_cues(mark, source, source)
            after = evaluate_at_feedback_cues(mark, candidate, source)
            uncertain = _uncertain_automatic_anchors(mark, source)
            if mark.kind == "note":
                verdict = "free_text_review"
            elif before["status"] == "satisfied":
                verdict = "protected_satisfied" if after["status"] == "satisfied" else "regression"
            elif before["status"] == "undecidable":
                verdict = "ambiguous_location"
            elif after["status"] == "satisfied":
                verdict = "provisional_gain" if uncertain else "gain"
            elif after["status"] == "undecidable":
                verdict = "ambiguous_location"
            else:
                verdict = "still_unresolved"
            clicked = [f"{a.review_frame}:{a.click_x},{a.click_y}"
                       for a in mark.anchors if a.click_x is not None]
            rows.append({
                "well": well, "first_review_frame": min(mark.frames(), default=0),
                "mark_id": mark.mark_id, "feedback_type": mark.kind,
                "review_frames": "|".join(map(str, mark.frames())),
                "clicked_frame_xy": "|".join(clicked),
                "cut_yx": json.dumps(mark.line, separators=(",", ":")) if mark.line else "",
                "source_status": before["status"], "round_one_status": after["status"],
                "verdict": verdict,
                "uncertain_automatic_anchors": "|".join(uncertain),
                "original_feedback_sentence": mark.sentence,
            })
    rows.sort(key=lambda row: (row["well"], row["first_review_frame"], row["mark_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_index(args.root.resolve(), args.parent_manifest, args.output)
    from collections import Counter
    print(json.dumps(dict(Counter(row["verdict"] for row in rows)), indent=2))


if __name__ == "__main__":
    main()
