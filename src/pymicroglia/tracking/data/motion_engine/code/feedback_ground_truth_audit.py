"""Audit completed motion feedback without changing the user's marks.

The marks are sparse constraints on accepted movies, not corrected label stacks.
This audit checks their source hashes, positional anchors, and unscored notes.
It deliberately does not feed feedback identities or coordinates to a producer.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from feedback_marks import evaluate_all, read_marks, read_sessions, sha256_of
from feedback_ground_truth_scoring import evaluate_at_feedback_cues


def _nearest_source_component(plane: np.ndarray, identity: int,
                              y: int, x: int) -> tuple[int, float, np.ndarray]:
    mask = plane == identity
    if not mask.any():
        raise ValueError(f"identity {identity} is absent from source frame")
    components, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    distance, nearest = ndimage.distance_transform_edt(~mask, return_indices=True)
    ny, nx = nearest[:, y, x]
    return int(components[ny, nx]), float(distance[y, x]), components


def audit_well(root: Path, well_dir: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    baseline = json.loads((well_dir / "WELL_BASELINE.json").read_text(encoding="utf-8"))
    well = baseline["well"]
    labels_path = root / str(baseline["accepted_parent"]).replace("\\", "/") / "out" / f"{well}.tif"
    actual_hash = sha256_of(labels_path)
    if actual_hash != baseline["labels_sha256"]:
        raise ValueError(f"{well}: accepted label hash disagrees with WELL_BASELINE.json")
    labels = tifffile.imread(labels_path)
    marks_path = well_dir / "feedback" / "marks.jsonl"
    marks = read_marks(marks_path)
    sessions = read_sessions(marks_path)
    if {m.labels_sha256 for m in marks} != {actual_hash}:
        raise ValueError(f"{well}: marks were not all made against accepted labels")
    seen_frames = {int(f) for session in sessions
                   for f in session.get("review_frames_viewed", [])}
    statuses = Counter(result["status"] for result in evaluate_all(marks, labels))
    cue_statuses = Counter(evaluate_at_feedback_cues(mark, labels, labels)["status"]
                           for mark in marks)
    marked_names = {anchor.identity for mark in marks
                    if mark.kind not in ("note", "ok")
                    for anchor in mark.anchors if anchor.identity > 0}
    foreground = labels > 0
    marked_name_pixels = int(np.count_nonzero(np.isin(labels, list(marked_names))))
    notes = [{"well": well, "mark_id": mark.mark_id,
              "review_frames": "|".join(map(str, mark.frames())),
              "identities": "|".join(map(str, mark.identities)),
              "note": mark.note, "sentence": mark.sentence}
             for mark in marks if mark.kind == "note"]
    disagreements = []
    uncertain_tracking_anchors = []
    cache: dict[tuple[int, int, int, int], tuple[int, float, np.ndarray]] = {}

    def component(frame: int, identity: int, y: int, x: int):
        key = frame, identity, y, x
        if key not in cache:
            cache[key] = _nearest_source_component(labels[frame], identity, y, x)
        return cache[key]

    for mark in marks:
        for index, anchor in enumerate(mark.anchors):
            if anchor.identity > 0 and anchor.role in ("reference", "control"):
                _, component_count = ndimage.label(
                    labels[anchor.frame_idx] == anchor.identity,
                    structure=np.ones((3, 3), dtype=np.uint8))
                if component_count > 1:
                    uncertain_tracking_anchors.append({
                        "well": well, "mark_id": mark.mark_id, "kind": mark.kind,
                        "anchor_index": index, "role": anchor.role,
                        "review_frame": anchor.review_frame,
                        "identity": anchor.identity,
                        "components_with_identity": component_count,
                        "saved_y": anchor.y, "saved_x": anchor.x,
                        "reason": "reference selected by identity-wide centroid"
                        if anchor.role == "reference" else
                        "followed identity has disconnected bodies",
                    })
            if anchor.identity <= 0 or anchor.click_y is None or anchor.click_x is None:
                continue
            frame = anchor.frame_idx
            clicked_comp, distance, components = component(
                frame, anchor.identity, anchor.click_y, anchor.click_x)
            saved_comp = int(components[anchor.y, anchor.x])
            if saved_comp != clicked_comp or distance > 2:
                disagreements.append({"well": well, "mark_id": mark.mark_id,
                                      "kind": mark.kind, "anchor_index": index,
                                      "review_frame": anchor.review_frame,
                                      "identity": anchor.identity,
                                      "disagreement": "saved_vs_clicked_component" if saved_comp != clicked_comp
                                      else "click_over_2px_from_label",
                                      "saved_y": anchor.y, "saved_x": anchor.x,
                                      "click_y": anchor.click_y, "click_x": anchor.click_x,
                                      "distance_px": round(distance, 3),
                                      "saved_component": saved_comp,
                                      "cue_component": clicked_comp})
        if mark.kind == "split" and mark.line and mark.anchors:
            body = mark.anchors[0]
            (y0, x0), (y1, x1) = mark.line
            mid_y, mid_x = round((y0 + y1) / 2), round((x0 + x1) / 2)
            line_comp, distance, components = component(
                body.frame_idx, body.identity, mid_y, mid_x)
            saved_comp = int(components[body.y, body.x])
            if saved_comp != line_comp or distance > 5:
                disagreements.append({"well": well, "mark_id": mark.mark_id,
                                      "kind": mark.kind, "anchor_index": 0,
                                      "review_frame": body.review_frame,
                                      "identity": body.identity,
                                      "disagreement": "saved_vs_cut_component" if saved_comp != line_comp
                                      else "cut_over_5px_from_label",
                                      "saved_y": body.y, "saved_x": body.x,
                                      "click_y": mid_y, "click_x": mid_x,
                                      "distance_px": round(distance, 3),
                                      "saved_component": saved_comp,
                                      "cue_component": line_comp})
            for index, side in enumerate(mark.anchors[1:], start=1):
                if side.role != "side":
                    continue
                side_comp = int(components[side.y, side.x])
                if side_comp != line_comp:
                    disagreements.append({"well": well, "mark_id": mark.mark_id,
                                          "kind": mark.kind, "anchor_index": index,
                                          "review_frame": side.review_frame,
                                          "identity": side.identity,
                                          "disagreement": "cut_side_on_other_component",
                                          "saved_y": side.y, "saved_x": side.x,
                                          "click_y": mid_y, "click_x": mid_x,
                                          "distance_px": round(distance, 3),
                                          "saved_component": side_comp,
                                          "cue_component": line_comp})
    summary = {"well": well, "frames": int(labels.shape[0]),
               "reviewed_frames": len(seen_frames), "active_marks": len(marks),
               "satisfied": statuses["satisfied"], "violated": statuses["violated"],
               "undecidable": statuses["undecidable"],
               "not_testable": statuses["not-testable"],
               "cue_violated": cue_statuses["violated"],
               "cue_undecidable": cue_statuses["undecidable"],
               "cue_satisfied": cue_statuses["satisfied"],
               "notes": len(notes), "marked_name_pixels": marked_name_pixels,
               "named_pixels": int(np.count_nonzero(foreground)),
               "marked_name_fraction": round(marked_name_pixels / max(1, int(np.count_nonzero(foreground))), 6),
               "anchor_disagreements": len(disagreements),
               "uncertain_tracking_anchors": len(uncertain_tracking_anchors),
               "labels_sha256": actual_hash, "marks_sha256": sha256_of(marks_path)}
    return summary, disagreements, notes, uncertain_tracking_anchors


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    summaries, disagreements, notes, uncertain_tracking_anchors = [], [], [], []
    for well_dir in sorted((root / "issues").glob("[0-9][0-9]_*")):
        if not (well_dir / "feedback" / "marks.jsonl").is_file():
            continue
        summary, rows, note_rows, tracking_rows = audit_well(root, well_dir)
        summaries.append(summary)
        disagreements.extend(rows)
        notes.extend(note_rows)
        uncertain_tracking_anchors.extend(tracking_rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "anchor_disagreements.csv", disagreements)
    write_csv(args.output_dir / "unscored_notes.csv", notes)
    write_csv(args.output_dir / "uncertain_tracking_anchors.csv", uncertain_tracking_anchors)
    print(json.dumps({"wells": len(summaries),
                      "active_marks": sum(s["active_marks"] for s in summaries),
                      "anchor_disagreements": len(disagreements),
                      "uncertain_tracking_anchors": len(uncertain_tracking_anchors),
                      "unscored_notes": len(notes)}, indent=2))


if __name__ == "__main__":
    main()
