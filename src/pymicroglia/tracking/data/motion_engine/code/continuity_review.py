"""Complete-field continuity review using the shared three-column renderer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_review_tiff import build_review_bundle


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    score_path = Path(upstream_dir)/"metrics.json" if upstream_dir is not None else None
    score = json.loads(score_path.read_text()) if score_path is not None and score_path.is_file() else None
    candidate_path = Path(params["candidate_labels_path"])
    bundle = build_review_bundle(
        baseline_labels_path=Path(params["baseline_labels_path"]),
        candidate_labels_path=candidate_path, raw_path=Path(params["raw_path"]),
        motion_path=None, output_dir=out.qc/"full_dataset", stem=candidate_path.stem,
        candidate_title=str(params.get("candidate_title", "CANDIDATE")),
        source_frame_offset=int(params.get("source_frame_offset", 0)),
        frame_interval_min=float(params.get("frame_interval_min", 30.0)),
        include_motion=False)
    manifest = json.loads(Path(bundle["manifest"]).read_text())
    reviewer = pd.read_csv(params["review_cases_path"]) if params.get("review_cases_path") else pd.DataFrame()
    reviewer["candidate_human_verdict"] = ""
    reviewer["candidate_notes"] = ""
    reviewer_path = out.out/"reviewer_labels.csv"
    reviewer.to_csv(reviewer_path, index=False)
    summary = dict(manifest["render"], attempt_id=params["attempt_id"], complete_field=True,
                   diagnostic_score=score, promotion_decision="not made by renderer")
    summary_path = out.out/"review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    outputs = {name:Path(bundle["output_dir"])/row["relative_path"]
               for name,row in bundle["outputs"].items()}
    outputs.update(reviewer_labels=reviewer_path, review_summary=summary_path,
                   review_manifest=Path(bundle["manifest"]))
    return dict(outputs=outputs, summary=summary)
