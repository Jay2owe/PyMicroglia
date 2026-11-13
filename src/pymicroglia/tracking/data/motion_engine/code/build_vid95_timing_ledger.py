"""Summarize the full VID95 tracking and review delivery times by well."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEMS = [f"95_{row}{col}" for row, cols in
         (("A", range(1, 7)), ("B", range(1, 7)), ("C", range(1, 3)))
         for col in cols]


def stage(stem: str, name: str) -> dict:
    run = f"full_vid95_clean_lookback_20260922_{stem}"
    path = ROOT / name / run / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def seconds(run: dict) -> float:
    return float(run.get("elapsed_seconds") or 0.0)


def evidence_seconds(stem: str) -> float:
    run = f"full_vid95_clean_lookback_20260922_{stem}"
    path = ROOT / "registered inputs/motion evidence" / run / "run.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return float(next(row["elapsed_seconds"] for row in manifest["movies"]
                      if row["stem"] == stem))


def main() -> None:
    destination = ROOT / "reviews/vid95_full_fresh_20260921_outlines/timing_ledger.csv"
    rows = []
    for stem in STEMS:
        input_check = stage(stem, "m0_inputs")
        anchor = stage(stem, "m2_anchor")
        track = stage(stem, "m3_track")
        events = stage(stem, "m4_events")
        stages = {p.name: stage(stem, p.name) for p in ROOT.iterdir()
                  if p.is_dir() and p.name.startswith("m") and p.name[1:2].isdigit()}
        ownership_names = [name for name in stages if name not in
                           {"m0_inputs", "m1_motion", "m2_anchor", "m3_track",
                            "m4_events", "m5_reconcile", "m6_review", "m23_provenance"}]
        review = stages["m6_review"]
        start = input_check.get("started_at_utc")
        finish = review.get("finished_at_utc")
        total = ((datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()
                 if start and finish else None)
        rows.append({"stem": stem,
                     "motion_evidence_seconds": evidence_seconds(stem),
                     "input_verification_seconds": seconds(input_check),
                     "tracking_seconds": sum(seconds(stages.get(name, {})) for name in
                                              ("m1_motion", "m2_anchor", "m3_track",
                                               "m4_events", "m5_reconcile")),
                     "identity_ownership_seconds": sum(seconds(stages[name]) for name in ownership_names),
                     "display_preparation_seconds": review.get("summary", {}).get("display_preparation_seconds"),
                     "tiff_rendering_seconds": review.get("summary", {}).get("tiff_rendering_seconds"),
                     "review_stage_seconds": seconds(review),
                     "total_seconds_to_openable_tiffs": total,
                     "complete_review_tiffs": int(bool(review.get("status") == "done")),
                     "first_stage_started_at_utc": start,
                     "review_tiffs_finished_at_utc": finish})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(destination)


if __name__ == "__main__":
    main()
