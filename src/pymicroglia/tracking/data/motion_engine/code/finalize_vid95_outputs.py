"""Audit completed VID95 media and update its single Research Gallery record."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parent / "Experiments/Tmem119-CreERT2/Tests/CD68-DIO-mCherry with HCQ"
CONTROLLER = ROOT / "reviews/vid95_full_fresh_20260921_completion_clean_lookback/status.json"
HERE = ROOT / "reviews/vid95_full_fresh_20260921_finalization"
BUNDLE = ROOT / "reviews/vid95_full_fresh_20260921_outlines"
RUN = ROOT / "analysis outputs/full_vid95_fresh_20260921"
STEMS = [f"95_{row}{col}" for row, cols in
         (("A", range(1, 7)), ("B", range(1, 7)), ("C", range(1, 3)))
         for col in cols]
REGISTER = (Path.home() / ".claude/skills/plot-that/scripts/register.py")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(stage: str, **extra: object) -> None:
    (HERE / "status.json").write_text(json.dumps({"stage": stage,
        "updated_at_utc": now(), **extra}, indent=2) + "\n", encoding="utf-8")
    print(stage, flush=True)


def run(name: str, command: list[str]) -> None:
    path = HERE / f"{name}.log"
    write(name, command=command, log=str(path))
    with path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; see {path}")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def gallery_update() -> None:
    artifacts = []
    grid_masters = []
    for stem in STEMS:
        grid_bundle = RUN / "figures/all-cell-trace-grid" / stem.lower().replace("_", "-")
        masters = list((grid_bundle / "fig").glob("*.svg"))
        if len(masters) != 1:
            raise ValueError(f"{stem}: expected one grid SVG, found {masters}")
        grid_masters.extend(masters)
        artifacts.append({"path": rel(masters[0]), "media_type": "plot",
                          "role": f"cell trace grid {stem}",
                          "caption": f"Fiji-processed red signal in tracked cells, {stem}; period tests are exploratory.",
                          "presentation_state": "draft", "produced_by": {"bundle": rel(grid_bundle)}})
        outline = BUNDLE / f"{stem}.tif"
        artifacts.append({"path": rel(outline), "media_type": "video",
                          "role": f"cell outlines {stem}",
                          "caption": f"All 99 retained frames with tracked outlines, {stem}.",
                          "presentation_state": "draft", "produced_by": {"bundle": rel(BUNDLE)}})
        review = ROOT / "m6_review" / f"full_vid95_clean_lookback_20260922_{stem}" / "out" / f"{stem}_full_field_review.tif"
        artifacts.append({"path": rel(review), "media_type": "video",
                          "role": f"full tracking review {stem}",
                          "caption": f"Complete 99-frame identity, motion and event review, {stem}; user review pending.",
                          "presentation_state": "draft", "produced_by": {
                              "script": "code/pipeline.py",
                              "input": "registered inputs/full_vid95_clean_lookback_20260922_config.json"}})
    if len(grid_masters) != 14:
        raise AssertionError("Grid count is incomplete")
    payload = {
        "update_note": "Completed all 14 fresh full-field VID95 runs, reviewed artifact counts and reference grid settings, and linked original media; biological cell-identity review remains pending.",
        "status": "complete", "presentation_state": "draft",
        "summary": "All 14 VID95 wells have 99 registered red-signal frames after excluding source frames 1-2 before Fiji extraction, fresh motion tracking, complete review TIFFs, outline stacks and cell trace grids.",
        "explanation": "Validated C1-derived translations align both raw channels. The first two source frames were removed before the supplied Fiji macro normalized and subtracted the processed green reference from the processed red signal. Two copies of the clean first retained frame provided lookback context for Motion's established two-frame input offset; neither copy enters the 99-frame outputs. The analysis grids show within-cell normalized traces, with uncorrected Lomb period tests on measured values before display scaling.",
        "next_step": "Jamie reviews the complete three-panel tracking TIFFs and cell identity assignments before accepting the grids or biological interpretations.",
        "related_analyses": ["ANA-20260831-020"],
        "data_refs": [
            {"path": str(EXPERIMENT / "AI_Exports/registered_unmixed_2026-07-23/raw_registered_two_channel_stacks_30min/raw_registered_stack_manifest.csv"), "role": "registered raw source manifest"},
            {"path": str(EXPERIMENT / "AI_Exports/registered_unmixed_2026-07-23/registration_validation_all_frames.csv"), "role": "all-frame registration validation"},
            {"path": "registered inputs/full_vid95_fresh_20260921/manifest.json", "role": "99-frame extracted movies"},
            {"path": "registered inputs/full_vid95_clean_lookback_20260922/manifest.json", "role": "clean, non-output tracker lookback frames"},
            {"path": "registered inputs/full_vid95_clean_lookback_20260922_config.json", "role": "hash-pinned Motion inputs"},
            {"path": "registered inputs/full_vid95_fresh_20260921_analysis.json", "role": "analysis plan"},
        ],
        "analysis_refs": [
            {"path": "code/trim_vid95_registered_channels.py", "role": "frame exclusion"},
            {"path": "code/run_vid95_extract.ijm", "role": "Fiji macro runner"},
            {"path": "code/pad_vid95_tracker_lookback.py", "role": "clean tracker lookback context"},
            {"path": "code/pipeline.py", "role": "motion tracker"},
            {"path": "code/build_full_vid95_analysis_config.py", "role": "analysis configuration"},
            {"path": "reviews/vid95_full_fresh_20260921_outlines/review_tiff_validation.csv", "role": "complete-movie review validation"},
            {"path": "reviews/vid95_full_fresh_20260921_outlines/timing_ledger.csv", "role": "processing timing"},
            {"path": "reviews/vid95_full_fresh_20260921_outlines/m8_speed_proof.json", "role": "byte-identical merge speed check"},
            {"path": "reviews/vid95_full_fresh_20260921_outlines/accepted_baseline_hash_check.json", "role": "protected accepted-output check"},
        ],
        "artifacts": artifacts,
        "limitations": [
            "Fiji contrast normalization changes pixel values; processed signal is not raw camera-unit intensity.",
            "Automated cell identities and events remain pending visual review.",
            "The 99-frame recording and three-cycle sufficiency default limit confidence for long periods; unresolved estimates are not evidence of arrhythmia.",
        ],
    }
    path = HERE / "gallery_update.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    run("gallery_update", [sys.executable, "research/_tools/cli.py", "--root",
                           str(ROOT), "update", "ANA-20260921-001", "--input", str(path)])


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    try:
        write("waiting_for_analysis_and_outlines")
        while True:
            if CONTROLLER.exists():
                state = json.loads(CONTROLLER.read_text(encoding="utf-8"))
                if state.get("stage") == "failed":
                    raise RuntimeError(f"processing controller failed: {state.get('error')}")
                if state.get("stage") == "done":
                    break
            time.sleep(60)
        run("verify_grids", [sys.executable, "-u", "code/verify_vid95_grids.py"])
        run("outline_bundle_check", [sys.executable, str(REGISTER), "check", str(BUNDLE)])
        run("outline_bundle_register", [sys.executable, str(REGISTER), "add", str(BUNDLE)])
        gallery_update()
        write("done", finished_at_utc=now())
    except BaseException as error:
        write("failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        raise


if __name__ == "__main__":
    main()
