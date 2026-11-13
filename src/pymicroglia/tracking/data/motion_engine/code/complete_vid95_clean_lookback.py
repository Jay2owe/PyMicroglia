"""Finish 14 complete 99-frame VID95 tracker runs using clean lookback frames."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reviews/vid95_full_fresh_20260921_completion_clean_lookback"
INPUT = ROOT / "registered inputs/full_vid95_clean_lookback_20260922"
CONFIG = ROOT / "registered inputs/full_vid95_clean_lookback_20260922_config.json"
ANALYSIS_CONFIG = ROOT / "registered inputs/full_vid95_fresh_20260921_analysis.json"
ANALYSIS_RUN = ROOT / "analysis outputs/full_vid95_fresh_20260921"
STEMS = [f"95_{row}{col}" for row, cols in
         (("A", range(1, 7)), ("B", range(1, 7)), ("C", range(1, 3)))
         for col in cols]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(stage: str, **extra: object) -> None:
    (HERE / "status.json").write_text(json.dumps({
        "stage": stage, "updated_at_utc": now(), **extra}, indent=2) + "\n",
        encoding="utf-8")
    print(stage, flush=True)


def run(name: str, command: list[str]) -> None:
    log = HERE / f"{name}.log"
    write(name, command=command, log=str(log))
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}: {log}")


def main() -> None:
    HERE.mkdir(exist_ok=False)
    try:
        for stem in STEMS:
            evidence_run = f"full_vid95_clean_lookback_20260922_{stem}"
            status = ROOT / "registered inputs/motion evidence" / evidence_run / "run.json"
            if status.exists() and json.loads(status.read_text()).get("status") == "done":
                continue
            run(f"motion_evidence_{stem}", [sys.executable, "-u",
                "code/prepare_full_vid95_motion.py", "--registered", str(INPUT),
                "--run", evidence_run, "--stems", stem])
        run("verify_clean_lookback_boundary", [sys.executable, "-u",
            "code/verify_vid95_clean_lookback.py"])
        run("pin_inputs", [sys.executable, "-u", "code/build_full_vid95_config.py",
                           "--variant", "clean_lookback"])
        run("track_all_wells", [sys.executable, "-u", "code/run_accepted_batch.py",
            "--config", str(CONFIG), "--run-prefix", "full_vid95_clean_lookback_20260922_",
            "--workers", "1", "--log-root", str(HERE / "tracker_wells"),
            "--stems", *STEMS])
        run("verify_review_tiffs", [sys.executable, "-u", "code/verify_vid95_deliverables.py"])
        run("build_timing_ledger", [sys.executable, "-u", "code/build_vid95_timing_ledger.py"])
        run("build_analysis_config", [sys.executable, "-u", "code/build_full_vid95_analysis_config.py"])
        run("analysis_doctor", [sys.executable, "-m", "analysis", "doctor",
             "--config", str(ANALYSIS_CONFIG)])
        run("analysis_run", [sys.executable, "-m", "analysis", "run",
             "--config", str(ANALYSIS_CONFIG), "--out", str(ANALYSIS_RUN)])
        run("cell_grids", [sys.executable, "-m", "analysis", "plots", str(ANALYSIS_RUN)])
        run("outline_videos", [sys.executable, "-u",
             "reviews/vid95_full_fresh_20260921_outlines/plot.py"])
        write("done", finished_at_utc=now())
    except BaseException as error:
        write("failed", error=f"{type(error).__name__}: {error}",
              finished_at_utc=now())
        raise


if __name__ == "__main__":
    main()
