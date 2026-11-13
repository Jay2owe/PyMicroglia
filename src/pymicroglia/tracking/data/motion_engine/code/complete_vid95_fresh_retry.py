"""Finish VID95 with A2 resumed from its completed M5 checkpoint."""

from __future__ import annotations

import json
import ctypes
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reviews/vid95_full_fresh_20260921_completion_retry"
STATUS = HERE / "status.json"
REST = ("95_A4", "95_A5", "95_A6", "95_B1", "95_B2", "95_B3",
        "95_B4", "95_B5", "95_B6", "95_C1", "95_C2")
PREFIX = "full_vid95_fresh_20260921_"
CONFIG = ROOT / "registered inputs/full_vid95_fresh_20260921_config.json"
ANALYSIS_CONFIG = ROOT / "registered inputs/full_vid95_fresh_20260921_analysis.json"
ANALYSIS_RUN = ROOT / "analysis outputs/full_vid95_fresh_20260921"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(state: dict, stage: str, **extra: object) -> None:
    state.update({"stage": stage, "updated_at_utc": now(), **extra})
    STATUS.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(stage, flush=True)


def run(state: dict, name: str, command: list[str]) -> None:
    path = HERE / f"{name}.log"
    write(state, name, command=command, log=str(path))
    with path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; see {path}")


def reviews_done(stems: tuple[str, ...]) -> bool:
    for stem in stems:
        path = ROOT / "m6_review" / (PREFIX + stem) / "run.json"
        if not path.exists():
            return False
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "done":
            return False
    return True


def first_failure(stems: tuple[str, ...]) -> str | None:
    for stem in stems:
        for path in ROOT.glob(f"m*/{PREFIX}{stem}/run.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("status") == "failed":
                return f"{stem}: {path.parent.parent.name} failed"
    return None


def wait_for_reviews(state: dict, stems: tuple[str, ...]) -> None:
    write(state, "waiting_for_review_tiffs_" + "_".join(stems))
    deadline = time.monotonic() + 8 * 60 * 60
    while not reviews_done(stems):
        failure = first_failure(stems)
        if failure:
            raise RuntimeError(failure)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Tracking wells {stems} did not finish within eight hours")
        time.sleep(30)


def resume_a3() -> None:
    pause_path = ROOT / "reviews/vid95_full_fresh_20260921_tracker_logs_completion/paused_processes.json"
    pause = json.loads(pause_path.read_text())
    pid = int(pause["95_A3"]["pid"])
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
    ntdll.NtResumeProcess.restype = ctypes.c_ulong
    handle = kernel.OpenProcess(0x0800, 0, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess failed for paused A3 PID {pid}")
    try:
        result = ntdll.NtResumeProcess(handle)
    finally:
        kernel.CloseHandle(handle)
    if result:
        raise RuntimeError(f"NtResumeProcess returned {result} for A3 PID {pid}")
    pause["95_A3"]["resumed_at_utc"] = now()
    pause_path.write_text(json.dumps(pause, indent=2) + "\n", encoding="utf-8")
    print(f"resumed 95_A3 process {pid}", flush=True)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=False)
    state = {"started_at_utc": now(), "stage": "starting"}
    try:
        wait_for_reviews(state, ("95_A1",))
        resume_a3()
        wait_for_reviews(state, ("95_A3",))
        run(state, "resume_a2_after_m5", [sys.executable, "-u", "code/pipeline.py",
             "--run", PREFIX + "95_A2_retry", "--stem", "95_A2",
             "--config", str(CONFIG), "--resume-after-m5-from", PREFIX + "95_A2"])
        run(state, "track_remaining", [sys.executable, "-u", "code/run_accepted_batch.py",
             "--config", str(CONFIG), "--run-prefix", PREFIX, "--workers", "1",
             "--log-root", str(HERE / "tracker_wells"), "--stems", *REST])
        run(state, "verify_review_tiffs", [sys.executable, "-u",
             "code/verify_vid95_deliverables.py"])
        run(state, "build_timing_ledger", [sys.executable, "-u",
             "code/build_vid95_timing_ledger.py"])
        run(state, "build_analysis_config", [sys.executable, "-u",
             "code/build_full_vid95_analysis_config.py"])
        run(state, "analysis_doctor", [sys.executable, "-m", "analysis", "doctor",
             "--config", str(ANALYSIS_CONFIG)])
        run(state, "analysis_run", [sys.executable, "-m", "analysis", "run",
             "--config", str(ANALYSIS_CONFIG), "--out", str(ANALYSIS_RUN)])
        run(state, "cell_grids", [sys.executable, "-m", "analysis", "plots",
             str(ANALYSIS_RUN)])
        run(state, "outline_videos", [sys.executable, "-u",
             "reviews/vid95_full_fresh_20260921_outlines/plot.py"])
        write(state, "done", finished_at_utc=now())
    except BaseException as error:
        write(state, "failed", error=f"{type(error).__name__}: {error}",
              finished_at_utc=now())
        raise


if __name__ == "__main__":
    main()
