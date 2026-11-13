"""Measure how much Java heap the motion engine needs for each recording.

    python heap_probe.py SENT_FOLDER --java JAVA_EXE --stacks B4_d1 B4_d4 --heap 24g \
        --recipe bright_dim_dense

SENT_FOLDER is the chain-trial folder that ``remote_chain_trials.py`` already
unpacked (``run/snapshots/after`` and ``run/fixtures/stacks``). The engine is
run exactly as the chain runs it except for its ``-Xmx`` cap, with Java's
garbage-collection log on, so the heap left after each collection (what the
recording really needs) can be read. Stacks run one after another; results go
to ``SENT_FOLDER/run/heap``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

_GC = re.compile(r"(\d+)M->(\d+)M\((\d+)M\)")


def _probe(run: Path, name: str, args, psutil, motion_runtime, series) -> dict:
    out = run / "heap" / f"{name}-{args.recipe}-{args.heap}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "gc.log"
    log.unlink(missing_ok=True)
    peak = {"wset": 0}
    real = subprocess.Popen

    def popen(command, *rest, **options):
        command = [part for part in command if not str(part).startswith("-Xmx")]
        command[1:1] = [f"-Xmx{args.heap}", f"-Xlog:gc:file={log.as_posix()}"]
        process = real(command, *rest, **options)

        def watch():
            try:
                handle = psutil.Process(process.pid)
                while process.poll() is None:
                    peak["wset"] = max(peak["wset"], handle.memory_info().peak_wset)
                    time.sleep(.5)
            except psutil.Error:
                pass
        threading.Thread(target=watch, daemon=True).start()
        return process

    result = {"stack": name, "recipe": args.recipe, "heap": args.heap, "java": args.java,
              "machine_bytes": psutil.virtual_memory().total}
    started = time.perf_counter()
    subprocess.Popen = popen
    try:
        with series.open_series(run / "fixtures" / "stacks" / f"{name}.tif") as opened:
            frames, channels, height, width = map(int, opened.shape)
            result |= {"frames": frames, "channels": channels, "height": height, "width": width,
                       "input_bytes": frames * height * width * opened.dtype.itemsize}
            poses, metadata = motion_runtime.estimate(opened, args.recipe, channel=args.channel)
        result |= {"ok": True, "recipe_id": metadata.get("recipe_id"),
                   "poses_sha256": hashlib.sha256(poses.tobytes()).hexdigest()}
    except Exception as exc:  # the outcome is the measurement
        result |= {"ok": False, "error": str(exc)[-1500:]}
    finally:
        subprocess.Popen = real
    result["seconds"] = round(time.perf_counter() - started, 1)
    sizes = [tuple(map(int, m.groups())) for m in _GC.finditer(log.read_text(errors="replace"))] if log.exists() else []
    result |= {"collections": len(sizes),
               "max_after_gc_mib": max((after for _, after, _ in sizes), default=None),
               "max_before_gc_mib": max((before for before, _, _ in sizes), default=None),
               "java_peak_wset_mib": peak["wset"] // 2**20}
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sent", type=Path)
    parser.add_argument("--java", required=True)
    parser.add_argument("--stacks", nargs="+", required=True)
    parser.add_argument("--heap", required=True)
    parser.add_argument("--recipe", default="bright_dim_dense")
    parser.add_argument("--channel", type=int, default=1)
    args = parser.parse_args()
    run = args.sent.resolve() / "run"
    sys.path.insert(0, str(run / "snapshots" / "after"))
    os.environ["AUTO_ORGANOTYPIC_JAVA"] = args.java
    import psutil
    from auto_organotypic import motion_runtime, series
    results = [_probe(run, name, args, psutil, motion_runtime, series) for name in args.stacks]
    (run / "heap" / f"summary-{args.heap}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
