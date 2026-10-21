"""Time real public measurement execution on bounded, deterministic pixel inputs.

No fake modules, saved result reuse, or scientific option overrides. The
generated outlines are measurement inputs, not a claim of detection/tracking.
Run each trial in a fresh process. --profile is diagnostic, not a timing trial.
"""
from __future__ import annotations
import argparse
import cProfile
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(48, 96, 192), default=48)
    parser.add_argument("--label", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--index-seed", type=Path)
    parser.add_argument("--index-mode", choices=("immediate", "batched"), default="batched")
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    trial = root / args.label
    if trial.exists():
        parser.error("Choose a fresh label; existing evidence is never overwritten")
    # Measured fixture output is 3 MiB at 48 frames. Include the copied index
    # and its atomic-write temporary plus 512 MiB of headroom.
    seed_bytes = args.index_seed.stat().st_size if args.index_seed else 0
    if args.index_seed and args.index_seed.suffix == ".gz":
        with args.index_seed.open("rb") as stream:
            stream.seek(-4, 2)
            seed_bytes = int.from_bytes(stream.read(4), "little")
    if shutil.disk_usage(root).free < 512 * 1024**2 + 2 * seed_bytes:
        parser.error("Insufficient space for fixture, copied index, atomic temporary and 512 MiB headroom")
    trial.mkdir()
    os.environ["PYMICROGLIA_STORE"] = str(trial / "store")
    os.environ["PYMICROGLIA_INDEX"] = str(trial / "index")
    if args.index_seed:
        (trial / "index").mkdir()
        if args.index_seed.suffix == ".gz":
            with gzip.open(args.index_seed, "rb") as source, (trial / "index/manifest.json").open("wb") as dest:
                shutil.copyfileobj(source, dest)
        else:
            shutil.copyfile(args.index_seed, trial / "index/manifest.json")
    seed_identity = digest(trial / "index/manifest.json") if args.index_seed else None
    import numpy as np
    import psutil
    import tifffile
    source_root = Path(__file__).resolve().parents[1] / "src/pymicroglia"
    sources = {str(p.relative_to(source_root)): digest(p) for p in source_root.rglob("*.py")}
    yy, xx = np.mgrid[:96, :128]
    rng = np.random.default_rng(111026)
    raw = np.zeros((args.frames, 96, 128), dtype=np.uint16)
    labels = np.zeros_like(raw)
    for frame in range(args.frames):
        plane = rng.normal(100, 5, yy.shape)
        for cell in range(6):
            cx = 24 + 40 * (cell % 3) + 2 * np.sin(frame / 17 + cell)
            cy = 25 + 46 * (cell // 3) + 2 * np.cos(frame / 13 + cell)
            mask = (xx - cx)**2 + (yy - cy)**2 <= (7 + np.sin(frame / 9 + cell))**2
            labels[frame, mask] = cell + 1
            plane[mask] += 600 + 250 * np.cos(2 * np.pi * frame / 48 + cell)
        raw[frame] = np.clip(plane, 0, 65535).astype(np.uint16)
    for name, pixels in (("raw", raw), ("labels", labels)):
        tifffile.imwrite(trial / (name + ".tif"), pixels, photometric="minisblack", metadata={"axes": "TYX"})
    del raw, labels
    inputs = {name: digest(trial / (name + ".tif")) for name in ("raw", "labels")}
    process = psutil.Process()
    memory = []
    stopped = threading.Event()
    def sample():
        while not stopped.is_set():
            memory.append(process.memory_info().rss)
            stopped.wait(.05)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    receipt = {"action": "measure", "kind": "execution", "status": "failed", "frames": args.frames,
               "cells": 6, "pixels": [96, 128], "synthetic": True,
               "settings": "all available measurement modules, live defaults; 30 minute frame interval",
               "source_sha256": sources, "input_sha256": inputs,
               "profiled": args.profile, "memory_sampling_seconds": .05}
    profile = cProfile.Profile() if args.profile else None
    index_saves = []
    started = time.perf_counter()
    try:
        if profile: profile.enable()
        import pymicroglia
        from pymicroglia import store
        from contextlib import nullcontext
        if args.index_mode == "immediate":
            store.manifest.batch_updates = nullcontext
        original_save = store.manifest.save
        def counted_save(data):
            now = time.perf_counter()
            try:
                return original_save(data)
            finally:
                index_saves.append(time.perf_counter() - now)
        store.manifest.save = counted_save
        from pymicroglia.run import run_action
        from pymicroglia import config
        assert config.index_root().resolve() == (trial / "index").resolve()
        assert config.store_root().resolve() == (trial / "store").resolve()
        result = run_action("measure", movies=[{"stem": "synthetic", "raw": str(trial / "raw.tif"),
                           "labels": str(trial / "labels.tif")}], output_dir=trial / "outputs",
                           run_label="measured", frame_interval_min=30, output_roots=[trial / "outputs"],
                           claim="Benchmark real measurement of deterministic synthetic pixels; no biological finding.")
        receipt["seconds"] = time.perf_counter() - started
        receipt["status"] = "measured"
        receipt["modules"] = result.get("movies", [])
    except Exception:
        receipt["seconds"] = time.perf_counter() - started
        receipt["error"] = traceback.format_exc()
    finally:
        if profile:
            profile.disable()
            profile.dump_stats(str(trial / "profile.pstats"))
        stopped.set()
        thread.join()
    receipt["peak_rss_bytes"] = max(memory + [process.memory_info().rss])
    receipt["index_mode"] = args.index_mode
    receipt["index_seed_sha256"] = seed_identity
    receipt["index_save_seconds"] = index_saves
    receipt["retained_rss_bytes"] = process.memory_info().rss
    receipt["tables"] = {str(p.relative_to(trial / "outputs")): digest(p) for p in (trial / "outputs").rglob("*.csv")}
    receipt["source_unchanged"] = all((source_root / name).is_file() and digest(source_root / name) == sha for name, sha in sources.items())
    receipt["inputs_unchanged"] = all(digest(trial / (name + ".tif")) == sha for name, sha in inputs.items())
    receipt["output_bytes"] = sum(p.stat().st_size for p in trial.rglob("*") if p.is_file())
    # This discovery index is a trial-local copy, never a source or result.
    # Keep its identity; later analysis reads each output's durable ledger.
    disposable_index = trial / "index/manifest.json"
    if disposable_index.is_file():
        receipt["temporary_index_sha256"] = digest(disposable_index)
        receipt["temporary_index_bytes"] = disposable_index.stat().st_size
        disposable_index.unlink()
    receipt["shared_store_code_sha256"] = digest(Path(store.manifest.__file__)) if "store" in locals() else None
    (trial / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"source_sha256", "modules", "tables"}}, indent=2))
    return 0 if receipt["status"] == "measured" and receipt["source_unchanged"] and receipt["inputs_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
