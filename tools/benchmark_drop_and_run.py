"""Time the drop-a-folder-and-run path, one case per fresh process.

    python tools/benchmark_drop_and_run.py snapshot --out DIR
    python tools/benchmark_drop_and_run.py fixtures --out DIR
    python tools/benchmark_drop_and_run.py list
    python tools/benchmark_drop_and_run.py case CASE --fixtures DIR --source SNAPSHOT|live --label L --out DIR [--profile]
    python tools/benchmark_drop_and_run.py trials CASE [CASE ...] --fixtures DIR --before SNAP --after SNAP --out DIR [--repeats 2]

Fixtures are built from real Incucyte vessel 95 frames already on this
computer: well Green and Red planes stacked into the time-and-channel layout
PyIncucyte exports, full field. Every case gets fresh mutable state copied or
hard-linked from them, so no trial sees another trial's labels, manifests or
results. The real per-image export folder is only ever read.

A receipt records wall and CPU seconds, peak working set, an output digest with
temporary roots replaced, and the files the imported packages came from, so a
trial that ran the wrong source copy cannot pass as evidence.
--profile is diagnostic, not a timing trial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PM = HERE.parents[1]
AO = PM.parents[1] / "Auto-Organotypic"
SEPARATE_95 = (PM.parent / "Experiments" / "Tmem119-CreERT2" / "Tests"
               / "CD68-DIO-mCherry with HCQ" / "Incucyte_vessel95_raw")
INTERVAL_S = 1800.0
CHANNELS = (("Green", 2), ("Red", 3))
#: name -> (well, first frame, frame count). Wells B4 and B5 are the ones whose
#: Green and Red planes are all on this computer; B4's second day is a third
#: distinct recording. Four days is all of B4 on this computer; it still stops
#: at the cells stage, which needs about 121 h (72 h settling plus the 48 h
#: rolling baseline), so no fixture here times the stages after cells.
STACKS = {"B4_d1": ("B4", 0, 49), "B5_d1": ("B5", 0, 49), "B4_d2": ("B4", 48, 49),
          "B4_d4": ("B4", 0, 193)}
VOLATILE = ("generated_at", "created_at", "updated_at", "started_at", "finished_at",
            "seconds", "elapsed", "elapsed_seconds", "mtime", "modified", "id", "run_id")


# --------------------------------------------------------------- identities
def tree_digest(root: Path) -> str:
    sha = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        sha.update(path.relative_to(root).as_posix().encode())
        sha.update(hashlib.sha256(path.read_bytes()).digest())
    return sha.hexdigest()


def snapshot(out: Path) -> None:
    if out.exists():
        raise SystemExit(f"{out} exists; snapshots are never overwritten")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(PM / "src" / "pymicroglia", out / "pymicroglia", ignore=ignore)
    shutil.copytree(AO / "src" / "auto_organotypic", out / "auto_organotypic", ignore=ignore)
    identity = {name: tree_digest(out / name) for name in ("pymicroglia", "auto_organotypic")}
    (out / "identity.json").write_text(json.dumps(identity, indent=1), encoding="utf-8")
    print(json.dumps(identity))


# ----------------------------------------------------------------- fixtures
def _frames(well: str, channel: int, start: int, count: int) -> list[Path]:
    planes = sorted(SEPARATE_95.glob(f"VID95_{well}_{channel}_*.tif"))
    if len(planes) < start + count:
        raise SystemExit(f"{well} channel {channel}: {len(planes)} planes, need {start + count}")
    return planes[start:start + count]


def _write_stack(path: Path, well: str, start: int, frames: int) -> None:
    import numpy as np
    import tifffile
    first = tifffile.imread(_frames(well, CHANNELS[0][1], start, 1)[0])
    data = np.empty((frames, len(CHANNELS)) + first.shape, first.dtype)
    for c, (_, index) in enumerate(CHANNELS):
        for t, plane in enumerate(_frames(well, index, start, frames)):
            data[t, c] = tifffile.imread(plane)
    tifffile.imwrite(path, data, imagej=True, metadata={
        "axes": "TCYX", "finterval": INTERVAL_S, "tunit": "s",
        "Labels": [name for name, _ in CHANNELS] * frames})


def fixtures(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stacks = out / "stacks"
    stacks.mkdir(exist_ok=True)
    for name, (well, start, frames) in STACKS.items():
        target = stacks / f"{name}.tif"
        if not target.exists():
            _write_stack(target, well, start, frames)
    receipt = {name: {"bytes": (stacks / f"{name}.tif").stat().st_size,
                      "sha256": hashlib.sha256((stacks / f"{name}.tif").read_bytes()).hexdigest()}
               for name in STACKS}
    receipt["source"] = str(SEPARATE_95)
    (out / "fixtures.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt, indent=1))


def _link_folder(fixture: Path, target: Path, names: list[tuple[str, str]]) -> Path:
    """A fresh folder of hard links: (new file name, source stack name)."""
    target.mkdir(parents=True)
    for new, source in names:
        os.link(fixture / "stacks" / f"{source}.tif", target / new)
    return target


# -------------------------------------------------------------------- cases
def _many(count: int) -> list[tuple[str, str]]:
    return [(f"W{n:03d}.tif", "B4_d1") for n in range(count)]


def _setup(case: str, fixture: Path, work: Path) -> dict:
    kind, _, size = case.rpartition(".")
    if size == "separate95":
        return {"folder": SEPARATE_95}
    if size in ("local1", "local12", "local96"):
        count = int(size[5:])
        return {"folder": _link_folder(fixture, work / "drop", _many(count)), "count": count}
    if size == "file1":
        folder = _link_folder(fixture, work / "drop", _many(12))
        return {"folder": folder, "file": folder / "W000.tif"}
    if size == "small":
        return {"folder": _link_folder(fixture, work / "drop", [("B4.tif", "B4_d1")])}
    if size == "typical":
        return {"folder": _link_folder(fixture, work / "drop", [("B4.tif", "B4_d4")])}
    if size == "growth":
        return {"folder": _link_folder(fixture, work / "drop",
                                       [("B4.tif", "B4_d1"), ("B5.tif", "B5_d1"), ("B4_day2.tif", "B4_d2")])}
    if size == "r20":
        folder = _link_folder(fixture, work / "drop", _many(1))
        for n in range(20):
            run = folder / "pymicroglia_analysis" / "auto_microglia" / f"run{n:02d}"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"outputs": {"tracked_measurement": {
                f"W000_registered_translation": {"folder": str(run / "tracked_measurement" / "W000" / "m")}}}}))
            os.utime(run / "manifest.json", (1_790_000_000 + n, 1_790_000_000 + n))
        return {"folder": folder}
    if size == "labels500":
        folder = _link_folder(fixture, work / "drop", _many(1))
        return {"folder": folder, "count": 500}
    if size == "real":
        return {"folder": work}
    if size == "stages5x12":
        return {"folder": _link_folder(fixture, work / "drop", _many(12))}
    raise SystemExit(f"unknown case {case}")


def _labels_for(rows) -> dict:
    return {row["key"]: {"name": f"Slice {n}", "condition": ("HCQ", "vehicle")[n % 2],
                         "treatment": "LPS", "sample_id": f"M{n // 4}"}
            for n, row in enumerate(rows)}


def _act(case: str, ctx: dict, work: Path):
    from pymicroglia.ui import intake
    from auto_organotypic import recording_labels
    kind = case.rpartition(".")[0]
    folder = str(ctx["folder"])
    if kind == "pm.inspect":
        return intake.inspect([str(ctx.get("file") or folder)])
    if kind == "pm.save_labels":
        found = intake.inspect([folder])
        ctx["_untimed"] = time.perf_counter()
        return intake.save_labels(found["folder"], _labels_for(found["recordings"]))
    if kind == "pm.run_params":
        found = intake.inspect([folder])
        ctx["_untimed"] = time.perf_counter()
        return intake.run_params(found["folder"], found["mode"], found["recordings"])
    if kind == "pm.results":
        return intake.results(folder)
    if kind == "pm.chain":
        from pymicroglia.pipelines import auto_microglia
        found = intake.inspect([folder])
        intake.save_labels(found["folder"], _labels_for(found["recordings"]))
        params = intake.run_params(found["folder"], found["mode"], found["recordings"])
        stages = ctx["stages"] = []
        last = [time.perf_counter()]

        def progress(event):
            now = time.perf_counter()
            stages.append({**{k: v for k, v in dict(event).items() if k != "seconds"},
                           "wall": round(now - last[0], 3)})
            last[0] = now
        ctx["_untimed"] = time.perf_counter()
        result = auto_microglia.run(**params, on_progress=progress, claim="speed pass")
        return {"result": result, "outputs": _tree(Path(params["folder"]))}
    if kind == "ao.inspect_inputs":
        from auto_organotypic import workspace
        paths = sorted(str(p) for p in Path(folder).glob("*.tif"))
        return workspace.inspect_inputs(paths)
    if kind == "ao.create_project":
        from auto_organotypic import workspace
        rows = workspace.inspect_inputs(sorted(str(p) for p in Path(folder).glob("*.tif")))
        for n, row in enumerate(rows):
            row["name"] = f"Slice {n}"
            row["annotations"] = {"condition": ("HCQ", "vehicle")[n % 2], "sample_id": f"M{n}"}
        # The server starts this warm-up when it opens (where it has one); by the
        # time somebody has dropped, named and labelled recordings it is done.
        from auto_organotypic import server
        getattr(server, "_warm_capabilities", lambda: None)()
        ctx["_untimed"] = time.perf_counter()
        project = workspace.create_project(work / "results", "Speed", rows)
        return {"records": project["records"], "labels": recording_labels.read(folder)}
    if kind == "ao.save_annotations":
        from auto_organotypic import workspace, workspace_outputs
        rows = workspace.inspect_inputs(sorted(str(p) for p in Path(folder).glob("*.tif")))
        project = workspace.create_project(work / "results", "Speed", rows)
        ctx["_untimed"] = time.perf_counter()
        last = None
        for n, record in enumerate(project["records"]):
            last = workspace_outputs.save_annotations(
                project["folder"], (last or project)["revision"], record["id"],
                {"condition": "HCQ", "sample_id": f"M{n}"})
        return {"labels": recording_labels.read(folder)}
    if kind == "ao.labels_by_stem":
        stems = [f"W{n:03d}" for n in range(ctx["count"])]
        recording_labels.save(folder, {f"{s}.tif": {"condition": "HCQ"} for s in stems})
        ctx["_untimed"] = time.perf_counter()
        return recording_labels.by_stem(Path(folder) / "pymicroglia_analysis" / "x",
                                        [s + "_registered_translation" for s in stems])
    if kind == "ao.index_add":
        # What every finished artefact costs the run: one claim into the index.
        from auto_organotypic.store import manifest
        for n in range(3):
            manifest.add({"path": str(work / f"artefact{n}.json"), "stage": "speed",
                          "digest": f"d{n}", "params": {"n": n}})
        index = manifest.load()
        return {"summary": manifest.summary(),
                "added": {k: {f: v for f, v in e.items() if f != "indexed"}
                          for k, e in index["artefacts"].items() if e.get("stage") == "speed"}}
    if kind == "ao.corrections":
        # Each stage start asks for every recording's corrections.
        from auto_organotypic import corrections
        paths = sorted(str(p) for p in Path(folder).glob("*.tif"))
        return [{path: [c.field for c in found] for path, found
                 in corrections.records_for_stage(stage, paths).items()}
                for stage in ("crop", "outline", "register", "cell_masks", "cells")]
    raise SystemExit(f"unknown case {case}")


CASES = [
    "pm.inspect.separate95", "pm.run_params.separate95",
    "pm.inspect.local1", "pm.inspect.local12", "pm.inspect.local96", "pm.inspect.file1",
    "pm.save_labels.local12", "pm.save_labels.local96",
    "pm.run_params.local12", "pm.run_params.local96",
    "pm.results.r20",
    "pm.chain.small", "pm.chain.typical", "pm.chain.growth",
    "ao.inspect_inputs.local12", "ao.inspect_inputs.local96",
    "ao.create_project.local12", "ao.save_annotations.local12",
    "ao.labels_by_stem.labels500",
    "ao.index_add.real", "ao.corrections.stages5x12",
]


def _tree(root: Path) -> list:
    return sorted([p.relative_to(root).as_posix(), p.stat().st_size]
                  for p in root.rglob("*") if p.is_file())


def _normal(value, roots):
    if isinstance(value, dict):
        return {k: _normal(v, roots) for k, v in sorted(value.items()) if k not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_normal(v, roots) for v in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for root, name in roots:
            value = value.replace(root, name).replace(root.replace("\\", "/"), name)
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)


def _private_store(fixture: Path, work: Path) -> None:
    """Give the trial its own artefact index, pixel store and decisions folder.

    The index starts as the frozen copy of this computer's real one (fixtures
    ``index/manifest.json``, 83 MB on 2026-10-30), because its size is what the
    bookkeeping costs depend on; a trial never reads or grows the live one, and
    never finds another trial's cached artefacts.
    """
    index = work / "store" / "index"
    index.mkdir(parents=True)
    frozen = fixture / "index" / "manifest.json"
    if frozen.is_file():
        shutil.copyfile(frozen, index / "manifest.json")
    for package in ("PYMICROGLIA", "AUTO_ORGANOTYPIC"):
        os.environ[f"{package}_INDEX"] = str(index)
        os.environ[f"{package}_STORE"] = str(work / "store" / "pixels")
        os.environ[f"{package}_DECISIONS"] = str(work / "store" / "decisions")


def run_case(args) -> int:
    if args.source != "live":
        sys.path[:0] = [str(Path(args.source).resolve())]
    os.environ.setdefault("PYMICROGLIA_AI", "0")
    # The learned cell mask needs trained weights and the package ships none;
    # this is the accepted base named in MCG_mask_model/README.md.
    os.environ.setdefault("PYMICROGLIA_MASK_WEIGHTS", str(
        PM / "development" / "single_frame_mask_unet" / "MCG_mask_model" / "runs"
        / "stage3_disguise" / "model.pt"))
    out = args.out.resolve()
    receipt_path = out / f"{args.case}--{args.label}.json"
    if receipt_path.exists():
        raise SystemExit("Choose a fresh label; existing evidence is never overwritten")
    import tempfile
    import psutil
    # One fixed folder per case: recording identities hash their path, so a
    # random folder name would make identical outputs look different.
    work = Path(args.scratch or tempfile.gettempdir()).resolve() / "work" / args.case
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    _private_store(args.fixtures.resolve(), work)
    ctx = _setup(args.case, args.fixtures.resolve(), work)
    import pymicroglia, auto_organotypic
    # A running server imports these once at start-up; a request never pays it.
    from pymicroglia.ui import intake  # noqa: F401
    from auto_organotypic import recording_labels, workspace, workspace_outputs, sources  # noqa: F401
    if args.case.startswith("pm.chain"):
        from pymicroglia.pipelines import auto_microglia  # noqa: F401
    imported = {"pymicroglia": pymicroglia.__file__, "auto_organotypic": auto_organotypic.__file__}
    process = psutil.Process()
    profile = None
    if args.profile:
        import cProfile
        profile = cProfile.Profile()
    error = None
    cpu0, t0 = time.process_time(), time.perf_counter()
    try:
        if profile:
            profile.enable()
        result = _act(args.case, ctx, work)
    except Exception as exc:  # recorded, not hidden
        import traceback
        result, error = None, traceback.format_exc()
    finally:
        if profile:
            profile.disable()
    t1, cpu1 = time.perf_counter(), time.process_time()
    start = ctx.get("_untimed", t0)
    roots = [(str(work), "<WORK>"), (str(args.fixtures.resolve()), "<FIXTURES>")]
    normal = _normal(result, roots)
    payload = json.dumps(normal, sort_keys=True, default=repr)
    receipt = {
        "case": args.case, "label": args.label, "source": args.source,
        "imported": imported, "seconds": round(t1 - start, 4),
        "setup_in_act_seconds": round(start - t0, 4), "cpu_seconds": round(cpu1 - cpu0, 4),
        "peak_wset_bytes": getattr(process.memory_info(), "peak_wset", None),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "error": error, "stages": ctx.get("stages"), "profiled": bool(profile),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.case}--{args.label}.output.json").write_text(payload, encoding="utf-8")
    if profile:
        profile.dump_stats(out / f"{args.case}--{args.label}.prof")
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: receipt[k] for k in ("case", "label", "seconds", "cpu_seconds",
                                              "peak_wset_bytes", "output_sha256")}))
    return 1 if error else 0


def trials(args) -> int:
    order = []
    for repeat in range(args.repeats):
        order += [("before", args.before), ("after", args.after)] if repeat % 2 == 0 else \
                 [("after", args.after), ("before", args.before)]
    failed = 0
    for case in args.cases:
        for n, (side, source) in enumerate(order):
            label = f"{args.tag}{side}{n}"
            command = [sys.executable, str(HERE), "case", case, "--fixtures", str(args.fixtures),
                       "--source", str(source), "--label", label, "--out", str(args.out)]
            if args.scratch:
                command += ["--scratch", str(args.scratch)]
            failed += subprocess.run(command).returncode != 0
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("snapshot"); one.add_argument("--out", type=Path, required=True)
    one = sub.add_parser("fixtures"); one.add_argument("--out", type=Path, required=True)
    sub.add_parser("list")
    one = sub.add_parser("case")
    one.add_argument("case", choices=CASES)
    for p in (one, two := sub.add_parser("trials")):
        p.add_argument("--fixtures", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--scratch", type=Path)
    one.add_argument("--source", required=True)
    one.add_argument("--label", required=True)
    one.add_argument("--profile", action="store_true")
    two.add_argument("cases", nargs="+", choices=CASES)
    two.add_argument("--before", required=True)
    two.add_argument("--after", required=True)
    two.add_argument("--repeats", type=int, default=2)
    two.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.out.resolve())
    elif args.command == "fixtures":
        fixtures(args.out.resolve())
    elif args.command == "list":
        print("\n".join(CASES))
    elif args.command == "case":
        return run_case(args)
    else:
        return trials(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
