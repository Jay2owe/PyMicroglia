"""Run drop-and-run chain trials on the lab workstation from one sent folder.

    python remote_chain_trials.py SENT_FOLDER --java JAVA_EXE --cases pm.chain.small \
        --repeats 2 --tag w1 [--after-zip final --inputs OTHER_SENT_FOLDER]

SENT_FOLDER holds what the desktop sent with ``workstation send``, flat:

- ``benchmark_drop_and_run.py``: the timing tool (``tools/`` in PyMicroglia);
- ``before.zip`` and ``after.zip``: source snapshots from its ``snapshot``
  command, each with the ``identity.json`` written when it was taken;
- ``fixtures.json`` plus the stacks it lists that the cases need, and
  ``manifest.json``, the frozen artefact index;
- ``model.pt``: the accepted cell-mask weights (``runs/stage3_disguise``).

``--after-zip NAME`` times ``NAME.zip`` as the after side instead of
``after.zip``. ``--inputs`` names a second sent folder searched for any file
missing from SENT_FOLDER, so a later trial sends only what is new.

Everything is checked before any timing: snapshot digests against their
identity files, stack hashes against ``fixtures.json``, the Java version and
the analysis imports. Receipts go to ``SENT_FOLDER/run/trials`` and
``summary.json`` beside them; inputs are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _harness(sent: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", sent / "benchmark_drop_and_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input(folders: list[Path], name: str) -> Path:
    """The first sent folder holding ``name``."""
    for folder in folders:
        if (folder / name).exists():
            return folder / name
    raise SystemExit(f"{name} is in none of {[str(f) for f in folders]}")


def _unpack(folders: list[Path], run: Path, bench, after_zip: str = "after") -> dict:
    """Snapshots unpacked once and proved identical to what was sent."""
    found = {}
    for side, name in (("before", "before"), ("after", after_zip)):
        target = run / "snapshots" / name
        if not target.exists():
            partial = target.with_name(name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            with zipfile.ZipFile(_input(folders, f"{name}.zip")) as archive:
                archive.extractall(partial)
            partial.rename(target)
        identity = json.loads((target / "identity.json").read_text(encoding="utf-8"))
        for name, digest in identity.items():
            if bench.tree_digest(target / name) != digest:
                raise SystemExit(f"{side} snapshot {name} does not match its identity.json")
        found[side] = {"folder": str(target), "identity": identity}
    return found


def _fixtures(folders: list[Path], run: Path, stacks: list[str]) -> Path:
    receipt = json.loads(_input(folders, "fixtures.json").read_text(encoding="utf-8"))
    fixtures = run / "fixtures"
    (fixtures / "stacks").mkdir(parents=True, exist_ok=True)
    (fixtures / "index").mkdir(exist_ok=True)
    for name in stacks:
        source, target = _input(folders, f"{name}.tif"), fixtures / "stacks" / f"{name}.tif"
        if _sha(source) != receipt[name]["sha256"]:
            raise SystemExit(f"{name}.tif does not match fixtures.json")
        if not target.exists():
            os.link(source, target)
    index = fixtures / "index" / "manifest.json"
    if not index.exists():
        os.link(_input(folders, "manifest.json"), index)
    return fixtures


def _java(java: str) -> str:
    result = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=30)
    return (result.stderr or result.stdout).splitlines()[0]


def _summary(out: Path, cases: list[str], tag: str, snapshots: dict) -> dict:
    summary = {"tag": tag, "snapshots": snapshots, "cases": {}}
    for case in cases:
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(out.glob(f"{case}--{tag}*.json")) if not p.name.endswith(".output.json")]
        sides = {}
        for row in rows:
            side = "after" if "after" in row["label"][len(tag):] else "before"
            folder = snapshots[side]["folder"]
            if not all(Path(f).resolve().is_relative_to(Path(folder).resolve()) for f in row["imported"].values()):
                raise SystemExit(f"{row['label']} imported code from outside its snapshot")
            sides.setdefault(side, []).append(row)
        summary["cases"][case] = {side: {
            "seconds": [r["seconds"] for r in group],
            "median_seconds": statistics.median(r["seconds"] for r in group),
            "peak_wset_bytes": max(r["peak_wset_bytes"] or 0 for r in group),
            "output_sha256": sorted({r["output_sha256"] for r in group}),
            "errors": [r["label"] for r in group if r["error"]],
            "stages": [r["stages"] for r in group]} for side, group in sides.items()}
        hashes = [set(v["output_sha256"]) for v in summary["cases"][case].values()]
        summary["cases"][case]["parity"] = len(hashes) == 2 and len(set.union(*hashes)) == 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sent", type=Path)
    parser.add_argument("--java", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--after-zip", default="after")
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    sent = args.sent.resolve()
    folders = [sent] + ([args.inputs.resolve()] if args.inputs else [])
    run = sent / "run"
    bench = _harness(sent)
    stacks = sorted({name for case in args.cases for name in
                     {"small": ["B4_d1"], "typical": ["B4_d4"],
                      "growth": ["B4_d1", "B5_d1", "B4_d2"]}[case.rpartition(".")[2]]})
    snapshots = _unpack(folders, run, bench, args.after_zip)
    fixtures = _fixtures(folders, run, stacks)
    import torch, tifffile, psutil, numpy  # noqa: F401  fail here, not mid-trial
    checks = {"java": _java(args.java), "python": sys.version, "torch": torch.__version__,
              "numpy": numpy.__version__, "stacks": stacks}
    print(json.dumps(checks), flush=True)
    env = os.environ | {"AUTO_ORGANOTYPIC_JAVA": args.java, "PYMICROGLIA_AI": "0",
                        "PYMICROGLIA_MASK_WEIGHTS": str(_input(folders, "model.pt"))}
    out = run / "trials"
    command = [sys.executable, str(sent / "benchmark_drop_and_run.py"), "trials", *args.cases,
               "--fixtures", str(fixtures), "--before", snapshots["before"]["folder"],
               "--after", snapshots["after"]["folder"], "--out", str(out),
               "--scratch", str(run / "scratch"), "--repeats", str(args.repeats), "--tag", args.tag]
    code = subprocess.run(command, env=env).returncode
    summary = _summary(out, args.cases, args.tag, snapshots) | {"checks": checks, "exit_code": code}
    (out / f"summary-{args.tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({case: {k: (v if k == "parity" else {"median_seconds": v["median_seconds"],
                                                          "errors": v["errors"]})
                             for k, v in row.items()} for case, row in summary["cases"].items()}, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
