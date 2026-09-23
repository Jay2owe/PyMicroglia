"""Run the frozen Motion engine shipped in the PyMicroglia wheel.

The engine is imported from an isolated per-run folder. Its historical flat
module names are restored after a run, so they do not stay in PyMicroglia's
import namespace.
No tracking threshold, assignment rule, or calibration is changed here.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from importlib import resources
from pathlib import Path

import tifffile

from .contract import TrackingResult

METHOD_VERSION = "2026-09-22-frozen-motion-engine-v1"
_IMPORT_LOCK = threading.RLock()


def _inside(path: object, root: Path) -> bool:
    try:
        return Path(str(path)).resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _run_frozen(root: Path, stem: str, run_name: str, config: Path,
                log: Path) -> None:
    code = root / "code"
    names = {path.stem for path in code.rglob("*.py")
             if path.stem != "__init__"}
    with _IMPORT_LOCK:
        previous = {name: sys.modules.pop(name) for name in names
                    if name in sys.modules}
        sys.path.insert(0, str(code))
        try:
            with log.open("w", encoding="utf-8") as handle:
                with redirect_stdout(handle), redirect_stderr(handle):
                    pipeline = importlib.import_module("pipeline")
                    pipeline.run_base(run_name, stem, config)
        finally:
            sys.path.remove(str(code))
            for name, module in list(sys.modules.items()):
                if _inside(getattr(module, "__file__", None), root):
                    sys.modules.pop(name, None)
            sys.modules.update(previous)


def _engine(folder: Path) -> Path:
    archive_resource = resources.files("pymicroglia.tracking").joinpath(
        "data/motion_engine.zip")
    content = archive_resource.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    target = folder / "engine"
    marker = target / "archive.sha256"
    if marker.is_file():
        if marker.read_text(encoding="ascii").strip() != digest:
            raise ValueError("this Motion run folder contains a different engine snapshot")
        if not (target / "code/pipeline.py").is_file():
            raise FileNotFoundError(target / "code/pipeline.py")
        return target
    if target.exists():
        raise FileExistsError(f"unverified Motion engine folder: {target}")
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive_resource.open("rb")) as archive:
        names = archive.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts
               for name in names):
            raise ValueError("Motion engine archive has unsafe paths")
        archive.extractall(target)
    marker.write_text(digest + "\n", encoding="ascii")
    return target


def run(inputs, folder, *, stem: str | None = None) -> TrackingResult:
    """Run all accepted Motion stages for one stem and return their files."""
    payload = json.loads(Path(inputs).read_text(encoding="utf-8"))
    stems = [str(value) for value in payload.get("stems", ())]
    if stem is None:
        if len(stems) != 1:
            raise ValueError("choose stem= for a handoff containing multiple recordings")
        stem = stems[0]
    if stem not in stems:
        raise ValueError(f"{stem!r} is not named in the Motion handoff")
    prepared = payload.get("prepared", {}).get(stem)
    if not prepared:
        raise ValueError(f"{stem}: Motion input stacks were not prepared")
    config = Path(prepared["config"])
    if not config.is_file():
        raise FileNotFoundError(config)

    root = _engine(Path(folder))
    run_name = f"pymicroglia_{stem}"
    accepted = root / "m22_accepted_history" / run_name
    if accepted.exists():
        raise FileExistsError(f"immutable Motion run already exists: {accepted}")
    log = Path(folder) / f"{stem}_motion.log"
    try:
        _run_frozen(root, stem, run_name, config, log)
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-25:]) if log.is_file() else ""
        raise RuntimeError(f"Motion failed for {stem}; see {log}\n{tail}") from exc
    raw = config.parent / prepared["pins"]["registered_raw"][
        "relative_to_registered_input_dir"]
    if not accepted.is_dir():
        raise FileNotFoundError(f"Motion finished without accepted labels: {accepted}")
    labels = accepted / "out" / f"{stem}.tif"
    with tifffile.TiffFile(raw) as opened:
        raw_frames = int(opened.series[0].shape[0])
    with tifffile.TiffFile(labels) as opened:
        label_frames = int(opened.series[0].shape[0])
    offset = raw_frames - label_frames
    if offset < 0 or offset > 2:
        raise ValueError(f"unexpected Motion frame alignment: {offset} leading frames")
    evidence = config.parent / prepared["pins"]["motion_composite"][
        "relative_to_motion_input_dir"]
    return TrackingResult.from_folder(accepted, stem, raw=raw,
                                      evidence=evidence,
                                      source_frame_offset=offset)
