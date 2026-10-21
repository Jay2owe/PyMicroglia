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
from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from importlib import resources
from pathlib import Path

import tifffile

from .contract import TrackingResult, DecisionTables, MOTION_DECISION_TABLES, MOTION_LABEL_TABLES

NATIVE_METHOD_VERSION = "2026-10-12-native-speed59-recording-audit192"
METHOD_VERSION = "2026-10-12-R218-A004-complete-corrections"
_IMPORT_LOCK = threading.RLock()


def _inside(path: object, root: Path) -> bool:
    try:
        return Path(str(path)).resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _run_frozen(root: Path, stem: str, run_name: str, config: Path,
                log: Path, *, bounded_memory: bool = False,working_array_min_bytes: int = 32*1024**2,
                resume: bool = False) -> None:
    code = root / "code"
    names = {path.stem for path in code.rglob("*.py")
             if path.stem != "__init__"}
    from .checkpoints import exclusive
    with _IMPORT_LOCK, exclusive(root/'tracking.lock'):
        previous = {name: sys.modules.pop(name) for name in names
                    if name in sys.modules}
        sys.path.insert(0, str(code))
        try:
            with log.open("a" if resume else "w", encoding="utf-8") as handle:
                with redirect_stdout(handle), redirect_stderr(handle):
                    def execute():
                        # Import after installing contexts, as Motion's CLI does:
                        # direct function imports must capture the wrapped versions.
                        from . import detector_acceleration
                        detector = importlib.import_module("raw_physical_hypotheses")
                        implementation = Path(detector_acceleration.__file__)
                        (root / 'detector_acceleration.json').write_text(json.dumps({
                            'operation': 'batch independent core windows',
                            'backend': 'cpu',
                            'source_sha256': hashlib.sha256(implementation.read_bytes()).hexdigest(),
                            'scientific_settings': 'inherited from native detector',
                        }, indent=2) + '\n')
                        with detector_acceleration.install(detector):
                            pipeline = importlib.import_module("pipeline")
                            from . import checkpoints, residency_frames
                            history = importlib.import_module("accepted_history")
                            with residency_frames.install(history, checkpoints, root):
                                _execute_pipeline(pipeline, code, root, run_name, stem, config,
                                                  bounded_memory, working_array_min_bytes, resume=resume)
                            _execute_corrections(root,stem,run_name,config,root.parent,resume=resume,
                                bounded_memory=bounded_memory,working_array_min_bytes=working_array_min_bytes)
                    optimized = importlib.import_module("optimized_pipeline")
                    optimized.main(["--no-evidence-cache", "--stage-timings",
                                    str(root / "accepted_stage_timings.jsonl")], runner=execute)
        finally:
            sys.path.remove(str(code))
            for name, module in list(sys.modules.items()):
                if _inside(getattr(module, "__file__", None), root):
                    sys.modules.pop(name, None)
            sys.modules.update(previous)


def _execute_corrections(root, stem, run_name, config, folder, *, resume=False,
                         bounded_memory=False,working_array_min_bytes=32*1024**2):
    from .completed_feedback import run_current
    if bounded_memory:
        from .correction_storage import bounded_corrections
        with bounded_corrections(root,minimum_bytes=working_array_min_bytes):
            return run_current(root,stem,run_name,config,folder,resume=resume)
    return run_current(root,stem,run_name,config,folder,resume=resume)


def _execute_pipeline(pipeline, code, root, run_name, stem, config,
                      bounded_memory, working_array_min_bytes, *, resume=False):
    if bounded_memory:
        from .engine_bounded import run as run_bounded
        from .frozen_io import install
        import tempfile
        common = importlib.import_module("common")
        with tempfile.TemporaryDirectory(prefix="motion-display-") as scratch:
            from auto_organotypic.array_storage import Pool,install as install_arrays
            pool=Pool(Path(scratch)/'working_arrays',minimum_bytes=working_array_min_bytes)
            import inspect
            storage_record=dict(minimum_bytes=working_array_min_bytes,temporary_directory=scratch,
                code_sha256={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for path in (__file__,inspect.getfile(run_bounded),inspect.getfile(install),inspect.getfile(Pool))},status='running')
            storage_path=root/'working_array_storage.json'
            storage_path.write_text(json.dumps(storage_record,indent=2))
            with install_arrays(pool):
                with install(common, code, Path(scratch),array_pool=pool):
                    run_bounded(pipeline, common, run_name, stem, config,array_pool=pool,resume=resume)
            storage_record.update(status='complete',allocations=pool.allocations,allocated_bytes=pool.allocated_bytes)
            storage_path.write_text(json.dumps(storage_record,indent=2))
    else:
        # Keep the ordinary runner's stage order, thread contexts and outputs.
        # Share immutable label snapshots between cell masks and buffer review
        # images on disk, as in the bounded runner, without moving arithmetic
        # arrays or changing the scientific operations.
        import tempfile
        from .frozen_io import install
        common = importlib.import_module('common')
        with tempfile.TemporaryDirectory(prefix='motion-native-storage-') as scratch:
            with install(common, code, Path(scratch)):
                pipeline.run_base(run_name, stem, config)


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


def run(inputs, folder, *, stem: str | None = None,
        bounded_memory: bool = False,working_array_min_bytes: int = 32*1024**2,
        resume: bool = False) -> TrackingResult:
    """Run Motion; resume=True reuses verified stages in bounded-memory mode.

    Completed stage files remain immutable. The interrupted stage is preserved
    before retry; Motion's accepted-history checkpoints are carried forward.
    Inputs, settings, software and producer code must match the saved run.
    """
    if resume and not bounded_memory:
        raise ValueError('Motion checkpoint resume requires bounded_memory=True')
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
    accepted = root / "m23_completed_feedback" / run_name
    if accepted.exists() and not resume:
        raise FileExistsError(f"immutable Motion run already exists: {accepted}")
    log = Path(folder) / f"{stem}_motion.log"
    try:
        if bounded_memory:
            if working_array_min_bytes<=0:raise ValueError('Working-array storage threshold must be positive')
            storage_options={} if working_array_min_bytes==32*1024**2 else dict(working_array_min_bytes=working_array_min_bytes)
            if resume:storage_options['resume']=True
            _run_frozen(root, stem, run_name, config, log, bounded_memory=True,**storage_options)
        else:
            _run_frozen(root, stem, run_name, config, log)
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-25:]) if log.is_file() else ""
        raise RuntimeError(f"Motion failed for {stem}; see {log}\n{tail}") from exc
    config_values = json.loads(config.read_text(encoding="utf-8"))
    registered_directory = (config.parent / config_values["registered_input_dir"]).resolve()
    motion_directory = (config.parent / config_values["motion_input_dir"]).resolve()
    raw = registered_directory / prepared["pins"]["registered_raw"][
        "relative_to_registered_input_dir"]
    if not accepted.is_dir():
        raise FileNotFoundError(f"Motion finished without accepted labels: {accepted}")
    labels = accepted / "out" / f"{stem}.tif"
    with tifffile.TiffFile(raw) as opened:
        raw_frames = int(opened.series[0].shape[0])
    with tifffile.TiffFile(labels) as opened:
        label_frames = int(opened.series[0].shape[0])
    offset = raw_frames - label_frames
    expected_offset = config_values.get("source_frame_offset", 0)
    if (isinstance(expected_offset, bool) or not isinstance(expected_offset, int)
            or expected_offset < 0 or offset != expected_offset):
        raise ValueError(f"Motion frame alignment {offset} differs from configured exclusion {expected_offset}")
    evidence = (motion_directory / prepared["pins"]["motion_composite"][
        "relative_to_motion_input_dir"]).resolve()
    result = TrackingResult.from_folder(accepted, stem, raw=raw,
                                      evidence=evidence,
                                      source_frame_offset=offset)
    corrections = accepted/'mid/completed_feedback'
    if corrections.is_dir():
        native = root/'m22_accepted_history'/run_name
        tables = [(name,str((native/'mid/accepted_history'/relative).resolve()),grain)
                  for name,relative,grain in MOTION_DECISION_TABLES]
        tables += [(name,str((native/'out'/relative).resolve()),grain)
                   for name,relative,grain in MOTION_LABEL_TABLES]
        tables += [('completed_feedback/'+path.relative_to(corrections).with_suffix('').as_posix(),
                    str(path.resolve()),()) for path in sorted(corrections.rglob('*.csv'))]
        result = replace(result,decisions=DecisionTables(corrections,tuple(tables)))
    return result
