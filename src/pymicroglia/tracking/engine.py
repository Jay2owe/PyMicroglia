"""Run the frozen Motion engine shipped in the PyMicroglia wheel.

The engine is imported from an isolated per-run folder. Its historical flat
module names are restored after a run, so they do not stay in PyMicroglia's
import namespace.
Versioned runtime repairs are recorded in the affected stage's provenance;
the frozen engine and its calibration remain unchanged.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
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
                            from . import checkpoints, residency_frames, repair_composition, lineage_overlaps, owner_consensus_conflicts, reciprocal_overlaps, owner_identity_accounting, stage_fallbacks, pipeline_fallbacks
                            history = importlib.import_module("accepted_history")
                            with residency_frames.install(history, checkpoints, root), \
                                    repair_composition.install(history, checkpoints, root), \
                                    lineage_overlaps.install(history, checkpoints, root), \
                                    owner_consensus_conflicts.install(history, checkpoints, root), \
                                    reciprocal_overlaps.install(history, checkpoints, root), \
                                    owner_identity_accounting.install(history, checkpoints, root), \
                                    stage_fallbacks.install(history, checkpoints, root), \
                                    pipeline_fallbacks.install(pipeline, importlib.import_module("common"), root):
                                _execute_pipeline(pipeline, code, root, run_name, stem, config,
                                                  bounded_memory, working_array_min_bytes, resume=resume)
                            # The final stage replays tracking, so it needs the same anchor policy.
                            with stage_fallbacks.anchor_names(stage_fallbacks.writer(root/('final_'+stage_fallbacks.AUDIT))):
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
    from .pipeline_fallbacks import final_corrections
    cfg = importlib.import_module('common').Config.load(Path(config))
    if bounded_memory:
        from .correction_storage import bounded_corrections
        with bounded_corrections(root,minimum_bytes=working_array_min_bytes) as pool:
            return final_corrections(lambda: run_current(root,stem,run_name,config,folder,resume=resume,array_pool=pool),
                                     root,stem,run_name,cfg,resume=resume)
    return final_corrections(lambda: run_current(root,stem,run_name,config,folder,resume=resume),
                             root,stem,run_name,cfg,resume=resume)


def _execute_pipeline(pipeline, code, root, run_name, stem, config,
                      bounded_memory, working_array_min_bytes, *, resume=False):
    if bounded_memory:
        from .engine_bounded import run as run_bounded
        from .frozen_io import install
        import tempfile
        common = importlib.import_module("common")
        with tempfile.TemporaryDirectory(prefix="motion-display-") as scratch:
            from auto_organotypic.array_storage import Pool,install as install_arrays
            pool=Pool(Path(scratch)/'working_arrays',minimum_bytes=working_array_min_bytes,temporary_ram_bytes=32*1024**2,released_bytes=1073741824)
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
        from .engine_bounded import run as run_staged
        common = importlib.import_module('common')
        with tempfile.TemporaryDirectory(prefix='motion-native-storage-') as scratch:
            with install(common, code, Path(scratch)):
                run_staged(pipeline, common, run_name, stem, config, native_review=True)


def _engine(folder: Path) -> Path:
    from . import shipped_engines
    source = shipped_engines.folder(shipped_engines.MOTION)
    digest = shipped_engines.fingerprint(source)
    target = folder / "engine"
    # The marker keeps its name so run folders keep their layout; it now holds the folder's fingerprint.
    marker = target / "archive.sha256"
    if marker.is_file():
        if marker.read_text(encoding="ascii").strip() != digest:
            raise ValueError("this Motion run folder contains a different engine snapshot")
        if not (target / "code/pipeline.py").is_file():
            raise FileNotFoundError(target / "code/pipeline.py")
        return target
    if target.exists():
        # Unpacking stopped before its marker: keep that copy and unpack again.
        from .checkpoints import set_aside
        set_aside(target, "incomplete")
    target.mkdir(parents=True)
    shipped_engines.copy(source, target)
    marker.write_text(digest + "\n", encoding="ascii")
    return target


#: Waits before each retry of a run stopped by memory, disk or a held file.
RETRY_WAITS = (60, 300)


def _transient(error: BaseException) -> bool:
    """Memory, disk and file-lock failures may clear; a retry resumes from saved stages."""
    return isinstance(error, (MemoryError, PermissionError, FileExistsError)) or (
        isinstance(error, OSError) and not isinstance(error, FileNotFoundError))


def _run_with_retries(root, stem, run_name, config, config_values, log, *, bounded_memory,
                      working_array_min_bytes, resume):
    """Run Motion; retry transient stops, then publish the last finished labels.

    Data problems inside a stage are handled by the stage fallbacks. Here a
    run stopped by memory, disk or a held file is retried in the bounded
    route, which uses less memory and resumes from verified stage
    checkpoints. If every attempt stops, the deepest finished stage's labels
    are published, marked degraded; only a run with no finished identity
    stage (inputs, motion or first tracking failed) still raises.
    """
    import gc
    import time
    attempt_bounded, attempt_resume = bounded_memory, resume
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            if attempt_bounded:
                storage_options={} if working_array_min_bytes==32*1024**2 else dict(working_array_min_bytes=working_array_min_bytes)
                if attempt_resume:storage_options['resume']=True
                _run_frozen(root, stem, run_name, config, log, bounded_memory=True,**storage_options)
            else:
                _run_frozen(root, stem, run_name, config, log)
            return
        except Exception as exc:
            error = exc
            if attempt < len(RETRY_WAITS) and _transient(exc):
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(f"\nRETRY {attempt + 1}: {type(exc).__name__}: {exc}; "
                                 f"resuming in the bounded route after {RETRY_WAITS[attempt]} s\n")
                gc.collect()
                time.sleep(RETRY_WAITS[attempt])
                attempt_bounded = attempt_resume = True
                continue
            break
    from .pipeline_fallbacks import publish_without_corrections
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace")
                     .splitlines()[-25:]) if log.is_file() else ""
    try:
        publish_without_corrections(
            root, stem, run_name, f"run stopped: {type(error).__name__}: {error}"[:2000],
            source_frame_offset=config_values.get("source_frame_offset", 0),
            frame_interval_min=config_values.get("frame_interval_min", 30.0),
            resume=attempt_resume)
    except RuntimeError:
        if isinstance(error, (ValueError, FileNotFoundError)):
            raise error  # input or settings problem: report it as itself
        raise RuntimeError(f"Motion failed for {stem} before any identity stage finished; "
                           f"see {log}\n{tail}") from error


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
    if bounded_memory and working_array_min_bytes<=0:
        raise ValueError('Working-array storage threshold must be positive')
    config_values = json.loads(config.read_text(encoding="utf-8"))
    _run_with_retries(root, stem, run_name, config, config_values, log, bounded_memory=bounded_memory,
                      working_array_min_bytes=working_array_min_bytes, resume=resume)
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
