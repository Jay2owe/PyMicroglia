"""Replay the complete duplicate-history operation from its saved predecessor.

Compares every returned array/table/metric with the frozen saved stage. No
tracking job or accepted file is modified. Output directory must be new.
"""
import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--operation', choices=('duplicate', 'flip'), default='duplicate')
    p.add_argument('--stem', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--storage-source', type=Path)
    p.add_argument('--threshold-mib', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    platform.platform = lambda *x, **k: 'Windows-duplicate-history-comparison'
    import numpy as np
    import pandas as pd
    import psutil
    import tifffile
    if psutil.virtual_memory().available < 3 * 1024**3:
        raise RuntimeError('Need 3 GiB free RAM before this isolated trial')
    if shutil.disk_usage(a.output.parent).free < 4 * 1024**3:
        raise RuntimeError('Need 4 GiB free disk before this isolated trial')
    a.output.mkdir(exist_ok=False)
    sys.path.insert(0, str((a.engine / 'code').resolve()))
    history = importlib.import_module('accepted_history')
    duplicate = importlib.import_module('duplicate_soma_exclusivity')
    common = importlib.import_module('common')
    config = common.Config.load(a.config)
    source = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    sources = json.loads((a.engine / 'm0_inputs' / a.run / 'out' / f'{a.stem}_sources.json').read_text())
    raw_path = Path(sources['files']['registered_raw']['path'])
    if a.operation == 'duplicate':
        predecessor = '42_issue091_recording_boundary_owner_cycle'
        successor = '43_issue092_complete_duplicate_lineage'
        stage_name = '54_complete_duplicate_lineage'
    else:
        predecessor = '51_well_a5_issue003_branched_motion_lineage_recovery'
        successor = '52_well_a5_issue003_complete_flip_recovery'
        stage_name = '67_complete_flip_recovery'
        if a.adapter:
            raise ValueError('Duplicate-only adapter cannot accelerate the flip stage')
    labels_path = source / (predecessor + '.tif')
    unclaimed_path = source / (predecessor + '_unclaimed.tif')
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(raw_path)[config.source_frame_offset:] if a.operation == 'duplicate' else None
    if raw is not None and raw.shape != labels.shape:
        raise ValueError(f'Unexpected raw alignment: {raw.shape} != {labels.shape}')
    inputs = [labels_path, unclaimed_path, a.config]
    if raw is not None:
        inputs.append(raw_path)
    for relative in ('26_latent_body_tracks/out/latent_track_points.csv',
                     '25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv'):
        origin = source / relative
        dest = a.output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, dest)
        inputs.append(origin)
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024**2), b''):
                h.update(block)
        return h.hexdigest()
    source_hashes = {str(path): sha(path) for path in (a.engine / 'code').glob('*.py')}
    source_hashes[str(Path(__file__).resolve())] = sha(Path(__file__))
    input_hashes = {str(path): sha(path) for path in inputs}
    context = nullcontext({})
    if a.adapter:
        sys.path.insert(0, str(a.adapter.resolve()))
        adapter = importlib.import_module('identity_reuse')
        context = adapter.install(duplicate)
        source_hashes[str(Path(adapter.__file__))] = sha(Path(adapter.__file__))
    process = psutil.Process()
    cpu = process.cpu_times()
    start = time.perf_counter()
    holder = {}
    stage_times = {}
    def execute():
        # The production context resolves recording-relative units and installs
        # accepted accelerations. Load settings inside that same context.
        effective_config = common.Config.load(a.config)
        with ExitStack() as stack:
            if a.storage_source:
                from benchmark_working_storage import load
                storage = load(a.storage_source.resolve())
                pool = storage.Pool(a.output / 'scratch', minimum_bytes=a.threshold_mib * 1024**2)
                stack.enter_context(storage.install(pool))
                holder['pool'] = pool
                for name in ('array_storage.py', 'array_selection.py'):
                    path = a.storage_source.with_name(name)
                    if path.is_file():
                        source_hashes[str(path)] = sha(path)
            counters = stack.enter_context(context)
            holder['counters'] = counters
            def timed(module, name):
                old = module.recover
                def call(*args, **kwargs):
                    then = time.perf_counter()
                    try:
                        return old(*args, **kwargs)
                    finally:
                        stage_times[name] = time.perf_counter()-then
                module.recover = call
                stack.callback(setattr, module, 'recover', old)
            if a.operation == 'duplicate':
                timed(duplicate, 'duplicate_projection_seconds')
                timed(history.complete_duplicate_lineage, 'lineage_allocation_seconds')
                holder['result'] = history._issue092_complete_duplicate_lineage(
                    labels, unclaimed, raw, effective_config, a.output)
            else:
                holder['result'] = history._well_a5_issue003_complete_flip_recovery(
                    labels, unclaimed, effective_config, a.output)
    optimized = importlib.import_module('optimized_pipeline')
    optimized.main(['--no-evidence-cache'], runner=execute)
    result, counters = holder.pop('result'), holder.pop('counters')
    seconds = time.perf_counter() - start
    after_cpu = process.cpu_times()
    memory = process.memory_info()
    comparison = {}
    for index, name in enumerate((successor + '.tif', successor + '_unclaimed.tif')):
        reference = tifffile.imread(source / name)
        comparison[name] = bool(np.array_equal(result[index], reference))
        del reference
    reference = source / stage_name / 'out'
    actual = a.output / stage_name / 'out'
    for path in actual.iterdir():
        comparison[path.name] = path.read_bytes() == (reference / path.name).read_bytes()
    report = dict(status='pass' if all(comparison.values()) else 'fail', operation=a.operation, seconds=seconds,
        cpu_seconds=after_cpu.user + after_cpu.system - cpu.user - cpu.system,
        peak_resident_bytes=getattr(memory, 'peak_wset', memory.rss), shape=list(labels.shape),
        comparison=comparison, counters=counters, stage_times=stage_times,
        storage=None if not a.storage_source else dict(source=str(a.storage_source),
            threshold_mib=a.threshold_mib, allocations=holder['pool'].allocations,
            allocated_bytes=holder['pool'].allocated_bytes),
        inputs=input_hashes, sources=source_hashes,
        scope='Complete selected identity-history operation; shared desktop; saved predecessor; no downstream rerun')
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','sources')}), flush=True)
    if report['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
