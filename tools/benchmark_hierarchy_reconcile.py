"""Replay the complete hierarchy stage using frozen predecessor and evidence."""
import argparse
from contextlib import ExitStack
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'config', 'output', 'storage-source'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--snapshot', type=Path)
    p.add_argument('--reference-output', type=Path)
    p.add_argument('--run', required=True)
    p.add_argument('--stem', required=True)
    p.add_argument('--no-profile',action='store_true')
    p.add_argument('--group',choices=['hierarchy','quarantine'],default='hierarchy')
    a = p.parse_args()
    for name in ('engine','config','output','storage_source','adapter','reference_output','snapshot'):
        value=getattr(a,name)
        if value is not None:setattr(a,name,value.resolve())
    platform.platform = lambda *args, **kwargs: 'Windows-hierarchy-replay'
    import numpy as np
    import psutil
    import tifffile
    if psutil.virtual_memory().available < 2*1024**3 or shutil.disk_usage(a.output.parent).free < 3*1024**3:
        raise RuntimeError('Insufficient memory/disk for hierarchy replay')
    a.output.mkdir(exist_ok=False)
    sys.path.insert(0, str(a.engine.resolve()/'code'))
    if a.snapshot:sys.path.insert(0,str(a.snapshot))
    import accepted_history as history
    import common
    import optimized_pipeline
    from benchmark_working_storage import load
    storage = load(a.storage_source.resolve())
    cfg = common.Config.load(a.config)
    source = a.engine/'m22_accepted_history'/a.run/'mid/accepted_history'
    root = a.output/'stage'
    shutil.copytree(source/'07_trimmed_evidence', root/'07_trimmed_evidence')
    predecessor='04_issue008.tif' if a.group=='hierarchy' else '03_issue007.tif'
    successor='05_issue009.tif' if a.group=='hierarchy' else '04_issue008.tif'
    labels = tifffile.imread(source/predecessor)
    raw=lag=None
    if a.group=='quarantine':
        raw=tifffile.imread(source/'07_trimmed_evidence/out'/f'{a.stem}_registered_raw.tif')
        lag=tifffile.imread(source/'07_trimmed_evidence/out'/f'{a.stem}_lag01_float.tif')
    holder = {}
    proc = psutil.Process()
    cpu, started = proc.cpu_times(), time.perf_counter()
    def execute():
        with ExitStack() as stack:
            pool = storage.Pool(a.output/'scratch', minimum_bytes=1024**2,
                temporary_ram_bytes=32*1024**2 if a.snapshot else 0)
            stack.enter_context(storage.install(pool))
            if a.snapshot:
                from pymicroglia.tracking import frozen_io
                stack.enter_context(frozen_io.install(common,a.engine/'code',a.output/'scratch',array_pool=pool))
            if a.adapter:
                sys.path.insert(0, str(a.adapter.resolve()))
                import hierarchy_metric_reuse as reuse
                holder['counters'] = stack.enter_context(reuse.install_loader(history))
            profiler = None if a.no_profile else cProfile.Profile()
            if profiler:profiler.enable()
            try:
                if a.group=='hierarchy':
                    candidate, _ = history._issue009(labels, None, None, a.stem, root, cfg.source_frame_offset,
                                                     frame_interval_min=cfg.values['frame_interval_min'])
                else:
                    candidate = history._issue008(labels, raw, lag, a.stem, root, cfg.source_frame_offset,
                                                     frame_interval_min=cfg.values['frame_interval_min'])
            finally:
                if profiler:
                    profiler.disable()
                    profiler.dump_stats(a.output/'profile.pstats')
            holder['labels_equal'] = bool(np.array_equal(candidate, tifffile.imread(source/successor)))
    optimized_pipeline.main(['--no-evidence-cache'], runner=execute)
    elapsed = time.perf_counter()-started
    after = proc.cpu_times()
    checks = {}
    unavailable_intermediate_references = []
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix not in ('.tif', '.csv', '.json'):
            continue
        relative = path.relative_to(root)
        expected = (a.reference_output or source)/relative
        if not expected.is_file() and not a.reference_output and path.suffix=='.tif':
            if relative.parts[0]=='10_hierarchy_inputs':expected=source/'04_issue008.tif'
            elif relative.parts[0]=='13_hierarchy_reconcile':expected=source/'05_issue009.tif'
            elif relative.parts[0]=='11_ordered_quarantine':
                unavailable_intermediate_references.append(str(relative))
                continue
            elif a.group=='quarantine' and relative.parts[0] in ('08_global_quarantine','09_general_merge_bridge'):
                unavailable_intermediate_references.append(str(relative))
                continue
        if path.suffix == '.tif':
            same = expected.is_file() and np.array_equal(tifffile.imread(path), tifffile.imread(expected))
        else:
            same = expected.is_file() and path.read_bytes() == expected.read_bytes()
        checks[str(relative)] = bool(same)
    memory = proc.memory_info()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(status='pass' if holder['labels_equal'] and checks and all(checks.values()) else 'fail',
        seconds=elapsed, cpu_seconds=after.user+after.system-cpu.user-cpu.system,
        peak_resident_bytes=getattr(memory,'peak_wset',memory.rss), checks=checks,
        labels_equal=holder['labels_equal'], counters=holder.get('counters'),
        unavailable_intermediate_references=unavailable_intermediate_references,
        inputs={str(path):sha(path) for path in [a.config, source/predecessor, *sorted((source/'07_trimmed_evidence/out').glob('*'))]},
        sources={str(path):sha(path) for path in [Path(__file__),a.storage_source,
            *(([a.adapter/'hierarchy_metric_reuse.py']) if a.adapter else [])]},
        profiling=not a.no_profile,
        group=a.group,
        scope='Complete '+a.group+' operation and all available emitted-file references; peak includes verification; frozen input preparation excluded')
    if a.snapshot:
        report['sources'].update({str(path):sha(path) for path in (a.snapshot/'pymicroglia/tracking').glob('*.py')})
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','sources')}))
    if report['status']!='pass':raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        if '--output' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--output')+1])
            if folder.is_dir():(folder/'error.log').write_text(traceback.format_exc())
        raise
