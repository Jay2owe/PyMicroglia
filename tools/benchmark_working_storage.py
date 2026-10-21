"""Compare disposable storage costs, exact values and bounded retained snapshots.

Run each implementation in its own process. Pass a frozen array_storage.py;
no scientific pipeline, accepted output or durable checkpoint is modified.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import tempfile
import time


def load(path):
    import auto_organotypic
    for name in ('array_selection','array_storage'):
        source=path.with_name(name+'.py')
        if not source.is_file():continue
        spec=importlib.util.spec_from_file_location('auto_organotypic.'+name,source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        setattr(auto_organotypic,name,module)
    return sys.modules['auto_organotypic.array_storage']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=16)
    parser.add_argument('--threshold-mib',type=int,default=1)
    parser.add_argument('--temporary-ram-mib',type=int,default=0)
    parser.add_argument('--snapshot-window',type=int,default=0)
    args=parser.parse_args()
    platform.platform=lambda *a,**k:'Windows-storage-comparison'
    import numpy as np
    import psutil
    storage=load(args.source.resolve())
    values=np.random.default_rng(95).integers(0,60,(471,471),dtype=np.uint16 if args.snapshot_window else np.int64)
    original_where=np.where
    expected=hashlib.sha256()
    for frame in range(args.frames):
        expected.update(original_where(values==(frame%60),values,0).tobytes())
    process=psutil.Process()
    before_io=process.io_counters();before_cpu=process.cpu_times()
    flushes=0
    original_flush=np.memmap.flush
    def flush(mapping,*a,**k):
        nonlocal flushes
        flushes+=1
        return original_flush(mapping,*a,**k)
    actual=hashlib.sha256()
    start=time.perf_counter()
    args.scratch.mkdir(parents=True,exist_ok=True)
    try:
        np.memmap.flush=flush
        with tempfile.TemporaryDirectory(prefix='storage-trial-',dir=args.scratch) as work:
            options=dict(minimum_bytes=args.threshold_mib*1024**2)
            if args.temporary_ram_mib:options['temporary_ram_bytes']=args.temporary_ram_mib*1024**2
            pool=storage.Pool(work,**options)
            with storage.install(pool):
                retained=[]
                for frame in range(args.frames):
                    result=np.where(values==(frame%60),values,0)
                    if args.snapshot_window:
                        result=pool.snapshot(result)
                        retained.append(result)
                        if len(retained)>args.snapshot_window:retained.pop(0)
                    actual.update(memoryview(result).cast('B'))
                    del result
                retained.clear()
            allocations=pool.allocations;allocated_bytes=pool.allocated_bytes
    finally:np.memmap.flush=original_flush
    seconds=time.perf_counter()-start
    memory=process.memory_info();io=process.io_counters();cpu=process.cpu_times()
    report=dict(status='pass' if actual.digest()==expected.digest() else 'fail',
        frames=args.frames,shape=list(values.shape),dtype=str(values.dtype),
        threshold_mib=args.threshold_mib,seconds=seconds,flush_calls=flushes,
        temporary_ram_mib=args.temporary_ram_mib,
        snapshot_window=args.snapshot_window,
        temporary_ram_peak_bytes=getattr(pool,'temporary_ram_peak_bytes',0),
        temporary_ram_allocations=getattr(pool,'temporary_ram_allocations',0),
        allocated_bytes=allocated_bytes,allocations=allocations,
        peak_resident_bytes=getattr(memory,'peak_wset',memory.rss),
        io_delta={key:getattr(io,key)-getattr(before_io,key) for key in io._fields},
        cpu_seconds=cpu.user+cpu.system-before_cpu.user-before_cpu.system,
        output_sha256=actual.hexdigest(),source=str(args.source.resolve()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        scope='Repeated frame selections and optional retained history snapshots, including allocation and cleanup; not complete tracking')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
