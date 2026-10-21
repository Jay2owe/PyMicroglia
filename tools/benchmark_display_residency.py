"""Render every native review pixel using bounded temporary display residency."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','config','snapshot','storage','labels','review','output','scratch'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stem',required=True)
    p.add_argument('--mode',choices=('baseline','candidate'),required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-display-residency-benchmark'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.snapshot.resolve())]
    import numpy as np
    import psutil
    import tifffile
    import common,pipeline
    from pymicroglia.tracking.frozen_io import overlay_renderer
    from benchmark_working_storage import load
    storage=load(a.storage.resolve());a.scratch.mkdir(parents=True,exist_ok=True)
    cfg=common.Config.load(a.config);paths=pipeline._pinned_paths(cfg,a.stem)
    expected=hashlib.sha256()
    with tifffile.TiffFile(a.review) as opened:
        shape=opened.series[0].shape
        for page in opened.pages:expected.update(page.asarray().tobytes())
    process=psutil.Process();cpu=process.cpu_times();started=time.perf_counter()
    pool=storage.Pool(a.scratch,minimum_bytes=1024**2,temporary_ram_bytes=32*1024**2)
    with storage.install(pool):
        def mapped(path,name):
            target=a.scratch/(name+'.dat')
            with tifffile.TiffFile(path) as opened:result=pool.adopt(opened.series[0].asarray(out=str(target)),target)
            result.flags.writeable=False
            return result
        raw=mapped(paths['registered_raw'],'raw');labels=mapped(a.labels,'labels')
        renderer=overlay_renderer(common,a.scratch)
        tick=time.perf_counter()
        with (pool.resident_dimensions(4) if a.mode=='candidate' else nullcontext()):
            result=renderer(raw,labels,thick=2)
        seconds=time.perf_counter()-tick
        actual=hashlib.sha256()
        for frame in result:actual.update(frame.tobytes())
        assert tuple(result.shape)==tuple(shape) and actual.digest()==expected.digest()
        del result,frame,raw,labels
        gc.collect()
    after=process.cpu_times();mem=process.memory_info()
    report=dict(status='pass',mode=a.mode,render_seconds=seconds,total_seconds=time.perf_counter()-started,
        cpu_seconds=after.user+after.system-cpu.user-cpu.system,peak_resident_bytes=mem.peak_wset,
        peak_private_bytes=mem.peak_pagefile,temporary_ram_peak_bytes=pool.temporary_ram_peak_bytes,
        mapped_allocations=pool.allocations,mapped_bytes=pool.allocated_bytes,pixels_sha256=actual.hexdigest(),
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 (Path(__file__),a.storage,a.snapshot/'pymicroglia/tracking/frozen_io.py')},
        scope='All 476 native review frames match the original TIFF pixels; no scientific stage recalculation')
    report['inputs']={}
    for path in (a.config,a.labels,a.review,paths['registered_raw']):
        with path.open('rb') as handle:report['inputs'][str(path)]=hashlib.file_digest(handle,'sha256').hexdigest()
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
