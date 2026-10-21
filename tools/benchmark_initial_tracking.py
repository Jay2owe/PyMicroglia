"""Replay initial assignment from the original full recording's observations."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine','config','snapshot','adapter','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--stem',required=True)
    p.add_argument('--native',action='store_true')
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-initial-tracking-replay'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.snapshot.resolve()),str(a.adapter.resolve())]
    import numpy as np
    import psutil
    import tifffile
    import tracking
    import common
    import pipeline
    import optimized_pipeline
    import tracking_feature_batch
    import lazy_motion_counts
    from pymicroglia.tracking import frozen_io
    from benchmark_working_storage import load
    storage=load(a.snapshot.resolve()/'array_storage.py')
    a.output.mkdir(exist_ok=False)
    cfg=common.Config.load(a.config)
    paths=pipeline._pinned_paths(cfg,a.stem)
    holder={}
    process=psutil.Process()
    if psutil.virtual_memory().available<3*1024**3:raise RuntimeError('Insufficient free RAM')
    source_files=[Path(__file__),a.config,*a.snapshot.rglob('*.py')]
    source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    started=time.perf_counter();cpu=process.cpu_times()
    def execute():
        with ExitStack() as stack:
            pool=storage.Pool(a.output/'scratch',minimum_bytes=1024**2,
                temporary_ram_bytes=32*1024**2 if a.native else 0)
            stack.enter_context(storage.install(pool))
            stack.enter_context(frozen_io.install(common,a.engine.resolve()/'code',a.output/'scratch',array_pool=pool))
            if not a.native:
                holder['batch']=stack.enter_context(tracking_feature_batch.install(tracking))
                holder['motion']=stack.enter_context(lazy_motion_counts.install(tracking,frozen_io.LabelMask))
                assert holder['motion']['supported']
            def mapped(path,name):
                target=a.output/'scratch'/(name+'.dat')
                with tifffile.TiffFile(path) as opened:
                    value=pool.adopt(opened.series[0].asarray(out=str(target)),target)
                value.flags.writeable=False
                return value
            raw=mapped(paths['registered_raw'],'raw')
            lag=mapped(paths['lag_float'],'lag')
            observations,table,anchor=pipeline._load_anchor_inputs(a.stem,a.run)
            tick=time.perf_counter()
            labels,links=tracking.track_from_anchor(observations,raw,lag,anchor,cfg.values['tracking'],table)
            holder['tracking_seconds']=time.perf_counter()-tick
            reference=a.engine/'m3_track'/a.run/'out'
            holder['labels_equal']=bool(np.array_equal(labels,tifffile.imread(reference/(a.stem+'_initial_identities.tif'))))
            output=a.output/'links.csv'
            links.to_csv(output,index=False)
            holder['links_equal']=output.read_bytes()==(reference/'links.csv').read_bytes()
            holder['shape']=list(labels.shape)
    optimized_pipeline.main(['--no-evidence-cache'],runner=execute)
    after=process.cpu_times();memory=process.memory_info()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(p)==value for p,value in source_hashes.items()),'Measured sources changed'
    report=dict(status='pass' if holder['labels_equal'] and holder['links_equal'] else 'fail',
                seconds=time.perf_counter()-started,cpu_seconds=after.user+after.system-cpu.user-cpu.system,
                peak_resident_bytes=memory.peak_wset,peak_private_bytes=memory.peak_pagefile,**holder,
                sources=source_hashes,
                scope='Complete initial assignments and links; excludes review rendering and later stages')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
