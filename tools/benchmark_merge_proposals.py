"""Replay full merge handling; compare masks and every recorded split score."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','config','snapshot','adapter','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--stem',required=True)
    p.add_argument('--geometry',action='store_true')
    p.add_argument('--local-foreground',action='store_true')
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-merge-proposal-replay'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.snapshot.resolve()),str(a.adapter.resolve())]
    import numpy as np
    import psutil
    import tifffile
    import common,pipeline,optimized_pipeline,persistent_merges
    import merge_proposal_storage
    from benchmark_working_storage import load
    from pymicroglia.tracking.correction_audit import array_sha256
    if psutil.virtual_memory().available<2*1024**3 or shutil.disk_usage(a.output.parent).free<4*1024**3:
        raise RuntimeError('Insufficient free memory/disk')
    a.output.mkdir(exist_ok=False)
    cfg=common.Config.load(a.config)
    paths=pipeline._pinned_paths(cfg,a.stem)
    source=a.engine/'m7_persistence'/a.run/'out'/(a.stem+'.tif')
    reference=a.engine/'m8_merge'/a.run
    files=[Path(__file__),Path(persistent_merges.__file__),Path(merge_proposal_storage.__file__),
           a.snapshot/'array_storage.py',a.config,source,paths['registered_raw'],paths['lag_float']]
    if a.geometry:
        import merge_geometry_reuse
        from pymicroglia.tracking.frozen_io import LabelMask
        files.append(Path(merge_geometry_reuse.__file__))
    if a.local_foreground:
        import merge_foreground_scope
        files.append(Path(merge_foreground_scope.__file__))
    def sha(path):
        with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
    hashes={str(f):sha(f) for f in files}
    holder={};process=psutil.Process();cpu=process.cpu_times();started=time.perf_counter()
    def execute():
        storage=load(a.snapshot/'array_storage.py')
        with ExitStack() as stack:
            pool=storage.Pool(a.output/'scratch',minimum_bytes=1024**2,temporary_ram_bytes=32*1024**2)
            stack.enter_context(storage.install(pool))
            if a.geometry:
                holder['geometry']=stack.enter_context(merge_geometry_reuse.install(persistent_merges,LabelMask))
                assert holder['geometry']['supported']
            holder['counters']=stack.enter_context(merge_proposal_storage.install(persistent_merges))
            assert holder['counters']['supported']
            if a.local_foreground:
                holder['foreground_scope']=stack.enter_context(merge_foreground_scope.install(persistent_merges))
                assert holder['foreground_scope']['supported']
            def mapped(path,name):
                target=a.output/'scratch'/(name+'.dat')
                with tifffile.TiffFile(path) as opened:
                    result=pool.adopt(opened.series[0].asarray(out=str(target)),target)
                result.flags.writeable=False
                return result
            labels=mapped(source,'labels');raw=mapped(paths['registered_raw'],'raw');lag=mapped(paths['lag_float'],'lag')
            parent_hash=array_sha256(labels)
            tick=time.perf_counter()
            result,events,inferred=persistent_merges.carry_identities_through_merges(
                labels,raw,lag,cfg.values['accepted_postprocessing']['persistent_merges'])
            holder['merge_seconds']=time.perf_counter()-tick
            holder['labels_equal']=bool(np.array_equal(result,tifffile.imread(reference/'out'/(a.stem+'.tif'))))
            holder['inferred_equal']=bool(np.array_equal(inferred,tifffile.imread(reference/'mid'/(a.stem+'_inferred_merge_pixels.tif'))))
            holder['input_unchanged']=array_sha256(labels)==parent_hash
            events.to_csv(a.output/'persistent_merge_events.csv',index=False)
            events.attrs['frame_rows'].to_csv(a.output/'persistent_merge_frame_partitions.csv',index=False)
            holder['tables_equal']={name:(a.output/name).read_bytes()==(reference/'out'/name).read_bytes()
                for name in ('persistent_merge_events.csv','persistent_merge_frame_partitions.csv')}
            holder['temporary_ram_peak_bytes']=pool.temporary_ram_peak_bytes
    optimized_pipeline.main(['--no-evidence-cache'],runner=execute)
    after=process.cpu_times();mem=process.memory_info()
    passed=holder['labels_equal'] and holder['inferred_equal'] and holder['input_unchanged'] and all(holder['tables_equal'].values())
    report=dict(status='pass' if passed else 'fail',seconds=time.perf_counter()-started,
        cpu_seconds=after.user+after.system-cpu.user-cpu.system,peak_resident_bytes=mem.peak_wset,
        peak_private_bytes=mem.peak_pagefile,**holder,sources=hashes,
        scope='Complete 476-frame merge decisions and scores; rendering skipped only when scientific content is identical')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
