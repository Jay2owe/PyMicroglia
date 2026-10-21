"""Profile complete shared two-pass tracking from saved native inputs."""
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
    p=argparse.ArgumentParser()
    for name in ('engine','config','output','snapshot'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--stem',required=True)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--reference-tables',type=Path)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--mapped-inputs',action='store_true')
    p.add_argument('--compact-host',action='store_true')
    p.add_argument('--temporary-ram-mib',type=int,default=0)
    p.add_argument('--measure-feature-reuse',action='store_true')
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-tracking-reconstruction-profile'
    import numpy as np
    import psutil
    import tifffile
    if psutil.virtual_memory().available<2*1024**3 or shutil.disk_usage(a.output.parent).free<3*1024**3:
        raise RuntimeError('Insufficient free memory/disk')
    a.output.mkdir(exist_ok=False)
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.snapshot.resolve())]
    import accepted_history as ah
    import common
    import pipeline
    import optimized_pipeline
    from pymicroglia.tracking import frozen_io
    from benchmark_working_storage import load
    storage=load(a.snapshot.resolve()/'array_storage.py')
    cfg=common.Config.load(a.config)
    paths=pipeline._pinned_paths(cfg,a.stem)
    raw=lag=None
    if not a.mapped_inputs:
        raw=tifffile.imread(paths['registered_raw']);lag=tifffile.imread(paths['lag_float'])
    lineage=a.engine/'m19_layered_lineage'/a.run
    physical_path=lineage/'mid'/f'{a.stem}_physical_somas.tif'
    reservation_path=a.engine/'m12_identity_reservation'/a.run/'out'/f'{a.stem}.tif'
    physical=reservation=observations=table=anchor=None
    if not a.mapped_inputs:
        physical=tifffile.imread(physical_path);reservation=tifffile.imread(reservation_path)
        observations,table,anchor=pipeline._load_anchor_inputs(a.stem,a.run)
    reserved=ah._field_motion_reservations(a.run)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    inputs={str(p):sha(p) for p in [a.config,physical_path,reservation_path,paths['registered_raw'],paths['lag_float']]}
    timings={};holder={};process=psutil.Process();cpu=process.cpu_times();start=time.perf_counter()
    memory_boundaries=[]
    def capture_memory(stage):
        memory_boundaries.append(dict(stage=stage,seconds=time.perf_counter()-start,**process.memory_info()._asdict()))
    capture_memory('inputs_loaded')
    def execute():
        nonlocal raw,lag,physical,reservation,observations,table,anchor
        config=common.Config.load(a.config)
        with ExitStack() as stack:
            options=dict(minimum_bytes=1024**2)
            if a.temporary_ram_mib:options['temporary_ram_bytes']=a.temporary_ram_mib*1024**2
            pool=storage.Pool(a.output/'scratch',**options)
            stack.enter_context(storage.install(pool))
            if a.mapped_inputs:
                original_read=tifffile.imread
                read_count=0
                def mapped_read(path,*args,**kwargs):
                    nonlocal read_count
                    if args or kwargs:return original_read(path,*args,**kwargs)
                    target=a.output/'scratch'/f'input_{read_count}.dat';read_count+=1
                    with tifffile.TiffFile(path) as opened:
                        value=pool.adopt(opened.series[0].asarray(out=str(target)),target)
                    value.flags.writeable=False
                    return value
                tifffile.imread=mapped_read
                stack.callback(setattr,tifffile,'imread',original_read)
                raw=mapped_read(paths['registered_raw']);lag=mapped_read(paths['lag_float'])
                physical=mapped_read(physical_path);reservation=mapped_read(reservation_path)
                observations,table,anchor=pipeline._load_anchor_inputs(a.stem,a.run)
                capture_memory('mapped_inputs_loaded')
            stack.enter_context(frozen_io.install(common,a.engine.resolve()/'code',a.output/'scratch',array_pool=pool))
            if a.measure_feature_reuse:
                from pymicroglia.tracking import tracking_feature_batch as measured_batch
                from tracking_numeric_reuse_probe import observe
                holder['numeric_reuse_probe']=stack.enter_context(observe(measured_batch))
            if a.adapter:
                sys.path.insert(0,str(a.adapter.resolve()))
                import tracking_feature_batch
                import tracking
                holder['batch']=stack.enter_context(tracking_feature_batch.install(tracking))
                if a.compact_host:
                    import host_partition_reuse
                    import oscillatory_reconciliation
                    holder['host']=stack.enter_context(host_partition_reuse.install(oscillatory_reconciliation,ah))
            holder['pool']=pool
            for name in ('prepare_tracking_observations','track_physical_somas','multilayer_identity_confidence_table',
                         'identity_confidence_table','optimise_lineage_graph','optimise_host_merge_graph',
                         'optimise_territory_ownership_graph','reconcile_same_territory_replacements'):
                original=getattr(ah,name)
                def call(*v,_fn=original,_name=name,**kw):
                    capture_memory(_name+'_start')
                    tick=time.perf_counter();result=_fn(*v,**kw)
                    capture_memory(_name+'_end')
                    timings.setdefault(_name,[]).append(time.perf_counter()-tick)
                    if _name=='track_physical_somas':
                        number=len(timings[_name])
                        for i,value in enumerate(result):
                            if isinstance(value,np.ndarray):
                                digest=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                (a.output/f'pass_{number}_labels.sha256').write_text(digest)
                            else:value.to_csv(a.output/f'pass_{number}_table_{i}.csv',index=False)
                    return result
                setattr(ah,name,call);stack.callback(setattr,ah,name,original)
            profiler=cProfile.Profile() if a.profile else None
            if profiler:profiler.enable()
            try:holder['labels']=ah._issue005(config,physical,reservation,raw,lag,observations,table,anchor,reserved)
            finally:
                if profiler:
                    profiler.disable();profiler.dump_stats(a.output/'profile.pstats')
            reference=a.engine/'m22_accepted_history'/a.run/'mid/accepted_history/01_issue005.tif'
            holder['final_equal']=bool(np.array_equal(holder['labels'],tifffile.imread(reference)))
            del holder['labels']
    optimized_pipeline.main(['--no-evidence-cache'],runner=execute)
    seconds=time.perf_counter()-start;after=process.cpu_times();memory=process.memory_info()
    same=holder['final_equal']
    comparison={}
    if a.reference_tables:
        for path in a.output.glob('pass_*'):
            comparison[path.name]=path.read_bytes()==(a.reference_tables/path.name).read_bytes()
    same=same and all(comparison.values())
    report=dict(status='pass' if same else 'fail',seconds=seconds,
                cpu_seconds=after.user+after.system-cpu.user-cpu.system,
                peak_resident_bytes=getattr(memory,'peak_wset',memory.rss),
                peak_private_bytes=getattr(memory,'peak_pagefile',None),memory_boundaries=memory_boundaries,
                shape=list(physical.shape),
                final_labels_equal=same,timings=timings,inputs=inputs,comparison=comparison,counters=holder.get('batch'),
                sources={str(p):sha(p) for p in (a.engine/'code').glob('*.py')},
                profiling=a.profile,mapped_inputs=a.mapped_inputs,host_counters=holder.get('host'),
                temporary_ram_mib=a.temporary_ram_mib,
                temporary_ram_peak_bytes=getattr(holder['pool'],'temporary_ram_peak_bytes',0),
                numeric_reuse_probe=holder.get('numeric_reuse_probe'),
                scope='Complete two-pass tracking and ownership reconciliation; shared desktop')
    report['sources'][str(Path(__file__))]=sha(Path(__file__))
    if a.measure_feature_reuse:report['sources'][str(Path(__file__).with_name('tracking_numeric_reuse_probe.py'))]=sha(Path(__file__).with_name('tracking_numeric_reuse_probe.py'))
    for path in (a.snapshot/'pymicroglia/tracking').glob('*.py'):
        report['sources'][str(path)]=sha(path)
    report['sources'][str(a.snapshot/'array_storage.py')]=sha(a.snapshot/'array_storage.py')
    if a.adapter:
        report['sources'][str(a.adapter/'tracking_feature_batch.py')]=sha(a.adapter/'tracking_feature_batch.py')
        helper=a.adapter/'diagnostic_shortlist.py'
        if helper.is_file():report['sources'][str(helper)]=sha(helper)
        if a.compact_host:report['sources'][str(a.adapter/'host_partition_reuse.py')]=sha(a.adapter/'host_partition_reuse.py')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','sources','memory_boundaries')}),flush=True)
    if not same:raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        if '--output' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--output')+1])
            if folder.is_dir():(folder/'error.log').write_text(traceback.format_exc())
        raise
