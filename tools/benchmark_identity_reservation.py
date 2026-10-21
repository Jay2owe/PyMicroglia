"""Replay complete native identity stages into isolated immutable folders."""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import types


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','config','snapshot','adapter','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--stem',required=True)
    p.add_argument('--native-history',action='store_true')
    p.add_argument('--stage',choices=('reservation','layered','lineage','history'),default='reservation')
    p.add_argument('--profile',action='store_true',help='Write a CPU profile of the stage call')
    a=p.parse_args()
    for name in ('engine','config','snapshot','adapter','output'):setattr(a,name,getattr(a,name).resolve())
    platform.platform=lambda *a,**k:'Windows-identity-reservation-replay'
    sys.path[:0]=[str(a.engine/'code'),str(a.snapshot),str(a.adapter)]
    import numpy as np
    import psutil
    import tifffile
    import common,pipeline,optimized_pipeline,motion_handoffs
    import motion_history_reuse
    from pymicroglia.tracking import frozen_io
    from benchmark_working_storage import load
    from compare_tracking_memory import image_signature,sha
    if psutil.virtual_memory().available<3*1024**3 or shutil.disk_usage(a.output.parent).free<6*1024**3:
        raise RuntimeError('Insufficient free memory/disk for isolated reservation replay')
    a.output.mkdir(exist_ok=False)
    cfg=common.Config.load(a.config);paths=pipeline._pinned_paths(cfg,a.stem)
    predecessor,stage_name={'reservation':('m9_alias','m12_identity_reservation'),
        'layered':('m12_identity_reservation','m19_layered_lineage'),
        'lineage':('m19_layered_lineage','m20_oscillatory_lineage'),
        'history':('m19_layered_lineage','m22_accepted_history')}[a.stage]
    parent=a.engine/predecessor/a.run/'out'/(a.stem+'.tif')
    reference=a.engine/stage_name/a.run
    files=[Path(__file__),Path(motion_handoffs.__file__),Path(motion_history_reuse.__file__),
           a.snapshot/'array_storage.py',a.config,parent,*paths.values()]
    hashes={str(f):sha(f) for f in files}
    hashes.update({str(f):sha(f) for f in (a.snapshot/'pymicroglia/tracking').glob('*.py')})
    class IsolatedRun(common.Run):
        pass
    namespace=dict(common.Run.__init__.__globals__);namespace['ROOT']=a.output/'engine'
    IsolatedRun.__init__=types.FunctionType(common.Run.__init__.__code__,namespace,
        '__init__',common.Run.__init__.__defaults__,common.Run.__init__.__closure__)
    pipeline.Run=IsolatedRun
    holder={};timings={};cpu_timings={};process=psutil.Process();cpu=process.cpu_times();started=time.perf_counter()
    def execute():
        storage=load(a.snapshot/'array_storage.py')
        with ExitStack() as stack:
            pool=storage.Pool(a.output/'scratch',minimum_bytes=1024**2,temporary_ram_bytes=32*1024**2)
            stack.enter_context(storage.install(pool))
            stack.enter_context(frozen_io.install(common,a.engine/'code',a.output/'scratch',array_pool=pool))
            if a.native_history:
                assert motion_handoffs.carry_established_identities_along_motion.__code__.co_filename=='<indexed-motion-history>'
                holder['history_mode']='frozen native composition'
            else:
                holder['history_counters']=stack.enter_context(motion_history_reuse.install(motion_handoffs,pipeline))
                assert holder['history_counters']['supported']
            original_read=tifffile.imread;maps={};read_count=0
            pinned={path.resolve() for path in paths.values()}
            def mapped(file,*args,**kwargs):
                nonlocal read_count
                if args or kwargs or not isinstance(file,(str,Path)):return original_read(file,*args,**kwargs)
                key=Path(file).resolve()
                cacheable=a.stage!='history' or key in pinned
                if not cacheable or key not in maps:
                    with tifffile.TiffFile(key) as opened:
                        series=opened.series[0]
                        if not pool.large(series.shape,series.dtype):return original_read(file)
                        target=a.output/'scratch'/('input_'+str(read_count)+'.dat');read_count+=1
                        value=pool.adopt(series.asarray(out=str(target)),target)
                        if not cacheable:return value
                        maps[key]=value
                        maps[key].flags.writeable=False
                return maps[key]
            tifffile.imread=mapped
            try:
                baseline=mapped(parent)
                if a.stage in ('reservation','layered','history'):observations,table,anchor=pipeline._load_anchor_inputs(a.stem,a.run)
                names=(('build_motion_pair_cache','carry_identities_through_motion_handoffs',
                    'retrack_fixed_objects','carry_established_identities_along_motion',
                    'carry_reserved_identities_through_motion_hosts','_identity_tables','outline_overlay')
                    if a.stage=='reservation' else ('track_physical_somas','physical_soma_tracking_evidence',
                    'correct_identity_ledger_transfers','_identity_tables','outline_overlay') if a.stage=='layered'
                    else ('optimise_lineage_graph','optimise_host_merge_graph',
                    'substantial_identity_conflicts','_identity_tables','outline_overlay') if a.stage=='lineage'
                    else ('run_accepted_history','_identity_tables'))
                for name in names:
                    if not hasattr(pipeline,name):continue
                    original=getattr(pipeline,name)
                    def timed(*args,_name=name,_original=original,**kwargs):
                        tick=time.perf_counter();cpu_tick=time.process_time();value=_original(*args,**kwargs)
                        timings.setdefault(_name,[]).append(time.perf_counter()-tick)
                        cpu_timings.setdefault(_name,[]).append(time.process_time()-cpu_tick)
                        return value
                    setattr(pipeline,name,timed)
                tick=time.perf_counter()
                profiler=None
                if a.profile:
                    import cProfile
                    profiler=cProfile.Profile();profiler.enable()
                if a.stage=='reservation':
                    result=pipeline.stage_m12_accepted(cfg,a.stem,a.run,paths,baseline,observations,table,anchor)
                elif a.stage=='layered':
                    from pymicroglia.tracking.checkpoints import Stages
                    # The m12 history fields, decoded read-only as in the history route.
                    decoder=Stages.__new__(Stages);decoder.root=a.engine;decoder.loader=mapped
                    history_inputs=pipeline.AcceptedHistoryInputs()
                    path=a.engine/'m12_identity_reservation'/a.run/'checkpoint.json'
                    manifest=json.loads(path.read_text());hashes[str(path)]=sha(path)
                    decoder._verify(manifest['files'])
                    for name,value in manifest['history'].items():
                        setattr(history_inputs,name,decoder._load(value))
                    holder['numeric_context']=getattr(np,'_get_promotion_state',lambda:None)()
                    result=pipeline.stage_m19_accepted(cfg,a.stem,a.run,paths,baseline,observations,table,anchor,
                                                       history_inputs=history_inputs)
                elif a.stage=='lineage':
                    assert pipeline.optimise_host_merge_graph.__closure__ is not None
                    # The actual runner executes this stage in a fresh worker.
                    # NumPy 2.1 uses a different scalar-promotion state there.
                    def lineage():
                        holder['numeric_context']=getattr(np,'_get_promotion_state',lambda:None)()
                        return pipeline.stage_m20_accepted(cfg,a.stem,a.run,paths,baseline)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result=executor.submit(lineage).result()
                else:
                    from concurrent.futures import Future
                    from pymicroglia.tracking.checkpoints import Stages
                    # Use only the decoder and verifier. Its normal constructor
                    # writes resume state, which this read-only replay must not do.
                    decoder=Stages.__new__(Stages);decoder.root=a.engine;decoder.loader=mapped
                    history_inputs=pipeline.AcceptedHistoryInputs()
                    for prior in ('m12_identity_reservation','m19_layered_lineage'):
                        path=a.engine/prior/a.run/'checkpoint.json'
                        manifest=json.loads(path.read_text());hashes[str(path)]=sha(path)
                        decoder._verify(manifest['files'])
                        for name,value in manifest['history'].items():
                            setattr(history_inputs,name,decoder._load(value))
                    path=a.engine/'m20_oscillatory_lineage'/a.run/'checkpoint.json'
                    manifest=json.loads(path.read_text());hashes[str(path)]=sha(path)
                    decoder._verify(manifest['files'])
                    completed=Future();completed.set_result(decoder._load(manifest['result']))
                    def history():
                        holder['numeric_context']=getattr(np,'_get_promotion_state',lambda:None)()
                        return pipeline.stage_m22_accepted_history(cfg,a.stem,a.run,paths,baseline,
                            observations,table,anchor,history_inputs=history_inputs,m20_result_future=completed)
                    tick=time.perf_counter()
                    # The bounded runner calls m22 on its own thread and leaves
                    # review_pending.json; the ordinary runner uses a worker.
                    # NumPy 2.1 promotion differs between them, so follow the
                    # route that produced the reference.
                    holder['history_thread']='caller' if (reference/'review_pending.json').is_file() else 'worker'
                    if holder['history_thread']=='caller':
                        result=history()
                    else:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            result=executor.submit(history).result()
                holder['stage_seconds']=time.perf_counter()-tick
                if profiler:
                    import pstats
                    profiler.disable();profiler.dump_stats(str(a.output/'profile.pstats'))
                    rows=[dict(function=f'{Path(f).name}:{line}:{fn}',calls=c[1],own=round(c[2],3),cumulative=round(c[3],3))
                          for (f,line,fn),c in pstats.Stats(profiler).stats.items()]
                    holder['by_own']=sorted(rows,key=lambda r:-r['own'])[:40]
                    holder['by_cumulative']=sorted(rows,key=lambda r:-r['cumulative'])[:60]
                holder['shape']=list(result[0].shape)
            finally:
                tifffile.imread=original_read
                if a.stage=='history':
                    assert set(maps).issubset(pinned)
                    holder['retained_input_paths']=[str(path) for path in maps]
                    holder['input_cache_policy']='Pinned immutable recording inputs only; intermediate reads released with their final consumer'
            holder['temporary_ram_peak_bytes']=pool.temporary_ram_peak_bytes
    optimized_pipeline.main(['--no-evidence-cache','--stage-timings',str(a.output/'stage_timings.jsonl')],runner=execute)
    stage_cpu=process.cpu_times()
    candidate=a.output/'engine'/stage_name/a.run
    def scientific(folder):return {str(p.relative_to(folder)):p for p in folder.rglob('*')
                                   if p.is_file() and p.suffix in ('.tif','.tiff','.csv')}
    expected,actual=scientific(reference),scientific(candidate);checks={}
    for name in sorted(set(expected)|set(actual)):
        checks[name]=(name in expected and name in actual and
            (image_signature(expected[name])==image_signature(actual[name]) if Path(name).suffix in ('.tif','.tiff')
             else expected[name].read_bytes()==actual[name].read_bytes()))
    after=process.cpu_times();memory=process.memory_info()
    source_unchanged=all(sha(Path(f))==value for f,value in hashes.items())
    report=dict(status='pass' if checks and all(checks.values()) and source_unchanged else 'fail',
        seconds=time.perf_counter()-started,cpu_seconds=after.user+after.system-cpu.user-cpu.system,
        cpu_before_comparison_seconds=stage_cpu.user+stage_cpu.system-cpu.user-cpu.system,
        peak_resident_bytes=memory.peak_wset,peak_private_bytes=memory.peak_pagefile,
        **holder,timings=timings,cpu_timings=cpu_timings,checks=checks,sources=hashes,source_unchanged=source_unchanged,
        scope=f'Complete {holder["shape"][0]}-frame {a.stage} stage including all saved tables, masks and native review pixels')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
