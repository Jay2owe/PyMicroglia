"""Replay complete dormant ownership/restoration with optional exact reuse."""
import argparse
from contextlib import ExitStack
import cProfile
import hashlib
import importlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','config','output','storage-source'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--run',required=True)
    p.add_argument('--stem',required=True)
    p.add_argument('--reference-tables',type=Path)
    p.add_argument('--reuse-ownership',action='store_true')
    p.add_argument('--reuse-path',action='store_true')
    p.add_argument('--verify-detector',action='store_true')
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-dormant-stage-benchmark'
    import numpy as np
    import psutil
    import tifffile
    if psutil.virtual_memory().available<3*1024**3 or shutil.disk_usage(a.output.parent).free<4*1024**3:
        raise RuntimeError('Insufficient free memory/disk for stage replay')
    a.output.mkdir(exist_ok=False)
    sys.path.insert(0,str(a.engine.resolve()/'code'))
    import accepted_history as ah
    import common
    import pipeline
    import optimized_pipeline
    from benchmark_working_storage import load
    storage=load(a.storage_source.resolve())
    cfg=common.Config.load(a.config)
    source=a.engine/'m22_accepted_history'/a.run/'mid/accepted_history'
    paths=[source/'07_issue011.tif',source/'07_issue011_unclaimed.tif',source/'06_issue010.tif']
    sources=json.loads((a.engine/'m0_inputs'/a.run/'out'/f'{a.stem}_sources.json').read_text())
    raw_path=Path(sources['files']['registered_raw']['path'])
    pinned=pipeline._pinned_paths(cfg,a.stem)
    lag_path=Path(pinned['lag_float'])
    paths.extend((raw_path,lag_path,a.config))
    labels,ledger,protected=map(tifffile.imread,paths[:3])
    raw=tifffile.imread(raw_path)[cfg.source_frame_offset:]
    lag=ah.align_transition_stack(tifffile.imread(lag_path),len(labels),cfg.source_frame_offset)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    inputs={str(p):sha(p) for p in paths}
    codes={str(p):sha(p) for p in (a.engine/'code').glob('*.py')}
    codes[str(Path(__file__))]=sha(Path(__file__))
    holder={}; timings={}
    process=psutil.Process();cpu=process.cpu_times();start=time.perf_counter()
    def execute():
        with ExitStack() as stack:
            pool=storage.Pool(a.output/'scratch',minimum_bytes=1024**2)
            stack.enter_context(storage.install(pool));holder['pool']=pool
            if a.adapter:
                sys.path.insert(0,str(a.adapter.resolve()))
                import dormant_component_reuse as reuse
                holder['counters']=stack.enter_context(reuse.install_loader(ah))
                codes[str(Path(reuse.__file__))]=sha(Path(reuse.__file__))
                if a.reuse_ownership:
                    import ownership_observation_reuse
                    holder['ownership_counters']=stack.enter_context(ownership_observation_reuse.install_loader(ah))
                    codes[str(Path(ownership_observation_reuse.__file__))]=sha(Path(ownership_observation_reuse.__file__))
                if a.reuse_path:
                    import dormant_path_reuse
                    holder['path_counters']=stack.enter_context(dormant_path_reuse.install_loader(ah))
                    codes[str(Path(dormant_path_reuse.__file__))]=sha(Path(dormant_path_reuse.__file__))
            loader=ah._issue_module
            old_cached=ah.cached_dormant_detect
            def measured_detector(module):
                function=old_cached(module)
                reference=getattr(module.detect,'__wrapped__',None)
                def detect(*values,**kw):
                    tick=time.perf_counter()
                    result=function(*values,**kw)
                    timings[module.__name__+'.detect']=time.perf_counter()-tick
                    for i,table in enumerate(result):
                        table.to_csv(a.output/f'detector_{i}.csv',index=False)
                    if a.verify_detector:
                        if reference is None:raise ValueError('Reference detector unavailable')
                        tick=time.perf_counter();before_cpu=process.cpu_times()
                        old=reference(*values,**kw)
                        timings['reference_detector_verification']=time.perf_counter()-tick
                        after_cpu=process.cpu_times()
                        holder['verification_cpu']=after_cpu.user+after_cpu.system-before_cpu.user-before_cpu.system
                        holder['detector_equal']=[l.to_csv(index=False)==r.to_csv(index=False) for l,r in zip(old,result)]
                        for i,table in enumerate(old):table.to_csv(a.output/f'reference_detector_{i}.csv',index=False)
                    return result
                return detect
            ah.cached_dormant_detect=measured_detector
            stack.callback(setattr,ah,'cached_dormant_detect',old_cached)
            def timed_load(*args,**kwargs):
                module=loader(*args,**kwargs)
                # Persist complete detector evidence for paired CSV comparison.
                for name in ('reconcile',):
                    if not hasattr(module,name): continue
                    original=getattr(module,name)
                    def timed(*v,_fn=original,_name=module.__name__+'.'+name,**kw):
                        tick=time.perf_counter()
                        result=_fn(*v,**kw)
                        timings[_name]=time.perf_counter()-tick
                        for i, value in enumerate(result):
                            if hasattr(value, 'to_csv'):
                                value.to_csv(a.output/f'{_name}_table_{i}.csv',index=False)
                        if _name.endswith('.detect'):
                            for i,table in enumerate(result):
                                table.to_csv(a.output/f'detector_{i}.csv',index=False)
                        return result
                    setattr(module,name,timed)
                return module
            ah._issue_module=timed_load
            stack.callback(setattr,ah,'_issue_module',loader)
            profiler=cProfile.Profile();profiler.enable()
            try:holder['result']=ah._issue012(labels,ledger,protected,raw,lag,cfg.source_frame_offset)
            finally:
                profiler.disable();profiler.dump_stats(a.output/'profile.pstats')
            holder['comparison']={}
            for i,name in enumerate(('08_issue012.tif','08_issue012_unclaimed.tif')):
                holder['comparison'][name]=bool(np.array_equal(holder['result'][i],tifffile.imread(source/name)))
            del holder['result']
    optimized_pipeline.main(['--no-evidence-cache'],runner=execute)
    seconds=time.perf_counter()-start;after=process.cpu_times()
    comparison=holder['comparison']
    if a.verify_detector:
        comparison.update({f'detector_{i}.csv':same for i,same in enumerate(holder['detector_equal'])})
    if a.reference_tables:
        for path in a.output.glob('detector_*.csv'):
            comparison[path.name]=path.read_bytes()==(a.reference_tables/path.name).read_bytes()
    memory=process.memory_info()
    report=dict(status='pass' if all(comparison.values()) else 'fail',seconds=seconds-timings.get('reference_detector_verification',0),
        total_seconds_including_verification=seconds,
        cpu_seconds=after.user+after.system-cpu.user-cpu.system-holder.get('verification_cpu',0),
        peak_resident_bytes=getattr(memory,'peak_wset',memory.rss),shape=list(labels.shape),
        comparison=comparison,timings=timings,counters=holder.get('counters'),ownership_counters=holder.get('ownership_counters'),path_counters=holder.get('path_counters'),
        storage=dict(allocations=holder['pool'].allocations,allocated_bytes=holder['pool'].allocated_bytes),
        inputs=inputs,sources=codes,scope='Complete dormant ownership/restoration; shared desktop; cProfile enabled; peak includes verification')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','sources')}),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        if '--output' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--output')+1])
            if folder.is_dir():(folder/'error.log').write_text(traceback.format_exc())
        raise
