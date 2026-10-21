"""Replay the complete five-operation territory/recovery group, read-only inputs."""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','snapshot','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--run',required=True);p.add_argument('--stem',required=True)
    a=p.parse_args()
    for name in ('engine','snapshot','config','output','adapter'):
        value=getattr(a,name)
        if value is not None:setattr(a,name,value.resolve())
    platform.platform=lambda *a,**k:'Windows-territory-recovery-replay'
    import numpy as np
    import pandas as pd
    import psutil
    import shutil
    import tifffile
    if psutil.virtual_memory().available<2*1024**3 or shutil.disk_usage(a.output.parent).free<5*1024**3:
        raise RuntimeError('Insufficient memory/disk for territory recovery replay')
    a.output.mkdir(exist_ok=False)
    sys.path[:0]=[str(a.snapshot),str(a.engine/'code')]
    if a.adapter:sys.path.insert(0,str(a.adapter))
    from pymicroglia.tracking import frozen_io
    from pymicroglia.tracking.correction_audit import array_sha256
    from benchmark_working_storage import load
    import accepted_history as ah
    import common,optimized_pipeline
    storage=load(a.snapshot/'array_storage.py')
    source=a.engine/'m22_accepted_history'/a.run/'mid/accepted_history'
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    source_paths=[Path(__file__),a.config,*sorted((a.snapshot/'pymicroglia/tracking').glob('*.py'))]
    if a.adapter:source_paths.append(a.adapter/'territory_peak_geometry.py')
    hashes={str(f):sha(f) for f in source_paths}
    cfg=common.Config.load(a.config);holder={};timings=[];process=psutil.Process();cpu=process.cpu_times();started=time.perf_counter()
    def execute():
        with ExitStack() as stack:
            pool=storage.Pool(a.output/'scratch',minimum_bytes=1024**2,temporary_ram_bytes=32*1024**2)
            stack.enter_context(storage.install(pool))
            stack.enter_context(frozen_io.install(common,a.engine/'code',a.output/'scratch',array_pool=pool))
            if a.adapter:
                import territory_memory,territory_peak_geometry
                holder['peak_counters']=stack.enter_context(territory_peak_geometry.install(territory_memory))
                assert holder['peak_counters']['supported']
            import territory_memory
            original_peak=territory_memory._centre_surround_peak
            holder['peak_calls']=0
            def measured_peak(*args,**kwargs):
                holder['peak_calls']+=1;return original_peak(*args,**kwargs)
            territory_memory._centre_surround_peak=measured_peak
            stack.callback(setattr,territory_memory,'_centre_surround_peak',original_peak)
            def read(path):
                hashes[str(path)]=sha(path)
                target=a.output/'scratch'/('input_'+str(len(hashes))+'.dat')
                with tifffile.TiffFile(path) as opened:value=pool.adopt(opened.series[0].asarray(out=str(target)),target)
                value.flags.writeable=False;return value
            labels=read(source/'11_issue017_rebase.tif');unclaimed=read(source/'11_issue017_rebase_unclaimed.tif')
            raw=read(source/'07_trimmed_evidence/out'/f'{a.stem}_registered_raw.tif')
            lag=read(source/'07_trimmed_evidence/out'/f'{a.stem}_lag01_float.tif')
            for name in ('reconcile_territory_memory','assign_persistent_unclaimed_bodies','reserve_local_lineages',
                         'reconcile_reciprocal_lineages','recover_stationary_gaps'):
                original=getattr(ah,name)
                def call(*args,_name=name,_original=original,**kwargs):
                    tick=time.perf_counter();cpu_tick=time.process_time();result=_original(*args,**kwargs)
                    row=dict(operation=_name,seconds=time.perf_counter()-tick,cpu_seconds=time.process_time()-cpu_tick,fields={})
                    for field in fields(result):
                        value=getattr(result,field.name)
                        if isinstance(value,np.ndarray):row['fields'][field.name]=dict(shape=list(value.shape),dtype=value.dtype.str,sha256=array_sha256(value))
                        elif isinstance(value,pd.DataFrame):
                            path=a.output/(_name+'_'+field.name+'.csv');value.to_csv(path,index=False);row['fields'][field.name]=dict(sha256=sha(path))
                    timings.append(row)
                    with (a.output/'operations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    return result
                setattr(ah,name,call);stack.callback(setattr,ah,name,original)
            def calculate():
                holder['numeric_context']=getattr(np,'_get_promotion_state',lambda:None)()
                tick=time.perf_counter();result=ah._issue023_tail(labels,unclaimed,raw,lag,cfg.source_frame_offset)
                holder['stage_seconds']=time.perf_counter()-tick
                holder['labels_equal']=array_sha256(result[0])==array_sha256(read(source/'12_issue023_generalized.tif'))
                holder['ledger_equal']=array_sha256(result[1])==array_sha256(read(source/'12_issue023_generalized_unclaimed.tif'))
            with ThreadPoolExecutor(max_workers=1) as executor:executor.submit(calculate).result()
            holder['temporary_ram_peak_bytes']=pool.temporary_ram_peak_bytes
    optimized_pipeline.main(['--no-evidence-cache'],runner=execute)
    after=process.cpu_times();memory=process.memory_info();unchanged=all(sha(Path(f))==v for f,v in hashes.items())
    report=dict(status='pass' if unchanged and holder['labels_equal'] and holder['ledger_equal'] else 'fail',
        seconds=time.perf_counter()-started,cpu_seconds=after.user+after.system-cpu.user-cpu.system,
        peak_resident_bytes=getattr(memory,'peak_wset',memory.rss),peak_private_bytes=getattr(memory,'peak_pagefile',None),
        sources=hashes,source_unchanged=unchanged,operations=timings,**holder,
        scope='Complete five-operation territory/recovery group; complete final labels and ledger compared with frozen original; emitted intermediate evidence retained for future comparisons')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','operations')}),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
