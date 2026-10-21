"""Replay complete correction evidence from read-only native checkpoints."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


LARGE=1024**2


def _layout(value):
    return 'F' if value.flags.f_contiguous and not value.flags.c_contiguous else 'C'


def save_evidence(data,folder):
    """Pickle the evidence tree with large plain and working arrays as .npy files (shared objects once)."""
    import pickle
    import numpy as np
    from auto_organotypic.array_storage import WorkingArray
    folder.mkdir(parents=True,exist_ok=False)
    files={};held=[]
    class Saver(pickle.Pickler):
        def persistent_id(self,value):
            if (type(value) in (np.ndarray,WorkingArray) and value.nbytes>=LARGE
                    and value.dtype.kind in 'biufc' and value.ndim):
                if id(value) not in files:
                    name=f'array_{len(files)}.npy'
                    np.save(folder/name,np.asarray(value),allow_pickle=False)
                    files[id(value)]=(name,type(value).__name__,_layout(value))
                    held.append(value)
                return files[id(value)]
            return None
    with open(folder/'evidence.pickle','wb') as handle:
        Saver(handle,protocol=5).dump(data)
    (folder/'arrays.json').write_text(json.dumps(sorted(files.values()),indent=1)+'\n')


def load_evidence(folder,pool):
    """The saved evidence tree; working arrays return to file-backed pool storage, plain arrays to RAM."""
    import pickle
    import numpy as np
    loaded={}
    class Loader(pickle.Unpickler):
        def persistent_load(self,key):
            name,kind,order=key
            if name not in loaded:
                source=np.load(folder/name,mmap_mode='r',allow_pickle=False)
                if kind=='WorkingArray':
                    result=pool.allocate(source.shape,source.dtype,order=order,force_disk=True)
                else:
                    result=np.ndarray(source.shape,dtype=source.dtype,order=order)
                target=np.asarray(result)
                row=source[0].nbytes if source.ndim>1 and len(source) else source.itemsize
                step=max(1,(64*1024**2)//max(1,row))
                for start in range(0,len(source),step):
                    target[start:start+step]=source[start:start+step]
                del source
                loaded[name]=result
            return loaded[name]
    with open(folder/'evidence.pickle','rb') as handle:
        return Loader(handle).load()


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','snapshot','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--stem',required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--reference-evidence',type=Path)
    p.add_argument('--evidence-only',action='store_true')
    p.add_argument('--profile-corrections',action='store_true')
    # Corrections-only replays: keep the fresh evidence once, then reuse it.
    # A loaded copy must reproduce the saved evidence fingerprints exactly.
    p.add_argument('--save-evidence',type=Path)
    p.add_argument('--load-evidence',type=Path)
    a=p.parse_args()
    for name in ('engine','snapshot','config','output','reference','reference_evidence','save_evidence','load_evidence'):
        value=getattr(a,name)
        if value is not None:setattr(a,name,value.resolve())
    platform.platform=lambda *a,**k:'Windows-complete-evidence-replay'
    import numpy as np
    import pandas as pd
    import psutil
    import shutil
    if psutil.virtual_memory().available<2*1024**3 or shutil.disk_usage(a.output.parent).free<4*1024**3:
        raise RuntimeError('Insufficient memory/disk for complete evidence replay')
    a.output.mkdir(exist_ok=False)
    sys.path[:0]=[str(a.snapshot),str(a.engine/'code')]
    from pymicroglia.tracking import completed_evidence,completed_feedback,correction_storage
    from pymicroglia.tracking.correction_audit import CorrectionAudit,array_sha256
    from benchmark_working_storage import load
    load(a.snapshot/'array_storage.py')
    import common
    import pipeline
    import optimized_pipeline
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sources={str(f):sha(f) for f in a.snapshot.rglob('*') if f.suffix in ('.py','.zip')}
    sources.update({str(f):sha(f) for f in (a.engine/'code').rglob('*.py')})
    # A reference without run.json (an original output folder) still checks every table.
    reference=None if a.reference is None or not (a.reference/'run.json').is_file() else json.loads((a.reference/'run.json').read_text())
    records=[];checks={};evidence_hashes={};timings={};holder={}
    proc=psutil.Process();start=time.perf_counter();cpu=proc.cpu_times()
    def fingerprint(value,key):
        if isinstance(value,np.ndarray):
            evidence_hashes[key]=dict(shape=list(value.shape),dtype=value.dtype.str,sha256=array_sha256(value))
        elif isinstance(value,pd.DataFrame):
            evidence_hashes[key]=dict(table_sha256=hashlib.sha256(value.to_csv(index=False).encode()).hexdigest())
        elif isinstance(value,dict):
            for name,child in value.items():fingerprint(child,key+'/'+str(name))
        elif isinstance(value,(list,tuple)):
            for i,child in enumerate(value):fingerprint(child,key+'/'+str(i))
        elif isinstance(value,np.generic):evidence_hashes[key]=value.item()
        else:evidence_hashes[key]=value
    def execute():
        # Only the owned output folder receives evidence, temporary arrays or
        # comparison tables. The input engine and its checkpoints are read-only.
        with correction_storage.bounded_corrections(a.output,minimum_bytes=1024**2) as pool:
            tick=time.perf_counter()
            if a.load_evidence:
                data=load_evidence(a.load_evidence,pool)
                timings['evidence_load_seconds']=time.perf_counter()-tick
            else:
                data=completed_evidence.generate(a.engine,a.stem,a.run,a.config,a.output,array_pool=pool)
                timings['complete_evidence_seconds']=time.perf_counter()-tick
            fingerprint(data,'evidence')
            (a.output/'evidence_hashes.json').write_text(json.dumps(evidence_hashes,indent=2)+'\n')
            if a.load_evidence:
                checks['loaded_evidence']=json.loads((a.load_evidence/'evidence_hashes.json').read_text())==evidence_hashes
                if not checks['loaded_evidence']:raise RuntimeError('Loaded evidence differs from its saved fingerprints')
            if a.save_evidence:
                tick=time.perf_counter()
                try:
                    save_evidence(data,a.save_evidence)
                    (a.save_evidence/'evidence_hashes.json').write_text(json.dumps(evidence_hashes,indent=2)+'\n')
                    timings['evidence_save_seconds']=time.perf_counter()-tick
                except Exception as error:
                    timings['evidence_save_error']=repr(error)
            if a.reference_evidence:
                expected=json.loads(a.reference_evidence.read_text())
                checks['complete_evidence']=expected==evidence_hashes
                (a.output/'evidence_differences.json').write_text(json.dumps({
                    key:dict(before=expected.get(key),after=evidence_hashes.get(key))
                    for key in expected.keys()|evidence_hashes.keys()
                    if expected.get(key)!=evidence_hashes.get(key)},indent=2)+'\n')
            if a.evidence_only:return
            audit=CorrectionAudit(data['labels'])
            last=time.perf_counter()
            profiler=None
            if a.profile_corrections:
                import cProfile
                profiler=cProfile.Profile();profiler.enable()
            def table_check(relative,table):
                path=a.output/'tables'/relative;path.parent.mkdir(parents=True,exist_ok=True)
                if isinstance(table,dict):path.write_text(json.dumps(table,indent=2)+'\n')
                else:table.to_csv(path,index=False)
                if a.reference:
                    old=a.reference/'mid/completed_feedback'/relative
                    checks[str(relative)]=old.is_file() and old.read_bytes()==path.read_bytes()
            def checkpoint(key,family,stage,labels,ledger,tables,arrays=None):
                nonlocal last
                records.append(dict(family=family,stage=stage,**audit.record(labels,ledger)))
                for name,table in tables.items():
                    table_check(Path(family)/stage/(name+('.json' if isinstance(table,dict) else '.csv')),table)
                now=time.perf_counter()
                with (a.output/'stage_timings.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(family=family,stage=stage,seconds=now-last))+'\n')
                if profiler:
                    profiler.dump_stats(a.output/'corrections.pstats')
                    profiler.enable()
                last=time.perf_counter()
            def inventory(stage,table):table_check(Path('inventory')/(stage+'.csv'),table)
            tick=time.perf_counter()
            result=completed_feedback.apply({a.stem:data},a.output,
                source_frame_offset=common.Config.load(a.config).source_frame_offset,
                callback=checkpoint,inventory_callback=inventory)
            if profiler:profiler.disable();profiler.dump_stats(a.output/'corrections.pstats')
            timings['complete_corrections_seconds']=time.perf_counter()-tick
            holder['final_label_sha256']=array_sha256(result[a.stem][0])
            holder['final_ledger_sha256']=array_sha256(result[a.stem][1])
            if reference:
                expected=[{k:v for k,v in row.items() if k!='seconds'} for row in reference['stages']]
                checks['all_ordered_boundaries']=expected==records
            (a.output/'boundaries.json').write_text(json.dumps(records,indent=2)+'\n')
    optimized_pipeline.main(['--no-evidence-cache'],runner=execute)
    after=proc.cpu_times();memory=proc.memory_info()
    unchanged=all(sha(Path(path))==digest for path,digest in sources.items())
    report=dict(status='pass' if unchanged and checks and all(checks.values()) else 'unverified' if not checks else 'fail',
        seconds=time.perf_counter()-start,cpu_seconds=after.user+after.system-cpu.user-cpu.system,
        timings=timings,checks=checks,boundary_count=len(records),source_unchanged=unchanged,sources=sources,
        config_sha256=sha(a.config),peak_resident_bytes=getattr(memory,'peak_wset',memory.rss),
        peak_private_bytes=getattr(memory,'peak_pagefile',None),**holder,
        profiling_corrections=a.profile_corrections,
        scope=('Complete fresh evidence only from read-only native checkpoints; accepted corrections, native tracking and final TIFF publication excluded'
               if a.evidence_only else 'Complete fresh evidence and accepted correction sequence from read-only native checkpoints; native tracking and final TIFF publication excluded'))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','checks')}),flush=True)
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':main()
