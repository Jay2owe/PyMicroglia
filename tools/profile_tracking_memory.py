"""Profile the complete packaged tracker in an isolated frozen process.

The launcher supplies an owned output directory, a package source snapshot and
a normal recording config. Scientific outputs are retained for exact comparison.
"""
from pathlib import Path
import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

import psutil


class BeforeHistoryComplete(Exception):
    """The explicitly requested twelve-stage scope has finished."""


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def worker(args):
    platform.platform=lambda *a,**k:'Windows-memory-profile'
    sys.path.insert(0,str(Path(args.source).resolve()))
    from pymicroglia.tracking import engine,completed_feedback,completed_evidence
    if args.storage_source:
        selection=args.storage_source.with_name('array_selection.py')
        if selection.is_file():
            selection_spec=importlib.util.spec_from_file_location('auto_organotypic.array_selection',selection)
            selection_module=importlib.util.module_from_spec(selection_spec)
            sys.modules[selection_spec.name]=selection_module
            selection_spec.loader.exec_module(selection_module)
        spec=importlib.util.spec_from_file_location('auto_organotypic.array_storage',args.storage_source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
    process=psutil.Process()
    lock=threading.Lock()
    def event(name,edge):
        m=process.memory_info()
        row=dict(time=time.time(),stage=name,edge=edge,thread=threading.get_ident(),
                 rss=m.rss,private=getattr(m,'private',m.vms),peak_rss=getattr(m,'peak_wset',m.rss))
        row['io']=process.io_counters()._asdict()
        row['cpu']=process.cpu_times()._asdict()
        with lock:
            with (args.report/'stages.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def wrap(original,name):
        def call(*a,**k):
            if args.before_history_only and name=='stage_m22_accepted_history':
                future=k.get('m20_result_future')
                if future is not None:future.result()
                raise BeforeHistoryComplete()
            event(name,'start')
            profiler = cProfile.Profile() if args.profile_stages and (
                name.startswith('stage_m') or name in ('correction_evidence','corrections')) else None
            if profiler:profiler.enable()
            try:return original(*a,**k)
            finally:
                if profiler:
                    profiler.disable()
                    profiler.dump_stats(args.report/(name.replace('/','_')+'.pstats'))
                event(name,'end')
        return call
    original=engine._execute_pipeline
    def native(pipeline,*a,**k):
        for name in dir(pipeline):
            if name.startswith('stage_m') and callable(getattr(pipeline,name)):
                setattr(pipeline,name,wrap(getattr(pipeline,name),name))
        return wrap(original,'native')(pipeline,*a,**k)
    engine._execute_pipeline=native
    completed_evidence.generate=wrap(completed_evidence.generate,'correction_evidence')
    apply=completed_feedback.apply
    def corrections(*a,**k):
        callback=k['callback']
        def checkpoint(key,family,stage,*values,**kw):
            result=callback(key,family,stage,*values,**kw)
            event('correction/'+family+'/'+stage,'boundary')
            return result
        k['callback']=checkpoint
        return wrap(apply,'corrections')(*a,**k)
    completed_feedback.apply=corrections
    root=engine._engine(args.output)
    whole_stage='earlier_tracking_group' if args.before_history_only else 'complete_tracking'
    event(whole_stage,'start')
    try:
        engine._run_frozen(root,args.movie,'profile',args.config,args.report/'run.log',
                           bounded_memory=args.bounded,working_array_min_bytes=args.working_array_min_bytes)
        event(whole_stage,'end')
        (args.report/'worker.json').write_text(json.dumps(dict(status='pass',
            engine_module=engine.__file__,pid=os.getpid(),output=str(root),method=engine.METHOD_VERSION),indent=2))
    except BeforeHistoryComplete:
        event(whole_stage,'end')
        (args.report/'worker.json').write_text(json.dumps(dict(status='pass',scope='Twelve earlier native stages only',
            engine_module=engine.__file__,pid=os.getpid(),output=str(root),method=engine.METHOD_VERSION),indent=2))
    except BaseException:
        (args.report/'worker.json').write_text(json.dumps(dict(status='fail',error=traceback.format_exc()),indent=2))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--movie',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--bounded',action='store_true')
    p.add_argument('--profile-stages',action='store_true',help='Write CPU profiles for each complete native stage and correction group')
    p.add_argument('--before-history-only',action='store_true',help='Finish all twelve earlier native stages, including concurrent lineage work, then stop before accepted history')
    p.add_argument('--working-array-min-bytes',type=int,default=32*1024**2)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--storage-source',type=Path,help='Frozen array-storage dependency for this trial')
    p.add_argument('--minimum-free-ram-gib',type=float,default=4)
    p.add_argument('--minimum-free-disk-gib',type=float,default=20)
    args=p.parse_args()
    if args.storage_source:args.storage_source=args.storage_source.resolve()
    for name in ('source','config','output','report'):setattr(args,name,getattr(args,name).resolve())
    if args.worker:return worker(args)
    # Admission belongs in the launcher so a failed check cannot accidentally
    # be followed by starting the expensive worker in another shell statement.
    available=psutil.virtual_memory().available
    disk_parent=args.output
    while not disk_parent.exists():disk_parent=disk_parent.parent
    free_disk=shutil.disk_usage(disk_parent).free
    if available<args.minimum_free_ram_gib*1024**3:
        raise SystemExit(f'Insufficient free RAM: {available} bytes')
    if free_disk<args.minimum_free_disk_gib*1024**3:
        raise SystemExit(f'Insufficient free disk: {free_disk} bytes')
    args.report.mkdir(parents=True,exist_ok=False)
    args.output.mkdir(parents=True,exist_ok=False)
    source_files={str(path.relative_to(args.source)):sha(path) for path in args.source.rglob('*') if path.suffix in ('.py','.zip')}
    (args.report/'inputs.json').write_text(json.dumps(dict(config=str(args.config),config_sha256=sha(args.config),
        source=str(args.source),files=source_files,bounded=args.bounded,
        profiling=args.profile_stages,before_history_only=args.before_history_only,
        working_array_min_bytes=args.working_array_min_bytes,
        storage_source=str(args.storage_source) if args.storage_source else None,
        storage_sha256=sha(args.storage_source) if args.storage_source else None,
        available_ram=psutil.virtual_memory().available),indent=2))
    command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    start=time.perf_counter()
    with (args.report/'launcher.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        proc=psutil.Process(child.pid)
        peak_rss=peak_private=peak_tree_rss=0
        with (args.report/'memory.jsonl').open('w') as trace:
            while child.poll() is None:
                try:
                    mem=proc.memory_info();tree=mem.rss
                    for desc in proc.children(recursive=True):
                        try:tree+=desc.memory_info().rss
                        except psutil.Error:pass
                    private=getattr(mem,'private',mem.vms)
                    peak_rss=max(peak_rss,getattr(mem,'peak_wset',mem.rss))
                    peak_private=max(peak_private,private);peak_tree_rss=max(peak_tree_rss,tree)
                    trace.write(json.dumps(dict(time=time.time(),seconds=time.perf_counter()-start,rss=mem.rss,private=private,tree_rss=tree,
                        io=proc.io_counters()._asdict(),cpu=proc.cpu_times()._asdict()))+'\n')
                    trace.flush()
                except psutil.Error:pass
                time.sleep(.2)
    if (args.report/'stages.jsonl').exists():
        for line in (args.report/'stages.jsonl').read_text().splitlines():
            event=json.loads(line)
            peak_rss=max(peak_rss,event['peak_rss'])
            peak_private=max(peak_private,event['private'])
    unchanged=all(sha(args.source/name)==digest for name,digest in source_files.items())
    report=dict(status='pass' if child.returncode==0 and unchanged else 'fail',exit_code=child.returncode,
        seconds=time.perf_counter()-start,peak_working_set_bytes=peak_rss,
        peak_private_bytes=peak_private,sampled_peak_process_tree_rss_bytes=peak_tree_rss,
        source_unchanged=unchanged,scope=('Complete twelve-stage earlier tracking group; accepted history, correction evidence, accepted corrections and final publication excluded'
            if args.before_history_only else 'Complete prepared-input tracking: native stages, fresh correction evidence, all accepted corrections and publication'))
    report['profiling']=args.profile_stages
    (args.report/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
