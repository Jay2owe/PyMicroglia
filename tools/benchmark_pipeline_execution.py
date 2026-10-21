"""Fresh public workflow executions on the existing controlled demo recipes.

No scientific functions or renderers are replaced. Setup and independent demo
assertions are outside action timing. Outputs and failures remain reviewable.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Other projects can remain under active development without invalidating these
# long serial comparisons. This override exists only in benchmark processes.
PINNED_DEPENDENCIES=ROOT/'development/speed-pass-20261011/complete/frozen-dependencies'
if (PINNED_DEPENDENCIES/'snapshot.json').is_file():
    sys.path.insert(0,str(PINNED_DEPENDENCIES))
    os.environ['PYTHONPATH']=str(PINNED_DEPENDENCIES)+os.pathsep+os.environ.get('PYTHONPATH','')
import types
tests = types.ModuleType('tests')
tests.__path__ = [str(ROOT/'tests')]
sys.modules['tests'] = tests


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity():
    return {p.relative_to(ROOT).as_posix(): digest(p)
            for p in (ROOT/'src/pymicroglia').rglob('*.py')}


def configure_ledger(baseline=False):
    from auto_organotypic.store import ledger, manifest
    if baseline:
        path=ROOT/'development/speed-pass-20261011/baseline/store_ledger.py'
        spec=importlib.util.spec_from_file_location('auto_organotypic.store._benchmark_before_ledger',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ledger.entry_for=module.entry_for
        # Keep public execution on the frozen dispatcher when measuring the
        # pre-pass path, even after the optimized dispatcher has been installed.
        runner_path=ROOT/'development/speed-pass-20261011/baseline/run.py'
        if runner_path.exists():
            name='pymicroglia._benchmark_before_run'
            old_spec=importlib.util.spec_from_file_location(name,runner_path)
            old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
            import pymicroglia.run as runner
            runner.run_recorded=old.run_recorded
        measure_path=ROOT/'development/speed-pass-20261011/baseline/measure_init.py'
        if measure_path.exists():
            name='pymicroglia.measure._benchmark_before_init'
            old_spec=importlib.util.spec_from_file_location(name,measure_path)
            old_measure=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old_measure)
            importlib.import_module('pymicroglia.measure').measure=old_measure.measure
        fingerprint_path=ROOT/'development/speed-pass-20261011/baseline/store_manifest_fingerprint.py'
        name='auto_organotypic.store._benchmark_before_manifest'
        old_spec=importlib.util.spec_from_file_location(name,fingerprint_path)
        old_manifest=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old_manifest)
        # Keep the current pending-write context; only the original lookup body
        # is the baseline treatment, with its fresh whole-index load on each hit.
        old_manifest.load=manifest.load
        old_manifest.save=manifest.save
        manifest.fingerprint=old_manifest.fingerprint
    else:path=Path(ledger.__file__)
    if not baseline:fingerprint_path=Path(manifest.__file__)
    runner_path=ROOT/('development/speed-pass-20261011/baseline/run.py' if baseline else 'src/pymicroglia/run.py')
    measure_path=ROOT/('development/speed-pass-20261011/baseline/measure_init.py' if baseline else 'src/pymicroglia/measure/__init__.py')
    return {'mode':'baseline' if baseline else 'current','sha256':digest(path),'source':str(path),
            'dispatcher_sha256':digest(runner_path),'measurement_entry_sha256':digest(measure_path),
            'source_fingerprint_sha256':digest(fingerprint_path),
            'parsed_lookup_sha256':None if baseline else digest(Path(manifest.__file__).with_name('_parsed_lookup.py')),
            'dependency_environment':dependency_environment()}


def dependency_environment():
    from pymicroglia import workbench
    return workbench.cw.run_records.software_identity()


MEMORY_SAMPLING={'rss_interval_seconds':.05,'descendant_discovery_seconds':1.0}
_owned_processes={}


def process_tree_rss(process, *, include_parent=True):
    """Sample known workers frequently without enumerating Windows every 50 ms.

    On this host enumeration consumed about 32 ms CPU per sample. Refresh child
    ownership once a second; read known-process RSS every 50 ms. Very short-lived
    workers can evade discovery, so these remain sampled, not guaranteed peaks.
    """
    import psutil
    key=(process.pid,process.create_time());now=time.monotonic()
    refreshed,children=_owned_processes.get(key,(0,()))
    if now-refreshed>=MEMORY_SAMPLING['descendant_discovery_seconds']:
        children=process.children(recursive=True)
        _owned_processes[key]=(now,children)
    total=process.memory_info().rss if include_parent else 0
    for child in children:
        try:
            if child.is_running():total+=child.memory_info().rss
        except psutil.Error:pass
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['rhythm','audit','relationships','behaviour','coordination','intervention'])
    parser.add_argument('--label', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--baseline-ledger',action='store_true')
    parser.add_argument('--reservation-history',type=Path,
        help='Existing native behaviour reservation records, identical for both trials; all analysis still recomputes')
    parser.add_argument('--case',default='responses',choices=['responses','timing','rhythms','insufficient'])
    args=parser.parse_args()
    base=ROOT/'development/speed-pass-20261011/complete'/args.label/args.family
    base.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(base).free < 700*1024**2:
        raise RuntimeError('Less than 700 MiB free for a bounded workflow trial')
    os.environ['PYMICROGLIA_STORE']=str(base/'store')
    os.environ['PYMICROGLIA_INDEX']=str(base/'index.json')
    os.environ['MPLBACKEND']='Agg'
    sys.executable=shutil.which('python') or sys.executable
    import psutil
    from pymicroglia import config
    from pymicroglia.run import run_action
    from pymicroglia.pipelines._requests import FAMILIES
    from pymicroglia._results import read_document
    ledger_identity=configure_ledger(args.baseline_ledger)
    demo=importlib.import_module('tests.pipelines.'+args.family+'.demo')
    receipt={'family':args.family,'label':args.label,'profiled':args.profile,
             'recipe':'existing synthetic demo; explicit demo options; no biological finding',
             'source_before':source_identity(),'ledger_implementation':ledger_identity,'operations':[]}
    process=psutil.Process();done=threading.Event();samples=[]
    def memory():
        while not done.wait(.05):
            try:samples.append((time.perf_counter(),process_tree_rss(process)))
            except psutil.Error:pass
    thread=threading.Thread(target=memory,daemon=True);thread.start()
    started=time.perf_counter()
    try:
        source=demo.prepare(base/'fixture')
        if args.reservation_history:
            if args.family!='behaviour':raise ValueError('Reservation fixture is scoped to behaviour validation')
            history=source/'run/pooled/tables/.pipeline-confirmation/behaviour-states'
            files=sorted(args.reservation_history.glob('*.json'))
            if not files or sum(p.stat().st_size for p in files)>1024**2:
                raise ValueError('Expected a bounded native reservation history')
            history.mkdir(parents=True)
            receipt['reservation_history']={p.name:digest(p) for p in files}
            for original in files:shutil.copy2(original,history/original.name)
            receipt['recipe']+='; identical existing reservation history; empty result store; all stages recomputed'
        # Preparation is authoritative for names and source tables.
        requests=[]
        for path in sorted(source.rglob('*.json')):
            value=read_document(path)
            if isinstance(value,dict) and value.get('pipeline') in {v[0] for v in FAMILIES.values()}:
                action=next(k for k,v in FAMILIES.items() if v[0]==value['pipeline'])
                if args.family=='coordination' and path.name!='supported-request.json':continue
                if args.family=='intervention' and path.parent.name!=args.case:continue
                requests.append((path,value,action))
        if len(requests)!=1:raise ValueError('Expected one complete primary recipe: '+str([str(p) for p,_,_ in requests]))
        path,request,action=requests[0]
        if args.family=='coordination':
            request['sample_summary']={'enabled':True,'aggregation':'mean','evidence':{'method':'none'}}
        run=path.parent/'run'
        if not run.exists():run=source/'run'
        presentation=path.parent/'presentation.json'
        if not presentation.exists():presentation=source/'presentation.json'
        params={'run':run,'pipeline_request':request,'output_dir':base/'outputs',
                'if_exists':'skip','claim':'Controlled synthetic performance verification; no biological finding.'}
        if presentation.exists():params['presentation']=read_document(presentation)
        # Include the saved visual families omitted by the compact demos. These
        # request extra native outputs; they do not relax a scientific setting.
        if args.family=='coordination':
            params.setdefault('presentation',{}).setdefault('overview',{})['views']=['coverage','effects','samples']
        if args.family=='intervention':
            params.setdefault('presentation',{}).setdefault('reports',{})['views']=['cell_report','trace_grid']
        receipt['presentation']=params.get('presentation',{})
        receipt['request']=request
        receipt['inputs']={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
        import cProfile
        profiler=cProfile.Profile() if args.profile else None
        offset=len(samples);t=time.perf_counter()
        if profiler:profiler.enable()
        try:result=run_action(action,**params)
        finally:
            if profiler:profiler.disable();profiler.dump_stats(str(base/'profile.pstats'))
        elapsed=time.perf_counter()-t
        states={k:{'status':v.outcome.status,'reason':v.outcome.reason} for k,v in result.results.items()}
        receipt['operations'].append({'id':'action:'+action,'kind':'execution',
            'status':'measured' if result.successful else 'failed','seconds':elapsed,
            'peak_rss_bytes':max([v for _,v in samples[offset:]] or [process.memory_info().rss]),
            'retained_rss_bytes':process.memory_info().rss,'states':states,
            'record_path':str(result.record_path)})
        if not result.successful:raise AssertionError(states)
        receipt['status']='completed'
    except BaseException:
        receipt['status']='failed';receipt['error']=traceback.format_exc()
    finally:
        done.set();thread.join()
        receipt['total_seconds']=time.perf_counter()-started
        receipt['memory_scope']='action process and owned descendants; sampled RSS sum'
        receipt['memory_sampling']=MEMORY_SAMPLING
        receipt['source_unchanged']=receipt['source_before']==source_identity()
        receipt['dependencies_unchanged']=ledger_identity['dependency_environment']==dependency_environment()
        receipt['outputs']={p.relative_to(base).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size}
            for p in (base/'outputs').rglob('*') if p.is_file()}
        (base/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str),encoding='utf-8')
        print(json.dumps({k:receipt[k] for k in ['family','status','total_seconds','operations']},default=str),flush=True)
        if receipt.get('error'):print(receipt['error'],flush=True)
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
