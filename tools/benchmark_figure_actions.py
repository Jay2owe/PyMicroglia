"""Execute public figure actions, including export, against frozen saved data.

Uses native package renderers and their audited figure/table export. Failures
remain failures; no mocks, fabricated eligibility, or plotting-only timings.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

from benchmark_pipeline_execution import ROOT, digest, source_identity, configure_ledger, dependency_environment, process_tree_rss, MEMORY_SAMPLING


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--actions',nargs='*')
    parser.add_argument('--saved-run',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--stem',default='parity_A1')
    parser.add_argument('--options',type=Path,help='Explicit per-action input selections; never scientific replacements')
    parser.add_argument('--baseline-ledger',action='store_true')
    args=parser.parse_args()
    base=ROOT/'development/speed-pass-20261011/complete'/args.label/'figures'
    base.mkdir(parents=True,exist_ok=True)
    os.environ['PYMICROGLIA_STORE']=str(base/'store')
    os.environ['PYMICROGLIA_INDEX']=str(base/'index.json')
    os.environ['MPLBACKEND']='Agg'
    sys.executable=shutil.which('python') or sys.executable
    import psutil
    from pymicroglia.run import run_action
    from pymicroglia.visualisation.figures import load
    from pymicroglia._results import figure_plans
    ledger_identity=configure_ledger(args.baseline_ledger)
    figures=load();run=args.saved_run or args.run or ROOT/'tests/fixtures/motion_parity/run'
    before=source_identity()
    plans=figure_plans(run) if args.saved_run else {}
    if args.saved_run and not plans:
        # Method-audit records presentation plans at its results root, while
        # the other families normally put them beside measurement inputs.
        candidates=list((run.parents[1]/'outputs').glob('*/.auto-organotypic/figure-plans.json'))
        if len(candidates)!=1:
            raise ValueError('No unique saved presentation plan for '+str(run))
        run=candidates[0].parent.parent
        plans=figure_plans(run)
    input_hashes={p.relative_to(run).as_posix():digest(p) for p in run.rglob('*')
                  if p.is_file() and p.suffix.lower() in {'.csv','.npz','.npy','.tif','.tiff','.json'}}
    input_snapshot=json.dumps(input_hashes,sort_keys=True)
    (base/'input-snapshot.json').write_text(input_snapshot,encoding='utf-8')
    input_identity=hashlib.sha256(input_snapshot.encode()).hexdigest()
    # Existing parity demo choices, required to select actual source identities
    # and a declared cycle, never to manufacture a statistical eligibility result.
    overrides={'cell_report_card':{'identity':1,'images':0,'metrics':['corrected_mean','area_px','turnover_index']},
        'predictability_clock':{'period_hours':24.},
        'tissue_tectonics':{'reporter_metric':'corrected_mean'},
        'mask_change_map':{'identity':1},'cell_time_sculpture':{'identity':1},'cell_trace_card':{'identity':1}}
    if args.options:
        for key,value in json.loads(args.options.read_text()).items():
            overrides.setdefault(key,{}).update(value)
    names=args.actions or (sorted({v['figure'].replace('-','_') for v in plans.values()})
                           if args.saved_run else [k for k,v in figures.items() if not v.saved])
    process=psutil.Process()
    for action in names:
        target=base/action
        target.mkdir(exist_ok=False)
        receipt={'id':'action:'+action,'kind':'execution','label':args.label,'source_before':before,
            'fixture':str(run),'recipe':'frozen synthetic Motion parity tables; native public renderer and export',
            'overrides':overrides.get(action,{}),'status':'failed','ledger_implementation':ledger_identity,
            'input_snapshot_sha256':input_identity,'input_snapshot':str(base/'input-snapshot.json')}
        if shutil.disk_usage(base).free<400*1024**2:
            receipt['error']='Less than 400 MiB headroom';(target/'receipt.json').write_text(json.dumps(receipt));break
        done=threading.Event();memory=[process.memory_info().rss]
        def monitor():
            while not done.wait(.05):memory.append(process_tree_rss(process))
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        started=time.perf_counter()
        try:
            params={'run':run,'output_dir':target/'output','claim':'Controlled synthetic export performance verification; no biological finding.'}
            if args.saved_run:
                matching=[r for r in plans.values() if r['figure'].replace('-','_')==action]
                if not matching:raise ValueError('No saved figure plan from the complete pipeline for '+action)
                params['item']=matching[0]['name']
            else:params.update(stem=args.stem or None,**overrides.get(action,{}))
            result=run_action(action,**params)
            receipt['seconds']=time.perf_counter()-started
            files=[Path(p) for p in result['figures']]
            if not files or not all(p.is_file() and p.stat().st_size for p in files):raise AssertionError('No exported figure')
            receipt['status']='measured';receipt['result']={k:str(v) for k,v in result.items()}
        except Exception:receipt['error']=traceback.format_exc();receipt['seconds']=time.perf_counter()-started
        finally:
            done.set();thread.join();gc.collect()
            receipt['peak_rss_bytes']=max(memory);receipt['retained_rss_bytes']=process.memory_info().rss
            receipt['memory_scope']='action process and owned descendants; sampled RSS sum'
            receipt['memory_sampling']=MEMORY_SAMPLING
            receipt['outputs']={p.relative_to(target).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size}
                                for p in (target/'output').rglob('*') if p.is_file()}
            receipt['source_unchanged']=before==source_identity()
            receipt['dependencies_unchanged']=ledger_identity['dependency_environment']==dependency_environment()
            (target/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str),encoding='utf-8')
            print(json.dumps({k:receipt[k] for k in ['id','status','seconds','peak_rss_bytes']}),flush=True)
            if receipt.get('error'):print(receipt['error'].splitlines()[-1],flush=True)


if __name__=='__main__':main()
