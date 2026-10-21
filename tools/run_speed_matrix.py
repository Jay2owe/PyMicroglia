"""Run explicitly listed benchmark commands serially, retaining every exit/log.

Resume skips only commands with exit zero; individual action receipts still
decide scientific success. No parallel timing trials and no unbounded retries.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from benchmark_pipeline_execution import ROOT


def compare_figures(args):
    """Compare each native export pair; discard only verified temporary copies."""
    if args[0]!='tools/benchmark_figure_actions.py' or '--baseline-ledger' in args:return []
    label=args[args.index('--label')+1]
    if '-after' not in label:return []
    base=ROOT/'development/speed-pass-20261011/complete'
    before=base/label.replace('-after','-before')/'figures'
    after=base/label/'figures';records=[]
    for current in sorted(after.glob('*/receipt.json')):
        original=before/current.parent.name/'receipt.json'
        if not original.exists():continue
        receipt=base/'comparisons'/(label+'-'+current.parent.name+'.json')
        result=subprocess.run([shutil.which('python'),'tools/compare_action_execution.py',
            str(original),str(current),str(receipt)],cwd=ROOT,capture_output=True,text=True)
        row={'comparison':str(receipt),'exit_code':result.returncode,'cleanup':[]}
        if result.returncode==0:
            for action_receipt in (original,current):
                target=(action_receipt.parent/'output').resolve()
                if not target.is_relative_to(base.resolve()) or target.name!='output' or target.parent.parent.name!='figures':
                    raise ValueError('Unsafe benchmark cleanup target '+str(target))
                if target.is_dir():
                    row['cleanup'].append({'path':str(target),'files':sum(p.is_file() for p in target.rglob('*'))})
            # Record the exact checked targets before removing these reproductions.
            cleanup=receipt.with_suffix('.cleanup.json')
            cleanup.write_text(json.dumps({**row,'status':'verified; cleanup pending'},indent=2))
            for item in row['cleanup']:shutil.rmtree(item['path'])
            cleanup.write_text(json.dumps({**row,'status':'complete; inputs and receipts retained'},indent=2))
        else:row['error']=result.stderr[-2000:]
        records.append(row)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path)
    args=parser.parse_args();plan=json.loads(args.plan.read_text(encoding='utf-8'))
    record=args.plan.with_suffix('.execution.json')
    done=json.loads(record.read_text()) if record.exists() else []
    for row in plan:
        if any(item['id']==row['id'] and item['exit_code']==0 for item in done):continue
        if shutil.disk_usage(ROOT).free<700*1024**2:raise OSError('Less than 700 MiB free for next trial')
        log=args.plan.parent/(row['id']+'.log')
        started=time.time();print('Starting '+row['id'],flush=True)
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([shutil.which('python'),*row['args']],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        done.append({'id':row['id'],'command':row['args'],'exit_code':result.returncode,
                     'elapsed_seconds':time.time()-started,'log':str(log)})
        if result.returncode==0:done[-1]['comparisons']=compare_figures(row['args'])
        record.write_text(json.dumps(done,indent=2),encoding='utf-8')
        print('Finished '+row['id']+' exit='+str(result.returncode),flush=True)


if __name__=='__main__':main()
