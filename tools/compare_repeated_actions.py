"""Summarize the fixed ABBA follow-up without dropping unfavorable observations."""
from pathlib import Path
import json
import shutil
import subprocess
from benchmark_pipeline_execution import ROOT,digest


def main():
    root=ROOT/'development/speed-pass-20261011/complete'
    design=json.loads((root/'regression-repeat-design.json').read_text())
    for action in design['actions']:
        pairs=[]
        for number in [1,2]:
            before=root/f'regression-repeat{number}-before/figures'/action/'receipt.json'
            after=root/f'regression-repeat{number}-after/figures'/action/'receipt.json'
            target=root/'comparisons'/f'regression-repeat{number}-after-{action}.json'
            old=json.loads(target.read_text()) if target.exists() else {}
            if not (old.get('parity') and old.get('before_receipt_sha256')==digest(before)
                    and old.get('after_receipt_sha256')==digest(after)):
                subprocess.run([shutil.which('python'),'tools/compare_action_execution.py',
                    str(before),str(after),str(target)],cwd=ROOT,check=False)
            record=json.loads(target.read_text())
            pairs.append((target,record))
        rows=[row for _,row in pairs]
        result={'id':'action:'+action,'kind':'execution',
            'status':'measured' if all(r['status']=='measured' for r in rows) else 'failed',
            'before_seconds':[value for r in rows for value in r['before_seconds']],
            'after_seconds':[value for r in rows for value in r['after_seconds']],
            'before_peak_rss_bytes':max(r['before_peak_rss_bytes'] for r in rows),
            'after_peak_rss_bytes':max(r['after_peak_rss_bytes'] for r in rows),
            'peak_rss_bytes':max(r['peak_rss_bytes'] for r in rows),
            'parity':all(r['parity'] for r in rows),
            'matched_stable_environment':all(r['matched_stable_environment'] for r in rows),
            'comparisons':[{'path':str(p),'sha256':digest(p)} for p,_ in pairs],
            'design':design,'interpretation':'Two counterbalanced pairs; all observations retained; no tight statistical latency guarantee.'}
        target=root/'comparisons'/('repeated-'+action+'.json')
        target.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result[k] for k in ['id','parity','before_seconds','after_seconds']}))


if __name__=='__main__':main()
