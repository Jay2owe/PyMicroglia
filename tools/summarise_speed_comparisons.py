"""Publish every action's timing and resource result, including unverified rows."""
import csv
import json
import statistics
from benchmark_pipeline_execution import ROOT


def main():
    root=ROOT/'development/speed-pass-20261011'
    evidence=json.loads((root/'coverage-evidence.json').read_text())
    rows=[]
    for item in evidence['operations']:
        before=item.get('before_seconds',[]);after=item.get('after_seconds',[])
        old=statistics.median(before) if before and all(v is not None for v in before) else None
        new=statistics.median(after) if after and all(v is not None for v in after) else None
        verified=item.get('parity') is True
        rows.append({'operation':item['id'],'verified':verified,
            'before_seconds':old,'after_seconds':new,'before_over_after':old/new if old and new else None,
            'before_trials':len(before),'after_trials':len(after),
            'before_peak_rss_bytes':item.get('before_peak_rss_bytes'),
            'after_peak_rss_bytes':item.get('after_peak_rss_bytes'),
            'peak_rss_bytes':item.get('peak_rss_bytes'),
            'classification':('unverified' if not verified else 'slower observed' if new>old*1.05
                else 'faster observed' if new<old/1.05 else 'similar observed'),
            'receipt':item.get('receipt','')})
    output=root/'complete/action-timings.csv'
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'required':len(rows),'verified':sum(r['verified'] for r in rows),
        'interpretation':'Single pairs are observations, not statistical guarantees. Review all cases separately.',
        'slower_observed':[r for r in rows if r['classification']=='slower observed'],
        'unverified':[r['operation'] for r in rows if not r['verified']]}
    (output.parent/'comparison-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
