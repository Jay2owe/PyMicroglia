"""Compare complete browser boundary snapshots, retaining every timing trial."""
import argparse
import copy
import json
from pathlib import Path
from benchmark_pipeline_execution import digest


def comparable(snapshot):
    result=copy.deepcopy(snapshot)
    if 'paints' in result:
        result['paints']=sorted(result['paints'])
    return result


def differences(left,right,path=''):
    if isinstance(left,dict) and isinstance(right,dict):
        return [item for key in sorted(set(left)|set(right))
                for item in differences(left.get(key),right.get(key),path+'/'+str(key))]
    if left==right:return []
    return [path]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();data=json.loads(args.receipt.read_text())
    stable=all(data.get(k) is True for k in ('source_unchanged','dependencies_unchanged','runtime_assets_unchanged'))
    stable=stable and len(data.get('browser_versions',[]))==4 and len(set(data['browser_versions']))==1
    clean=not data.get('failure') and not data.get('errors')
    restarts=data.get('restart_checks',[])
    restarted=len(restarts)==4 and all(r.get('passed') for r in restarts)
    args.output.mkdir(parents=True,exist_ok=True)
    for identity in ['initial-usable-screen','selected-item-full-detail','growth','restart','external-edit']:
        rows=[r for r in data['operations'] if r['id']=='workflow:'+identity]
        before=[r for r in rows if r['label']=='before'];after=[r for r in rows if r['label']=='after']
        reference=comparable(rows[0]['snapshot']) if rows else None
        mismatch=[{'label':r['label'],'paths':differences(reference,comparable(r['snapshot']))}
                  for r in rows if comparable(r['snapshot'])!=reference]
        success=len(before)==len(after)==2 and all(r.get('status')=='measured' for r in rows)
        result={'id':'workflow:'+identity,'kind':'execution','status':'measured' if success else 'failed',
                'before_seconds':[r['seconds'] for r in before],'after_seconds':[r['seconds'] for r in after],
                'peak_rss_bytes':max([r.get('peak_rss_bytes',0) for r in rows] or [0]),
                'before_peak_rss_bytes':max([r.get('peak_rss_bytes',0) for r in before] or [0]),
                'after_peak_rss_bytes':max([r.get('peak_rss_bytes',0) for r in after] or [0]),
                'matched_stable_environment':stable,'browser_errors':data.get('errors',[]),
                'parity':bool(success and stable and clean and not mismatch and restarted),
                'owned_server_restart_verified':restarted,
                'snapshot_differences':mismatch,'receipt_source':str(args.receipt),
                'receipt_source_sha256':digest(args.receipt),
                'comparison':'Exact full-detail data, selection, geometry and requested paint inventory; paint ordering ignored'}
        (args.output/('workbench-'+identity+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in {'snapshot_differences'}}))


if __name__=='__main__':main()
