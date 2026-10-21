"""Match workflow tables by stage, ignoring only generated cache-folder names."""
import argparse
import json
from pathlib import Path
import re
from compare_action_execution import equivalent
from benchmark_pipeline_execution import digest


def metadata_references(receipt, data):
    """Normalize only references proven against retained native output files."""
    from datetime import datetime
    from pymicroglia._results import read_document
    from pymicroglia.pipelines._contracts import content_id
    aliases={'master':{},'slug':{},'confirmation_id':{}}
    confirmations=[]
    for name,record in data['outputs'].items():
        path=receipt.parent/name
        if path.suffix=='.svg' and re.search(r'-[0-9a-f]{12}-p\d+',path.stem):
            if digest(path)!=record['sha256']:raise ValueError('Report changed since execution: '+name)
            normalized=re.sub(r'-[0-9a-f]{12}-p','-<binding>-p',path.name)
            aliases['master'][path.name]=normalized
            aliases['master'][json.dumps(path.name)]=json.dumps(normalized)
            aliases['slug'][path.stem]=Path(normalized).stem
        if '/science/' in name and path.name=='confirmation_record.json':
            if digest(path)!=record['sha256']:raise ValueError('Confirmation changed since execution')
            document=read_document(path)
            identity=document.pop('confirmation_id')
            if identity!=content_id(document):raise ValueError('Invalid native confirmation identity')
            if document.get('consumption') is not None:
                datetime.fromisoformat(document['consumption']['opened_utc'])
                document['consumption']['opened_utc']='<reservation-opened-time>'
            canonical=content_id(document)
            aliases['confirmation_id'][identity]=canonical
            confirmations.append(canonical)
    return aliases,sorted(confirmations)


def inventory(path):
    data=json.loads(path.read_text());files={}
    for name in data['outputs']:
        p=Path(name)
        if p.suffix.lower() not in {'.csv','.npy','.npz','.tif','.tiff'}:continue
        parts=list(p.parts)
        # Linked portable reports duplicate the already-compared source tables.
        if 'linked-results-index' in parts or 'audit-index' in parts:continue
        key='/'.join(part for part in parts if not re.fullmatch('[0-9a-f]{32}|[0-9a-f]{64}',part))
        # Some report names contain the first eight characters of their actual
        # invocation directory. Normalize only that verified relationship.
        for part in parts:
            if re.fullmatch('[0-9a-f]{32}',part):
                key=key.replace('-'+part[:8]+'-p','-<invocation>-p')
        key=re.sub(r'-[0-9a-f]{12}-p', '-<binding>-p',key)
        if key in files:raise ValueError('Ambiguous stage output '+key)
        files[key]=path.parent/name
        if digest(files[key])!=data['outputs'][name]['sha256']:
            raise ValueError('Output changed since its execution receipt: '+name)
    return data,files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path);parser.add_argument('after',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();a,af=inventory(args.before);b,bf=inventory(args.after)
    left_aliases,left_confirmations=metadata_references(args.before,a)
    right_aliases,right_confirmations=metadata_references(args.after,b)
    checks=[]
    if left_confirmations or right_confirmations:
        checks.append({'file':'native confirmation records','equal':left_confirmations==right_confirmations,
            'comparison':'verified content identities; complete records equal except reservation opening timestamp'})
    for key in sorted(set(af)|set(bf)):
        equal,method=(False,'missing counterpart') if key not in af or key not in bf else equivalent(
            af[key],bf[key],args.before.parent,args.after.parent,(left_aliases,right_aliases))
        checks.append({'file':key,'equal':bool(equal),'comparison':method})
    ar,br=a['operations'][0],b['operations'][0]
    scientific_suffixes={'.csv','.tif','.tiff','.npy','.npz'}
    ai={k:v for k,v in a.get('inputs',{}).items() if Path(k).suffix.lower() in scientific_suffixes}
    bi={k:v for k,v in b.get('inputs',{}).items() if Path(k).suffix.lower() in scientific_suffixes}
    input_checks=[]
    for key in sorted(set(ai)|set(bi)):
        equal=key in ai and key in bi and ai[key]==bi[key]
        method='identical recorded input hashes'
        if not equal and key in ai and key in bi:
            left=args.before.parent/'fixture'/key;right=args.after.parent/'fixture'/key
            if digest(left)!=ai[key] or digest(right)!=bi[key]:
                method='input changed since execution'
            else:
                equal,method=equivalent(left,right,args.before.parent,args.after.parent)
        input_checks.append({'source':key,'equal':bool(equal),'comparison':method})
    states={k:v['status'] for k,v in ar['states'].items()}=={k:v['status'] for k,v in br['states'].items()}
    unprofiled=not a.get('profiled') and not b.get('profiled')
    success=a['status']==b['status']=='completed' and ar['id']==br['id'] and unprofiled
    environment=a.get('ledger_implementation',{}).get('dependency_environment')
    stable=bool(environment) and environment==b.get('ledger_implementation',{}).get('dependency_environment')
    stable=stable and all(row.get('source_unchanged') and row.get('dependencies_unchanged') for row in (a,b))
    same_sampling=a.get('memory_sampling')==b.get('memory_sampling')
    same_reservations=a.get('reservation_history')==b.get('reservation_history')
    stable=stable and same_sampling
    stable=stable and a.get('source_before')==b.get('source_before')
    result={'id':ar['id'],'kind':'execution','status':'measured' if success else 'failed',
        'before_seconds':[ar['seconds']],'after_seconds':[br['seconds']],
        'peak_rss_bytes':max(ar['peak_rss_bytes'],br['peak_rss_bytes']),
        'before_peak_rss_bytes':ar['peak_rss_bytes'],'after_peak_rss_bytes':br['peak_rss_bytes'],
        'same_request':a['request']==b['request'],'same_stage_statuses':states,
        'same_presentation':a.get('presentation')==b.get('presentation'),
        'unprofiled_trials':unprofiled,
        'same_memory_sampling':same_sampling,
        'same_reservation_history':same_reservations,
        'input_checks':input_checks,
        'matched_stable_environment':bool(stable),
        'parity':success and stable and states and same_reservations and a['request']==b['request']
            and a.get('presentation')==b.get('presentation') and bool(input_checks)
            and all(c['equal'] for c in input_checks) and bool(checks) and all(c['equal'] for c in checks),
        'checks':checks,'before_receipt':str(args.before),'after_receipt':str(args.after),
        'before_receipt_sha256':digest(args.before),'after_receipt_sha256':digest(args.after)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    return 0 if result['parity'] else 1


if __name__=='__main__':raise SystemExit(main())
