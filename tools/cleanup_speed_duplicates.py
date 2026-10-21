"""Remove verified candidate reproductions after coverage and all consumers finish."""
from pathlib import Path
import argparse
import json
import shutil
from benchmark_pipeline_execution import ROOT,digest


def current_record(target, records):
    """Later actions can legitimately update a shared result-folder manifest."""
    checksum=digest(target)
    matching=[row for row in records if row['sha256']==checksum]
    if not matching:
        raise ValueError('Reproduction changed after verification: '+str(target))
    return matching[-1]


def compact_large_tables(base):
    """Keep exact recovery copies of large retained tables after their consumers."""
    import gzip
    import hashlib
    folder=base/'compact-evidence'
    folder.mkdir(exist_ok=True)
    journal=folder/'recovery.json'
    result=json.loads(journal.read_text()) if journal.exists() else {
        'recipe':'Decompress archive into the recorded original path; require original_sha256 and bytes before using it.',
        'files':[]}
    known={row['original'] for row in result['files']}
    for receipt in sorted(base.glob('*/**/receipt.json')):
        data=json.loads(receipt.read_text())
        if 'source_before' not in data:continue
        trial=(base/receipt.relative_to(base).parts[0]).resolve()
        for name,record in data.get('outputs',{}).items():
            target=(receipt.parent/name).resolve()
            if not target.is_relative_to(trial):raise ValueError('Unsafe retained output '+str(target))
            if target.suffix!='.csv' or not target.is_file() or target.stat().st_size<10*1024**2:continue
            relative=target.relative_to(base).as_posix()
            if relative in known:continue
            checksum=digest(target)
            if checksum!=record['sha256']:raise ValueError('Retained output changed '+str(target))
            archive=folder/(checksum+'.csv.gz')
            if not archive.exists():
                temporary=archive.with_suffix('.partial')
                with target.open('rb') as source,temporary.open('wb') as output:
                    with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as packed:
                        shutil.copyfileobj(source,packed,1024*1024)
                temporary.replace(archive)
            verified=hashlib.sha256();size=0
            with gzip.open(archive,'rb') as source:
                while block:=source.read(1024*1024):verified.update(block);size+=len(block)
            if verified.hexdigest()!=checksum or size!=target.stat().st_size:
                raise ValueError('Recovery verification failed '+str(target))
            row={'original':relative,'original_sha256':checksum,'bytes':size,
                 'archive':archive.relative_to(base).as_posix(),'archive_sha256':digest(archive),
                 'archive_bytes':archive.stat().st_size,'execution_receipt':receipt.relative_to(base).as_posix(),
                 'status':'reconstruction verified; removal pending'}
            result['files'].append(row);journal.write_text(json.dumps(result,indent=2))
            target.unlink();row['status']='compacted';known.add(relative)
            journal.write_text(json.dumps(result,indent=2))
    return {'retained_tables':len(result['files']),
            'original_bytes':sum(row['bytes'] for row in result['files']),
            'archive_bytes':sum(p.stat().st_size for p in folder.glob('*.gz')),
            'recovery':str(journal)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    root=ROOT/'development/speed-pass-20261011';base=(root/'complete').resolve()
    gate=json.loads((root/'coverage-check.json').read_text())
    if not gate.get('complete') or gate.get('covered')!=148:
        raise ValueError('The complete 148-operation gate must pass first')
    import psutil
    for process in psutil.process_iter(['pid','cmdline']):
        command=process.info['cmdline'] or []
        if any(str(x).endswith('run_speed_matrix.py') for x in command) and any('speed-pass-20261011' in str(x) for x in command):
            try:
                if Path(process.cwd()).resolve()==ROOT.resolve():
                    raise ValueError('A benchmark matrix still needs its outputs')
            except psutil.NoSuchProcess:pass
    candidates={};protected=set();receipts={}
    for source in sorted((base/'comparisons').glob('*.json')):
        row=json.loads(source.read_text())
        if not row.get('after_receipt'):continue
        receipt=Path(row['after_receipt'])
        if not receipt.is_absolute():receipt=ROOT/receipt
        receipt=receipt.resolve()
        if not receipt.is_relative_to(base):continue
        label=receipt.relative_to(base).parts[0]
        if not label.endswith('-after'):continue
        data=receipts.setdefault(str(receipt),json.loads(receipt.read_text()))
        accepted=row.get('parity') is True and row.get('status')=='measured'
        if accepted and row.get('after_receipt_sha256')!=digest(receipt):
            raise ValueError('Comparison no longer authenticates its execution receipt')
        trial=(base/label).resolve()
        for name,details in data.get('outputs',{}).items():
            target=(receipt.parent/name).resolve()
            if not target.is_relative_to(trial) or target==receipt:
                raise ValueError('Unsafe reproduction target '+str(target))
            if not accepted:
                protected.add(target);continue
            if target.is_file():
                candidates.setdefault(target,[]).append({'path':str(target),'bytes':target.stat().st_size,
                    'sha256':details['sha256'],'execution_receipt':str(receipt),'comparison':str(source)})
    rows=[]
    for target,records in candidates.items():
        if target in protected:continue
        rows.append(current_record(target,records))
    output=base/'duplicate-cleanup.json'
    result={'apply':args.apply,'status':'verified; deletion pending' if args.apply else 'dry run',
            'files':rows,'bytes':sum(row['bytes'] for row in rows),
            'free_before':shutil.disk_usage(base).free,
            'retained':'Original inputs, accepted baseline outputs, recipes, execution/comparison hashes, screenshots and failure evidence'}
    output.write_text(json.dumps(result,indent=2))
    if args.apply:
        for row in rows:Path(row['path']).unlink()
        result['retained_table_compaction']=compact_large_tables(base)
        result.update(status='complete',free_after=shutil.disk_usage(base).free)
        output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':main()
