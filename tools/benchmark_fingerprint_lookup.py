"""Interleaved fresh-process lookup trials on real small/large index contents.

Only eight tiny local sources are accessed. The frozen large index's artefact
descriptions are workload ballast; no referenced laboratory image is opened.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time

from benchmark_pipeline_execution import ROOT, configure_ledger, digest


def worker(folder, mode):
    os.environ['PYMICROGLIA_INDEX']=str(folder)
    from pymicroglia import config
    from auto_organotypic.store import manifest
    import psutil
    identity=configure_ledger(mode=='before')
    assert config.index_root().resolve()==folder.resolve()
    sources=sorted((folder/'sources').glob('*.txt'))
    process=psutil.Process();memory=[process.memory_info().rss];done=threading.Event()
    def sample():
        while not done.wait(.05):memory.append(process.memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    started=time.perf_counter()
    try:
        values=[manifest.fingerprint(p).as_dict() for _ in range(2) for p in sources]
        elapsed=time.perf_counter()-started
        memory.append(process.memory_info().rss)
    finally:
        done.set();monitor.join()
    print(json.dumps({'mode':mode,'seconds':elapsed,'lookups':len(values),
        'peak_rss_bytes':max(memory),'retained_rss_bytes':memory[-1],
        'results_sha256':hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest(),
        'implementation':identity,'index_sha256':digest(folder/'manifest.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path);parser.add_argument('--mode')
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.mode)
    root=ROOT/'development/speed-pass-20261011'
    base=root/'complete/fingerprint-lookups';base.mkdir(exist_ok=False)
    if shutil.disk_usage(base).free<1024**3:raise OSError('One GiB free required for bounded index trials')
    from auto_organotypic.store import keys,manifest
    inputs={
        'pipeline':root/'complete/pipeline-rhythm-before/rhythm/index.json/manifest.json',
        'large':root/'execution/index-seed.json.gz',
    }
    recipes={}
    for case,source in inputs.items():
        folder=base/case;folder.mkdir();(folder/'sources').mkdir()
        opener=gzip.open if source.suffix=='.gz' else open
        with opener(source,'rt',encoding='utf-8') as stream:data=json.load(stream)
        data['fingerprints']={}
        for n in range(8):
            path=folder/'sources'/f'input{n}.txt'
            path.write_text(('local synthetic input '+str(n)+'\n')*30,encoding='utf-8')
            data['fingerprints'][manifest._slot(path)]=keys.fingerprint(path).as_dict()
        target=folder/'manifest.json';target.write_text(json.dumps(data,indent=2),encoding='utf-8')
        recipes[case]={'seed':str(source),'seed_sha256':digest(source),'index_bytes':target.stat().st_size,
            'index_sha256':digest(target),'source_hashes':{p.name:digest(p) for p in (folder/'sources').glob('*.txt')}}
        del data;gc.collect()
    results=[]
    for case in inputs:
        for number,mode in enumerate(['before','after','after','before']):
            completed=subprocess.run([shutil.which('python'),str(Path(__file__).resolve()),
                '--worker',str(base/case),'--mode',mode],capture_output=True,text=True)
            (base/f'{case}-{number}-{mode}.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
            if completed.returncode:raise RuntimeError(completed.stderr[-2000:])
            row=json.loads(completed.stdout.strip().splitlines()[-1]);row['case']=case
            results.append(row)
            (base/'trials.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:row[k] for k in ['case','mode','seconds','peak_rss_bytes']}),flush=True)
    comparisons={}
    for case in inputs:
        rows=[r for r in results if r['case']==case]
        assert len({r['results_sha256'] for r in rows})==1
        assert {r['index_sha256'] for r in rows}=={recipes[case]['index_sha256']}
        before=statistics.median(r['seconds'] for r in rows if r['mode']=='before')
        after=statistics.median(r['seconds'] for r in rows if r['mode']=='after')
        comparisons[case]={'before_seconds':before,'after_seconds':after,
            'speedup':before/after,'parity':True,'recipe':recipes[case]}
    receipt={'comparison':comparisons,'trials':results,'scope':'16 source fingerprint lookups; not a whole-action or UI claim'}
    (base/'comparison.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
