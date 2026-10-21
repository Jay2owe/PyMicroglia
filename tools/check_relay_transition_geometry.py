"""Prove exact relay motion values and complete lineage candidate scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','config','labels','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stem',required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-relay-motion-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    import common,pipeline,oscillatory_reconciliation as module
    import relay_transition_geometry as reuse
    import host_partition_reuse
    rng=np.random.default_rng(811);checks=0
    original=module._transition_evidence
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for case in range(60):
            old=rng.random((29,35))>.7;new=rng.random(old.shape)>.7
            lag=rng.normal(0,3,old.shape).astype(np.float32)
            if case%5==0:old[:]=False
            if case%7==0:new[:]=False
            if case%11==0:lag[:,0]=np.nan;lag[:,1]=np.inf;lag[:,2]=-np.inf
            if case%2:old=old.T;new=new.T;lag=lag.T
            expected=original(old,new,lag)
            actual=reuse.evidence(old,new,lag,module)
            assert expected.keys()==actual.keys()
            for key in expected:
                if isinstance(expected[key],float):assert np.float64(expected[key]).tobytes()==np.float64(actual[key]).tobytes(),(case,key)
                else:assert expected[key]==actual[key],(case,key)
            checks+=1
    cfg=common.Config.load(a.config);paths=pipeline._pinned_paths(cfg,a.stem)
    labels=tifffile.imread(a.labels);lag=tifffile.imread(paths['lag_float'])
    params=cfg.values['accepted_postprocessing']['oscillatory_lineage']
    def execute():
        with host_partition_reuse.install(module):
            tick=time.perf_counter();expected=module.graph_lineage_candidates(labels,lag,params['lineage_graph'],params['minimum_substantial_core_px']);before=time.perf_counter()-tick
            with reuse.install(module) as counters:
                assert counters['supported']
                tick=time.perf_counter();actual=module.graph_lineage_candidates(labels,lag,params['lineage_graph'],params['minimum_substantial_core_px']);after=time.perf_counter()-tick
            assert expected.to_csv(index=False)==actual.to_csv(index=False)
            assert actual.to_csv(index=False).encode()==a.reference.read_bytes()
            return before,after,len(actual),counters
    with ThreadPoolExecutor(max_workers=1) as executor:before,after,rows,counters=executor.submit(execute).result()
    report=dict(status='pass',frames=len(labels),edge_cases=checks,rows=rows,
        before_seconds=before,after_seconds=after,counters=counters,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),Path(reuse.__file__),Path(module.__file__))},
        scope='Complete lineage candidate table including all relay transitions, original saved table and edge-case scalar bits; excludes host partitions and rendering')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
