"""Check all ownership masks, state updates and action tables exactly."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-ownership-cache-checks'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import pandas as pd
    import accepted_history as ah
    import ownership_observation_reuse as adapter
    module=ah._issue_module(ah.I012,'m1_reconcile_domain_ownership')
    cfg=ah._params('issue012','domain_ownership')['ownership']
    def equal(left,right):
        if isinstance(left,np.ndarray):assert np.array_equal(left,right,equal_nan=True)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:equal(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            assert len(left)==len(right)
            for l,r in zip(left,right):equal(l,r)
        elif isinstance(left,float) and np.isnan(left):assert np.isnan(right)
        else:assert left==right
    cases=[]
    rng=np.random.default_rng(8521)
    for case in range(12):
        labels=np.zeros((9,41,47),dtype=np.uint16)
        for i in range(1,9):
            y,x=rng.integers(0,33),rng.integers(0,39)
            labels[:,y:y+7,x:x+7]=i
        labels[3:6,labels[0]==1]=2
        lag=rng.normal(0,3,(8,41,47)).astype('float32')
        ledger=np.zeros_like(labels)
        states={i:dict(identity=i,host_identity=2,start_source_frame=1,age=case,
                      reference_area=40,anchor_mask=labels[0]==i,host_mask=labels[0]>0)
                for i in range(1,9)}
        state2=deepcopy(states);rows=[];rows2=[]
        old=module.apply_dormancy(labels[0],labels[1],ledger[1],lag[0],cfg,states,2,rows,{})
        with adapter.install(module) as stats:
            new=module.apply_dormancy(labels[0],labels[1],ledger[1],lag[0],cfg,state2,2,rows2,{})
        assert len(stats['supported'])==2
        equal((old,states,rows),(new,state2,rows2))
        old=module.reconcile(labels,ledger,lag,{'O','R','M','U'},cfg,0)
        with adapter.install(module):new=module.reconcile(labels,ledger,lag,{'O','R','M','U'},cfg,0)
        equal(old[:2],new[:2])
        for l,r in zip(old[2:],new[2:]):
            pd.testing.assert_frame_equal(l,r,check_exact=True)
            assert l.to_csv(index=False)==r.to_csv(index=False)
        cases.append(dict(case=case,actions=len(new[2]),registry=len(new[3])))
    report=dict(status='pass',cases=cases,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(adapter.__file__),Path(module.__file__)]})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='pass',cases=cases)))


if __name__=='__main__':main()
