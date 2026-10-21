"""Compare body counts, including bridged cores and disconnected same-name cells."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','labels','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-body-conflict-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    import global_soma_ledger as module
    import conflict_core_batch as reuse
    original=module.substantial_identity_conflicts;rng=np.random.default_rng(21)
    bodies=np.zeros((4,33,39),np.int32)
    bodies[:,2:12,2:12]=1;bodies[:,19:29,20:31]=1
    bodies[1,8:24,9:23]=1
    bodies[2,0:5,30:39]=2;bodies[2,24:33,0:7]=2
    bodies[3]=1
    cases=[bodies,bodies[:,:,::-1],bodies.transpose(0,2,1),np.zeros((2,1,2),np.int32)]
    negatives=bodies.copy();negatives[negatives==1]=-9;cases.append(negatives)
    huge=bodies.astype(np.uint64);huge[huge==1]=np.uint64(2**63+1);cases.append(huge)
    cases += [rng.integers(0,3,(2,17,19),dtype=np.int16) for _ in range(6)]
    checks=0
    for labels in cases:
        for minimum in (-1,0,1,4,12,60):
            expected=original(labels,minimum)
            with reuse.install(module) as counters:actual=module.substantial_identity_conflicts(labels,minimum)
            assert expected.to_csv(index=False)==actual.to_csv(index=False),(checks,minimum)
            checks+=1
    labels=tifffile.imread(a.labels)
    tick=time.perf_counter();expected=original(labels);before=time.perf_counter()-tick
    with reuse.install(module) as counters:
        tick=time.perf_counter();actual=module.substantial_identity_conflicts(labels);after=time.perf_counter()-tick
    assert expected.to_csv(index=False)==actual.to_csv(index=False)
    assert module.substantial_identity_conflicts is original
    report=dict(status='pass',frames=len(labels),edge_cases=checks,rows=len(actual),
        before_seconds=before,after_seconds=after,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),Path(reuse.__file__),Path(module.__file__))},
        input_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),
        scope='Complete conflict table and adversarial body/core connectivity; same existing threshold and exact output')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
