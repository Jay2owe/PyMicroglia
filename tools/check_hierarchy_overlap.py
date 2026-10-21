"""Compare pixel intersection counts on real frames and adversarial arrays."""
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
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-hierarchy-overlap-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    import accepted_history as history
    import hierarchy_overlap_reuse as reuse
    import hierarchy_component_batch as components
    module=history._issue_module(history.I009,'m6_hierarchy_reconcile');geometry=module.geometry
    original=geometry.overlap;rng=np.random.default_rng(901)
    arrays=[np.asarray(x,dtype=dtype) for dtype in ('int64','uint64') for x in ([],[0],[0,1,4],[4,1,0],[1,1,2])]
    arrays += [np.sort(rng.choice(900,size=i,replace=False)) for i in range(1,100,4)]
    arrays += [np.asarray([2**63,2**64-1],dtype=np.uint64),np.asarray([-2**63,-1,2**63-1],dtype=np.int64)]
    cases=[geometry.Component(0,i,1,x,len(x),0.,0.) for i,x in enumerate(arrays)]
    before=after=0.;pairs=0
    def check(previous,current):
        nonlocal before,after,pairs
        tick=time.perf_counter();expected=[original(x,y) for x in previous for y in current];before+=time.perf_counter()-tick
        tick=time.perf_counter();actual=[geometry.overlap(x,y) for x in previous for y in current];after+=time.perf_counter()-tick
        assert expected==actual
        pairs+=len(expected)
    labels=tifffile.imread(a.labels)
    with reuse.install(geometry) as counters:
        assert counters['supported'];check(cases,cases)
        previous=[]
        for t,frame in enumerate(labels):
            current=components.components(frame,t,1,geometry.Component,history._cropped_frame_components)
            if previous:check(previous,current)
            previous=current
    assert geometry.overlap is original
    report=dict(status='pass',frames=len(labels),edge_arrays=len(cases),pairs=pairs,
        before_seconds=before,after_seconds=after,counters=counters,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__),Path(reuse.__file__),Path(components.__file__),Path(geometry.__file__))},
        input_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),scope='All adjacent-frame component pair overlap counts and adversarial fallback cases; not a stage timing')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
