"""Verify complete evidence tables, mapping tables and invalid-map errors."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--stem',required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-fixed-object-evidence-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import pandas as pd
    import tifffile
    import persistence,global_soma_ledger as ledger,pipeline
    import fixed_object_overlap as reuse
    original=(persistence.green_evidence_for_fixed_objects,persistence._frame_bijections,ledger.physical_soma_tracking_evidence)
    def evaluate(fn,*args):
        try:return ('value',fn(*args).to_csv(index=False))
        except Exception as error:return ('error',type(error).__name__,str(error))
    rng=np.random.default_rng(741)
    cases=[]
    for dtype in (np.int16,np.uint16,np.uint64):
        for kind in range(6):
            source=rng.integers(0,9,(3,11,13),dtype=np.uint16).astype(dtype)
            if dtype==np.int16:source=source-4
            target=source.copy()
            if kind==1:target[target!=0]+=20
            if kind==2:target[0,:2,:2]=0
            if kind==3:target[0,0,0]=31
            if kind==4:target[target!=0]=1
            if kind==5 and dtype==np.uint64:
                target[target!=0]+=np.uint64(2**63)
            table=pd.DataFrame([dict(t=t,local_id=int(v),green_track_ids='2;5') for t,f in enumerate(source) for v in np.unique(f)])
            cases.append((source,target,table))
    for source,target,table in cases:
        expected=[evaluate(original[0],source,target,table),evaluate(original[1],source,target),evaluate(original[2],source,target,source,table)]
        with reuse.install(persistence,ledger,pipeline):
            actual=[evaluate(persistence.green_evidence_for_fixed_objects,source,target,table),evaluate(persistence._frame_bijections,source,target),evaluate(ledger.physical_soma_tracking_evidence,source,target,source,table)]
        assert expected==actual
    observations,table,anchor=pipeline._load_anchor_inputs(a.stem,a.run)
    parent=a.engine/'m12_identity_reservation'/a.run/'out'/(a.stem+'.tif')
    physical_path=a.engine/'m19_layered_lineage'/a.run/'mid'/(a.stem+'_physical_somas.tif')
    accepted=tifffile.imread(parent);physical=tifffile.imread(physical_path)
    arguments=[(physical,observations,table),(accepted,accepted),(physical,accepted,observations,table)]
    timings=[]
    for index,args in enumerate(arguments):
        tick=time.perf_counter();expected=evaluate(original[index],*args);before=time.perf_counter()-tick
        with reuse.install(persistence,ledger,pipeline) as counters:
            fn=(persistence.green_evidence_for_fixed_objects,persistence._frame_bijections,ledger.physical_soma_tracking_evidence)[index]
            tick=time.perf_counter();actual=evaluate(fn,*args);after=time.perf_counter()-tick
        assert expected[0]=='value' and expected==actual
        timings.append(dict(operation=original[index].__name__,before_seconds=before,after_seconds=after,
            csv_sha256=hashlib.sha256(actual[1].encode()).hexdigest()))
    assert original==(persistence.green_evidence_for_fixed_objects,persistence._frame_bijections,ledger.physical_soma_tracking_evidence)
    report=dict(status='pass',frames=len(physical),edge_cases=len(cases),timings=timings,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__),Path(reuse.__file__),Path(persistence.__file__),Path(ledger.__file__))},
        inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (parent,physical_path)},
        scope='Three complete 476-frame evidence/mapping tables, unusual labels, ordered error messages and restoration; not complete tracking')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('sources','inputs')}),flush=True)


if __name__=='__main__':main()
