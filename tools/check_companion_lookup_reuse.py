"""Differential companion lookups and complete excursion discovery."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    p=argparse.ArgumentParser()
    for key in ('engine','adapter','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-companion-lookup-checks'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import pandas as pd
    import bounded_owner_excursion as module
    import companion_lookup_reuse as adapter
    rng=np.random.default_rng(936);cases=[]
    for case in range(12):
        rows=[];labels=np.zeros((24,47,59),np.uint16)
        for track in range(1,13):
            y,x=5+(track//4)*10,5+(track%4)*12
            labels[:,y-3:y+4,x-3:x+4]=track
            labels[8:11,y-3:y+4,x-3:x+4]=track%12+1
            for frame in range(24):
                rows.append(dict(track_id=track,frame=frame,x=float(x),y=float(y),radius_px=3.,
                    strong=True,in_encounter=False,state='observed',
                    physically_visible=bool(rng.integers(0,2)),candidate_owner=int(rng.integers(0,13))))
        attached=pd.DataFrame(rows)
        if case%2:attached=pd.concat([attached,attached.iloc[[case]]],ignore_index=True)
        stats=dict(queries=0,indexed_rows=0);index=adapter.CompanionIndex(attached,module,stats)
        for _ in range(30):
            args=(int(rng.integers(1,15)),int(rng.integers(0,15)),rng.integers(0,24,6).tolist(),
                  int(rng.integers(0,12)),int(rng.integers(0,4)),float(rng.uniform(0,8)))
            assert module._companion(attached,*args)==index.query(*args)
        expected=module.discover(labels,attached,{})
        with adapter.install(module) as counters:actual=module.discover(labels,attached,{})
        assert counters['supported'] and counters['queries']>0
        for l,r in zip(expected,actual):
            pd.testing.assert_frame_equal(l,r,check_exact=True)
            assert l.to_csv(index=False)==r.to_csv(index=False)
        cases.append(dict(case=case,lookup_queries=stats['queries'],discover_queries=counters['queries'],audit_rows=len(actual[0])))
    report=dict(status='pass',cases=cases,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(module.__file__),Path(adapter.__file__)]})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='pass',cases=len(cases))))


if __name__=='__main__':main()
