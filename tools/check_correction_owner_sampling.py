"""Check batched local ownership against every scalar vote and table value."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('snapshot','adapter','correction-root','labels','points','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--module',choices=['separable_merge_recovery','resident_takeover'],default='separable_merge_recovery')
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-owner-sampling-check'
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0]=[str(a.snapshot.resolve()),str(a.adapter.resolve())]
    from pymicroglia.tracking import completed_feedback
    import correction_owner_sampling as reuse
    rng=np.random.default_rng(3652);cases=0;observations=0;before=after=0.
    with completed_feedback.methods(a.correction_root):
        module=importlib.import_module(a.module);original=module.attach_owners
        with reuse.install(module) as counters:
            assert counters['supported'] and module.attach_owners is not original
            def compare(points,labels):
                nonlocal cases,observations,before,after
                tick=time.perf_counter();old=original(points,labels);before+=time.perf_counter()-tick
                tick=time.perf_counter();new=module.attach_owners(points,labels);after+=time.perf_counter()-tick
                pd.testing.assert_frame_equal(old,new,check_exact=True)
                assert old.to_csv(index=False)==new.to_csv(index=False)
                cases+=1;observations+=len(points)
            for trial in range(50):
                labels=rng.choice([0,0,0,1,2,7,81],size=(4,29,31)).astype(np.uint16)
                points=pd.DataFrame(dict(frame=rng.integers(-1,4,120),
                    x=rng.uniform(0,30.99,120),y=rng.uniform(0,28.99,120),
                    radius_px=rng.uniform(-2,12,120),state=rng.choice(['observed','latent_visible','missing'],120)))
                if trial%5==0:points.index=np.repeat(np.arange(60),2)
                if trial%7==0:labels[:,10:15,10:15]=0
                compare(points,labels)
            labels=tifffile.imread(a.labels);points=pd.read_csv(a.points,low_memory=False)
            full_before,full_after=before,after;compare(points,labels)
            full_before=before-full_before;full_after=after-full_after
            compare(points.iloc[:0],labels)
        assert module.attach_owners is original
    report=dict(status='pass',cases=cases,points=observations,before_seconds=before,after_seconds=after,
        recording_before_seconds=full_before,recording_after_seconds=full_after,counters=counters,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),Path(module.__file__),Path(reuse.__file__),a.labels,a.points]},
        scope='All scalar owner votes, row/index order, dtypes, visibility and exact CSV values; full correction sequence still required')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}))


if __name__=='__main__':main()
