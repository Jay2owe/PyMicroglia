"""Check batched local ownership against every scalar vote and table value.

Repeated calls on movies edited in a few frames, and on point subsets, check
the per-frame vote reuse: every call is compared with the original sampler.
"""
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
    p.add_argument('--plane-budget',type=int,help='bytes of frame copies the sampler may hold (0 forces frame digests)')
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-owner-sampling-check'
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0]=[str(a.snapshot.resolve()),str(a.adapter.resolve())]
    from pymicroglia.tracking import completed_feedback
    import correction_owner_sampling as reuse
    if a.plane_budget is not None:reuse.PLANE_BUDGET=a.plane_budget
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
                if trial%3==0:
                    labels[int(rng.integers(0,4))]=rng.choice([0,1,2,7,81],size=(29,31)).astype(np.uint16)
                    compare(points,labels);compare(points.sample(frac=.5,random_state=trial),labels)
            labels=tifffile.imread(a.labels);points=pd.read_csv(a.points,low_memory=False)
            full_before,full_after=before,after;compare(points,labels)
            full_before=before-full_before;full_after=after-full_after
            compare(points.iloc[:0],labels)
            edited=labels.copy()
            for t in rng.choice(len(labels),5,replace=False):
                names=np.unique(edited[t]);names=names[names>0]
                edited[t][edited[t]==names[0]]=names[-1]
                edited[t][:40,:40]=0
            compare(points,edited);compare(points.sample(frac=.3,random_state=1),edited)
            compare(points,labels);compare(points,edited.astype(np.int32))
        assert module.attach_owners is original
        assert counters['reused']>0 and counters['measured']>0,counters
    report=dict(status='pass',plane_budget=reuse.PLANE_BUDGET,cases=cases,points=observations,before_seconds=before,after_seconds=after,
        recording_before_seconds=full_before,recording_after_seconds=full_after,counters=counters,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),Path(module.__file__),Path(reuse.__file__),a.labels,a.points]},
        scope='All scalar owner votes, row/index order, dtypes, visibility and exact CSV values; full correction sequence still required')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}))


if __name__=='__main__':main()
