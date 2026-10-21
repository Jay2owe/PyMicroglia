"""Compare batched overlap counts and complete motion evidence tables."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','config','adapter','labels','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stem',required=True)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-motion-overlap-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import pandas as pd
    import psutil
    import tifffile
    import common,pipeline,motion_handoffs as module
    import motion_overlap_batch as reuse
    rng=np.random.default_rng(2440);queries=0
    for i in range(80):
        frame=rng.integers(-3,9,(23,31),dtype=np.int64)
        components=rng.integers(-2,13,frame.shape,dtype=np.int32)
        if i%4==0:
            frame=frame.astype(np.uint64)
        counts=reuse.Counts(frame,components)
        for component in range(-2,15):
            mask=components==component
            for minimum,coverage in ((1,0.),(3,.05),(8,.2)):
                expected=[]
                for identity in set(map(int,np.unique(frame[mask])))-{0}:
                    body=frame==identity;overlap=int(np.count_nonzero(body&mask))
                    if overlap>=minimum and overlap/max(int(body.sum()),1)>=coverage:expected.append(identity)
                assert counts.eligible(component,minimum,coverage)==expected
                queries+=1
    cfg=common.Config.load(a.config);paths=pipeline._pinned_paths(cfg,a.stem)
    labels=tifffile.imread(a.labels);lag=tifffile.imread(paths['lag_float'])
    params=cfg.values['accepted_postprocessing']['identity_reservation']['motion_handoff']
    tick=time.perf_counter();cache=module.build_motion_pair_cache(lag,params);cache_seconds=time.perf_counter()-tick
    process=psutil.Process()
    before_cpu=process.cpu_times();tick=time.perf_counter()
    expected=module.motion_pair_evidence_for_fixed_objects(labels,lag,params,pair_cache=cache)
    before=time.perf_counter()-tick;end_cpu=process.cpu_times()
    old_cpu=end_cpu.user+end_cpu.system-before_cpu.user-before_cpu.system
    original=module.motion_pair_evidence_for_fixed_objects
    with reuse.install(module) as counters:
        assert counters['supported']
        before_cpu=process.cpu_times();tick=time.perf_counter()
        actual=module.motion_pair_evidence_for_fixed_objects(labels,lag,params,pair_cache=cache)
        after=time.perf_counter()-tick;end_cpu=process.cpu_times()
        new_cpu=end_cpu.user+end_cpu.system-before_cpu.user-before_cpu.system
    assert module.motion_pair_evidence_for_fixed_objects is original
    assert actual.to_csv(index=False)==expected.to_csv(index=False)
    # Existing evidence and a restricted transition selection retain the same
    # names and ordering; they are execution inputs, never production targets.
    transitions={0,1,3}
    prior=pd.DataFrame([dict(t=0,local_id=1,green_track_ids='7;9'),dict(t=1,local_id=2,green_track_ids='nan')])
    expected_small=original(labels[:8],lag[:7],params,existing=prior,transitions=transitions,pair_cache=cache[:7])
    with reuse.install(module):
        actual_small=module.motion_pair_evidence_for_fixed_objects(labels[:8],lag[:7],params,existing=prior,transitions=transitions,pair_cache=cache[:7])
    assert actual_small.to_csv(index=False)==expected_small.to_csv(index=False)
    memory=process.memory_info()
    report=dict(status='pass',count_queries=queries,frames=len(labels),rows=len(actual),
        pair_cache_seconds=cache_seconds,before_seconds=before,after_seconds=after,
        before_cpu_seconds=old_cpu,after_cpu_seconds=new_cpu,peak_resident_bytes=memory.peak_wset,
        counters=counters,table_sha256=hashlib.sha256(actual.to_csv(index=False).encode()).hexdigest(),
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 (Path(__file__),Path(reuse.__file__),Path(module.__file__))},
        scope='Complete motion-evidence operation on 476-frame native labels; shared pair cache preparation timed separately')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
