"""Verify direct history values and mutable presence against the original loops."""
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
    platform.platform=lambda *a,**k:'Windows-motion-history-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    import common,pipeline,motion_handoffs as module
    import motion_history_reuse as reuse
    cfg=common.Config.load(a.config);paths=pipeline._pinned_paths(cfg,a.stem)
    labels=tifffile.imread(a.labels);raw=tifffile.imread(paths['registered_raw'])
    rng=np.random.default_rng(2098);before=after=0.;queries=0
    def equal(expected,actual):
        if expected is None or actual is None:
            assert expected is actual
            return
        assert expected.keys()==actual.keys()
        for key,value in expected.items():
            if isinstance(value,np.ndarray):assert value.tobytes()==actual[key].tobytes(),key
            else:assert np.float64(value).tobytes()==np.float64(actual[key]).tobytes(),key
    for t in rng.integers(0,len(labels),180):
        identities=np.unique(labels[t]);identity=int(rng.choice(identities))
        frames=int(rng.integers(0,14))
        tick=time.perf_counter();expected=module._recent_identity_state(labels,raw,identity,t,frames);before+=time.perf_counter()-tick
        tick=time.perf_counter();actual=reuse.recent_state(labels,raw,identity,t,frames,module._shape_descriptor);after+=time.perf_counter()-tick
        equal(expected,actual);queries+=1
    small=rng.integers(0,9,(67,13,17),dtype=np.uint16)
    # Include long runs, absent identities, gaps and zero background.
    small[:]=0;small[:,2:5,2:6]=1;small[31:35,2:5,2:6]=2
    stats=dict(age_queries=0,presence_refreshes=0);index=reuse.Presence(small,stats)
    for _ in range(100):
        t=int(rng.integers(len(small)))
        small[t,rng.integers(0,13,9),rng.integers(0,17,9)]=rng.integers(0,9,9)
        index.refresh(t)
        for identity in range(10):
            for frame in (t,0,len(small)-1,-1):
                assert index.age(identity,frame)==module._segment_age(small,identity,frame)
    lag=tifffile.imread(paths['lag_float'])
    params=cfg.values['accepted_postprocessing']['identity_reservation']['motion_handoff']
    # Complete prefix checks invalidation under the real production mutation
    # order. Full-stage verification follows; this is not a whole-run claim.
    length=24
    original=module.carry_established_identities_along_motion
    expected=original(labels[:length],raw[:length],lag[:length-1],params)
    with reuse.install(module) as counters:
        assert counters['supported']
        actual=module.carry_established_identities_along_motion(labels[:length],raw[:length],lag[:length-1],params)
    assert module.carry_established_identities_along_motion is original
    np.testing.assert_array_equal(expected[0],actual[0])
    for a_table,b_table in zip(expected[1:],actual[1:]):assert a_table.to_csv(index=False)==b_table.to_csv(index=False)
    report=dict(status='pass',recent_queries=queries,before_seconds=before,after_seconds=after,
        mutable_presence_queries=stats['age_queries'],prefix_frames=length,
        prefix_events=len(actual[1]),prefix_audit=len(actual[2]),counters=counters,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 (Path(__file__),Path(reuse.__file__),Path(module.__file__))})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
