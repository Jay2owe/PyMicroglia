"""Compare every motion component and descriptor on a complete recording."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','config','adapter','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stem',required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-motion-lobe-batch-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import psutil
    import tifffile
    import common,pipeline,motion_handoffs as module
    import motion_lobe_batch as reuse
    cfg=common.Config.load(a.config);paths=pipeline._pinned_paths(cfg,a.stem)
    params=cfg.values['accepted_postprocessing']['identity_reservation']['motion_handoff']
    rng=np.random.default_rng(2712)
    cases=[np.zeros((9,11),bool),np.ones((9,11),bool),np.eye(12,dtype=bool)]
    for _ in range(20):
        case=rng.random((31,39))>.7;cases.extend([case,case.T,case[::2]])
    before=after=0.;observations=0;calls=0
    def check(binary,minimum):
        nonlocal before,after,observations,calls
        tick=time.perf_counter();expected=module._motion_lobes(binary,minimum);before+=time.perf_counter()-tick
        tick=time.perf_counter();actual=reuse.lobes(binary,minimum,module,module._motion_lobes);after+=time.perf_counter()-tick
        np.testing.assert_array_equal(expected[0],actual[0]);assert len(expected[1])==len(actual[1])
        for old,new in zip(expected[1],actual[1]):
            assert old.component==new.component and old.area==new.area
            assert old.position.tobytes()==new.position.tobytes(),('position',old,new)
            assert old.shape.tobytes()==new.shape.tobytes(),('shape',old,new)
            observations+=1
        calls+=1
    for case in cases:
        for minimum in (1,3,8):check(case,minimum)
    lag=tifffile.imread(paths['lag_float'])
    threshold=float(params['motion_lobe_log2']);minimum=int(params['minimum_motion_lobe_px'])
    for frame in lag:
        finite=np.isfinite(frame)
        check((frame<=-threshold)&finite,minimum)
        check((frame>=threshold)&finite,minimum)
    original=module._motion_lobes
    with reuse.install(module) as counters:
        assert counters['supported'] and module._motion_lobes is not original
    assert module._motion_lobes is original
    report=dict(status='pass',transitions=len(lag),edge_cases=len(cases),calls=calls,
        observations=observations,before_seconds=before,after_seconds=after,
        peak_resident_bytes=psutil.Process().memory_info().peak_wset,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 (Path(__file__),Path(reuse.__file__),Path(module.__file__))},
        scope='Every motion component map, count, centroid and shape bit on the complete recording plus edge cases')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
