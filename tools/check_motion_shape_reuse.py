"""Compare every motion-region pairing on a complete recording."""
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
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-motion-shape-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import psutil
    import tifffile
    import common,pipeline,motion_handoffs as module
    import motion_shape_reuse as reuse
    cfg=common.Config.load(a.config);paths=pipeline._pinned_paths(cfg,a.stem)
    lag=tifffile.imread(paths['lag_float'])
    params=cfg.values['accepted_postprocessing']['identity_reservation']['motion_handoff']
    before=after=0.;pairs=0;original=module._shape_descriptor
    with reuse.install(module) as counters:
        assert counters['supported'];cached=module._shape_descriptor
        mask=np.array([[True,True],[False,True]])
        expected_shape=original(mask);first=cached(mask);first[:]=-9
        assert cached(mask).tobytes()==expected_shape.tobytes()
        for frame in lag:
            module._shape_descriptor=original
            tick=time.perf_counter();expected=module.pair_blue_red_regions(frame,params);before+=time.perf_counter()-tick
            module._shape_descriptor=cached
            tick=time.perf_counter();actual=module.pair_blue_red_regions(frame,params);after+=time.perf_counter()-tick
            for left,right in zip(expected[:2],actual[:2]):np.testing.assert_array_equal(left,right)
            assert list(expected[2])==list(actual[2])
            for identity,(left,features) in expected[2].items():
                right,got=actual[2][identity]
                assert left.component==right.component and left.area==right.area
                assert left.position.tobytes()==right.position.tobytes()
                assert left.shape.tobytes()==right.shape.tobytes()
                assert features.keys()==got.keys()
                for key,value in features.items():assert np.float64(value).tobytes()==np.float64(got[key]).tobytes()
                pairs+=1
    assert module._shape_descriptor is original
    report=dict(status='pass',transitions=len(lag),pairs=pairs,before_seconds=before,after_seconds=after,
        counters=counters,peak_resident_bytes=psutil.Process().memory_info().peak_wset,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 (Path(__file__),Path(reuse.__file__),Path(module.__file__))},
        scope='Complete blue/red motion pairing, all labels and every feature bit; alternating paired trials')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
