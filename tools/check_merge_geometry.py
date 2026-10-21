"""Prove every merge geometry field against every frame plus edge cases."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','snapshot','adapter','labels','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-merge-geometry-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.snapshot.resolve()),str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    import persistent_merges as module
    import merge_geometry_reuse as reuse
    from pymicroglia.tracking.frozen_io import LabelMask
    labels=tifffile.imread(a.labels)
    rng=np.random.default_rng(1835)
    cases=[np.zeros((8,11),np.uint16),np.array([[65000,0],[0,1]],np.uint16),
           np.array([[-1,0],[0,1]],np.int16)]
    for _ in range(10):
        frame=rng.integers(0,8,(19,27),dtype=np.uint16)
        cases.extend([frame,frame.T,frame[:,::2]])
    original=module._frame_geometry
    before_seconds=after_seconds=0.;count=0
    for frame in list(labels)+cases:
        tick=time.perf_counter();expected=original(frame);before_seconds+=time.perf_counter()-tick
        tick=time.perf_counter();actual=reuse.geometry(frame,LabelMask,original);after_seconds+=time.perf_counter()-tick
        assert list(actual)==list(expected)
        for identity,row in expected.items():
            got=actual[identity]
            assert row['area']==got['area'] and row['bounds']==got['bounds']
            assert row['position'].tobytes()==got['position'].tobytes()
            np.testing.assert_array_equal(np.asarray(row['mask']),np.asarray(got['mask']))
            count+=1
    with reuse.install(module,LabelMask) as stats:
        assert stats['supported']
        frame=cases[-1].copy();value=module._frame_geometry(frame)
        frozen={k:np.asarray(v['mask']).copy() for k,v in value.items()}
        frame[:]=0
        for k,v in value.items():np.testing.assert_array_equal(np.asarray(v['mask']),frozen[k])
    assert module._frame_geometry is original
    report=dict(status='pass',frames=len(labels),edge_cases=len(cases),observations=count,
        before_seconds=before_seconds,after_seconds=after_seconds,snapshot_immutable=True,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 (Path(__file__),Path(reuse.__file__),Path(module.__file__))})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
