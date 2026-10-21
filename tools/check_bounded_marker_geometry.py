"""Check exact nearest-marker ties, radius boundaries and outside-image points."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-marker-geometry-check'
    sys.path[:0]=[str(a.adapter.resolve()),str((a.engine/'code').resolve())]
    import numpy as np
    import bounded_owner_excursion as module
    import bounded_marker_geometry as adapter
    original=module._marker;rng=np.random.default_rng(820);cases=0;before=after=0.
    with adapter.install(module) as counters:
        assert counters['supported']
        for trial in range(1200):
            height,width=(471,471) if trial>=200 else (31,37)
            mask=rng.random((height,width))<(.001 if trial%2 else .05)
            if trial%17==0:mask[:]=False
            if trial%19==0:mask[:]=True
            point=SimpleNamespace(x=float(rng.uniform(-20,width+20)),y=float(rng.uniform(-20,height+20)),
                radius_px=float(rng.choice([0.,2.,4.,8.,np.inf])) if trial%3 else float(rng.uniform(1,30)))
            if trial%13==0:
                mask[:]=False;point.x=width//2+.5;point.y=height//2+.5
                mask[height//2,width//2]=mask[height//2+1,width//2+1]=True
                point.radius_px=float(np.sqrt(.5))
            tick=time.perf_counter();expected=original(mask,point);before+=time.perf_counter()-tick
            tick=time.perf_counter();actual=module._marker(mask,point);after+=time.perf_counter()-tick
            assert actual==expected,(trial,point,expected,actual)
            cases+=1
    assert module._marker is original
    report=dict(status='pass',cases=cases,before_seconds=before,after_seconds=after,counters=counters,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [Path(__file__),Path(module.__file__),Path(adapter.__file__)]},scope='Exact marker coordinates, ties and no-match outcomes; no complete-stage timing claim')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
