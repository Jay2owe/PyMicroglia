"""Verify exact centre/surround values, boundary behaviour and peak ties."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','raw','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-territory-peak-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    import territory_memory as module
    import territory_peak_geometry as reuse
    original=module._centre_surround_peak;rng=np.random.default_rng(712)
    before=after=0.;cases=0
    def check(image,point,settings):
        nonlocal before,after,cases
        tick=time.perf_counter();expected=original(image,*point,*settings);before+=time.perf_counter()-tick
        tick=time.perf_counter();actual=module._centre_surround_peak(image,*point,*settings);after+=time.perf_counter()-tick
        assert expected[:2]==actual[:2],(point,settings,expected,actual)
        assert np.float64(expected[2]).tobytes()==np.float64(actual[2]).tobytes(),(point,settings,expected,actual)
        cases+=1
    with warnings.catch_warnings(),reuse.install(module) as stats:
        warnings.simplefilter('ignore',RuntimeWarning);assert stats['supported']
        for dtype in ('float32','float64','uint16'):
            for shape in ((1,9),(9,1),(17,21),(40,31)):
                image=rng.uniform(0,100,shape).astype(dtype)
                for point in ((0.,0.),(shape[1]-1.,shape[0]-1.),(4.5,5.5),(-3.,2.)):
                    for settings in ((3,1,2,4),(2,0,0,0),(0,2,5,3),(-1,1,2,3),(4,-2,-3,-5)):
                        check(image,point,settings)
            tied=np.ones((15,17),dtype=dtype)
            check(tied,(7.,7.),(3,2,3,5))
            if dtype.startswith('float'):
                tied[4,6]=np.nan;tied[10,8]=np.inf;check(tied,(7.,7.),(3,2,3,5))
        movie=tifffile.imread(a.raw)
        for index in range(len(movie)):
            point=(float(rng.uniform(0,movie.shape[2]-1)),float(rng.uniform(0,movie.shape[1]-1)))
            check(movie[index],point,(10,3,5,8))
    assert module._centre_surround_peak is original
    report=dict(status='pass',cases=cases,frames=len(movie),before_seconds=before,after_seconds=after,counters=stats,
        scope='Ordered peak coordinates and score bits on all raw frames plus edge/tie/nonfinite cases; no full-stage timing claim',
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),Path(reuse.__file__),Path(module.__file__))},
        input_sha256=hashlib.sha256(a.raw.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}))


if __name__=='__main__':main()
