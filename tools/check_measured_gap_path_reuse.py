"""Exact arbitrary-order prefix queries and measured-gap decisions."""
import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--module',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.adapter.resolve()))
    import numpy as np
    import measured_gap_path_reuse as reuse
    spec=importlib.util.spec_from_file_location('measured_gap_original',a.module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.trace_measured_core_path;rng=np.random.default_rng(409)
    before=after=0.;queries=0;decisions=0
    def exact(left,right):
        assert len(left)==len(right)
        for (lf,lp),(rf,rp) in zip(left,right):
            assert lf==rf and lp.dtype==rp.dtype and lp.tobytes()==rp.tobytes()
    for case in range(36):
        points={t:np.asarray([[10.+.05*t,12.+.03*t,3.],[30.,30.,2.]],dtype='float32' if case%2 else 'float64') for t in range(180)}
        if case%3==0:
            for t in range(11,180,13):points.pop(t,None)
        if case%4==0:points[79]=np.asarray([[13.95,14.37,3.],[13.95,14.37,3.]])
        start=179 if case%2 else 0
        initial=np.asarray([10.+.05*start,12.+.03*start,3.])
        endpoints=list(rng.permutation(180))
        tick=time.perf_counter();expected=[original(points,initial,start,int(end)) for end in endpoints];before+=time.perf_counter()-tick
        with reuse.install(module) as stats:
            assert stats['supported']
            tick=time.perf_counter();actual=[module.trace_measured_core_path(points,initial,start,int(end)) for end in endpoints];after+=time.perf_counter()-tick
            assert stats['peak_cache_bytes']<=32*1024**2
            assert stats['extended_frames']<=stats['requested_frames']
            for left,right in zip(expected,actual):exact(left,right)
            # Returning to a previous target and changing the geometry object
            # must recompute rather than reusing another frame's path.
            exact(original(dict(points),initial,start,endpoints[0]),module.trace_measured_core_path(dict(points),initial,start,endpoints[0]))
        queries+=len(endpoints)
    @dataclass
    class Source:
        mask:object
        t:int
    points={t:np.asarray([[10.+.05*t,12.+.03*t,3.]]) for t in range(100)}
    mask=np.ones((40,40),bool);observation={'mask':mask}
    native_memory=module.MeasuredSomaGapMemory(points,1,[])
    requests=[(Source(mask,int(source)),int(target)) for target in (70,90,40,90) for source in rng.permutation(30)]
    expected=[native_memory.core_scaled_gap_supported(source,observation,target) for source,target in requests]
    with reuse.install(module) as memory_stats:
        cached_memory=module.MeasuredSomaGapMemory(points,1,[])
        actual=[cached_memory.core_scaled_gap_supported(source,observation,target) for source,target in requests]
        assert expected==actual;decisions=len(actual)
    assert module.trace_measured_core_path is original
    report=dict(status='pass',queries=queries,decisions=decisions,before_seconds=before,after_seconds=after,
        last_trace_counters=stats,memory_counters=memory_stats,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),a.module,Path(reuse.__file__))},
        scope='Complete path values, arbitrary endpoint order, ties, gaps, both directions, dtype preservation, and source-core decisions; full recording evidence still required')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}))


if __name__=='__main__':main()
