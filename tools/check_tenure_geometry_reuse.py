"""Check every component and exact backward-link choice on a complete movie."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','labels','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-tenure-geometry-check'
    import numpy as np
    import tifffile
    sys.path[:0]=[str(a.adapter.resolve()),str((a.engine/'code').resolve())]
    import accepted_history as history
    import tenure_geometry_reuse as reuse
    module=history._issue_module(history.I008,'m16_persistent_quarantine')
    helper=sys.modules['m12_component_tenure_audit']
    old_components,old_overlap=helper.frame_components,helper.overlap_pixels
    old_chain=module.physical_chain_backward
    rng=np.random.default_rng(65);components=0;frames=[];before=after=0.
    movie=tifffile.imread(a.labels)
    with reuse.geometry(helper) as geometry:
        assert geometry['supported'] and module.frame_components is helper.frame_components
        for i,frame in enumerate(movie):
            tick=time.perf_counter();old=old_components(frame,i,1);before+=time.perf_counter()-tick
            tick=time.perf_counter();new=helper.frame_components(frame,i,1);after+=time.perf_counter()-tick
            assert len(old)==len(new)
            for first,second in zip(old,new):
                for key,value in vars(first).items():
                    actual=getattr(second,key)
                    if isinstance(value,np.ndarray):assert value.dtype==actual.dtype and np.array_equal(value,actual)
                    else:assert type(value)==type(actual) and value==actual,(key,value,actual)
                components+=1
            frames.append(new)
        flat=[component for frame in frames for component in frame]
        for _ in range(10000):
            first,second=(flat[int(i)] for i in rng.integers(0,len(flat),2))
            assert old_overlap(first,second)==helper.overlap_pixels(first,second)
        queries=[]
        for frame in range(5,len(frames),11):
            for component in sorted(frames[frame],key=lambda item:-item.area)[:2]:
                queries.append((frame,component,dict(owner_match_radius_px=15.)))
        queries=queries+list(reversed(queries))
        tick=time.perf_counter();expected=[old_chain(frames,*query) for query in queries];path_before=time.perf_counter()-tick
        with reuse.paths(module) as paths:
            assert paths['supported']
            tick=time.perf_counter();actual=[module.physical_chain_backward(frames,*query) for query in queries];path_after=time.perf_counter()-tick
            for old,new in zip(expected,actual):assert [id(v) for v in old]==[id(v) for v in new]
            assert paths['reused']>0
    assert helper.frame_components is old_components and helper.overlap_pixels is old_overlap
    assert module.physical_chain_backward is old_chain
    report=dict(status='pass',frames=len(movie),components=components,overlap_pairs=10000,
        path_queries=len(queries),component_before_seconds=before,component_after_seconds=after,
        path_before_seconds=path_before,path_after_seconds=path_after,geometry=geometry,paths=paths,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),Path(reuse.__file__),Path(module.__file__),Path(helper.__file__),a.labels]},
        scope='Every complete-movie component field/pixel; scalar overlaps; ordered exact backward references. Complete ownership stage still required')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}))


if __name__=='__main__':main()
