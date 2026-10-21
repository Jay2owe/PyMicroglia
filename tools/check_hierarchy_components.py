"""Verify every component's identity, local number, pixels and centroid."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
from dataclasses import make_dataclass


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','labels','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-hierarchy-components-check'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    import accepted_history as history
    import hierarchy_component_batch as reuse
    component_type=make_dataclass('Component',['frame','identity','number','pixels','area','x','y'])
    original=history._cropped_frame_components
    labels=tifffile.imread(a.labels)
    rng=np.random.default_rng(276)
    cases=[rng.integers(0,10,(27,31),dtype=np.uint16) for _ in range(8)]
    cases.extend([np.zeros((1,1),np.uint16),np.ones((1,9),np.uint16),labels[0].T,labels[0].astype(np.int32)])
    before=after=0.0;compared=0
    with reuse.install(history) as stats:
        assert stats['supported']
        for frame in [*cases,*labels]:
            for minimum in (1,8):
                tick=time.perf_counter();expected=original(frame,3,minimum,component_type)
                before+=time.perf_counter()-tick
                tick=time.perf_counter();actual=history._cropped_frame_components(frame,3,minimum,component_type)
                after+=time.perf_counter()-tick
                assert len(expected)==len(actual)
                for old,new in zip(expected,actual):
                    assert (old.frame,old.identity,old.number,old.area)==(new.frame,new.identity,new.number,new.area)
                    assert old.pixels.tobytes()==new.pixels.tobytes()
                    assert np.float64(old.x).tobytes()==np.float64(new.x).tobytes()
                    assert np.float64(old.y).tobytes()==np.float64(new.y).tobytes()
                    compared+=1
    assert history._cropped_frame_components is original
    report=dict(status='pass',frames=len(labels),extra_cases=len(cases),components=compared,
                before_seconds=before,after_seconds=after,counters=stats,
                scope='Both component size settings on every full recording frame, plus sparse shapes/transposes; exact local numbering, pixels and floating centroids')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
