"""Compare every detector event and mask with the frozen ownership detector."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',required=True)
    a=p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0]=[str(a.adapter.resolve()),str((a.engine/'code').resolve())]
    import accepted_history as history
    import quarantine_overlap_batch as reuse
    module=history._issue_module(history.I008,'m7_global_quarantine')
    cfg=history._params('issue008','global_quarantine')['quarantine']
    def equal(first,second):
        assert len(first)==len(second),(len(first),len(second))
        for old,new in zip(first,second):
            assert old.keys()==new.keys()
            for key,value in old.items():
                other=new[key]
                if isinstance(value,np.ndarray):assert value.dtype==other.dtype and np.array_equal(value,other),key
                else:assert type(value)==type(other) and value==other,(key,value,other)
    rng=np.random.default_rng(1987);cases=0;events=0;before=after=0.
    for trial in range(50):
        labels=np.zeros((8,35,39),dtype=np.int32)
        for t in range(8):
            for i in range(7):
                y,x=rng.integers(0,28),rng.integers(0,32)
                labels[t,y:y+7,x:x+7]=rng.choice([-3,1,2,7,42])
        if trial%2==0:labels[4:]=labels[3];labels[4:][labels[4:]==1]=8
        params=dict(cfg,large_area_px=1,minimum_overlap_px=1,minimum_iou=0,
            maximum_centroid_shift_px=100,minimum_owner_history_pixels=0,
            minimum_history_margin_pixels=-100000,minimum_prior_identity_frames=0,
            maximum_replacement_history_pixels=100000,minimum_destination_persistence_frames=0,
            maximum_old_identity_current_area_px=100000)
        tick=time.perf_counter();old=module.detect_events(labels,params,trial%3);before+=time.perf_counter()-tick
        with reuse.install(module) as counters:
            assert counters['supported']
            tick=time.perf_counter();new=module.detect_events(labels,params,trial%3);after+=time.perf_counter()-tick
        equal(old,new);cases+=1;events+=len(old)
    path=a.engine/'m22_accepted_history'/a.run/'mid/accepted_history/03_issue007.tif'
    labels=tifffile.imread(path)
    tick=time.perf_counter();old=module.detect_events(labels,cfg,0);long_before=time.perf_counter()-tick
    with reuse.install(module) as counters:
        tick=time.perf_counter();new=module.detect_events(labels,cfg,0);long_after=time.perf_counter()-tick
    equal(old,new)
    report=dict(status='pass',edge_cases=cases,edge_events=events,edge_before_seconds=before,
        edge_after_seconds=after,frames=len(labels),events=len(old),before_seconds=long_before,
        after_seconds=long_after,counters=counters,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),Path(module.__file__),Path(reuse.__file__),path]},
        scope='Complete detector events and masks; application and complete history group excluded')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sources'}))


if __name__=='__main__':main()
