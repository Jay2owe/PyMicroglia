"""Differential checks for immutable terminal-tail measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-feature-reuse-checks'
    sys.path[:0]=[str((a.engine/'code').resolve()),str(a.adapter.resolve())]
    import numpy as np
    import pandas as pd
    import complete_flip_recovery as module
    import terminal_feature_reuse as adapter
    cases=[]
    for seed in range(12):
        rng=np.random.default_rng(seed)
        labels=np.zeros((15,35,47),dtype=np.uint16)
        points=[]
        for frame in range(15):
            for owner in range(1,6):
                y,x=rng.integers(0,30),rng.integers(0,42)
                labels[frame,y:y+5,x:x+5]=owner
                if seed%3==0:labels[frame,34,owner]=owner
                points.append(dict(frame=frame,candidate_owner=owner,track_id=owner,
                    strong=(frame+owner)%3!=0,x=x,y=y,radius_px=2.0))
        # Track handoffs cause every source-tail measurement to be exercised.
        visible=pd.DataFrame(points)
        visible.loc[visible.frame>8,'track_id']=(visible.loc[visible.frame>8,'track_id']%5)+1
        params={}
        expected=module._terminal_successor_proposals(labels,visible,params,1.5)
        with adapter.install(module) as counters:
            actual=module._terminal_successor_proposals(labels,visible,params,1.5)
            assert actual==expected
            assert counters['source_supported'] and counters['feature_misses']>0
        labels[10]=0
        expected=module._terminal_successor_proposals(labels,visible,params,1.5)
        with adapter.install(module):
            assert module._terminal_successor_proposals(labels,visible,params,1.5)==expected
        cases.append(dict(seed=seed,proposal_count=len(actual),counters=counters))
    original=module._terminal_successor_proposals
    def unknown(*args,**kwargs):return ['future scientific version']
    module._terminal_successor_proposals=unknown
    with adapter.install(module) as counters:
        assert not counters['source_supported']
        assert module._terminal_successor_proposals()==['future scientific version']
    module._terminal_successor_proposals=original
    result=dict(status='pass',cases=cases,edited_inputs_checked=True,unknown_source_fallback=True,
        source_sha256=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=len(cases),report=str(a.output))))


if __name__=='__main__':main()
