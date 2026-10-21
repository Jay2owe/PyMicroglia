"""Compare indexed nomination with the original row filters, including ties."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.adapter.resolve()))
    import numpy as np
    import pandas as pd
    from reconnection_candidate_index import ResidentIndex
    rng=np.random.default_rng(237)
    groups={}
    for track in [7,3,2,11,5,21,9,8,29,15]:
        frame=rng.integers(-3,60,90)
        if track%2:frame=frame.astype(float)+.2
        groups[track]=pd.DataFrame(dict(frame=frame,candidate_owner=rng.integers(0,6,len(frame))))
    index=ResidentIndex(groups)
    compared=0
    for _ in range(500):
        end=int(rng.integers(-4,55));start=end+int(rng.integers(2,20))
        before,after=rng.choice(list(groups),2,replace=False)
        owner=int(rng.integers(0,7))
        gap=list(range(end+1,start))
        expected=[]
        for track,group in groups.items():
            if track in {before,after}:continue
            middle=group[group.frame.astype(int).isin(gap)]
            if len(middle)==len(gap) and middle.candidate_owner.astype(int).eq(owner).any():
                expected.append((track,group))
        actual=index.select(before,after,end,start,owner)
        assert [v[0] for v in expected]==[v[0] for v in actual]
        assert all(left[1] is right[1] for left,right in zip(expected,actual))
        compared+=1
    a.output.write_text(json.dumps(dict(status='pass',queries=compared,
        scope='Exact membership, original group order and same DataFrame objects; duplicate, unsorted and truncated fractional frames'),indent=2)+'\n')
    print(json.dumps(dict(status='pass',queries=compared)))


if __name__=='__main__':main()
