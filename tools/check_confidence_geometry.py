"""Differential confidence tests and an optional full-recording comparison."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--labels',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--stem')
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-confidence-geometry-check'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import pandas as pd
    import lineage_confidence as module
    import confidence_geometry_reuse as reuse
    names=('identity_confidence_table','multilayer_identity_confidence_table')
    rng=np.random.default_rng(17)
    cases=[]
    def check(labels,raw,params,name):
        tick=time.perf_counter()
        expected=[getattr(module,n)(labels,raw,params) for n in names]
        before=time.perf_counter()-tick
        tick=time.perf_counter()
        with reuse.install(module) as stats:
            assert stats['supported']
            actual=[getattr(module,n)(labels,raw,params) for n in names]
        after=time.perf_counter()-tick
        for old,new in zip(expected,actual):
            pd.testing.assert_frame_equal(old,new,check_exact=True)
            assert old.to_csv(index=False)==new.to_csv(index=False)
        cases.append(dict(name=name,frames=len(labels),before_seconds=before,after_seconds=after))
    for i in range(9):
        labels=rng.integers(0,8,(6,23,27),dtype=np.uint16)
        raw=rng.normal(5,2,labels.shape).astype(np.float32)
        if i%2:labels=labels.transpose(0,2,1);raw=raw.transpose(0,2,1)
        if i==2:labels[labels==7]=65000
        if i==3:labels[:]=0
        if i==4:raw[0,0,0]=np.nan
        if i==5:labels=labels.astype(np.int32);labels[labels==7]=-1
        check(labels,raw,dict(confidence_background_ring_px=i%3-1),str(i))
    # Disk-backed labels and signal, as the bounded runner supplies them.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='confidence-mapped-',ignore_cleanup_errors=True) as folder:
        for i in range(6):
            labels=rng.integers(0,8,(5,41,37),dtype=np.uint16)
            raw=rng.normal(5,2,labels.shape).astype(np.float32)
            if i==4:labels[:,:,20:]=0
            mapped=[]
            for name,value in (('labels',labels),('raw',raw)):
                m=np.memmap(Path(folder)/f'{name}_{i}.dat',mode='w+',dtype=value.dtype,shape=value.shape)
                m[...]=value;mapped.append(m)
            check(mapped[0],mapped[1],dict(confidence_background_ring_px=i%3),f'mapped_{i}')
            del mapped,m
    # Sparse blob movies, where most smaller objects are far apart, at several edge-gap thresholds.
    for i in range(8):
        labels=np.zeros((4,60,70),np.uint16)
        for t in range(4):
            for name in range(1,25):
                y,x=int(rng.integers(0,56)),int(rng.integers(0,66))
                labels[t,y:y+int(rng.integers(1,5)),x:x+int(rng.integers(1,5))]=name if rng.random()<.9 else 0
        raw=rng.normal(5,2,labels.shape).astype(np.float32)
        check(labels,raw,dict(near_smaller_edge_gap_px=(0.,1.,3.,8.,20.,-2.,float('nan'),float('inf'))[i]),f'blobs_{i}')
    # The pruned edge-gap test against the original list on random rows, including
    # non-finite, huge and empty inputs.
    near_cases=near_equal=0
    def original_near(row,smaller,near):
        gaps=[max(float(np.linalg.norm(row["position"]-other["position"]))-float(row["radius"])-float(other["radius"]),0.0) for other in smaller]
        return bool(gaps and min(gaps)<=near)
    for trial in range(3000):
        n=int(rng.integers(0,12))
        spread=float((5,50,500,1e15)[trial%4])
        smaller=[dict(position=rng.uniform(0,spread,2),radius=float(rng.uniform(0,6))) for _ in range(n)]
        row=dict(position=rng.uniform(0,spread,2),radius=float(rng.uniform(0,6)))
        if trial%17==0 and n:smaller[0]['position'][1]=np.nan
        if trial%19==0:row['radius']=float('inf')
        if trial%23==0 and n:smaller[-1]['radius']=float('nan')
        near=float((0.,2.,8.,float('nan'),-1.,30.)[trial%6])
        near_cases+=1
        near_equal+=original_near(row,smaller,near)==reuse.near_smaller(row,smaller,near)
    assert near_cases==near_equal,(near_cases,near_equal)
    cases.append(dict(name='near_smaller_rows',cases=near_cases,equal=near_equal))
    if a.labels:
        import common,pipeline,tifffile
        labels=tifffile.imread(a.labels)
        cfg=common.Config.load(a.config)
        raw=tifffile.imread(pipeline._pinned_paths(cfg,a.stem)['registered_raw'])
        raw=raw[cfg.source_frame_offset:cfg.source_frame_offset+len(labels)]
        check(labels,raw,{},'complete_recording')
    a.output.write_text(json.dumps(dict(status='pass',cases=cases,
        scope='Both exact confidence tables; full-field frames, no recording-specific selection'),indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=cases)))


if __name__=='__main__':main()
