"""Verify bounded integer enumeration and measure a complete label movie."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('source','labels','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('paired','baseline','candidate'),default='paired')
    p.add_argument('--reference',type=Path)
    p.add_argument('--per-frame',action='store_true')
    p.add_argument('--scratch',type=Path)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-integer-unique-check'
    import numpy as np
    import psutil
    import tifffile
    from benchmark_working_storage import load
    storage=load(a.source.resolve())
    rng=np.random.default_rng(927)
    arrays=[]
    for dtype in ('int8','uint8','int16','uint16','int32','uint32','int64','uint64','>i2','>u2'):
        values=rng.integers(0,80,(17,19)).astype(dtype)
        if np.dtype(dtype).kind=='i':values-=np.array(40,dtype=dtype)
        arrays.extend([values,values.T,values[:0]])
    arrays.extend([np.array([2**64-4,2**64-1,2**64-4],dtype=np.uint64),
                   np.array([-2**63,-2**63+4,-2**63],dtype=np.int64),
                   np.array([-2**63,2**63-1],dtype=np.int64),
                   np.zeros(1024**2+17,dtype=np.uint16)])
    def equal(value):
        for counts in (False,True):
            expected=np.unique(value,return_counts=counts)
            actual=storage.integer_unique(value,np.unique,return_counts=counts)
            old=expected if counts else (expected,)
            new=actual if counts else (actual,)
            for x,y in zip(old,new):
                assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
    for value in arrays:equal(value)
    labels=tifffile.imread(a.labels)
    if a.per_frame:
        measurements=[]
        for counts in (False,True):
            before=after=0.;digest=hashlib.sha256()
            for frame in labels:
                tick=time.perf_counter();expected=np.unique(frame,return_counts=counts);before+=time.perf_counter()-tick
                tick=time.perf_counter();actual=storage.integer_unique(frame,np.unique,return_counts=counts);after+=time.perf_counter()-tick
                old=expected if counts else (expected,);new=actual if counts else (actual,)
                for x,y in zip(old,new):
                    assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                    digest.update(y.tobytes())
            measurements.append(dict(return_counts=counts,before_seconds=before,after_seconds=after,sha256=digest.hexdigest()))
        report=dict(status='pass',cases=len(arrays),frames=len(labels),measurements=measurements,
            source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
            scope='Independent frame integer enumeration; paired timings, exact values/counts and edge cases')
        if a.mode=='candidate':
            original=np.unique
            with tempfile.TemporaryDirectory(prefix='frame-unique-',dir=a.scratch or a.output.parent) as scratch:
                pool=storage.Pool(scratch)
                with storage.install(pool):
                    for frame in labels:
                        for counts in (False,True):
                            expected=original(frame,return_counts=counts)
                            actual=np.unique(frame,return_counts=counts)
                            old=expected if counts else (expected,);new=actual if counts else (actual,)
                            for x,y in zip(old,new):assert x.dtype==y.dtype and x.tobytes()==y.tobytes()
            assert np.unique is original
            report['installed_dispatch']='all frames exact and restored'
        a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    measurements=[];hashes={}
    for counts in (False,True):
        before=after=None
        if a.mode!='candidate':
            tick=time.perf_counter();expected=np.unique(labels,return_counts=counts);before=time.perf_counter()-tick
        if a.mode=='paired':
            tick=time.perf_counter();actual=storage.integer_unique(labels,np.unique,return_counts=counts);after=time.perf_counter()-tick
            old=expected if counts else (expected,);new=actual if counts else (actual,)
            for x,y in zip(old,new):assert x.dtype==y.dtype and x.tobytes()==y.tobytes()
        elif a.mode=='candidate':
            original=np.unique
            with tempfile.TemporaryDirectory(prefix='unique-check-',dir=a.scratch or a.output.parent) as scratch:
                pool=storage.Pool(scratch)
                with storage.install(pool):
                    tick=time.perf_counter();actual=np.unique(labels,return_counts=counts);after=time.perf_counter()-tick
            assert np.unique is original
        else:actual=expected
        values=actual if counts else (actual,)
        hashes[str(counts)]=[dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for v in values]
        measurements.append(dict(return_counts=counts,before_seconds=before,after_seconds=after))
    if a.reference:assert hashes==json.loads(a.reference.read_text())['hashes']
    report=dict(status='pass',cases=len(arrays),shape=list(labels.shape),measurements=measurements,
        peak_resident_bytes=psutil.Process().memory_info().peak_wset,
        peak_private_bytes=psutil.Process().memory_info().peak_pagefile,mode=a.mode,hashes=hashes,
        source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        scope='Exact unique values/counts, including signed/unsigned extremes and endian types; complete movie enumeration, not complete stage')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
