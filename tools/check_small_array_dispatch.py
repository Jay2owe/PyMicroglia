"""Compare mapped-array arithmetic before/after avoiding tiny dtype probes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
from benchmark_working_storage import load


def trial(source,scratch):
    import numpy as np
    module=load(source)
    receipts=[]
    with tempfile.TemporaryDirectory(prefix='small-array-',dir=scratch) as folder:
        pool=module.Pool(folder,minimum_bytes=1024**2)
        with module.install(pool):
            for dtype in (np.uint16,np.int64,np.float32,np.float64,np.complex128):
                raw=np.arange(96,dtype=float).astype(dtype).reshape(4,24)
                mapped=pool.array(raw,force=True)
                for operand in (mapped,mapped[:,::2],mapped[:0],mapped.T):
                    plain=pool.original['asarray'](operand)
                    for function,extra in ((np.add,2),(np.subtract,1),(np.multiply,.125),(np.equal,3)):
                        for options in ({},{'order':'F'}):
                            actual=function(operand,extra,**options);expected=function(plain,extra,**options)
                            assert actual.dtype==expected.dtype and actual.shape==expected.shape
                            assert actual.tobytes()==expected.tobytes()
                            receipts.append((actual.dtype.str,actual.shape,actual.flags.c_contiguous,actual.flags.f_contiguous,
                                hashlib.sha256(actual.tobytes()).hexdigest()))
                # Explicit dtype and out must retain the original path.
                target=np.empty_like(raw,dtype=np.float64 if dtype!=np.complex128 else np.complex128)
                np.add(mapped,1,out=target);assert target.tobytes()==(raw+1).astype(target.dtype).tobytes()
                converted=np.add(mapped,1,dtype=target.dtype)
                assert converted.dtype==target.dtype and converted.tobytes()==np.add(raw,1,dtype=target.dtype).tobytes()
                del mapped,operand,plain,actual
            raw=np.arange(471,dtype=np.float64);mapped=pool.array(raw,force=True)
            tick=time.perf_counter()
            for _ in range(12000):
                value=np.add(mapped,1.25)
                value=np.less(mapped,200)
            elapsed=time.perf_counter()-tick
            del value,mapped
    return dict(seconds=elapsed,receipts=receipts)


def main():
    p=argparse.ArgumentParser()
    for name in ('before','after','scratch','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-small-array-dispatch'
    old=trial(a.before.resolve(),a.scratch);new=trial(a.after.resolve(),a.scratch)
    assert old['receipts']==new['receipts']
    report=dict(status='pass',cases=len(old['receipts']),before_seconds=old['seconds'],after_seconds=new['seconds'],
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),a.before,a.after]},
        scope='Exact bytes, shape, dtype and layout of small arithmetic outputs; explicit output and dtype paths preserved; kernel timing only')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
