"""Real full-field observation storage, plus repeated-frame growth coverage."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc

platform.platform=lambda *a,**k:'Windows-observation-memory'
import numpy as np
import psutil
import tifffile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--mode',choices=['dense','shared'],required=True)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.code.resolve()))
    import tracking
    from pymicroglia.tracking.frozen_io import observation_reader
    labels=tifffile.imread(args.labels);raw=tifffile.imread(args.raw)
    count=sum(np.count_nonzero(np.unique(frame)) for frame in labels)*args.repeat
    expected=count*labels.shape[1]*labels.shape[2]
    if psutil.virtual_memory().available<expected*2+1024**3:
        raise MemoryError('Insufficient headroom for dense observation comparator')
    reader=tracking.frame_observations if args.mode=='dense' else observation_reader(tracking.frame_observations)
    tracemalloc.start();start=time.perf_counter()
    rows=[reader(labels[t],raw[t]) for _ in range(args.repeat) for t in range(len(labels))]
    retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    seconds=time.perf_counter()-start;process=psutil.Process();mem=process.memory_info()
    digest=hashlib.sha256()
    for frame in rows:
        for identity,row in sorted(frame.items()):
            digest.update(str(identity).encode())
            for key,value in sorted(row.items()):
                digest.update(key.encode())
                if key=='mask' or isinstance(value,np.ndarray):digest.update(np.asarray(value).tobytes())
                else:digest.update(repr(value).encode())
    report=dict(status='pass',mode=args.mode,repeat=args.repeat,shape=list(labels.shape),
        observation_count=int(count),seconds=seconds,peak_allocation_bytes=peak,
        retained_allocation_bytes=retained,peak_working_set_bytes=getattr(mem,'peak_wset',mem.rss),
        private_bytes=getattr(mem,'private',mem.vms),observations_sha256=digest.hexdigest(),
        scope='Observation storage only; repeat>1 repeats recorded frames for allocation growth, not a tracking accuracy experiment')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
