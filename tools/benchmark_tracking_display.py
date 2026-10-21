"""Measure exact display contrast or full-field rendering in a frozen adapter."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import platform
import sys
import tempfile
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--operation',choices=['contrast','review'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--code',type=Path)
    p.add_argument('--raw',type=Path)
    p.add_argument('--labels',type=Path)
    p.add_argument('--frames',type=int,default=99)
    args=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-display-comparison'
    sys.path.insert(0,str(args.source.resolve()))
    import numpy as np
    import psutil
    from pymicroglia.tracking import display_scale,frozen_io
    if not Path(display_scale.__file__).resolve().is_relative_to(args.source.resolve()):
        raise RuntimeError('Wrong display implementation imported')
    if args.operation=='contrast':
        raw=np.random.default_rng(19).standard_normal((args.frames,467,520),dtype=np.float32)
        raw*=np.float32(70);raw+=np.float32(150)
        inputs=dict(synthetic=True,shape=list(raw.shape),dtype=str(raw.dtype),seed=19)
    else:
        import tifffile
        sys.path.insert(0,str(args.code.resolve()))
        common=importlib.import_module('common')
        labels=tifffile.imread(args.labels)
        raw=tifffile.imread(args.raw)[-len(labels):]
        inputs=dict(synthetic=False,raw=str(args.raw.resolve()),labels=str(args.labels.resolve()),shape=list(raw.shape))
    digest=hashlib.sha256(memoryview(raw).cast('B')).hexdigest()
    process=psutil.Process();before=process.cpu_times()
    start=time.perf_counter()
    if args.operation=='contrast':
        result=dict(high=display_scale.positive_display_high(raw))
    else:
        args.scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='display-trial-',dir=args.scratch) as scratch:
            image=frozen_io.overlay_renderer(common,scratch)(raw,labels)
            pixels=hashlib.sha256()
            for frame in image:pixels.update(memoryview(frame).cast('B'))
            result=dict(pixel_sha256=pixels.hexdigest(),shape=list(image.shape))
            del frame, image
    elapsed=time.perf_counter()-start
    memory=process.memory_info();cpu=process.cpu_times()
    report=dict(status='complete',operation=args.operation,seconds=elapsed,
        cpu_seconds=cpu.user+cpu.system-before.user-before.system,
        peak_resident_bytes=getattr(memory,'peak_wset',memory.rss),result=result,
        inputs=inputs,input_array_sha256=digest,source=str(args.source.resolve()),
        source_sha256={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (display_scale,frozen_io)},
        scope='Display operation only, including mapped image cleanup; shared desktop')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
