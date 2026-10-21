"""Compare preparation storage overhead using frozen sources and identical inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    platform.platform = lambda *a, **k: 'Windows-preparation-comparison'
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import psutil
    import tifffile
    from pymicroglia.tracking import prepare_stream, motion_preparation
    assert Path(prepare_stream.__file__).resolve().is_relative_to(args.source.resolve())
    labels = tifffile.imread(args.labels)
    raw = tifffile.imread(args.raw)[-len(labels):]
    input_hash = hashlib.sha256(memoryview(raw).cast('B')).hexdigest()
    if not args.native:
        raw = raw.astype(np.float32)
    del labels
    flush = np.memmap.flush
    calls = [0]
    def counted(*a, **k):
        calls[0] += 1
        return flush(*a, **k)
    np.memmap.flush = counted
    process = psutil.Process()
    before = process.cpu_times()
    start = time.perf_counter()
    files = {}
    with tempfile.TemporaryDirectory(prefix='prepare-trial-', dir=args.scratch) as scratch:
        result = prepare_stream.build('sample', raw, None if args.native else args.labels,
            Path(scratch)/'output', frame_interval_min=30, dataset='storage comparison',
            native_signal=args.native)
        analysis_seconds = time.perf_counter()-start
        for path in sorted((Path(scratch)/'output').glob('*.tif')):
            h = hashlib.sha256()
            with tifffile.TiffFile(path) as opened:
                shape = list(opened.series[0].shape)
                dtype = str(opened.series[0].dtype)
                for page in opened.pages:
                    h.update(memoryview(page.asarray()).cast('B'))
            files[path.name] = dict(shape=shape, dtype=dtype, pixel_sha256=h.hexdigest())
        config = json.loads(Path(result['config']).read_text())
        files['motion_config.json'] = config
        # Elapsed analysis excludes verification; total includes verification and cleanup.
    elapsed = time.perf_counter()-start
    memory = process.memory_info()
    cpu = process.cpu_times()
    report = dict(status='complete', native=args.native, seconds=analysis_seconds,
        including_verification_seconds=elapsed,
        cpu_seconds=cpu.user+cpu.system-before.user-before.system,
        peak_resident_bytes=getattr(memory,'peak_wset',memory.rss), flush_calls=calls[0],
        input_sha256=input_hash, shape=list(raw.shape), results=files,
        source_sha256={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (prepare_stream,motion_preparation)})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__ == '__main__':
    main()
