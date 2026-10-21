"""Compare every frame's territory geometry with the frozen implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'labels', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    platform.platform = lambda *a, **k: 'Windows-territory-differential'
    sys.path[:0] = [str(args.engine.resolve()/'code'), str(args.adapter.resolve())]
    import psutil
    import tifffile
    import territory_reconciliation as module
    import territory_observation_reuse as reuse
    from check_territory_observation_reuse import same_rows
    labels = tifffile.imread(args.labels)
    original = module._observations
    before = after = cached = 0.0
    observations = 0
    with reuse.install(module) as counters:
        assert counters['supported']
        for frame in labels:
            tick = time.perf_counter()
            expected = original(frame)
            before += time.perf_counter()-tick
            tick = time.perf_counter()
            actual = module._observations(frame)
            after += time.perf_counter()-tick
            same_rows(expected, actual)
            tick = time.perf_counter()
            repeated = module._observations(frame)
            cached += time.perf_counter()-tick
            same_rows(expected, repeated)
            observations += len(expected)
    report = dict(status='pass', frames=len(labels), observations=observations,
                  original_seconds=before, candidate_seconds=after, cached_seconds=cached,
                  counters=counters, peak_resident_bytes=psutil.Process().memory_info().peak_wset,
                  inputs={str(args.labels.resolve()):hashlib.sha256(args.labels.read_bytes()).hexdigest()},
                  sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(module.__file__), Path(reuse.__file__), Path(__file__)]},
                  scope='Every frame and all observation fields; isolated geometry, not full-stage timing')
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','sources')}))


if __name__ == '__main__':
    main()
