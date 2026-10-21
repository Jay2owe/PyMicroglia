"""Compare the frozen optical physical_paths with the one-pass adapter, exactly.

The audited optical_basin_soma_paths.py is imported from one frozen revision.
physical_paths runs on real windows (the accepted labels as proposals, those
labels where an earlier history stage holds foreground as the measured movie,
and the labels measured against themselves, as separated-body ownership calls
it), given as plain arrays and read-only memmaps, and on random synthetic
movies including empty, one-pixel and edge-touching frames. Nodes (dtype,
shape, values), records (values and types), groups (order) and blocked must
equal the original's.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time


def same(a, b):
    import numpy as np
    na, ra, ga, ba = a
    nb, rb, gb, bb = b
    assert na.dtype == nb.dtype and na.shape == nb.shape and np.array_equal(na, nb), 'nodes'
    assert ra == rb, 'records'
    assert [{k: type(v) for k, v in r.items()} for r in ra[1:]] == [{k: type(v) for k, v in r.items()} for r in rb[1:]], 'types'
    assert list(ga.items()) == list(gb.items()), 'groups'
    assert ba == bb and all(type(x) is int for x in bb), 'blocked'


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'version', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--window', type=int, default=60)
    p.add_argument('--starts', default='0,200,500,900')
    p.add_argument('--synthetic', type=int, default=300)
    a = p.parse_args()
    import numpy as np
    import tifffile
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels_all = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    earlier_all = tifffile.imread(folder / '08_issue012.tif')
    sys.path[:0] = [str(a.version.resolve()), str(a.adapter.resolve())]
    import optical_basin_soma_paths as module
    import optical_physical_paths as adapter
    assert adapter._digest(module) in adapter.VERSIONS
    original = module.physical_paths
    scratch = Path(tempfile.mkdtemp(prefix='optical-paths-'))
    maps = []

    def mapped(name, array):
        path = scratch / (name + '.dat')
        array.tofile(path)
        result = np.memmap(path, dtype=array.dtype, mode='r', shape=array.shape)
        maps.append(result)
        return result
    cases = []
    for start in sorted({min(int(s), max(len(labels_all) - a.window, 0)) for s in a.starts.split(',')}):
        window = slice(start, min(start + a.window, len(labels_all)))
        labels = np.ascontiguousarray(labels_all[window])
        measured = np.where(earlier_all[window] > 0, labels, 0).astype(labels.dtype)
        cases.append(('real_%d' % start, labels, measured))
        cases.append(('self_%d' % start, labels, labels))
        cases.append(('memmap_%d' % start, mapped('l%d' % start, labels), mapped('m%d' % start, measured)))
    rng = np.random.default_rng(1)
    for trial in range(a.synthetic):
        T, H, W = int(rng.integers(1, 6)), int(rng.integers(1, 40)), int(rng.integers(1, 40))
        dtype = (np.int32, np.int64, np.uint16, np.uint8)[trial % 4]
        frames = np.zeros((T, H, W), dtype)
        for t in range(T):
            for _ in range(int(rng.poisson(rng.uniform(0, 12)))):
                y, x, r = int(rng.integers(-3, H + 3)), int(rng.integers(-3, W + 3)), int(rng.integers(1, 6))
                yy, xx = np.ogrid[:H, :W]
                m = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
                if rng.random() < .5:
                    m &= rng.random((H, W)) < .8
                frames[t][m] = rng.integers(1, int(rng.integers(1, 30)) + 1)
        if trial % 9 == 0:
            frames[:] = 0
        observed = ((rng.random((T, H, W)) < .3) * rng.integers(0, 5, (T, H, W))).astype(dtype)
        cases.append(('synthetic_%d' % trial, frames, observed))
    report = dict(status='pass', cases={}, seconds=dict(original=0.0, adapted=0.0))
    for name, proposals, observed in cases:
        tick = time.perf_counter()
        expected = original(proposals, observed)
        middle = time.perf_counter()
        stats = {}
        with adapter.install(stats):
            assert module.physical_paths.__wrapped__ is original
            actual = module.physical_paths(proposals, observed)
        end = time.perf_counter()
        assert module.physical_paths is original
        same(expected, actual)
        assert stats['fast'] == 1 and stats['fallback'] == 0, (name, stats)
        report['seconds']['original'] += middle - tick
        report['seconds']['adapted'] += end - middle
        if not name.startswith('synthetic'):
            report['cases'][name] = dict(frames=len(proposals), records=len(expected[1]) - 1,
                                         groups=len(expected[2]), blocked=len(expected[3]),
                                         seconds=dict(original=middle - tick, adapted=end - middle))
    report['synthetic'] = a.synthetic
    for array in maps:
        array._mmap.close()
    maps.clear()
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'])))


if __name__ == '__main__':
    main()
