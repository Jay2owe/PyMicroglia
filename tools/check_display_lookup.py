"""Compare the lookup outline renderer with the buffered renderer it replaces.

Both renderers come from one composed snapshot: frozen_io.overlay_renderer is
the unchanged buffered renderer and display_lookup.lookup_renderer the new
one. Every case compares the complete RGB movies byte for byte.
"""
import argparse, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def same(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return all(np.array_equal(a[t], b[t]) for t in range(len(a)))


def synthetic(rng, dtype, frames, size):
    labels = np.zeros((frames, size, size), np.uint16)
    for t in range(frames):
        for identity in range(1, int(rng.integers(1, 30))):
            r, c = rng.integers(0, size, 2)
            radius = int(rng.integers(1, 6))
            labels[t, max(0, r - radius):r + radius, max(0, c - radius):c + radius] = identity
    if dtype == np.float32:
        raw = (rng.random((frames, size, size)) * 900 - 50).astype(np.float32)
    else:
        info = np.iinfo(dtype)
        raw = rng.integers(info.min, int(info.max) + 1, (frames, size, size), dtype=np.int64).astype(dtype)
    if rng.random() < .3:
        raw[raw < np.percentile(raw, 40)] = 0
    return raw, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=300)
    parser.add_argument('--no-real', action='store_true')
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.snapshot.resolve())]
    import common
    from pymicroglia.tracking import frozen_io, display_lookup
    scratch = Path(tempfile.mkdtemp(prefix='display-lookup-'))
    old = frozen_io.overlay_renderer(common, scratch)
    rows, failures = [], []
    rng = np.random.default_rng(20261020)
    old_seconds = new_seconds = 0.0
    for case in range(a.cases):
        dtype = [np.uint8, np.uint16, np.uint16, np.int16, np.float32][case % 5]
        frames = int(rng.choice([1, 2, 3, 5, 9, 17, 40]))
        raw, labels = synthetic(rng, dtype, frames, int(rng.choice([7, 16, 33, 64])))
        thick = int(rng.choice([1, 2, 2, 3, 4]))
        extra = {}
        if rng.random() < .5:
            extra['inferred'] = rng.random(labels.shape) < .4
        if rng.random() < .5:
            extra['unresolved'] = rng.random(labels.shape) < .2
        block = int(rng.choice([1, 2000, 40000, display_lookup.BLOCK_BYTES]))
        new = display_lookup.lookup_renderer(common, scratch, block_bytes=block)
        t0 = time.perf_counter(); expected = old(raw, labels, thick=thick, **extra)
        t1 = time.perf_counter(); actual = new(raw, labels, thick=thick, **extra)
        t2 = time.perf_counter()
        old_seconds += t1 - t0; new_seconds += t2 - t1
        if not same(expected, actual):
            failures.append(dict(case=case, dtype=str(np.dtype(dtype)), frames=frames, thick=thick, block=block))
    rows.append(dict(kind='synthetic', cases=a.cases, old_seconds=old_seconds, new_seconds=new_seconds))
    # Every 16-bit value through the table, in contiguous and strided frames.
    values = np.arange(65536, dtype=np.uint16).reshape(1, 256, 256)
    for raw in (values, np.ascontiguousarray(values[:, ::-1, ::-1]), np.repeat(values, 3, axis=0)[:, :, ::2]):
        labels = np.zeros(raw.shape, np.uint16)
        expected = old(raw, labels); actual = display_lookup.lookup_renderer(common, scratch)(raw, labels)
        if not same(expected, actual):
            failures.append(dict(case='all_uint16_values', shape=list(raw.shape)))
    rows.append(dict(kind='all_uint16_values', cases=3))
    if not a.no_real:
        raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
        new = display_lookup.lookup_renderer(common, scratch)
        for name in ('01_issue005.tif', '11_issue017_rebase.tif'):
            labels = tifffile.imread(HISTORY / name)
            unresolved = np.zeros(labels.shape, bool); unresolved[:, 100:140, 200:260] = True
            for extra in ({}, dict(inferred=labels % 3 == 0, unresolved=unresolved)):
                t0 = time.perf_counter(); expected = old(raw, labels, **extra)
                t1 = time.perf_counter(); actual = new(raw, labels, **extra)
                t2 = time.perf_counter()
                ok = same(expected, actual)
                rows.append(dict(kind='real', labels=name, masks=bool(extra), equal=ok,
                                 old_seconds=t1 - t0, new_seconds=t2 - t1))
                if not ok:
                    failures.append(dict(case='real', labels=name, masks=bool(extra)))
                del expected, actual
    report = dict(status='pass' if not failures else 'fail', rows=rows, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
