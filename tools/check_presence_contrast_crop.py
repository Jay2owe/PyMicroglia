"""Compare boxed presence contrast with the full-frame original, exactly.

Real cell masks and unions from an accepted-history stack with the registered
raw movie, edge-touching and empty masks, ring widths 1-6 and non-integer
settings; both returned floats must have the same bits.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--cases', type=int, default=6000)
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import presence as module
    import presence_contrast_crop as crop
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '10_blob_residency.tif')
    raw = tifffile.imread(folder / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    rng = np.random.default_rng(3)
    cases = []
    while len(cases) < a.cases:
        t = int(rng.integers(len(labels)))
        ids = np.unique(labels[t]); ids = ids[ids > 0]
        choice = int(rng.integers(10))
        if choice < 6:
            mask = labels[t] == int(rng.choice(ids))
        elif choice < 8:
            mask = np.isin(labels[t], rng.choice(ids, 3))
        elif choice == 8:
            mask = np.zeros(labels.shape[1:], bool)
            y, x = int(rng.choice([0, labels.shape[1] - 3])), int(rng.integers(0, labels.shape[2] - 3))
            mask[y:y + 3, x:x + 3] = True
        else:
            mask = np.zeros(labels.shape[1:], bool)
        reach = int(rng.integers(1, 7)) if rng.random() < 0.95 else float(rng.choice([2.0, 0.0]))
        cases.append((mask, raw[t], {'contrast_ring_px': reach}))
    original = module.local_contrast
    tick = time.perf_counter(); expected = [original(*c) for c in cases]; before = time.perf_counter() - tick
    with crop.install(module) as stats:
        assert stats['supported']
        tick = time.perf_counter(); actual = [module.local_contrast(*c) for c in cases]; after = time.perf_counter() - tick
    assert module.local_contrast is original
    for e, f in zip(expected, actual):
        assert np.array(e, np.float64).tobytes() == np.array(f, np.float64).tobytes(), (e, f)
    report = dict(status='pass', cases=len(cases), before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
