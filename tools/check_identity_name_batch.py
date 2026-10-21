"""Compare batched retired-name reuse and boxed deferred restoration with the originals, exactly.

Real accepted-history stacks supply tracked, reference and candidate movies;
physical objects are per-frame connected components of the tracked foreground.
Synthetic movies add ties, novel and retired names, high-confidence sets,
several dtypes, empty frames and identity-count failures. Labels, tables and
exceptions must match.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def outcome(function, *args):
    import numpy as np
    try:
        labels, table = function(*args)
    except Exception as error:
        return ('error', type(error).__name__, str(error))
    labels = np.asarray(labels)
    import hashlib
    return ('ok', labels.dtype.str, labels.shape, hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(), list(table.columns),
            [tuple(row) for row in table.itertuples(index=False)], [str(t) for t in table.dtypes])


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--synthetic', type=int, default=400)
    a = p.parse_args()
    import numpy as np
    import tifffile
    from scipy import ndimage as ndi
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import global_soma_ledger as ledger
    import lineage_confidence as confidence
    import identity_name_batch as batch
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    real = {name: tifffile.imread(folder / name) for name in
            ('01_issue005.tif', '04_issue008.tif', '08_issue012.tif')}
    rng = np.random.default_rng(5)
    cases = []
    # 160-frame windows keep three real cases within a few hundred MiB.
    one, two, three = (stack[150:310] for stack in real.values())
    for tracked, reference in ((one, two), (two, three), (three, one)):
        physical = np.stack([ndi.label(frame > 0)[0] for frame in tracked]).astype(np.int32)
        ids = np.unique(reference); ids = ids[ids > 0]
        high = set(int(v) for v in rng.choice(ids, len(ids) // 2, replace=False))
        # Rename a third of the tracked names so novel/retired names exist.
        shifted = tracked.copy()
        moved = rng.choice(np.unique(tracked)[1:], len(ids) // 3, replace=False)
        for value in moved:
            shifted[tracked == value] = int(value) + 5000
        cases.append(('restore', (shifted, physical, reference, high)))
        cases.append(('reuse', (shifted, reference, 1)))
    for number in range(a.synthetic):
        shape = (int(rng.integers(1, 7)), int(rng.integers(4, 30)), int(rng.integers(4, 30)))
        dtype = [np.uint16, np.int32, np.int64, np.uint8][number % 4]
        top = int(rng.integers(1, 12))
        tracked = rng.integers(0, top + 1, shape) * (rng.random(shape) < 0.6)
        reference = rng.integers(0, top + 1, shape) * (rng.random(shape) < 0.6)
        if rng.random() < 0.3:
            tracked = tracked + (tracked > 0) * int(rng.integers(0, 6))
        physical = np.stack([ndi.label(frame > 0)[0] for frame in tracked])
        if rng.random() < 0.2:
            physical = rng.integers(0, 6, shape)
        if rng.random() < 0.1:
            physical[0] = 0
        high = set(int(v) for v in rng.integers(1, top + 3, int(rng.integers(0, 6))))
        tracked, reference = tracked.astype(dtype), reference.astype(dtype)
        cases.append(('restore', (tracked, physical.astype([np.int32, np.uint16][number % 2]), reference, high)))
        cases.append(('reuse', (tracked, reference, int(rng.choice([0, 1, 3])))))
    originals = {'reuse': ledger.reuse_retired_identity_names, 'restore': confidence.restore_deferred_identity_layer}
    before = after = 0.0
    expected = []
    with batch.install(ledger, confidence) as stats:
        assert stats['supported']
        replaced = {'reuse': ledger.reuse_retired_identity_names, 'restore': confidence.restore_deferred_identity_layer}
        assert all(replaced[k] is not originals[k] for k in replaced)
        for number, (kind, args) in enumerate(cases):
            tick = time.perf_counter(); e = outcome(originals[kind], *args); before += time.perf_counter() - tick
            tick = time.perf_counter(); f = outcome(replaced[kind], *args); after += time.perf_counter() - tick
            assert e == f, (number, kind, e[:3], f[:3])
            expected.append(e)
    assert ledger.reuse_retired_identity_names is originals['reuse']
    report = dict(status='pass', cases=len(cases), real_cases=6, real_window_frames=160,
                  outcomes={kind: sum(1 for (k, _), e in zip(cases, expected) if k == kind and e[0] == 'ok')
                            for kind in originals},
                  errors=sum(1 for e in expected if e[0] == 'error'),
                  rows=sum(len(e[5]) for e in expected if e[0] == 'ok'),
                  before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
