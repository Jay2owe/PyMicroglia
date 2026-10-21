"""Compare boxed Issue 017 raw line support with the original, exactly.

Real registered raw frames with identity masks from a recorded accepted-history
stack, over gaps of 2 to 40 frames, plus synthetic movies with discs at and
beyond the borders, tiny and huge masks, float and integer raw values and
empty masks: the three returned values must match bit for bit, through the
history loader as well.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def bits(values):
    import numpy as np
    return tuple(np.float64(v).tobytes() for v in values)


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--cases', type=int, default=600)
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import accepted_history as history
    import classify_disc_boxes as boxes
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '10_blob_residency.tif')
    raw = tifffile.imread(folder / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    rng = np.random.default_rng(31)
    cases = []
    while len(cases) < a.cases:
        pre = int(rng.integers(0, len(labels) - 3))
        post = min(len(labels) - 1, pre + int(rng.integers(1, 41)))
        ids = np.intersect1d(np.unique(labels[pre]), np.unique(labels[post]))[1:]
        others = np.unique(labels[post])[1:]
        if not len(ids) or not len(others):
            continue
        first = int(rng.choice(ids))
        second = first if rng.random() < 0.7 else int(rng.choice(others))
        cases.append((raw, pre, post, np.asarray(labels[pre] == first), np.asarray(labels[post] == second),
                      float(rng.choice([0.3, 0.5, 0.8]))))
    for number in range(300):
        shape = (int(rng.integers(5, 12)), int(rng.integers(3, 40)), int(rng.integers(3, 40)))
        movie = rng.uniform(0, 500, shape)
        if number % 2:
            movie = movie.astype(np.uint16)
        pre_mask = rng.random(shape[1:]) < float(rng.choice([0.01, 0.1, 0.8]))
        post_mask = rng.random(shape[1:]) < float(rng.choice([0.01, 0.1, 0.8]))
        if number % 7 == 0:
            pre_mask[:] = False
            pre_mask[0, 0] = True
        if number % 11 == 0:
            post_mask[:] = False
            post_mask[-1, -1] = True
        cases.append((movie, 0, int(rng.integers(0, shape[0])), pre_mask, post_mask, float(rng.choice([0.0, 0.5, 2.0]))))
    with boxes.install_loader(history) as loaded:
        module = history._issue_module(history.I017, 'm2_classify_evidence')
        assert loaded and loaded[0]['supported'], loaded
        original = module._classify_disc_original
        replaced = module.raw_line_support

        def outcome(function, case):
            try:
                return ('ok', bits(function(*case)))
            except Exception as error:
                return ('error', type(error).__name__, str(error))
        tick = time.perf_counter(); expected = [outcome(original, c) for c in cases]; before = time.perf_counter() - tick
        tick = time.perf_counter(); actual = [outcome(replaced, c) for c in cases]; after = time.perf_counter() - tick
    assert module.raw_line_support is original
    for number, (e, f) in enumerate(zip(expected, actual)):
        assert e == f, (number, e, f)
    report = dict(status='pass', cases=len(cases), real_cases=a.cases,
                  errors=sum(e[0] == 'error' for e in expected),
                  before_seconds=before, after_seconds=after, counters=loaded[0])
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
