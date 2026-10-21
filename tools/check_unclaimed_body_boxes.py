"""Compare boxed unclaimed-body tracking with the original, exactly.

Every unclaimed source identity of three recorded unclaimed stacks, plus
synthetic domains with several components, ties, single pixels and frames with
no substantial body, at several thresholds. Selected masks, summaries and
refusals must match.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def outcome(function, *args):
    import hashlib
    import numpy as np
    try:
        selected, track = function(*args)
    except AssertionError as error:
        return ('refused', str(error))
    return ('ok', hashlib.sha256(np.ascontiguousarray(selected).tobytes()).hexdigest(), selected.dtype.str,
            {k: (type(v).__name__, repr(v)) for k, v in track.items()})


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import unclaimed_body_identity as module
    import unclaimed_body_boxes as boxes
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    rng = np.random.default_rng(29)
    cases = []
    for name in ('06_issue010_unclaimed.tif', '08_issue012_unclaimed.tif', '09_seat_identity_unclaimed.tif'):
        stack = tifffile.imread(folder / name)
        for identity in np.unique(stack)[1:]:
            domain = stack == identity
            cases.append((domain, int(rng.choice([1, 10, 30])), float(rng.choice([0.0, 0.25, 0.5]))))
    for _ in range(400):
        shape = (int(rng.integers(1, 6)), int(rng.integers(4, 30)), int(rng.integers(4, 30)))
        domain = rng.random(shape) < float(rng.choice([0.05, 0.2, 0.5]))
        cases.append((domain, int(rng.integers(0, 6)), float(rng.choice([0.0, 0.5, 1.0]))))
    original = module._track_single_body
    tick = time.perf_counter(); expected = [outcome(original, *c) for c in cases]; before = time.perf_counter() - tick
    with boxes.install(module) as stats:
        assert stats['supported']
        tick = time.perf_counter(); actual = [outcome(module._track_single_body, *c) for c in cases]; after = time.perf_counter() - tick
    assert module._track_single_body is original
    for number, (e, f) in enumerate(zip(expected, actual)):
        assert e == f, number
    report = dict(status='pass', cases=len(cases), real_identities=len(cases) - 400,
                  refused=sum(e[0] == 'refused' for e in expected),
                  before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
