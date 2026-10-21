"""Compare the cropped complete-flip boundary step with the engine's own.

Synthetic masks cover empty, overlapping, multi-part, frame-edge and
non-boolean inputs; real masks pair identities of neighbouring frames of
accepted label stacks. Results must be equal floats, infinities included.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def blobs(rng, shape):
    mask = np.zeros(shape, bool)
    for _ in range(int(rng.integers(0, 5))):
        r, c = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        h, w = rng.integers(1, 8, 2)
        mask[max(0, r - h):r + h, max(0, c - w):c + w] = True
    if rng.random() < .3:
        mask &= rng.random(shape) < .7
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=4000)
    parser.add_argument('--real', type=int, default=3000)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import complete_flip_recovery as module
    import flip_boundary_crop as adapter
    original = module._mask_boundary_step
    rng = np.random.default_rng(20261020)
    failures, counts = [], dict(synthetic=0, real=0, infinite=0, zero=0)
    times = dict(original=0.0, cropped=0.0)

    def compare(left, right, label):
        t0 = time.perf_counter(); expected = original(left, right)
        with adapter.install(module) as stats:
            assert stats['supported']
            t1 = time.perf_counter(); actual = module._mask_boundary_step(left, right)
            t2 = time.perf_counter()
        times['original'] += t1 - t0; times['cropped'] += t2 - t1
        counts['infinite'] += expected == float('inf'); counts['zero'] += expected == 0.0
        if not (type(expected) is type(actual) and expected == actual):
            failures.append(dict(case=label, expected=repr(expected), actual=repr(actual)))

    for case in range(a.cases):
        shape = tuple(int(v) for v in rng.choice([5, 17, 40, 96], 2))
        left, right = blobs(rng, shape), blobs(rng, shape)
        if case % 7 == 0:
            right &= ~left
        if case % 11 == 0:
            compare(left.astype(np.uint8), right.astype(np.uint8), f'uint8-{case}')
        else:
            compare(left, right, f'synthetic-{case}')
        counts['synthetic'] += 1
    for name in ('01_issue005.tif', '11_issue017_rebase.tif'):
        labels = tifffile.imread(HISTORY / name)
        for case in range(a.real // 2):
            t = int(rng.integers(0, len(labels) - 1))
            here = np.unique(labels[t]); here = here[here > 0]
            after = np.unique(labels[t + 1]); after = after[after > 0]
            if not len(here) or not len(after):
                continue
            p = int(rng.choice(here)); s = int(rng.choice(after))
            if rng.random() < .5:
                compare(labels[t] == p, labels[t + 1] == s, f'{name}-{t}-{p}-{s}')
            else:
                compare(labels[t + 1] == s, labels[t] == p, f'{name}-{t}-{s}-{p}')
            counts['real'] += 1
    report = dict(status='pass' if not failures else 'fail', counts=counts,
                  seconds=times, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
