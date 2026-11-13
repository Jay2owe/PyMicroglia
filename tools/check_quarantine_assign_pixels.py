"""Compare Issue 009 ordered-quarantine assign with the pixel-index version.

Real components of the Issue 009 input stack are assigned to their own frame,
to neighbouring frames (where they cover other cells and background), under
existing, new and zero identities, on uint16, int32 and disk-backed stacks.
Components with unsorted, repeated, out-of-frame or no pixels keep the
original (and raise the same errors). Labels after the call, the returned
component's every field (pixel bytes, dtype, centroid bits), the observed
identity and the changed count must all be equal.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def outcome(function, labels, frame, component, identity):
    try:
        made, observed, changed = function(labels, frame, component, identity)
    except Exception as error:
        return ('error', type(error).__name__, str(error), labels.tobytes())
    return ('ok', labels.dtype.str, labels.tobytes(), made.frame, made.identity, made.number,
            made.pixels.dtype.str, made.pixels.tobytes(), made.area,
            np.float64(made.x).tobytes(), np.float64(made.y).tobytes(), observed, changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=400)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import accepted_history as history
    import quarantine_assign_pixels as adapter
    module = history._issue_module(history.HIERARCHY_OPS, 'ordered_quarantine')
    m1 = module.m1
    original = module.assign
    with adapter.install(module) as stats:
        assert stats['supported']
        replaced = module.assign
    assert module.assign is original
    stack = tifffile.imread(HISTORY / '04_issue008.tif')[:40]
    rng = np.random.default_rng(9)
    folder = Path(tempfile.mkdtemp())
    failures, seconds, cases = [], dict(original=0.0, replaced=0.0), 0
    with adapter.install(module) as stats:
        replaced = module.assign
        for number in range(a.cases):
            source = int(rng.integers(len(stack)))
            items = m1.frame_components(stack[source], source, 1)
            component = items[int(rng.integers(len(items)))]
            frame = int(np.clip(source + rng.integers(-3, 4), 0, len(stack) - 1))
            ids = np.unique(stack[frame])[1:]
            identity = int([component.identity, int(rng.choice(ids)), 60000, 0][number % 4])
            kind = number % 6
            if kind == 4:
                component = m1.Component(source, component.identity, component.number,
                                         component.pixels[::-1].copy(), component.area, component.x, component.y)
            elif kind == 5 and number % 12 == 5:
                component = m1.Component(source, component.identity, component.number,
                                         component.pixels[:0].copy(), 0, 0.0, 0.0)
            elif kind == 5:
                component = m1.Component(source, component.identity, component.number,
                                         np.r_[component.pixels, component.pixels[-1]], component.area,
                                         component.x, component.y)
            base = stack.astype(np.int32) if number % 3 == 1 else stack
            first, second = base.copy(), base.copy()
            if number % 3 == 2:
                disks = []
                for name, value in (('a', first), ('b', second)):
                    held = np.memmap(folder / f'{name}{number}.dat', value.dtype, 'w+', shape=value.shape)
                    held[:] = value
                    disks.append(held)
                first, second = disks
            t0 = time.perf_counter(); e = outcome(original, first, frame, component, identity)
            t1 = time.perf_counter(); f = outcome(replaced, second, frame, component, identity)
            t2 = time.perf_counter()
            seconds['original'] += t1 - t0; seconds['replaced'] += t2 - t1
            cases += 1
            if e != f:
                failures.append(dict(number=number, kind=kind, expected=str(e[:2]), actual=str(f[:2])))
        counters = dict(stats)
    assert counters['pixels'] > 0 and counters['fallback'] > 0, counters
    report = dict(status='pass' if not failures else 'fail', cases=cases, seconds=seconds,
                  counters=counters, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
