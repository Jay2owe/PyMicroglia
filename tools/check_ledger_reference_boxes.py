"""Compare boxed soma-ledger references with the frozen function, exactly.

python tools/check_ledger_reference_boxes.py <engine code folder> <labels.tif> <report.json>

Every bookend window of the real movie (anchors every third frame, the
configured 16 frames each side, minimum cores 0, 1, 12 and 40), random stacks
(uint8 to int64 names, touching and split components, one-pixel and thin
names), and other inputs (negative names, bool, float, 3-D frames, a memmap, a
list, windows outside the movie or reversed, a float minimum) must give the
same rows (keys, values and types, order) or the same error.
"""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import tifffile

code, labels_path, report_path = map(Path, sys.argv[1:4])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'
sys.path[:0] = [str(code.resolve()), str(here)]
ledger = importlib.import_module('global_soma_ledger')
import ledger_reference_boxes

held = [ledger._substantial_identity_references]
stats = {}
seconds = dict(before=0., after=0.)


def outcome(function, key, *args):
    tick = time.perf_counter()
    try:
        value = function(*args)
    except Exception as error:
        value = (type(error), str(error))
    seconds[key] += time.perf_counter() - tick
    return value


def same(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        return a == b
    return a == b and [[(k, type(v)) for k, v in row.items()] for row in a] == [
        [(k, type(v)) for k, v in row.items()] for row in b]


cases = equal = 0
with ledger_reference_boxes.install(ledger, stats):
    adapted = ledger._substantial_identity_references
    installed = adapted is not held[0] and adapted.__wrapped__ is held[0]

    def compare(*args):
        global cases, equal
        cases += 1
        equal += same(outcome(held[0], 'before', *args), outcome(adapted, 'after', *args))

    labels = tifffile.imread(labels_path)
    for anchor in range(0, len(labels), 3):
        compare(labels, anchor - 16, anchor + 16, (0, 1, 12, 40)[anchor % 4])
    real = dict(seconds)
    rng = np.random.default_rng(1174)
    for trial in range(200):
        dtype = [np.uint8, np.uint16, np.int32, np.int64][trial % 4]
        stack = np.zeros((3, 40, 50), dtype)
        for t in range(3):
            for name in rng.choice(np.arange(1, 120), int(rng.integers(0, 12)), replace=False):
                y, x = int(rng.integers(0, 38)), int(rng.integers(0, 48))
                h, w = int(rng.integers(1, 12)), int(rng.integers(1, 12))
                stack[t, y:y + h, x:x + w] = name
                if rng.random() < .3:
                    stack[t, rng.integers(0, 40), rng.integers(0, 50)] = name
        compare(stack, int(rng.integers(-2, 2)), int(rng.integers(0, 4)), int(rng.integers(0, 15)))
    stack = labels[:4].astype(np.int32)
    stack[1, 5, 5] = -3
    compare(stack, 0, 3, 12)
    compare(labels[:3] > 0, 0, 2, 12)
    compare(labels[:3].astype(float), 0, 2, 12)
    compare(labels[:2, None], 0, 1, 12)
    compare([labels[0].tolist()], 0, 0, 12)
    compare(labels, 200, 230, 12)
    compare(labels, 10, 5, 12)
    compare(labels, -40, -20, 12)
    compare(labels, 3, 5, 12.5)
    compare(labels, 3.0, 5, 12)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        memmap = np.memmap(Path(folder) / 'm.dat', labels.dtype, 'w+', shape=labels[:6].shape)
        memmap[:] = labels[:6]
        compare(memmap, 0, 5, 12)
        del memmap
restored = ledger._substantial_identity_references is held[0]
checks = dict(installed=bool(installed and restored), cases=cases, equal=int(equal), stats=stats,
              real_movie_seconds={k: round(v, 3) for k, v in real.items()})
passed = checks['installed'] and cases == equal and stats['boxed'] > 0 and stats['fallback'] > 0
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
