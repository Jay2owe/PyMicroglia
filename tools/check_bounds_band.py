"""Compare banded occupied bounds with array_bounds.occupied_bounds, exactly.

python tools/check_bounds_band.py <engine code folder> <report.json>

Random boolean masks of every small shape and real-sized frames (empty, one
pixel, full, edge rows and columns, sparse and dense), non-contiguous and
transposed views, and other inputs (integer, 3-D, memmap, list) must give the
same tuple (value and element types), None or error; occupied_coordinates must
still give numpy.nonzero's arrays through the installed adapter; install and restore.
"""
import importlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

code, report_path = Path(sys.argv[1]), Path(sys.argv[2])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'
sys.path[:0] = [str(code.resolve()), str(here)]
module = importlib.import_module('array_bounds')
import bounds_band

# Held in lists: the install also re-points module-level aliases, this script's included.
held = [module.occupied_bounds, module.occupied_coordinates]
stats = dict(versions=0, banded=0, fallback=0)
with bounds_band.install(stats):
    adapted = module.occupied_bounds
    installed = adapted is not held[0] and adapted.__wrapped__ is held[0]
    rng = np.random.default_rng(1161)

    def outcome(function, value):
        try:
            return function(value)
        except Exception as error:
            return type(error)

    def same(a, b):
        if isinstance(a, type) or a is None:
            return a is b
        if isinstance(a, tuple) and a and isinstance(a[0], np.ndarray):
            return all(x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)
        return a == b and [type(v) for v in a] == [type(v) for v in b]

    cases = equal = 0
    masks = []
    for h in range(0, 9):
        for w in range(0, 9):
            for density in (0, .05, .4, 1):
                masks.append(rng.random((h, w)) < density)
    for _ in range(300):
        h, w = int(rng.choice([1, 50, 471])), int(rng.choice([1, 77, 471]))
        mask = np.zeros((h, w), bool)
        for _ in range(int(rng.integers(0, 4))):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            mask[y:y + int(rng.integers(1, 30)), x:x + int(rng.integers(1, 30))] = True
        if rng.random() < .1:
            mask[rng.integers(0, h), :] = True
        masks.append(mask)
        masks.append(mask[::2, 1::3])
        masks.append(mask.T)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        memmap = np.memmap(Path(folder) / 'm.dat', bool, 'w+', shape=(20, 30))
        memmap[3:5, 7:9] = True
        others = [masks[-3].astype(np.uint8), np.stack([masks[-3]] * 2), memmap, [[0, 1], [1, 0]], np.zeros((0, 4), bool)]
        for value in masks + others:
            cases += 1
            equal += same(outcome(held[0], value), outcome(adapted, value))
            cases += 1
            # occupied_coordinates promises numpy.nonzero's coordinates.
            equal += same(outcome(np.nonzero, value), outcome(module.occupied_coordinates, value))
        del memmap
restored = module.occupied_bounds is held[0]
checks = dict(installed=bool(installed and restored), cases=cases, equal=int(equal), stats=stats)
passed = checks['installed'] and cases == equal and stats['versions'] == 1 and stats['banded'] > 0 and stats['fallback'] > 0
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
