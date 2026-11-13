"""Compare link_paint_boxes painting before and after the Boolean box write, exactly.

python tools/check_link_paint_boolean.py <before link_paint_boxes.py> <report.json>

Both painters paint the same label-mask and packed masks into plain, memmap
and working-array targets of several dtypes, with in-range, empty-box and
absent identities; every target must end with the same bytes, and the same
error must be raised for out-of-range values.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

before_path, report_path = map(Path, sys.argv[1:3])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'
sys.path.insert(0, str(here.parent))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


before = load(before_path, 'paint_before')
after = load(here / 'link_paint_boxes.py', 'paint_after')
storage = load(here / 'array_storage.py', 'paint_storage')


class LabelMask:
    def __init__(self, labels, identity):
        self.labels, self.identity, self.shape = labels, identity, labels.shape


class Packed:
    def __init__(self, bounds):
        self.bounds = bounds


class Module:
    _PackedObservationMask = Packed

    def __init__(self):
        self.frame = None

    def _materialize_observations(self, frame):
        return self.frame


def outcome(function, target, mask, value):
    try:
        function(target, mask, value)
        return ('ok', target.dtype.str, np.asarray(target).tobytes())
    except Exception as error:
        return (type(error).__name__, str(error))


rng = np.random.default_rng(123)
cases = equal = 0
kinds = {}
paths = dict(boxed=0, packed=0, fallback=0)
folder = Path(tempfile.mkdtemp())
pool = storage.Pool(folder / 'pool', minimum_bytes=1)
for trial in range(400):
    h, w = int(rng.integers(1, 60)), int(rng.integers(1, 60))
    labels = np.zeros((h, w), np.int32)
    for name in range(1, 6):
        y, x = rng.integers(0, h), rng.integers(0, w)
        dy, dx = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        region = labels[y:y + dy, x:x + dx]
        region[rng.random(region.shape) < .7] = name
    labels.flags.writeable = False
    identity = int(rng.integers(1, 8))  # 6 and 7 are absent names
    dtype = rng.choice(['uint8', 'uint16', 'int32', 'int64', 'float32'])
    value = int(rng.choice([identity, 300, -1, 70000])) if trial % 5 == 0 else identity
    # Packed mask: a full Boolean frame with a box, rebuilt by materialize.
    y0, x0 = int(rng.integers(0, h)), int(rng.integers(0, w))
    y1, x1 = int(rng.integers(y0, h + 1)), int(rng.integers(x0, w + 1))
    packed = np.zeros((h, w), bool)
    packed[y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0)) < .6
    for kind in ('plain', 'memmap', 'working'):
        results = []
        for module in (before, after):
            stats = dict(boxed=0, packed=0, fallback=0, snapshots=0, copied=0, copy_fallback=0)
            fake = Module()
            fake.frame = {0: {'mask': packed}}
            paint, copy, materialize = module.painter(LabelMask, stats, fake)
            materialize({0: {'mask': Packed((y0, y1, x0, x1))}})
            base = (np.arange(h * w).reshape(h, w) % 50).astype(dtype)
            if kind == 'plain':
                movie = np.array(np.stack([base, base]))
            elif kind == 'memmap':
                movie = np.memmap(folder / f'm{trial}_{len(results)}.dat', dtype=base.dtype, mode='w+', shape=(2, h, w))
                movie[:] = base
            else:
                movie = pool.array(np.stack([base, base]), force=True)
            one = outcome(paint, movie[1], LabelMask(labels, identity), value)
            two = outcome(paint, movie[0], packed, value)
            results.append((one, two))
            for key in paths:
                paths[key] += stats[key]
            for item in (one, two):
                kinds[item[0]] = kinds.get(item[0], 0) + 1
        cases += 1
        equal += results[0] == results[1]
pool.close()
passed = cases == equal and paths['boxed'] > 0 and paths['packed'] > 0 and len(kinds) > 1
report = dict(status='passed' if passed else 'failed', cases=cases, equal=equal, outcomes=kinds, paths=paths)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
