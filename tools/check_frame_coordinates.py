"""Compare one-pass frame_coordinates with the previous adapter and the original expression, exactly.

python tools/check_frame_coordinates.py <before confidence_geometry_reuse.py> <labels.tif> <report.json>

Every real frame, and random frames (uint8 to int64, negative names, names
above the frame size, empty, all background, one pixel, transposed and
strided views, memmaps, boolean, float and 3-D input) must give the same
(name, coordinates) sequence as sorted(np.unique) with
np.column_stack(np.nonzero(frame == name)): names and their types, coordinate
values, dtype, shape and order, or the same error. The previous adapter must
agree everywhere except its known error on zero-size frames. frame_centres
must give np.mean(coordinates, axis=0) of the same names, bit for bit.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import tifffile

before_path, labels_path, report_path = map(Path, sys.argv[1:4])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


before = load(before_path, 'confidence_before')
after = load(here / 'confidence_geometry_reuse.py', 'confidence_after')


def original(frame):
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        yield identity, np.column_stack(np.nonzero(frame == identity))


seconds = dict(original=0., before=0., after=0., centres=0.)
centre_equal = []


def outcome(function, key, frame):
    tick = time.perf_counter()
    try:
        value = [(identity, coordinates) for identity, coordinates in function(frame)]
    except Exception as error:
        value = (type(error), str(error))
    seconds[key] += time.perf_counter() - tick
    return value


def same(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        return a == b
    return len(a) == len(b) and all(
        type(i) is type(j) and i == j and x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
        for (i, x), (j, y) in zip(a, b))


cases = equal = previous_empty_errors = previous_other = 0


def compare(frame):
    # The previous adapter raised ValueError on zero-size frames (find_objects'
    # reduction), where the original yields nothing; that is counted, not passed.
    global cases, equal, previous_empty_errors, previous_other
    cases += 1
    old = outcome(original, 'original', frame)
    previous = outcome(before.frame_coordinates, 'before', frame)
    if not same(old, previous):
        if np.size(frame) == 0 and isinstance(previous, tuple) and previous[0] is ValueError:
            previous_empty_errors += 1
        else:
            previous_other += 1
    equal += same(old, outcome(after.frame_coordinates, 'after', frame))
    centres = outcome(lambda f: list(after.frame_centres(f)), 'centres', frame)
    expected = old if isinstance(old, tuple) else [(i, np.mean(c, axis=0)) for i, c in old]
    if isinstance(expected, tuple) or isinstance(centres, tuple):
        centre_equal.append(expected == centres)
    else:
        centre_equal.append(len(expected) == len(centres) and all(
            type(i) is type(j) and i == j and x.dtype == y.dtype and x.shape == y.shape
            and x.tobytes() == y.tobytes() for (i, x), (j, y) in zip(expected, centres)))


labels = tifffile.imread(labels_path)
for t in range(len(labels)):
    compare(labels[t])
real = dict(seconds)
rng = np.random.default_rng(1173)
for trial in range(300):
    dtype = [np.uint8, np.uint16, np.int16, np.int32, np.int64, np.uint32][trial % 6]
    shape = (int(rng.integers(0, 14)), int(rng.integers(0, 14)))
    names = [0, 0, 0, 1, 2, 5, 9, 120] + ([-3, -1] if np.dtype(dtype).kind == 'i' else [])
    if trial % 7 == 0:
        names.append(int(np.iinfo(dtype).max))
    frame = rng.choice(names, size=shape).astype(dtype)
    compare(frame)
    compare(frame.T)
    compare(frame[::2, 1::2])
compare(np.zeros((5, 6), np.uint16))
single = np.zeros((5, 6), np.uint16)
single[4, 5] = 3
compare(single)
compare(labels[0] > 0)
compare(labels[0].astype(float))
compare(labels[:2, :40, :40])
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    memmap = np.memmap(Path(folder) / 'm.dat', labels.dtype, 'w+', shape=labels.shape[1:])
    memmap[:] = labels[5]
    compare(memmap)
    del memmap

checks = dict(cases=cases, equal=int(equal), centres_equal=int(sum(centre_equal)), previous_empty_frame_errors=previous_empty_errors,
              previous_other_differences=previous_other,
              real_movie_seconds={k: round(v, 3) for k, v in real.items()})
passed = cases == equal == sum(centre_equal) and previous_other == 0
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
