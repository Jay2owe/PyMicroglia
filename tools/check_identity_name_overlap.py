"""Compare identity_name_batch's retired-name reuse before and after the novel-pixel overlap count, exactly.

python tools/check_identity_name_overlap.py <before identity_name_batch.py> <labels.tif> <report.json>

Random small stacks (uint8/uint16/int32/int64, memmaps, minimum overlaps 0-3,
frames without novel names, names over 255 that do not fit uint8) and the
real 99-frame movie with names retired, renamed and split must give the same
relabelled stack (values, dtype, type), audit table, fallback count and error.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import tifffile

before_path, labels_path, report_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


before = load(before_path, 'identity_name_before')
after = load(here / 'identity_name_batch.py', 'identity_name_after')


def original(candidate, accepted, minimum):
    raise LookupError('fallback')


old_stats, new_stats = dict(fallback=0, batched=0), dict(fallback=0, batched=0)
old = before.reuse_retired_identity_names(original, old_stats)
new = after.reuse_retired_identity_names(original, new_stats)
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
    if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], type):
        return a == b
    (x, audit_x), (y, audit_y) = a, b
    return (type(x) is type(y) and x.dtype == y.dtype and np.array_equal(x, y)
            and audit_x.equals(audit_y) and list(audit_x.columns) == list(audit_y.columns)
            and list(audit_x.dtypes) == list(audit_y.dtypes))


rng = np.random.default_rng(1171)
cases = equal = 0


def compare(candidate, accepted, minimum=1):
    global cases, equal
    cases += 1
    equal += same(outcome(old, 'before', candidate.copy(), accepted, minimum),
                  outcome(new, 'after', candidate.copy(), accepted, minimum))


for trial in range(400):
    dtype = [np.uint8, np.uint16, np.int32, np.int64][trial % 4]
    shape = (int(rng.integers(1, 5)), int(rng.integers(1, 12)), int(rng.integers(1, 12)))
    names = rng.choice([0, 0, 1, 2, 3, 5, 9, 40, 200], size=shape)
    accepted = names.astype(dtype)
    candidate = names.copy()
    candidate[rng.random(shape) < .15] = 0
    swap = rng.random(shape) < .4
    candidate[swap] = rng.choice([6, 7, 41, 250] + ([300] if dtype is not np.uint8 else []), size=int(swap.sum()))
    if trial % 9 == 0:
        candidate[0] = accepted[0]
    compare(candidate.astype(dtype), accepted, int(trial % 4))

small = dict(seconds)
labels = tifffile.imread(labels_path)
ids = np.unique(labels)
ids = ids[ids > 0]
renamed = labels.copy()
top = int(ids[-1])
for k, identity in enumerate(rng.choice(ids, 12, replace=False)):
    frames = slice(int(rng.integers(0, len(labels) // 2)), None)
    part = renamed[frames]
    part[part == identity] = top + 1 + k
renamed[labels == ids[0]] = 0
compare(renamed, labels)
compare(renamed, labels, 50)
compare(renamed.astype(np.int32), labels.astype(np.int64))
compare(labels, labels)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    memmap = np.memmap(Path(folder) / 'm.dat', renamed.dtype, 'w+', shape=renamed.shape)
    memmap[:] = renamed
    compare(memmap, labels)
    del memmap

real = {key: round(seconds[key] - small[key], 3) for key in seconds}
passed = cases == equal and old_stats == new_stats and new_stats['batched'] > 0 and new_stats['fallback'] > 0
report = dict(status='passed' if passed else 'failed', cases=cases, equal=int(equal),
              stats=dict(before=old_stats, after=new_stats), seconds=real)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
