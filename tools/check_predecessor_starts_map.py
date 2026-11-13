"""Compare preserve_predecessors with its one-Series start lookup, exactly.

python tools/check_predecessor_starts_map.py <engine code folder> <labels.tif> <points.csv> <report.json>

The frozen module on the real 99-frame movie and track points, on subsets,
shuffled rows, tracks renamed to floats, frames shifted and an empty table:
the merged points (values, dtypes, index), the predecessor rows and the scored
table must be identical, or the same error raised.
"""
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tifffile

code, labels_path, points_path, report_path = map(Path, sys.argv[1:5])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'
sys.path[:0] = [str(code.resolve()), str(here)]
module = importlib.import_module('predecessor_name_memory_a002')
import predecessor_starts_map

held = [module.preserve_predecessors]
seconds = dict(before=0., after=0.)


def outcome(function, key, *args):
    tick = time.perf_counter()
    try:
        value = function(*args)
    except Exception as error:
        value = (type(error), str(error))
    seconds[key] += time.perf_counter() - tick
    return value


def frames_same(a, b):
    return a.equals(b) and list(a.dtypes) == list(b.dtypes) and a.index.equals(b.index) and list(a.columns) == list(b.columns)


def same(a, b):
    if isinstance(a[0], type) or isinstance(b[0], type):
        return a == b
    return frames_same(a[0], b[0]) and a[1] == b[1] and frames_same(a[2], b[2])


labels = tifffile.imread(labels_path)
points = pd.read_csv(points_path)
rng = np.random.default_rng(1175)
tables = [points, points.sample(frac=1, random_state=1), points.iloc[:0]]
for trial in range(6):
    tables.append(points.sample(frac=float(rng.uniform(.3, .9)), random_state=10 + trial))
shifted = points.copy()
shifted['frame'] = np.minimum(shifted.frame + (shifted.track_id % 3 == 0), len(labels) - 1)
tables.append(shifted)
floating = points.copy()
floating['track_id'] = floating.track_id.astype(float)
tables.append(floating)
stats = {}
cases = equal = 0
with predecessor_starts_map.install(stats):
    adapted = module.preserve_predecessors
    installed = adapted is not held[0] and adapted.__wrapped__ is held[0]
    for table in tables:
        cases += 1
        equal += same(outcome(held[0], 'before', labels, table), outcome(adapted, 'after', labels, table))
restored = (module.preserve_predecessors is held[0] and '_starts_series' not in vars(module)
            and '_predecessor_tracks' not in vars(module))
checks = dict(installed=bool(installed and restored), cases=cases, equal=int(equal), stats=stats,
              seconds={k: round(v, 3) for k, v in seconds.items()})
# Recorded tables skip the per-track frame sort; shuffled rows and float tracks sort.
passed = (checks['installed'] and cases == equal and stats['versions'] == 1
          and stats['groups_in_order'] > 0 and stats['groups_sorted'] > 0)
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
