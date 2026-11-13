"""Compare process inventory track features with and without the one-pass adapter.

python tools/check_process_track_features.py <process_inventory_tracked.py> <report.json>
Imports the frozen module (with its folder first on the path), then compares
_track_features before and after install on scored tables: integer, float and
nullable track ids, missing track ids, ties for the top owner, owners only on
invisible rows, NaN owners on invisible rows, object and boolean flag columns,
duplicated and shuffled indexes, a single track, and an empty table. Results
must be equal dicts in the same key order with the same value types; a NaN
owner on a visible row must raise the same error type.
"""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

module_path, report_path = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(module_path.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'))
tracked = importlib.import_module(module_path.stem)
import process_track_features

rng = np.random.default_rng(5)


def table(n, tracks, owners, kind):
    frame = pd.DataFrame({
        'track_id': rng.integers(0, tracks, n),
        'frame': rng.integers(0, 50, n),
        'physically_visible': rng.random(n) < .7,
        'accepted_owner': rng.integers(0, owners, n),
        'strong': rng.random(n) < .5,
    })
    if kind == 'float':
        frame['track_id'] = frame.track_id.astype(float)
        frame.loc[frame.sample(frac=.05, random_state=1).index, 'track_id'] = np.nan
    elif kind == 'nullable':
        frame['track_id'] = frame.track_id.astype('Int64')
        frame.loc[frame.sample(frac=.05, random_state=2).index, 'track_id'] = pd.NA
    elif kind == 'object_flags':
        frame['physically_visible'] = frame.physically_visible.map({True: 1, False: 0}).astype(object)
        frame['strong'] = frame.strong.astype(int)
    elif kind == 'float_owner_nan_invisible':
        frame['accepted_owner'] = frame.accepted_owner.astype(float)
        frame.loc[~frame.physically_visible, 'accepted_owner'] = np.nan
    elif kind == 'duplicated_index':
        frame.index = rng.integers(0, max(1, n // 3), n)
    elif kind == 'shuffled':
        frame = frame.sample(frac=1, random_state=3)
    return frame


def outcome(function, scored):
    try:
        return function(scored)
    except Exception as error:
        return type(error)


def same(a, b):
    if isinstance(a, type) or isinstance(b, type):
        return a is b
    return (list(a) == list(b) and all(
        list(a[k]) == list(b[k]) and all(a[k][f] == b[k][f] and type(a[k][f]) is type(b[k][f]) for f in a[k])
        for k in a))


original = tracked._track_features
cases = []
for kind in ('int', 'float', 'nullable', 'object_flags', 'float_owner_nan_invisible', 'duplicated_index', 'shuffled'):
    for n, tracks, owners in ((2000, 60, 4), (2000, 300, 3), (500, 5, 40), (40, 40, 2)):
        cases.append((f'{kind}_{n}_{tracks}_{owners}', table(n, tracks, owners, kind)))
cases.append(('single_track', table(300, 1, 3, 'int')))
cases.append(('empty', table(0, 1, 2, 'int')))
invisible_only = table(400, 20, 5, 'int')
invisible_only['physically_visible'] = False
cases.append(('invisible_only', invisible_only))
nan_visible = table(400, 20, 5, 'float_owner_nan_invisible')
nan_visible.loc[nan_visible.index[0], ['physically_visible', 'accepted_owner']] = [True, np.nan]
cases.append(('nan_owner_visible', nan_visible))

stats = {}
results = {}
with process_track_features.install(tracked, stats):
    adapted = tracked._track_features
    installed = adapted is not original
    for name, scored in cases:
        before = scored.copy()
        expected = outcome(original, scored)
        got = outcome(adapted, scored)
        results[name] = bool(same(expected, got) and scored.equals(before))
restored = tracked._track_features is original
passed = installed and restored and all(results.values()) and stats['tied'] > 0 and stats['direct'] > 0
report = dict(status='passed' if passed else 'failed', installed=installed, restored=restored,
              cases=len(results), mismatched=[k for k, v in results.items() if not v], stats=stats)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
