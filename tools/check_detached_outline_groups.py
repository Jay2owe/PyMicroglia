"""Compare detached shared-outline discovery with its anchored-track grouping, exactly.

python tools/check_detached_outline_groups.py <engine code folder> <labels.tif> <points.csv> <report.json>

discover runs on the real movie and track points (scored as correct scores
them), with the default parameters and with lower anchor, visible-frame and
radius thresholds so candidate pairs reach the later tests, and on shuffled,
subset and float-track tables (the original statement): the candidate lists
must be identical (values, types, arrays bit for bit), or the same error raised.
"""
from dataclasses import replace
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
module = importlib.import_module('recover_detached_soma_shared_outline')
import detached_outline_groups

# Held in a list: the adapter re-points every module-level name bound to the function.
held = [module.discover]
seconds = dict(before=0., after=0.)


def canonical(value):
    if isinstance(value, np.ndarray):
        return ('array', value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, pd.DataFrame):
        # Object columns compare by each value's repr (their bytes are addresses).
        cells = (tuple(map(repr, value.itertuples())) if any(t == object for t in value.dtypes)
                 else value.to_numpy().tobytes() if value.size else b'')
        return ('frame', tuple(map(str, value.dtypes)), tuple(value.columns), cells, tuple(value.index))
    if isinstance(value, dict):
        return ('dict', tuple((k, canonical(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(canonical(v) for v in value))
    return (type(value).__name__, repr(value))


def same_frame(a, b):
    # Object columns compare by value (their bytes are object addresses).
    return (a.equals(b) and a.index.equals(b.index) and list(a.columns) == list(b.columns)
            and list(a.dtypes) == list(b.dtypes))


def outcome(function, key, *args):
    tick = time.perf_counter()
    try:
        value = canonical(function(*args))
    except Exception as error:
        value = (type(error).__name__, str(error))
    seconds[key] += time.perf_counter() - tick
    return value


labels = tifffile.imread(labels_path)
points = pd.read_csv(points_path)
scored = module.physical.attach_owners(points, labels)
for field in ("physically_visible", "strong"):
    scored[field] = scored[field].astype(bool)
raw = labels.astype(np.float32)
unclaimed = np.zeros_like(labels)
base = module.Params()
settings = [base, replace(base, minimum_anchor_frames=max(1, base.minimum_anchor_frames // 2)),
            replace(base, minimum_anchor_frames=1, minimum_visible_frames=1, minimum_soma_radius_px=0.)]
tables = [scored, scored.sample(frac=1, random_state=3), scored.sample(frac=.6, random_state=4)]
floating = scored.copy()
floating['track_id'] = floating.track_id.astype(float)
tables.append(floating)
stats = {}
cases = equal = 0
found = []
with detached_outline_groups.install(stats):
    adapted = module.discover
    installed = adapted is not held[0] and adapted.__wrapped__ is held[0]
    for table in tables:
        for params in settings:
            cases += 1
            before = outcome(held[0], 'before', labels, unclaimed, raw, table, params)
            after = outcome(adapted, 'after', labels, unclaimed, raw, table, params)
            equal += before == after
            found.append(len(before[1]) if before[0] == 'list' else before[0])
    # Every table discover can read: the anchored tracks' groups, compared directly.
    helper = vars(module)['_anchored_groups']
    group_cases = group_equal = 0
    for table in tables:
        anchors = module.dual._references(table, module.dual.Params())
        full = {int(track): group[group.physically_visible].sort_values("frame")
                for track, group in table.groupby("track_id")}
        for params in settings:
            mine = helper(table, anchors, params.minimum_anchor_frames)
            needed = {int(t) for t, a in anchors.items() if int(a["support_frames"]) >= params.minimum_anchor_frames}
            group_cases += 1
            # Integer tracks get only the anchored groups; other tables keep every group.
            expected = needed & set(full) if table['track_id'].dtype.kind in 'iu' else set(full)
            group_equal += (set(mine) == expected
                            and all(same_frame(mine[k], full[k]) for k in mine))
    # correct's per-track visible rows: the helper against the statement it
    # replaced, on these tables and on repeated and float frames.
    rows_helper = vars(module)['_visible_track_rows']
    repeated = pd.concat([scored, scored.iloc[::7]])
    float_frames = scored.assign(frame=scored.frame.astype(float))
    row_cases = row_equal = 0
    for table in tables + [repeated, float_frames, scored.iloc[0:0]]:
        expected = {int(track): list(group[group.physically_visible]
                                     .sort_values("frame").itertuples(index=False))
                    for track, group in table.groupby("track_id")}
        row_cases += 1
        row_equal += canonical(rows_helper(table)) == canonical(expected)
    # The whole correction, with the default and the lowest thresholds.
    held_correct = module.correct.__wrapped__
    correct_cases = correct_equal = 0
    protected = np.zeros(labels.shape, bool)
    for params in (settings[0], settings[2]):
        correct_cases += 1
        correct_equal += (outcome(held_correct, 'before', labels, unclaimed, raw, points, protected, params)
                          == outcome(module.correct, 'after', labels, unclaimed, raw, points, protected, params))
restored = (module.discover is held[0] and '_anchored_groups' not in vars(module)
            and not hasattr(module.correct, '__wrapped__') and '_visible_track_rows' not in vars(module))
# An isolated import can leave the module only as a composition's global; its
# own correct reads discover from the module's globals.
import types
holder = types.ModuleType('detached_outline_holder')
holder.detached = module
sys.modules[holder.__name__] = holder
sys.modules.pop('recover_detached_soma_shared_outline')
held_stats = {}
with detached_outline_groups.install(held_stats):
    held_patched = holder.detached.discover.__wrapped__ is held[0]
    held_equal = (outcome(held[0], 'before', labels, unclaimed, raw, scored, base)
                  == outcome(holder.detached.discover, 'after', labels, unclaimed, raw, scored, base))
held_restored = holder.detached.discover is held[0] and '_anchored_groups' not in vars(module)
sys.modules['recover_detached_soma_shared_outline'] = module
sys.modules.pop(holder.__name__)
checks = dict(installed=bool(installed and restored), cases=cases, equal=int(equal), group_cases=group_cases, group_equal=int(group_equal), candidates=found, stats=stats,
              row_cases=row_cases, row_equal=int(row_equal), correct_cases=correct_cases, correct_equal=int(correct_equal),
              held_module=dict(patched=held_patched, equal=held_equal, restored=held_restored, stats=held_stats),
              seconds={k: round(v, 3) for k, v in seconds.items()})
passed = (checks['installed'] and cases == equal and group_cases == group_equal and stats['versions'] == 1 and stats['anchored'] > 0
          and row_cases == row_equal and correct_cases == correct_equal and stats['correct_versions'] == 1
          and stats['sorted_once'] > 0 and stats['per_track'] >= 2
          and held_patched and held_equal and held_restored and held_stats['versions'] == 1 and held_stats['anchored'] == 1)
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
