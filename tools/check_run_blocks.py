"""Compare owner-run grouping with its contiguous-block slicing, exactly.

python tools/check_run_blocks.py <correction code folder> <labels.tif> <points.csv> <report.json>

_runs on every real track (sorted, shuffled, with a nullable owner column
holding missing values, and empty), _anchors' run numbers on every real track
(also unsorted, unsigned, float frames, huge owners), recurrent_core_memory_a002._anchors, and
the three modules' correct on the real movie: every run must have the same
index, columns, dtypes and values, and every result the same values or the
same error. A module held only by a composition is also patched and restored.
"""
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import pandas as pd
import tifffile

code, labels_path, points_path, report_path = map(Path, sys.argv[1:5])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'
sys.path[:0] = [str(code.resolve()), str(here)]
coobserved = importlib.import_module('coobserved_bookended_soma_memory_a001')
recurrent = importlib.import_module('recurrent_core_memory_a002')
retired = importlib.import_module('repeated_retired_owner_memory_a002')
import run_blocks

held = dict(runs=coobserved._runs, anchors=recurrent._anchors, retired_runs=retired._runs)
seconds = dict(before=0., after=0.)


def same(a, b):
    if isinstance(a, pd.DataFrame):
        return (isinstance(b, pd.DataFrame) and a.equals(b) and a.index.equals(b.index)
                and list(a.columns) == list(b.columns) and list(a.dtypes) == list(b.dtypes)
                and a.index.dtype == b.index.dtype and a.attrs == b.attrs)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return type(a) is type(b) and a == b


def outcome(function, key, *args):
    tick = time.perf_counter()
    try:
        value = ('value', function(*args))
    except Exception as error:
        value = ('error', (type(error).__name__, str(error)))
    seconds[key] += time.perf_counter() - tick
    return value


labels = tifffile.imread(labels_path)
points = pd.read_csv(points_path)
scored = coobserved.owner.attach_owners(points, labels)
visible = scored[scored.physically_visible.astype(bool)]
groups = [group for _, group in visible.groupby('track_id')]
rng = np.random.default_rng(31)
extra = []
for group in groups[:200:5]:
    extra.append(group.sample(frac=1, random_state=int(rng.integers(1 << 30))))
    nullable = group.copy()
    nullable['candidate_owner'] = nullable['candidate_owner'].astype('Int64')
    if len(nullable) > 2:
        nullable.iloc[1, nullable.columns.get_loc('candidate_owner')] = pd.NA
    extra.append(nullable)
extra += [visible.iloc[:0], visible.iloc[:1]]
stats = {}
results = dict(runs=0, runs_equal=0, ends=0, ends_equal=0)
with run_blocks.install(stats):
    patched = (coobserved._runs.__wrapped__ is held['runs'] and recurrent._anchors.__wrapped__ is held['anchors']
               and retired._runs is coobserved._runs)
    for group in groups + extra:
        results['runs'] += 1
        results['runs_equal'] += same(outcome(held['runs'], 'before', group), outcome(coobserved._runs, 'after', group))
    # _anchors' run numbers, on frame-sorted tracks as it makes them and on
    # unsorted, unsigned, float and huge-valued columns.
    numbers = vars(recurrent)['_run_numbers']
    variants = []
    for group in groups:
        variants.append(group.sort_values('frame'))
    for group in groups[:200:7]:
        variants += [group, group.assign(candidate_owner=group.candidate_owner.astype(np.uint32)),
                     group.assign(frame=group.frame.astype(float)),
                     group.assign(candidate_owner=group.candidate_owner.astype(np.int64) + (1 << 60))]
    for group in variants:
        if group.empty:
            continue
        results['numbers'] = results.get('numbers', 0) + 1
        expected = (group.candidate_owner.ne(group.candidate_owner.shift()) | group.frame.diff().ne(1)).cumsum()
        actual = numbers(group)
        results['numbers_equal'] = results.get('numbers_equal', 0) + bool(
            actual.dtype == expected.dtype and actual.index.equals(expected.index)
            and np.array_equal(actual.to_numpy(), expected.to_numpy()))
    raw = labels.astype(np.float32)
    protected = np.zeros(labels.shape, bool)
    long_minimum = recurrent.established._minimum(len(labels), recurrent.established.DEFAULTS['minimum_exclusive_anchor_movie_fraction'])
    short_minimum = recurrent.established._minimum(len(labels), recurrent.established.DEFAULTS['minimum_source_anchor_movie_fraction'])
    prepared = recurrent.established.soma.prepare_strong_somata_by_frame(scored)
    for minimum in (short_minimum, 1):
        results['ends'] += 1
        results['ends_equal'] += same(outcome(held['anchors'], 'before', scored, labels, long_minimum, minimum, prepared),
                                      outcome(recurrent._anchors, 'after', scored, labels, long_minimum, minimum, prepared))
    corrections = {}
    for name, module in (('coobserved', coobserved), ('recurrent', recurrent), ('retired', retired)):
        restore = {key: value for key, value in vars(module).items() if key in ('_runs', '_anchors')}
        # The original correct: the module's names set back for this call only.
        current = dict(restore)
        for key in restore:
            setattr(module, key, held['anchors'] if key == '_anchors' else held['runs'])
        before = outcome(module.correct, 'before', labels.copy(), raw, points, protected)
        for key, value in current.items():
            setattr(module, key, value)
        after = outcome(module.correct, 'after', labels.copy(), raw, points, protected)
        results['ends'] += 1
        equal = same(before, after)
        results['ends_equal'] += equal
        corrections[name] = dict(equal=bool(equal), outcome=before[0],
                                 applications=len(before[1][-1]) if before[0] == 'value' else None)
restored = (coobserved._runs is held['runs'] and recurrent._anchors is held['anchors'] and retired._runs is held['runs']
            and '_run_blocks' not in vars(coobserved) and '_run_blocks' not in vars(recurrent)
            and '_run_numbers' not in vars(recurrent) and '_run_frames' not in vars(coobserved)
            and '_run_frames' not in vars(recurrent))

holder = types.ModuleType('run_blocks_holder')
holder.coobserved = coobserved
sys.modules[holder.__name__] = holder
sys.modules.pop('coobserved_bookended_soma_memory_a001')
held_stats = {}
with run_blocks.install(held_stats):
    held_patched = holder.coobserved._runs.__wrapped__ is held['runs']
    held_equal = all(same(held['runs'](g), holder.coobserved._runs(g)) for g in groups[:50])
held_restored = holder.coobserved._runs is held['runs']
sys.modules['coobserved_bookended_soma_memory_a001'] = coobserved
sys.modules.pop(holder.__name__)

checks = dict(patched=bool(patched), restored=bool(restored), results=results, corrections=corrections, stats=stats,
              held_module=dict(patched=held_patched, equal=held_equal, restored=held_restored, stats=held_stats),
              seconds={k: round(v, 3) for k, v in seconds.items()})
passed = (patched and restored and results['runs'] == results['runs_equal'] and results['ends'] == results['ends_equal']
          and stats['versions'] == 2 and stats['sliced'] > 0 and stats['grouped'] > 0
          and stats['in_order'] > 0 and stats['sorted'] > 0
          and results['numbers'] == results['numbers_equal'] and stats['numbered'] > 0 and stats['numbers_original'] > 0
          and held_patched and held_equal and held_restored)
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2, default=str) + '\n')
print(json.dumps(report, default=str)[:1500])
raise SystemExit(0 if passed else 1)
