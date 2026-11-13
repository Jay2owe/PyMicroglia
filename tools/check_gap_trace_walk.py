"""Check the correction gap transport walk against the frozen walk.

python tools/check_gap_trace_walk.py <completed_feedback_engine folder> <report.json>
Imports both audited transport versions from the extracted correction engine,
installs the adapter, and compares every walk (frames and core bytes) with the
frozen function on shared geometries: repeated starts with different ends,
both directions, exact ties, empty frames, integer-valued cores and
non-finite starts. Also checks restoration and that other modules are left alone.
"""
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np

engine = Path(sys.argv[1])
report_path = Path(sys.argv[2])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'))
import gap_trace_walk


def load(stage, name):
    for flat in ('measured_soma_gap_memory', name):
        sys.modules.pop(flat, None)
    sys.path.insert(0, str(engine / 'frozen/final' / stage))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


def geometry(rng, frames, per_frame, ties):
    result = {}
    for t in range(frames):
        if rng.random() < .15:
            continue  # missing observation
        n = int(rng.integers(1, per_frame))
        g = np.column_stack([rng.uniform(0, 60, n), rng.uniform(0, 60, n), rng.uniform(2, 9, n)])
        if ties and n > 1 and rng.random() < .3:
            g[1] = g[0]  # an exact tie stops the walk
        if rng.random() < .2:
            g = np.round(g)
        result[t] = g
    return result


def outcome(function, *args):
    try:
        return function(*args)
    except Exception as error:
        return type(error)


def same(a, b):
    if isinstance(a, type) or isinstance(b, type):
        return a is b
    return len(a) == len(b) and all(
        ta == tb and type(ta) is type(tb) and ca.dtype == cb.dtype and ca.tobytes() == cb.tobytes()
        for (ta, ca), (tb, cb) in zip(a, b))


checks = {}
for stage, name in (('r215_a003', 'native_gap_owner_transport_a003'),
                    ('r213_a010', 'native_gap_owner_transport_a010')):
    module = load(stage, name)
    original = module.trace_measured_core_path
    stats = {}
    compared = mismatched = 0
    with gap_trace_walk.install(module, stats):
        walk = module.trace_measured_core_path
        installed = walk is not original and walk.__wrapped__ is original
        rng = np.random.default_rng(7)
        for case in range(40):
            g = geometry(rng, 80, 7, ties=case % 2 == 0)
            frames = sorted(g)
            for _ in range(60):
                t0 = int(rng.choice(frames))
                initial = g[t0][int(rng.integers(len(g[t0])))]
                for end in rng.integers(0, 80, 4):
                    for start, stop in ((t0, int(end)), (np.int64(t0), np.int64(end))):
                        compared += 1
                        mismatched += not same(walk(g, initial, start, stop), original(g, initial, start, stop))
        g = geometry(np.random.default_rng(3), 30, 5, ties=True)
        for initial in (np.array([np.nan, 1., 2.]), [1, 2, 3], np.array([1., 2.])):
            compared += 1
            mismatched += not same(outcome(walk, g, initial, 0, 20), outcome(original, g, initial, 0, 20))
        compared += 1
        mismatched += not same(walk({}, [1., 2., 3.], 5, 0), original({}, [1., 2., 3.], 5, 0))
    restored = module.trace_measured_core_path is original
    checks[stage] = dict(installed=installed, restored=restored, compared=compared,
                         mismatched=mismatched, stats={k: v for k, v in stats.items() if k != 'peak_cache_bytes'})

other = types.ModuleType('other_caller')
other.__file__ = __file__
other.trace_measured_core_path = original
with gap_trace_walk.install(other) as unchanged:
    checks['other_caller_untouched'] = other.trace_measured_core_path is original and unchanged['versions'] == 0

passed = (all(c['installed'] and c['restored'] and c['compared'] > 1000 and not c['mismatched']
              for k, c in checks.items() if k.startswith('r2'))
          and checks['other_caller_untouched'])
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
