"""Compare vectorised measured soma cores with the frozen measured_cores_by_frame, exactly.

python tools/check_measured_cores.py --frozen <observed_soma_reservation.py folder> --adapters <identity_candidate> --output <report.json>

Inputs: the real latent track points (whole, per-frame subsets, shuffled,
with duplicated rows, signed zeros, NaN and infinite positions, half-way
coordinates), random tables, and tables the adapter must hand back (float
frames, object columns, repeated column names, missing columns, huge
coordinates, the visibility column instead of physical visibility). Results
must be equal dictionaries with the same key order, list order and Python
value types, or the same error.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / ('development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/'
                  'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history')


def outcome(function, points):
    try:
        value = function(points)
        return 'value', [(type(k), k, [tuple((type(v), v) for v in row) for row in rows]) for k, rows in value.items()]
    except Exception as error:
        return 'error', type(error).__name__, str(error)


def main():
    parser = argparse.ArgumentParser()
    for name in ('frozen', 'adapters', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(a.frozen.resolve()), str(a.adapters.resolve())]
    module = importlib.import_module('observed_soma_reservation')
    import measured_cores
    stats = dict(vectorised=0, original=0)
    original = module.measured_cores_by_frame
    fast = measured_cores.adapt(original, stats)
    real = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    rng = np.random.default_rng(11)
    tables = [('real', real), ('real-shuffled', real.sample(frac=1, random_state=3)),
              ('real-early', real[real.frame < 40])]
    doubled = pd.concat([real.head(3000), real.head(3000)])
    tables.append(('duplicated-rows', doubled))
    odd = real.head(4000).copy()
    odd.loc[odd.index[::7], 'x'] = -0.0
    odd.loc[odd.index[1::7], 'x'] = 0.0
    odd.loc[odd.index[2::11], 'y'] = np.nan
    odd.loc[odd.index[3::13], 'x'] = np.inf
    odd.loc[odd.index[4::5], 'y'] = np.floor(odd.y[odd.index[4::5]]) + .5
    tables.append(('signed-zero-nan-inf-halves', odd))
    tables.append(('visibility', real.head(2000).drop(columns=[c for c in ('physically_visible',) if c in real]).assign(
        visibility=rng.integers(0, 2, 2000))))
    for trial in range(80):
        n = int(rng.integers(0, 300))
        t = pd.DataFrame(dict(frame=rng.integers(0, 12, n), track_id=rng.integers(1, 9, n),
                              x=np.round(rng.uniform(-3, 40, n) * 2) / 2, y=np.round(rng.uniform(-3, 40, n) * 2) / 2,
                              strong=rng.random(n) < .8, state=rng.choice(['observed', 'predicted'], n, p=[.8, .2]),
                              physically_visible=rng.random(n) < .9))
        if trial % 5 == 0 and n:
            t = pd.concat([t, t.sample(frac=.3, random_state=trial)])
        if trial % 9 == 0:
            t.index = rng.permutation(len(t)) * 2
        tables.append((f'random-{trial}', t))
    fallback = [('float-frames', real.head(2000).assign(frame=real.frame.head(2000).astype(float))),
                ('object-x', real.head(500).assign(x=real.x.head(500).astype(object))),
                ('float32-y', real.head(500).assign(y=real.y.head(500).astype(np.float32))),
                ('missing-state', real.head(500).drop(columns='state')),
                ('huge', real.head(500).assign(x=real.x.head(500) * 1e17)),
                ('repeated-column', pd.concat([real.head(300), real.head(300)[['x']]], axis=1))]
    tables += fallback
    cases, equal, seconds = 0, 0, dict(original=0.0, adapted=0.0)
    failures = []
    for name, table in tables:
        t0 = time.perf_counter()
        expected = outcome(original, table)
        t1 = time.perf_counter()
        actual = outcome(fast, table)
        t2 = time.perf_counter()
        seconds['original'] += t1 - t0
        seconds['adapted'] += t2 - t1
        cases += 1
        if expected == actual:
            equal += 1
        else:
            failures.append(name)
    passed = equal == cases and stats['vectorised'] > 0 and stats['original'] >= 5
    report = dict(status='pass' if passed else 'fail', cases=cases, equal=equal, stats=stats,
                  seconds={k: round(v, 2) for k, v in seconds.items()}, failures=failures)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
