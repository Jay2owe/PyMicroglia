"""Compare _recording_step_limit with its remembered form, exactly.

Real accepted tracker points with owner and other columns changed, missing
track names and frames, integer, unsigned and boolean columns, 60 random
subsets with moved positions, negative radii and shifted frames, (the
remembered value must equal a fresh original call), rows reordered, frames
tied within tracks, radii made non-finite, strong as objects and a missing
column. Every call of the wrapper must return the original's value (NaN where
NaN), and changed read columns must never reuse an earlier value.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import recording_step_memo as adapter
    spec = importlib.util.spec_from_file_location(adapter.MODULE, a.code / (adapter.MODULE + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[adapter.MODULE] = module
    spec.loader.exec_module(module)
    history = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    points = pd.read_csv(history / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points['physically_visible'] = points.visibility.gt(0)
    rng = np.random.default_rng(125)
    stats = {key: 0 for key in ('calls', 'reused', 'measured', 'vectorised', 'original', 'fallback')}
    original, wrapper = adapter.adapt(module, stats, {})
    cases = []
    base = points.copy()
    cases.append(('real', base, 'measured'))
    owners = base.copy()
    owners['accepted_owner'] = rng.integers(0, 50, len(owners))
    owners['state'] = 'observed'
    cases.append(('other columns changed', owners, 'reused'))
    reindexed = base.copy()
    reindexed.index = rng.permutation(len(reindexed)) + 7
    cases.append(('index changed', reindexed, 'reused'))
    cases.append(('rows reordered', base.sample(frac=1, random_state=3), 'measured'))
    tied = base.copy()
    pick = rng.choice(len(tied), len(tied) // 20, replace=False)
    tied.loc[tied.index[pick], 'frame'] = tied.frame.iloc[pick].to_numpy() + 1
    cases.append(('tied frames within tracks', tied, 'measured'))
    nan = base.copy()
    nan.loc[nan.index[rng.choice(len(nan), 500, replace=False)], 'radius_px'] = np.nan
    nan.loc[nan.index[rng.choice(len(nan), 50, replace=False)], 'x'] = np.inf
    cases.append(('non-finite radii and positions', nan, 'measured'))
    moved = base.copy()
    moved.loc[moved.index[0], 'x'] = np.nextafter(moved.x.iloc[0], np.inf)
    cases.append(('one position one ulp along', moved, 'measured'))
    weak = base.copy()
    weak['strong'] = False
    cases.append(('no strong points', weak, 'measured'))
    as_object = base.copy()
    as_object['strong'] = as_object.strong.astype(object)
    cases.append(('strong as objects', as_object, 'fallback'))
    float_frame = base.copy()
    float_frame['frame'] = float_frame.frame.astype(float)
    cases.append(('float frames', float_frame, 'measured'))
    nan_track = base.copy()
    nan_track['track_id'] = nan_track.track_id.astype(float)
    nan_track.loc[nan_track.index[:40], 'track_id'] = np.nan
    cases.append(('missing track names', nan_track, 'measured'))
    nan_frame = float_frame.copy()
    nan_frame.loc[nan_frame.index[5:30], 'frame'] = np.nan
    cases.append(('missing frames', nan_frame, 'measured'))
    integer_xy = base.copy()
    integer_xy['x'] = integer_xy.x.round().astype(np.int32)
    integer_xy['radius_px'] = integer_xy.radius_px.round().astype(np.int64)
    cases.append(('integer positions and radii', integer_xy, 'measured'))
    unsigned = base.copy()
    unsigned['track_id'] = unsigned.track_id.astype(np.uint32)
    unsigned['strong'] = unsigned.strong.astype(np.int8)
    cases.append(('unsigned tracks, integer flags', unsigned, 'measured'))
    boolean_x = base.copy()
    boolean_x['x'] = boolean_x.x > boolean_x.x.median()
    cases.append(('boolean positions', boolean_x, 'measured'))
    for trial in range(60):
        table = base.sample(frac=float(rng.uniform(.05, 1)), random_state=trial)
        table = table.copy()
        table['x'] = table.x + rng.normal(0, 3, len(table))
        if trial % 3 == 0:
            table.loc[table.index[rng.choice(len(table), max(1, len(table) // 50), replace=False)], 'radius_px'] = -1.0
        if trial % 5 == 0:
            table['frame'] = table.frame + rng.integers(0, 2, len(table))
        cases.append((f'random {trial}', table, 'measured'))
    report = dict(status='pass', rows=len(points), cases=[], seconds=dict(original=0.0, wrapper=0.0))
    for name, table, expected_route in cases:
        before = dict(stats)
        t0 = time.perf_counter()
        expected = original(table)
        t1 = time.perf_counter()
        actual = wrapper(table)
        t2 = time.perf_counter()
        again = wrapper(table)
        report['seconds']['original'] += t1 - t0
        report['seconds']['wrapper'] += t2 - t1
        assert same(expected, actual) and same(expected, again), (name, expected, actual, again)
        route = next(k for k in ('measured', 'reused', 'fallback') if stats[k] > before[k])
        assert route == expected_route, (name, route)
        report['cases'].append(dict(case=name, value=expected, route=route))
    missing = base.drop(columns='physically_visible')
    for function in (original, wrapper):
        try:
            function(missing)
        except AttributeError as error:
            message = str(error)
        else:
            raise AssertionError('missing column accepted')
    report['cases'].append(dict(case='missing column', route='fallback', error=message))
    report['stats'] = stats
    assert stats['vectorised'] > 0 and stats['original'] > 0, stats
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
