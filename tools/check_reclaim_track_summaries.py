"""Compare reclaim_track_summaries with delayed_projection_reclaim.discover, exactly.

discover runs on its accepted A1 stage input (57) and on several other label
movies from the accepted history, in plain and memory-pool forms, with the
accepted parameters and with looser ones that let more pairs reach the profile
and successor checks. The audit and scored tables, any error and every warning
must equal the original's; apply then runs on both audits and its labels and
tables must match too.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
STACKS = ('57_well_b2_issue001_terminal_projection_diversion', '58_well_b2_issue002_delayed_projection_reclaim',
          '33_issue063_post_split_excursion_recovery', '87_well_b2_issue018_recording_start_two_seat_inheritance')


def answer(function, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = ('ok', function(*args))
        except Exception as error:
            result = ('error', (type(error).__name__, str(error)))
    return result, [(w.category.__name__, str(w.message)) for w in caught]


def same(a, b):
    if type(a) is not type(b):
        return f'type {type(a).__name__} != {type(b).__name__}'
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return 'array dtype or shape'
        return None if np.array_equal(np.asarray(a), np.asarray(b), equal_nan=a.dtype.kind in 'fc') else 'array values'
    if isinstance(a, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        except AssertionError as error:
            return 'frame ' + str(error)[:300]
        return None
    if isinstance(a, dict):
        if list(a) != list(b):
            return 'dict keys'
        for key in a:
            problem = same(a[key], b[key])
            if problem:
                return f'{key}: {problem}'
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return 'length'
        for index, (x, y) in enumerate(zip(a, b)):
            problem = same(x, y)
            if problem:
                return f'[{index}] {problem}'
        return None
    if isinstance(a, float):
        return None if np.float64(a).tobytes() == np.float64(b).tobytes() else f'float {a!r} != {b!r}'
    return None if a == b else f'{a!r} != {b!r}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import tifffile
    import array_storage
    import delayed_projection_reclaim as reclaim
    import delayed_projection_reclaim_integration as integration
    import reclaim_track_summaries as adapter
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    _, accepted = integration._configured(config)
    loose = copy.deepcopy(accepted)
    for key in ('minimum_ending_anchor_movie_fraction', 'minimum_successor_run_movie_fraction',
                'minimum_projection_anchor_movie_fraction'):
        loose[key] = 0.0
    for key in ('maximum_birth_slack_movie_fraction', 'maximum_end_slack_movie_fraction',
                'maximum_reclaim_delay_movie_fraction'):
        loose[key] = 0.5
    loose['minimum_shared_ending_owner_fraction'] = 0.0
    a.scratch.mkdir(parents=True, exist_ok=True)
    pool = array_storage.Pool(a.scratch / 'pool')
    serial = [0]

    def pooled(stack):
        path = a.scratch / f'stack_{serial[0]}.dat'
        serial[0] += 1
        mapped = np.memmap(path, dtype=stack.dtype, mode='w+', shape=stack.shape)
        mapped[:] = stack
        mapped.flush()
        return pool.adopt(mapped, path)

    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    encounters = pd.read_csv(HISTORY / '26_latent_body_tracks/out/encounter_frames.csv')
    report = dict(failures=[], cases={})

    def fail(**item):
        if len(report['failures']) < 30:
            report['failures'].append(item)

    pool_raw = pooled(raw)
    for stem in STACKS:
        labels = tifffile.imread(HISTORY / (stem + '.tif'))
        cases = [('plain', 'accepted', lambda: labels.copy(), raw, accepted)]
        if stem == STACKS[0]:
            cases += [('pool', 'accepted', lambda: pooled(labels), pool_raw, accepted),
                      ('plain', 'loose', lambda: labels.copy(), raw, loose)]
        else:
            cases += [('plain', 'loose', lambda: labels.copy(), raw, loose)]
        for form, setting, make, movie, params in cases:
            name = f'{stem}:{form}:{setting}'
            t0 = time.perf_counter()
            expected = answer(reclaim.discover, make(), movie, points.copy(), encounters.copy(), params)
            t1 = time.perf_counter()
            with adapter.install(reclaim) as stats:
                assert stats['supported'] == ['delayed_projection_reclaim'], stats
                t2 = time.perf_counter()
                actual = answer(reclaim.discover, make(), movie, points.copy(), encounters.copy(), params)
                t3 = time.perf_counter()
                case_stats = dict(stats)
            restored = not hasattr(reclaim.discover, '__wrapped__') and '_reclaim_memo_factory' not in vars(reclaim)
            problem = same(expected, actual)
            if problem:
                fail(case=name, step='discover', problem=problem)
            rows = 0
            if expected[0][0] == 'ok' and actual[0][0] == 'ok':
                audit, scored = expected[0][1]
                rows = len(audit)
                eligible = int(audit.eligible.astype(bool).sum()) if len(audit) else 0
                applied = [answer(reclaim.apply, make(), movie, result[0][1][0], result[0][1][1], params)
                           for result in (expected, actual)]
                problem = same(*applied)
                if problem:
                    fail(case=name, step='apply', problem=problem)
            else:
                eligible = 0
            report['cases'][name] = dict(status=expected[0][0], warnings=len(expected[1]), rows=rows, eligible=eligible,
                                         restored=restored, original_seconds=round(t1 - t0, 2),
                                         adapted_seconds=round(t3 - t2, 2), stats=case_stats)
        del labels
    report['status'] = 'pass' if (not report['failures'] and all(c['restored'] for c in report['cases'].values())
                                  and all(c['stats']['calls'] for c in report['cases'].values())
                                  and any(c['stats']['smooth_reused'] for c in report['cases'].values())) else 'fail'
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report, indent=1, default=str))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
