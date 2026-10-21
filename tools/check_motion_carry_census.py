"""Compare established-identity motion carry with the boxed lost-label census.

Real label stacks (stage inputs saved by an earlier-stage run), the pinned raw
and lag images and the configured motion settings; the pair cache is built
once by the engine and shared. The engine function without adapters is the
reference; the corrected labels and both tables must be equal, plainly and
with disk-backed working arrays.
"""
import argparse, importlib, json, sys, time, types
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
INPUTS = ROOT / 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--labels', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import motion_handoffs as module
    import motion_history_reuse as adapter
    package = types.ModuleType('candidate_adapters')
    package.__path__ = [str(a.adapters.resolve()), str(a.adapters.resolve().parent / 'speed_candidate')]
    sys.modules['candidate_adapters'] = package
    array_storage = importlib.import_module('candidate_adapters.array_storage')
    config = json.loads((INPUTS / 'motion_config.json').read_text(encoding='utf-8'))
    params = config['accepted_postprocessing']['identity_reservation']['motion_handoff']
    raw = tifffile.imread(INPUTS / '95_full_A1.tif')
    lag = tifffile.imread(INPUTS / '95_full_A1_lag01_float.tif')
    cache = module.build_motion_pair_cache(lag, params)
    original = module.carry_established_identities_along_motion
    results, failures = [], []
    for path in a.labels:
        labels = tifffile.imread(path)
        for storage in ('plain', 'working'):
            with ExitStack() as scope:
                if storage == 'working':
                    pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=0)
                    scope.enter_context(array_storage.install(pool))
                t0 = time.perf_counter()
                expected = original(labels, raw, lag, params, pair_cache=cache)
                t1 = time.perf_counter()
                with adapter.install(module) as stats:
                    assert stats['supported']
                    actual = module.carry_established_identities_along_motion(labels, raw, lag, params, pair_cache=cache)
                t2 = time.perf_counter()
                same = (np.array_equal(np.asarray(expected[0]), np.asarray(actual[0]))
                        and expected[0].dtype == actual[0].dtype)
                for left, right in zip(expected[1:], actual[1:]):
                    try:
                        pd.testing.assert_frame_equal(left, right, check_exact=True)
                    except AssertionError:
                        same = False
                row = dict(labels=path.name, storage=storage, equal=bool(same), events=len(expected[1]),
                           audit=len(expected[2]), original_s=round(t1 - t0, 2), adapted_s=round(t2 - t1, 2),
                           census_boxed=stats['census_boxed'], census_whole=stats['census_whole'])
                results.append(row)
                print(row, flush=True)
                if not same:
                    failures.append(f'{path.name}-{storage}')
                del expected, actual
    report = dict(status='pass' if not failures else 'fail', results=results, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
