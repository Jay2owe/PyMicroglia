"""Compare stationary run linking with the direct two-diameter median.

The pair median is compared with np.median on random, tied, extreme and NaN
values. build_component_runs is then run original vs adapted on components of
real accepted stacks at several link settings; every run's members, flags and
median step must match exactly.
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def same_float(a, b):
    return type(a) is type(b) and ((math.isnan(a) and math.isnan(b)) or a == b)


def summary(runs):
    return [(run.run_id, run.identity, [part.key for part in run.members], run.stationary,
             repr(run.median_step_diameters)) for run in runs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import stationary_reconciliation as module
    import stationary_gap_recovery, territory_memory
    import stationary_pair_median as adapter
    failures = []
    rng = np.random.default_rng(20261020)
    values = list(rng.random(20000) * 40) + list(rng.integers(1, 30, 4000).astype(float))
    values += [0.0, -0.0, 1e308, -1e308, 5e-324, float('inf'), float('-inf'), float('nan')]
    pairs = 0
    for _ in range(60000):
        x, y = float(rng.choice(values)), float(rng.choice(values))
        pairs += 1
        if not same_float(np.median([x, y]), adapter.pair_median(x, y)):
            failures.append(f'pair {x!r} {y!r}')
    for x, y in [(1.0, 2.0), (1, 2), (np.float32(1.5), 2.0)]:
        if not same_float(np.median([x, y]), adapter.pair_median(x, y)):
            failures.append(f'typed pair {x!r} {y!r}')
    rows = []
    times = dict(original=0.0, adapted=0.0)
    settings = [dict(maximum_step_diameters=0.5, minimum_area_similarity=0.5, minimum_normalized_area_correlation=0.3,
                     maximum_median_step_diameters=0.25),
                dict(maximum_step_diameters=2.0, minimum_area_similarity=0.0, minimum_normalized_area_correlation=0.0,
                     maximum_median_step_diameters=1.0)]
    for name in ('01_issue005.tif', '11_issue017_rebase.tif'):
        components = module.extract_components(tifffile.imread(HISTORY / name), 12)
        for number, params in enumerate(settings):
            for minimum_frames in (3, 20):
                t0 = time.perf_counter(); expected = summary(module.build_component_runs(components, params, minimum_frames))
                with adapter.install(module, stationary_gap_recovery, territory_memory) as stats:
                    assert stats['supported'] and stats['aliases'] == 2
                    assert territory_memory.build_component_runs is module.build_component_runs
                    t1 = time.perf_counter(); actual = summary(module.build_component_runs(components, params, minimum_frames))
                    t2 = time.perf_counter()
                times['original'] += t1 - t0; times['adapted'] += t2 - t1
                rows.append(dict(case=f'{name}-{number}-{minimum_frames}', runs=len(expected),
                                 stationary=sum(r[3] for r in expected)))
                if expected != actual:
                    failures.append(f'runs {name}-{number}-{minimum_frames}')
    assert stationary_gap_recovery.build_component_runs is module.build_component_runs
    report = dict(status='pass' if not failures else 'fail', pairs=pairs, seconds=times, rows=rows,
                  failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
