"""Compare the complete-flip nested-alias search with per-source reuse.

Synthetic movies with recurrent nested names, and a real accepted label stack
(plain and disk-backed) with a point table of every cell's centroid, are run
through the engine's _nested_alias_proposals and the adapted copy at several
thresholds. Proposal lists must be equal.
"""
import argparse, json, sys, time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def points_for(labels, rng):
    rows = []
    for t in range(len(labels)):
        frame = np.asarray(labels[t])
        for identity in np.unique(frame)[1:]:
            yy, xx = np.nonzero(frame == identity)
            rows.append(dict(frame=t, candidate_owner=int(identity), track_id=int(identity),
                             strong=bool(rng.random() < .8), x=float(xx.mean()), y=float(yy.mean()),
                             radius_px=4.0))
    return pd.DataFrame(rows)


def synthetic(rng):
    frames = int(rng.integers(12, 30))
    labels = np.zeros((frames, 40, 50), np.uint16)
    for t in range(frames):
        labels[t, 5:15, 5:15] = 1
        if (t // 3) % 2 and 2 < t < frames - 3:
            labels[t, 5:15, 5:15] = 2
            if rng.random() < .3:
                labels[t, 5:8, 5:8] = 1
        y, x = rng.integers(18, 34), rng.integers(18, 44)
        labels[t, y:y + 5, x:x + 5] = 3 + int(rng.integers(0, 2))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import complete_flip_recovery as module
    import nested_alias_reuse as adapter
    import array_storage
    original = module._nested_alias_proposals
    failures, rows = [], []
    times = dict(original=0.0, adapted=0.0)
    settings = [{}, dict(minimum_nested_predecessor_movie_fraction=0.1, minimum_nested_source_runs=2,
                         minimum_nested_joint_coverage=0.5, maximum_nested_cooccurrence_movie_fraction=0.5,
                         minimum_nested_strong_fraction=0.1)]

    def compare(labels, visible, params, field_step, label):
        t0 = time.perf_counter(); expected = original(labels, visible, params, field_step)
        with adapter.install(module) as stats:
            assert stats['supported']
            t1 = time.perf_counter(); actual = module._nested_alias_proposals(labels, visible, params, field_step)
            t2 = time.perf_counter()
        assert module._nested_alias_proposals is original
        times['original'] += t1 - t0; times['adapted'] += t2 - t1
        accepted = sum(not row['_local_reasons'] for row in expected)
        rows.append(dict(case=label, proposals=len(expected), accepted=accepted))
        if expected != actual:
            failures.append(label)

    rng = np.random.default_rng(20261020)
    for case in range(60):
        labels = synthetic(rng)
        visible = points_for(labels, rng)
        compare(labels, visible, settings[case % 2], float(rng.choice([0.5, 2.0, 50.0])), f'synthetic-{case}')
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    visible = points_for(stack, rng)
    for number, params in enumerate(settings):
        compare(stack, visible, params, 1.5, f'real-plain-{number}')
    with ExitStack() as scope:
        pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
        scope.enter_context(array_storage.install(pool))
        working = np.array(stack)
        assert isinstance(working, array_storage.WorkingArray)
        compare(working, visible, settings[1], 1.5, 'real-working-1')
    # Boundary steps one at a time against the engine's own step: real
    # neighbouring frames (cells, absent identities, the same identity) and
    # random multi-part frames, with one memo shared by all calls on a stack.
    steps = 0
    for movie in (stack, (rng.integers(0, 5, (6, 30, 40)) * (rng.random((6, 30, 40)) < .35)).astype(np.int32)):
        memo = {}
        for _ in range(2500 if movie is stack else 600):
            t = int(rng.integers(0, len(movie) - 1))
            ids = list(map(int, np.unique(movie[t])[1:])) + list(map(int, np.unique(movie[t + 1])[1:])) + [0, 99999]
            first, second = int(rng.choice(ids)), int(rng.choice(ids))
            left, right = (t, t + 1) if rng.random() < .5 else (t + 1, t)
            expected = module._mask_boundary_step(movie[left] == first, movie[right] == second)
            actual = adapter.boundary_step(movie, left, first, right, second, memo, module._mask_boundary_step)
            steps += 1
            if not (expected == actual and type(expected) is type(actual)):
                failures.append(f'step-{left}-{first}-{right}-{second}')
    rows.append(dict(case='boundary-steps', proposals=steps, accepted=0))
    report = dict(status='pass' if not failures else 'fail', seconds=times, boundary_steps=steps,
                  real=[r for r in rows if r['case'].startswith('real')],
                  synthetic=dict(cases=sum(r['case'].startswith('synthetic') for r in rows),
                                 proposals=sum(r['proposals'] for r in rows if r['case'].startswith('synthetic')),
                                 accepted=sum(r['accepted'] for r in rows if r['case'].startswith('synthetic'))),
                  failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
