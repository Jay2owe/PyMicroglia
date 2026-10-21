"""Compare matrix link rows with the per-state Batch.build and prepare, exactly.

Every pair tuple (values and types), every prepared admission dictionary and
every far-pair decision must match for random assignments built from the
configured tracking parameters with edge cases: reserved and preferred
identities, missing reference values, computed shape descriptors, float32 and
integer scalars, zero areas, gaps, shared green tracks, retained-source
constraints and confidence sets.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time


def same(a, b):
    import numpy as np
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, float):
        return a == b or (a != a and b != b)
    return a == b


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--assignments', type=int, default=1500)
    a = p.parse_args()
    import numpy as np
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import tracking
    import common
    import tracking_feature_batch as tfb
    layered = common.Config.load(a.config).values['accepted_postprocessing']['layered_lineage']
    bases = [dict(layered['tracking']), dict(layered['comparison_tracking'])]
    rng = np.random.default_rng(12)
    report = dict(assignments=0, states=0, pairs=0, plain=0, far=0, matrix_fallbacks=0)
    tick = time.perf_counter()
    for number in range(a.assignments):
        params = dict(bases[number % 2])
        if rng.random() < 0.3:
            params['motion_reservation_identities'] = [int(v) for v in rng.integers(1, 30, 4)]
        if rng.random() < 0.3:
            params['high_confidence_identity_ids'] = [int(v) for v in rng.integers(1, 30, 6)]
            params['low_confidence_preferred_identity_reward'] = 0.2
        if rng.random() < 0.3:
            params['reversal_aware_motion'] = True
        if rng.random() < 0.2:
            params['maximum_recovery_distance_px'] = float(rng.choice([5.0, 30.0, np.inf]))
        target = int(rng.integers(5, 400))
        n_obs = int(rng.integers(1, 45))
        observations = []
        for k in range(n_obs):
            observations.append({
                'position': rng.uniform(0, 300, 2), 'area': float(rng.choice([0.0, 1.0, rng.uniform(5, 400)])),
                'mean': float(rng.uniform(0, 3000)), 'shape': rng.uniform(0, 4, 3),
                'preferred_identity': int(rng.choice([0, 0, int(rng.integers(1, 30))])),
                'local_id': k + 1, 'green_track_ids': set(int(v) for v in rng.integers(0, 12, int(rng.integers(0, 3))))})
        if rng.random() < 0.3:
            params['_retained_source_constraints'] = {(int(rng.integers(1, 30)), int(rng.integers(1, n_obs + 1))): {'x': 1}
                                                      for _ in range(3)}
        states = []
        for k in range(int(rng.integers(1, 60))):
            mask = np.zeros((20, 20), bool)
            mask[rng.integers(0, 20, 12), rng.integers(0, 20, 12)] = True
            mean = rng.uniform(0, 3000)
            states.append(tracking.IdentityState(
                identity=int(rng.integers(1, 30)), t=target - int(rng.choice([1, 1, 1, 2, 5, -1, -3])),
                mask=mask, position=rng.uniform(0, 300, 2),
                area=int(rng.choice([0, 1, int(rng.integers(5, 400))])) if rng.random() < 0.8 else float(rng.uniform(0, 400)),
                mean=np.float32(mean) if rng.random() < 0.3 else float(mean),
                velocity=rng.normal(0, 4, 2) * (rng.random() < 0.8),
                green_track_ids=set(int(v) for v in rng.integers(0, 12, int(rng.integers(0, 3)))),
                reference_area=float(rng.uniform(5, 400)) if rng.random() < 0.6 else np.nan,
                reference_mean=float(rng.uniform(0, 3000)) if rng.random() < 0.6 else np.nan,
                reference_shape=rng.uniform(0, 4, 3) if rng.random() < 0.6 else None))
        counters = defaultdict(int)
        old = tfb.Batch(observations, target, params, tracking, counters)
        new = tfb.Batch(observations, target, params, tracking, counters)
        for batch in (old, new):
            batch.defer_far = True; batch.cache_wrapped = False
            batch.cache_parameter_keys = ()
        new.matrix_states = list(states)
        for state in states:
            old.build(state)
            new.build(state)
            if new.matrix_row is None:
                report['matrix_fallbacks'] += 1
            assert len(old.rows) == len(new.rows)
            for i in range(len(old.rows)):
                e, f = old.rows[i], new.rows[i]
                assert len(e) == len(f) and all(same(x, y) for x, y in zip(e, f)), (number, i, e, f)
                if new.matrix_row is not None:
                    assert bool(new.matrix.far[new.matrix_row, i]) == (e[5] > e[7])
                    report['far'] += int(e[5] > e[7])
            assert old.plain.keys() == new.plain.keys(), (number, sorted(old.plain), sorted(new.plain))
            for key in old.plain:
                assert old.plain[key].keys() == new.plain[key].keys()
                assert all(same(old.plain[key][k], new.plain[key][k]) for k in old.plain[key]), (number, key)
            report['pairs'] += len(old.rows)
            report['plain'] += len(old.plain)
            report['states'] += 1
        report['assignments'] += 1
    assert report['matrix_fallbacks'] < report['states'] * 0.05, report
    report.update(status='pass', seconds=time.perf_counter() - tick)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
