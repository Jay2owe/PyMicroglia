"""Compare occupancy reservations with and without the overlap-box adapter, exactly.

Random one-frame contacts (shifted copies of state masks, fresh blobs, masks on
the frame edge), random gap steps and gates, and states or observations
without bounds are reserved by the frozen tracking._occupancy_reservations and
by the installed adapter; the returned links must have the same repr.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--trials', type=int, default=6000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    sys.path[:0] = [str((a.engine / 'code').resolve()), str(a.adapter.resolve())]
    import tracking
    import occupancy_overlap_boxes as adapter
    original = tracking._occupancy_reservations
    rng = np.random.default_rng(5)

    def blob(shape):
        mask = np.zeros(shape, bool)
        y, x = int(rng.integers(0, shape[0] - 2)), int(rng.integers(0, shape[1] - 2))
        h, w = int(rng.integers(1, 12)), int(rng.integers(1, 12))
        mask[y:y + h, x:x + w] = rng.random((len(range(y, min(y + h, shape[0]))),
                                             len(range(x, min(x + w, shape[1]))))) < .85
        if not mask.any():
            mask[y, x] = True
        return mask

    def bounds_of(mask):
        ys, xs = np.nonzero(mask)
        return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)

    cases = []
    for trial in range(a.trials):
        shape = (int(rng.integers(12, 48)), int(rng.integers(12, 48)))
        n_states, n_observations = int(rng.integers(0, 9)), int(rng.integers(0, 9))
        missing = trial % 7 == 0
        states = []
        for index in range(n_states):
            mask = blob(shape)
            bounds = None if missing and rng.random() < .3 else bounds_of(mask)
            states.append(tracking.IdentityState(index + 1, 0, mask, np.zeros(2), int(mask.sum()), 1.0,
                                                 np.zeros(2), set(), bounds=bounds))
        observations = []
        for index in range(n_observations):
            mask = blob(shape)
            if states and rng.random() < .7:
                source = states[int(rng.integers(0, n_states))].mask
                mask = np.roll(np.roll(source, int(rng.integers(-2, 3)), 0), int(rng.integers(-2, 3)), 1)
                if not mask.any():
                    mask = blob(shape)
            observation = {'mask': mask, 'area': int(mask.sum()), 'local_id': index + 1}
            if not (missing and rng.random() < .3):
                observation['bounds'] = bounds_of(mask)
            observations.append(observation)
        features = {}
        for row in range(n_states):
            gap = int(rng.choice([1, 1, 1, 2, 0]))
            for col in range(n_observations):
                features[row, col] = {'gap_steps': gap, 'allowed': True, 'cost': float(rng.random())}
        params = dict(occupancy_reservation=bool(trial % 11),
                      occupancy_minimum_intersection_px=int(rng.integers(1, 6)),
                      occupancy_minimum_intersection_over_union=float(rng.choice([0.0, 0.1, 0.5])),
                      occupancy_minimum_bidirectional_coverage=float(rng.choice([0.0, 0.3, 0.7])),
                      occupancy_minimum_iou_margin=float(rng.choice([0.0, 0.2])),
                      occupancy_maximum_competitor_coverage=float(rng.choice([0.2, 1.0])))
        cases.append((states, observations, features, params))
    tick = time.perf_counter()
    expected = [repr(original(*case)) for case in cases]
    middle = time.perf_counter()
    with adapter.install(tracking) as stats:
        assert stats['supported'] and tracking._occupancy_reservations.__wrapped__ is original
        actual = [repr(tracking._occupancy_reservations(*case)) for case in cases]
    end = time.perf_counter()
    assert tracking._occupancy_reservations is original
    mismatches = sum(x != y for x, y in zip(expected, actual))
    assert mismatches == 0, mismatches
    report = dict(status='pass', trials=len(cases), nonempty=sum(x != '[]' for x in expected),
                  links=sum(x.count('(') for x in expected), seconds=dict(original=middle - tick, adapted=end - middle),
                  stats=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
