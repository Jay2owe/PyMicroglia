"""Compare complete tracker assignments with and without the far-pair table, exactly.

The engine's own _assign (no adapters) is the reference. Random frames give
real observations and states with masks and bounds; settings start from the
configured layered-lineage tracking and comparison tracking and vary the
recovery and fast-motion bounds, red-led thresholds and state ages,
occupancy reservation, high-confidence names, small-cell quarantine and
confidence layers, retained-source constraints and gaps. Links (row, column
and every feature value), unmatched states and observations, and the
rejection audit must match.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time


def exact(a, b, where=()):
    import numpy as np
    if isinstance(a, float) or isinstance(b, float):
        assert type(a) is type(b) and np.float64(a).tobytes() == np.float64(b).tobytes(), (where, a, b)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape, where
        assert np.array_equal(a, b, equal_nan=True), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), (where, list(a), list(b))
        for key in a:
            exact(a[key], b[key], where + (key,))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), (where, a, b)
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, where + (i,))
    else:
        assert type(a) is type(b) and a == b, (where, a, b)


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cases', type=int, default=400)
    p.add_argument('--only', type=int, help='compare one case number (diagnosis)')
    p.add_argument('--no-sparse', action='store_true', help='disable the far-pair table (diagnosis)')
    a = p.parse_args()
    import numpy as np
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import common
    import tracking as module
    import tracking_feature_batch as adapter
    if a.no_sparse:
        try:
            import sparse_pair_table
        except ImportError:
            sparse_pair_table = type(sys)("sparse_pair_table")
        sparse_pair_table.install = lambda *args, **kwargs: (lambda: None)
    layered = common.Config.load(a.config.resolve()).values['accepted_postprocessing']['layered_lineage']
    bases = [dict(layered['tracking']), dict(layered['comparison_tracking'])]
    rng = np.random.default_rng(4401)
    report = dict(cases=0, links=0, pairs=0, sparse_far_pairs=0, materialized=0, layered=0, red_led=0)
    before = after = 0.0
    original_assign = module._assign

    def frame(count, shape):
        labels = np.zeros(shape, np.uint16)
        for i in range(1, count + 1):
            y, x = int(rng.integers(0, shape[0] - 6)), int(rng.integers(0, shape[1] - 6))
            h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
            labels[y:y + h, x:x + w] = i
        return labels

    for case in range(a.cases):
        shape = (int(rng.integers(60, 160)), int(rng.integers(60, 160)))
        target = 4
        current = frame(int(rng.integers(1, 45)), shape)
        earlier = current.copy() if rng.random() < 0.5 else frame(int(rng.integers(1, 45)), shape)
        if rng.random() < 0.5:
            earlier = np.roll(earlier, int(rng.integers(-3, 4)), axis=int(rng.integers(2)))
        raw = rng.uniform(1, 100, shape)
        observations = list(module.frame_observations(current, raw).values())
        previous = list(module.frame_observations(earlier, raw).values())
        if not observations or not previous:
            continue
        for o in observations:
            o['green_track_ids'] = {int(v) for v in rng.integers(0, 6, int(rng.integers(0, 3)))}
            o['preferred_identity'] = int(rng.choice([0, int(o['local_id']), int(rng.integers(1, 50))]))
            if rng.random() < 0.1:
                o['preferred_identity_protected'] = True
        # States lie on the tracking direction's earlier side, so every
        # one-frame pair shares one transition image, as in track_direction.
        direction = 1 if rng.random() < 0.5 else -1
        states = []
        for i, o in enumerate(previous):
            states.append(module.IdentityState(
                identity=int(o['local_id']) if rng.random() < 0.8 else int(rng.integers(1, 50)),
                t=target - direction * int(rng.choice([1, 1, 1, 1, 2, 4])), mask=o['mask'],
                position=o['position'] + rng.normal(0, 2, 2), area=o['area'], mean=o['mean'],
                velocity=rng.normal(0, 3, 2) * (rng.random() < 0.8),
                green_track_ids={int(v) for v in rng.integers(0, 6, int(rng.integers(0, 3)))},
                age=int(rng.integers(0, 5)), bounds=o['bounds'],
                reference_area=float(o['area']) if rng.random() < 0.6 else np.nan,
                reference_mean=float(o['mean']) if rng.random() < 0.6 else np.nan,
                reference_shape=o['shape'] if rng.random() < 0.6 else None))
        lag = rng.choice([-2.0, -1.6, -0.5, 0.0, 0.5, 1.6, 2.0, np.nan], (8,) + shape).astype(np.float32)
        params = dict(bases[case % 2])
        params.update(miss_cost=1.8, birth_cost=1.8, minimum_area_ratio=0.1, maximum_area_ratio=10.0,
                      maximum_mean_log2_change=4.0, motion_weight=2.0, _soma_rejection_events=[])
        params['retain_independently_observed_soma'] = False
        params['maximum_recovery_distance_px'] = float(rng.choice([8.0, 15.0, 30.0, 60.0]))
        if rng.random() < 0.3:
            params['fast_motion_max_link_px'] = float(rng.choice([0.0, 10.0, 40.0, 120.0]))
        if rng.random() < 0.6:
            params['red_led_motion'] = True
            if rng.random() < 0.3:
                for key in ('red_led_minimum_source_loss_coverage', 'red_led_minimum_destination_gain_coverage',
                            'red_led_high_minimum_source_loss_coverage', 'red_led_high_minimum_destination_gain_coverage'):
                    params[key] = float(rng.choice([0.0, -0.1, 0.7]))
                params['red_led_minimum_supportive_px'] = int(rng.choice([0, 12]))
            params['red_led_minimum_state_age'] = int(rng.choice([0, 1, 3]))
        else:
            params['red_led_motion'] = False
        if rng.random() < 0.4:
            params['occupancy_reservation'] = True
            params['occupancy_minimum_intersection_px'] = 2
        if rng.random() < 0.5:
            params['high_confidence_identity_ids'] = [int(v) for v in rng.integers(1, 50, 8)]
            params['small_cell_quarantine'] = bool(rng.random() < 0.5)
        if rng.random() < 0.4:
            params['confidence_layer_order'] = ['large', 'medium', 'small_or_dim']
            params['confidence_layer_by_identity'] = {int(v): str(rng.choice(['large', 'medium', 'small_or_dim']))
                                                      for v in rng.integers(1, 50, 20)}
            params['small_cell_quarantine'] = bool(rng.random() < 0.5)
            report['layered'] += 1
        if rng.random() < 0.3:
            params['motion_reservation_identities'] = [int(v) for v in rng.integers(1, 50, 4)]
        if rng.random() < 0.3:
            params['_retained_source_constraints'] = {
                (int(rng.integers(1, 50)), int(rng.integers(1, len(observations) + 1))): {'reason': 'fixture'}
                for _ in range(4)}
        report['red_led'] += int(bool(params.get('red_led_motion')))
        if a.only is not None and case != a.only:
            continue
        expected_params = deepcopy(params)
        tick = time.perf_counter()
        expected = original_assign(states, observations, target, lag, direction, expected_params)
        before += time.perf_counter() - tick
        actual_params = deepcopy(params)
        with adapter.install(module) as stats:
            assert stats['supported'] and (a.no_sparse or stats.get('sparse_supported')), stats
            tick = time.perf_counter()
            result = module._assign(states, observations, target, lag, direction, actual_params)
            links = [(row, col, dict(value)) for row, col, value in result[0]]
            after += time.perf_counter() - tick
        assert module._assign is original_assign
        if a.only is not None:
            for (r0, c0, v0), (r1, c1, v1) in zip(expected[0], links):
                v0 = dict(v0)
                diff = {k: (v0.get(k), v1.get(k)) for k in set(v0) | set(v1) if repr(v0.get(k)) != repr(v1.get(k))}
                if diff or (r0, c0) != (r1, c1):
                    print('link', (r0, c0), (r1, c1), diff, 'state', states[r0].identity, states[r0].t, 'preferred', observations[c0].get('preferred_identity'))
            print({k: v for k, v in params.items() if not k.startswith('_') or k == '_retained_source_constraints'})
        exact([(row, col, dict(value)) for row, col, value in expected[0]], links, (case, 'links'))
        assert expected[1] == result[1] and expected[2] == result[2], case
        exact(expected_params['_soma_rejection_events'], actual_params['_soma_rejection_events'], (case, 'audit'))
        assert sorted(expected_params) == sorted(actual_params), case
        report['cases'] += 1
        report['links'] += len(links)
        report['pairs'] += len(states) * len(observations)
        report['sparse_far_pairs'] += stats['sparse_far_pairs']
        report['materialized'] += stats['deferred_materialized']
    if a.only is None and not a.no_sparse:
        assert report['sparse_far_pairs'] > report['pairs'] * 0.2, report
    report.update(status='pass', before_seconds=before, after_seconds=after)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
