"""Adversarial differential checks for operation-local duplicate history reuse."""
import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    platform.platform = lambda *a, **k: 'Windows-duplicate-differential-check'
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import numpy as np
    import pandas as pd
    m = importlib.import_module('duplicate_soma_exclusivity')
    adapter = importlib.import_module('identity_reuse')
    rng = np.random.default_rng(127)
    cases = 0
    for shape in ((1, 1), (9, 11), (31, 29)):
        for repetition in range(8):
            labels = rng.integers(0, 5, size=(4, *shape), dtype=np.int32)
            rows = [dict(frame=f, accepted_owner=owner, physically_visible=True,
                         x=float(rng.uniform(-4, shape[1]+4)),
                         y=float(rng.uniform(-4, shape[0]+4)), radius_px=float(rng.uniform(1, 4)),
                         evidence=float(rng.integers(1, 4)), track_id=i)
                    for f in range(4) for owner in range(1, 5) for i in range(3)]
            points = pd.DataFrame(rows)
            for minimum in (1, 4):
                frames = None if repetition % 2 else np.array([3, 0, 2])
                expected = m._components(labels, points, minimum, frames)
                with adapter.install(m):
                    actual = m._components(labels, points, minimum, frames)
                assert list(expected) == list(actual)
                for key, values in expected.items():
                    assert len(values) == len(actual[key])
                    for left, right in zip(values, actual[key]):
                        for field in dataclasses.fields(left):
                            l, r = getattr(left, field.name), getattr(right, field.name)
                            assert np.array_equal(l, r), (key, field.name, l, r)
                cases += 1
    original_recover = m.recover
    original_cost = m._match_cost
    cost_calls = [0]
    def counted_cost(*args):
        cost_calls[0] += 1
        return original_cost(*args)
    m._match_cost = counted_cost
    timings = []
    try:
        for frames in (24, 96, 476):
            components = {}
            for frame in range(frames):
                for owner in (1, 18):
                    # Ties, absent frames and branches exercise stable choice and gap search.
                    components[(frame, owner)] = [] if frame % 11 == 4 else [
                        m.Component(frame, owner, i+1, np.zeros((1,1), dtype=np.int32), i+1,
                                    12, float(i*5), float(frame % 3), 2., (i,), 1., float(i*5), 0., 2.)
                        for i in range(3)]
            seeds = [part for values in components.values() for part in values]
            def trace_all(params):
                return [[(v.frame, v.owner, v.index) for v in m._trace(
                    seed, components, frames, params)] for seed in seeds]
            m.recover = trace_all
            params = dict(maximum_path_gap_fraction_of_movie=.04, maximum_path_step_cost=2.75)
            cost_calls[0] = 0
            start = time.perf_counter()
            expected = m.recover(params)
            old_seconds, old_calls = time.perf_counter()-start, cost_calls[0]
            cost_calls[0] = 0
            start = time.perf_counter()
            with adapter.install(m) as counters:
                actual = m.recover(params)
                # Each call must rebuild links even when the caller edits evidence/settings.
                for values in components.values():
                    for part in values:
                        part.x *= 3
                altered = m.recover(dict(params, maximum_path_step_cost=.3))
            candidate_seconds = time.perf_counter()-start
            assert actual == expected
            assert altered == m.recover(dict(params, maximum_path_step_cost=.3))
            timings.append(dict(frames=frames, baseline_seconds=old_seconds,
                baseline_match_calls=old_calls, candidate_seconds_two_calls=candidate_seconds,
                candidate_match_calls_two_calls=counters['match_cost_calls'], **counters))
            cases += 2
    finally:
        m.recover, m._match_cost = original_recover, original_cost
    report = dict(status='pass', differential_cases=cases, growth_checks=timings)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
