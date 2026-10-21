"""Compare memoised gap-tolerant bracket searches with the frozen _bracket, exactly.

Random searches on the real step-87 attached point table: real and absent
tracks, owner pairs taken from the table and arbitrary, forward and backward
frame ranges past both movie ends, empty ranges, frame lists, and several purity/coverage thresholds. The
same searches also run through the engine's reciprocal lookup cache.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def same(expected, actual):
    if expected is None or actual is None:
        return expected is None and actual is None
    return (type(expected[0]) is type(actual[0]) and expected[0] == actual[0]
            and expected[1].equals(actual[1]) and expected[2].equals(actual[2]))


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stack', default='87_well_b2_issue018_recording_start_two_seat_inheritance.tif')
    p.add_argument('--searches', type=int, default=20000)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import gap_tolerant_reciprocal_seat_exchange as module
    import gap_bracket_memo as memo
    accepted = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(accepted / a.stack)
    points = pd.read_csv(accepted / '26_latent_body_tracks/out/latent_track_points.csv')
    attached = module.attach_owners(points, labels).sort_values(["frame", "track_id"]).reset_index(drop=True)
    index = attached.set_index(["track_id", "frame"], drop=False)
    rng = np.random.default_rng(8)
    tracks = attached.track_id.astype(int).unique()
    owners = attached.candidate_owner.astype(int).unique()
    frames = len(labels)
    searches = []
    for _ in range(a.searches):
        if rng.random() < 0.7:
            row = attached.iloc[int(rng.integers(len(attached)))]
            frame = int(row.frame)
            others = attached[attached.frame == frame]
            other = others.iloc[int(rng.integers(len(others)))]
            pair = (int(row.track_id), int(other.track_id))
            owner_pair = (int(other.candidate_owner), int(row.candidate_owner))
        else:
            frame = int(rng.integers(0, frames))
            pair = (int(rng.choice(tracks)), int(rng.integers(0, 4000)))
            owner_pair = (int(rng.choice(owners)), int(rng.integers(0, 600)))
        gap = int(rng.choice([1, 3, 15, 40]))
        span = (range(frame - 1, max(-1, frame - gap - 1), -1) if rng.random() < 0.5
                else range(frame + 1, min(frames + 3, frame + gap + 1)))
        if rng.random() < 0.05:
            span = list(span)  # a frame list takes the stepwise search
        elif rng.random() < 0.01:
            span = range(frame, frame)  # empty
        threshold = [(0.8, 0.5), (0.0, 0.0), (0.95, 0.9)][int(rng.choice(3, p=[0.8, 0.1, 0.1]))]
        searches.append((*pair, *owner_pair, span, *threshold))
    tick = time.perf_counter()
    expected = [module._bracket(index, *s) for s in searches]
    before = time.perf_counter() - tick
    with memo.install(module) as stats:
        assert stats['supported']
        tick = time.perf_counter()
        actual = [module._bracket(index, *s) for s in searches]
        after = time.perf_counter() - tick
        repeated = [module._bracket(index, *s) for s in searches]
        # Through the engine's private-namespace clone with cached _point/_valid.
        from reciprocal_lookup_speed import cached_reciprocal_discovery
        exec("def _probe_discover(index, searches):\n"
             "    return [_bracket(index, *s) for s in searches], _bracket", module.__dict__)
        saved = module.discover
        module.discover = module._probe_discover
        try:
            with cached_reciprocal_discovery(module):
                cloned, clone = module.discover(index, searches)
        finally:
            module.discover = saved
            del module._probe_discover
        assert clone.__code__ is memo._bracket.__code__ and clone.__globals__ is not module.__dict__
    found = sum(e is not None for e in expected)
    assert all(same(e, f) for e, f in zip(expected, actual))
    assert all(same(e, f) for e, f in zip(expected, repeated))
    assert all(same(e, f) for e, f in zip(expected, cloned))
    report = dict(status='pass', searches=len(searches), brackets_found=found,
                  before_seconds=before, after_seconds=after, memo=dict(stats['memo']))
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
