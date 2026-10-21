"""Compare cached component geometry and complete restored paths exactly."""
import argparse
import json
from pathlib import Path
import platform
import sys

platform.platform = lambda *a, **k: 'Windows-dormant-path-check'
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'report'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.engine.resolve()/'code'))
    sys.path.insert(0, str(a.adapter.resolve()))
    import accepted_history as history
    import dormant_path_reuse as reuse
    module = history._issue_module(history.I012, 'm9_reconcile_dormant_releases')
    original = module.reconcile
    rng = np.random.default_rng(1013)
    checks = []
    for case in range(8):
        labels = np.zeros((16, 41, 53), np.int32)
        for t in range(len(labels)):
            labels[t, 3:9, 3+t:9+t] = 1 if t < 3 else 3
            labels[t, 13:20, 16:24] = 2
            labels[t, 35:41, 47:53] = 9000001
            for _ in range(7):
                y, x = rng.integers(0, 39), rng.integers(0, 51)
                labels[t, y:y+2, x:x+2] = rng.integers(4, 8)
        if case % 2:
            labels = labels.transpose(0, 2, 1)
        if case == 6:
            labels[:] = 0
        stats = dict(queries=0, hits=0, builds=0, frame_indexes=0, peak_cache_bytes=0)
        cache = reuse.OutlineCache(labels, module.ndi, module.components, stats, budget=32768)
        for t in range(len(labels)):
            for identity in (0, 1, 2, 3, 7, 100, 9000001):
                for minimum in (1, 4, 12):
                    expected = module.components(labels[t], identity, minimum)
                    actual = cache.components(labels[t], identity, minimum)
                    assert len(expected) == len(actual)
                    for left, right in zip(expected, actual):
                        assert {k:v for k,v in left.items() if k != 'mask'} == {k:v for k,v in right.items() if k != 'mask'}
                        assert np.array_equal(left['mask'], right['mask'])
        raw = rng.normal(100, 10, labels.shape).astype(np.float32)
        ledger = np.zeros_like(labels)
        selected = pd.DataFrame([dict(event_id='first', event_type='weak_name_takeover',
            identity=1, candidate_identity=3, host_identity=3,
            start_source_imagej_frame=4, release_source_imagej_frame=4,
            established_area_px=36, established_raw_mean=100)]
            # A second path over the same frames reuses each frame's component list.
            + ([dict(event_id='second', event_type='weak_name_takeover',
                     identity=2, candidate_identity=2, host_identity=2,
                     start_source_imagej_frame=6, release_source_imagej_frame=6,
                     established_area_px=56, established_raw_mean=100)] if case % 3 else []))
        if case == 7:
            selected = selected.iloc[:0]
        cfg = dict(minimum_component_area_px=1, component_search_radius_px=20,
                   maximum_track_gap_frames=3, maximum_component_cost=5,
                   minimum_track_area_ratio=0.01, maximum_track_area_ratio=3)
        frozen = labels.copy()
        expected = original(labels, ledger, raw, selected, 0, cfg)
        with reuse.install(module) as counters:
            assert counters['supported']
            actual = module.reconcile(labels, ledger, raw, selected, 0, cfg)
        assert module.reconcile is original
        assert np.array_equal(labels, frozen)
        for old, new in zip(expected[:2], actual[:2]):
            assert np.array_equal(old, new)
        pd.testing.assert_frame_equal(expected[2], actual[2], check_exact=True)
        assert expected[2].to_csv(index=False) == actual[2].to_csv(index=False)
        checks.append(dict(case=case, actions=len(actual[2]), counters=counters))
    report = dict(status='pass', cases=checks,
                  scope='Exact geometry, borders, sparse identity values, transposed inputs, empty inputs, complete paths and actions')
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='pass', cases=len(checks))))


if __name__ == '__main__':
    main()
