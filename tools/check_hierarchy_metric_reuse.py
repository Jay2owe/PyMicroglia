"""Check reused fractions and disjoint-range overlap against frozen metrics."""
import argparse
import json
from pathlib import Path
import platform
import sys

platform.platform = lambda *a, **k: 'Windows-hierarchy-metric-check'
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.engine.resolve()/'code'), str(args.adapter.resolve())]
    import accepted_history as history
    import hierarchy_metric_reuse as reuse
    module = history._issue_module(history.I009, 'm6_hierarchy_reconcile')
    geometry = module.geometry
    original, original_run, original_overlap = module.metrics, module.run, geometry.overlap
    rng = np.random.default_rng(903)
    lag = rng.normal(0, 3, (6, 31, 37)).astype(np.float32)
    count = 0
    def probe():
        nonlocal count
        for t in range(len(lag)):
            components = []
            for identity in range(24):
                pixels = np.sort(rng.choice(lag[t].size, size=identity+1, replace=False))
                components.append(geometry.Component(t, identity, 1, pixels, len(pixels),
                    float((pixels % 37).mean()), float((pixels // 37).mean())))
            for threshold in (1.5, 0.0, 2.0):
                for old in components:
                    for new in components:
                        expected = original(old, new, lag[t], threshold, lag[t].shape)
                        actual = module.metrics(old, new, lag[t], threshold, lag[t].shape)
                        assert expected.keys() == actual.keys()
                        for key in expected:
                            if isinstance(expected[key], float):
                                assert np.float64(expected[key]).tobytes() == np.float64(actual[key]).tobytes()
                            else:assert expected[key] == actual[key]
                        assert original_overlap(old, new) == geometry.overlap(old, new)
                        count += 1
    module.run = probe
    try:
        with reuse.install(module) as counters:
            assert counters['supported']
            module.run()
        assert module.metrics is original and geometry.overlap is original_overlap
    finally:
        module.run = original_run
    report = dict(status='pass', exact_pairs=count, counters=counters,
                  scope='Exact fractions, geometry and overlap; threshold/frame changes; context restoration')
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
