"""Compare lazy-mask motion evidence including nonfinite signal and cache scope."""
import argparse
import json
from pathlib import Path
import platform
import sys


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    platform.platform = lambda *a, **k: 'Windows-lazy-motion-check'
    sys.path[:0] = [str(a.engine.resolve()/'code'), str(a.adapter.resolve())]
    import numpy as np
    import tracking
    import lazy_motion_counts
    from pymicroglia.tracking.frozen_io import LabelMask
    rng = np.random.default_rng(2016)
    original = tracking.motion_support
    comparisons = 0
    import tempfile
    holder = tempfile.TemporaryDirectory(prefix='lazy-motion-', ignore_cleanup_errors=True)
    scratch = holder.name
    with lazy_motion_counts.install(tracking, LabelMask) as stats:
        assert stats['supported']
        for i in range(120):
            labels = rng.integers(0, 5, (31, 37), dtype=np.uint16)
            later = rng.integers(0, 5, (31, 37), dtype=np.uint16)
            if i % 3 == 0:
                labels[:, 15:] = 0
                later[:, :20] = 0
            if i % 4 == 0:
                labels = labels.T
                later = later.T
            lag = rng.choice([np.nan, np.inf, -np.inf, -1.5, -1., 0., 1., 1.5, 2.], labels.shape)
            mapped = None
            if i % 2:
                # A disk-backed transition image, as in the bounded runner.
                mapped = np.memmap(Path(scratch)/f'lag_{i}.dat', mode='w+', dtype=lag.dtype, shape=lag.shape)
                mapped[...] = lag
            for old_id in range(1, 5):
                for new_id in range(1, 5):
                    old, new = labels == old_id, later == new_id
                    def bounds(mask):
                        y, x = np.nonzero(mask)
                        return (int(y.min()), int(y.max())+1, int(x.min()), int(x.max())+1)
                    old_bounds, new_bounds = bounds(old), bounds(new)
                    left, right = LabelMask(labels, old_id), LabelMask(later, new_id)
                    expected = original(left, right, lag, old_bounds, new_bounds)
                    signal = lag if mapped is None else mapped
                    cache = {}
                    for first, second in ((left,right), (old,right), (left,new), (old,new)):
                        for repeat in range(2):
                            actual = tracking.motion_support(first, second, signal, old_bounds, new_bounds, cache)
                            assert expected == actual, (i, old_id, new_id, expected, actual)
                            for key, value in expected.items():
                                if isinstance(value, float):
                                    assert np.float64(value).tobytes() == np.float64(actual[key]).tobytes()
                            comparisons += 1
    assert tracking.motion_support is original
    del mapped, signal
    import gc; gc.collect()
    holder.cleanup()
    a.output.write_text(json.dumps(dict(status='pass', comparisons=comparisons, counters=stats,
        scope='Exact lazy and ordinary masks, plain and disk-backed transition images, thresholds, nonfinite signals, disjoint bounds, transposes, repeat caches'),indent=2)+'\n')
    print(json.dumps(dict(status='pass', comparisons=comparisons)))


if __name__ == '__main__':
    main()
