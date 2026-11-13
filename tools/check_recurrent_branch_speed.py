"""Compare recurrent_branch_inventory.scan with its kept-tree/gap and block-absorption rewrite, exactly.

The whole accepted A1 movie with its registered raw movie and tracker points,
a version with names swapped in a band of frames, a memmapped copy and
shuffled point rows. Both returned tables must be identical (values, dtypes,
order and CSV text), far-box skips, reused gaps and block reads must occur,
and the adapter must restore the module.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'correction-root', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve()), str(a.adapter.resolve())]
    import common
    import pipeline
    sys.path.insert(0, str((a.correction_root / 'code').resolve()))
    import recurrent_branch_inventory as inventory
    import recurrent_branch_speed as speed
    cfg = common.Config.load(a.config)
    run = a.engine / 'm22_accepted_history' / a.run
    labels = tifffile.imread(run / 'out' / f'{a.stem}.tif')
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[:len(labels)]
    points = pd.read_csv(run / 'mid/accepted_history/26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    rng = np.random.default_rng(1077)
    swapped = labels.copy()
    for t in range(100, 160):
        names = np.unique(swapped[t])[1:]
        a_, b_ = rng.choice(names, 2, replace=False)
        first, second = swapped[t] == a_, swapped[t] == b_
        swapped[t][first], swapped[t][second] = b_, a_
    folder = Path(tempfile.mkdtemp(prefix='branch_speed_', dir=os.environ.get('LOCALAPPDATA')))
    mapped = np.memmap(folder / 'labels.dat', dtype=labels.dtype, mode='w+', shape=labels.shape)
    mapped[...] = labels
    cases = [('accepted', labels, points), ('swapped', swapped, points), ('memmap', mapped, points),
             ('shuffled', labels, points.sample(frac=1, random_state=4))]
    original = inventory.scan
    report = dict(status='pass', cases={}, seconds=dict(original=0.0, adapted=0.0))
    with speed.install(inventory) as stats:
        adapted = inventory.scan
        assert adapted.__wrapped__ is original
        for name, movie, table in cases:
            t0 = time.perf_counter()
            expected = original(movie, raw, table)
            t1 = time.perf_counter()
            actual = adapted(movie, raw, table)
            t2 = time.perf_counter()
            report['seconds']['original'] += t1 - t0
            report['seconds']['adapted'] += t2 - t1
            for old, new in zip(expected, actual):
                pd.testing.assert_frame_equal(old, new, check_exact=True)
                assert old.to_csv(index=False) == new.to_csv(index=False)
            report['cases'][name] = [len(expected[0]), len(expected[1])]
            print(name, report['cases'][name], dict(stats), flush=True)
    assert inventory.scan is original
    assert stats['gaps_far'] > 0 and stats['gaps_measured'] > 0 and stats['absorption_block'] > 0, stats
    assert any(rows for rows, _ in report['cases'].values())
    report['stats'] = stats
    del mapped
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
