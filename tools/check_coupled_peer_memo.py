"""Compare coupled observed name return with and without the own-outline peer memo, exactly.

The frozen coupled_observed_name_return_a002.correct runs on real windows: the
accepted labels, the registered raw movie and tracker points, with nothing
protected and with pixels that differ from an earlier history stage protected.
The adapted correct must return the same labels (dtype, values) and the same
audit and application tables (columns, dtypes, values, order), or raise the
same error.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def run(function, *args):
    try:
        return ('value', function(*args))
    except Exception as error:
        return ('raised', type(error).__name__, str(error))


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--window', type=int, default=120)
    p.add_argument('--starts', default='0,150,300')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    cfg = common.Config.load(a.config)
    raw_all = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])
    for name in ('common', 'pipeline'):
        sys.modules.pop(name, None)
    sys.path.remove(str((a.engine / 'code').resolve()))
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import coupled_observed_name_return_a002 as module
    import coupled_peer_memo as adapter
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels_all = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    earlier_all = tifffile.imread(folder / '08_issue012.tif')
    points_all = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    original = module.correct
    report = dict(status='pass', cases=[], seconds=dict(original=0.0, adapted=0.0))
    for start in sorted({min(int(s), max(len(labels_all) - a.window, 0)) for s in a.starts.split(',')}):
        window = slice(start, start + a.window)
        labels = np.ascontiguousarray(labels_all[window])
        raw = np.ascontiguousarray(raw_all[window])
        points = points_all[(points_all.frame >= start) & (points_all.frame < start + a.window)].copy()
        points['frame'] -= start
        recorded = points.reset_index(drop=True)
        # Shuffled rows reach every track out of frame order, so its row lists take the sort.
        for kind, protected, points in (('none', np.zeros(labels.shape, bool), recorded),
                                        ('history', labels != earlier_all[window], recorded),
                                        ('shuffled', np.zeros(labels.shape, bool),
                                         recorded.sample(frac=1, random_state=start))):
            tick = time.perf_counter()
            expected = run(original, labels, raw, points, protected)
            middle = time.perf_counter()
            stats = {}
            with adapter.install(stats):
                assert module.correct.__wrapped__ is original and stats['versions'] == 1
                actual = run(module.correct, labels, raw, points, protected)
            end = time.perf_counter()
            assert module.correct is original
            assert expected[0] == actual[0], (start, kind, expected[:1], actual[:1])
            if expected[0] == 'raised':
                assert expected == actual, (start, kind)
                rows = None
            else:
                (x, xa, xp), (y, ya, yp) = expected[1], actual[1]
                assert x.dtype == y.dtype and np.array_equal(x, y), (start, kind)
                for left, right in ((xa, ya), (xp, yp)):
                    pd.testing.assert_frame_equal(left, right, check_exact=True)
                rows = dict(audit=len(xa), applications=len(xp),
                            applied=int((xa.outcome == 'applied').sum()) if len(xa) else 0)
            report['seconds']['original'] += middle - tick
            report['seconds']['adapted'] += end - middle
            report['cases'].append(dict(start=start, protected=kind, outcome=expected[0], rows=rows,
                                        reused=stats['reused'], in_order=stats['groups_in_order'],
                                        sorted=stats['groups_sorted'],
                                        seconds=dict(original=middle - tick, adapted=end - middle)))
            print(report['cases'][-1], flush=True)
    assert any(c['in_order'] for c in report['cases']) and any(c['sorted'] for c in report['cases'])
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'])))


if __name__ == '__main__':
    main()
