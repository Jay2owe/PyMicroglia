"""Compare equal-value component accounting on disk-backed frames with and without the plain view, exactly.

Real accepted label frames (two stacks) are held as the storage pool's
disk-backed arrays, as numpy memmaps and as plain arrays, in uint16, int32 and
int64. label_value_components must return the same labels (values, dtype and
type), owners and areas; restore_spatial_reciprocal_soma_paths_a002.Frame must
build the same ids, owners, areas, centres and point lists; the whole-stack
excess counts must agree. Float frames keep the original. The adapter must
restore every alias on exit.
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
    for name in ('engine', 'correction-root', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=60)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.correction_root / 'code').resolve()), str(a.adapter.resolve())]
    import component_accounting as accounting
    import restore_spatial_reciprocal_soma_paths_a002 as reciprocal
    import array_storage
    import component_accounting_view as view
    original = accounting.label_value_components
    history = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    movies = [tifffile.imread(history / '99_well_b3_issue013_anchored_projection_owner_relay.tif', key=range(a.frames)),
              tifffile.imread(history / '08_issue012.tif', key=range(200, 200 + a.frames))]
    points = pd.read_csv(history / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    folder = Path(tempfile.mkdtemp(prefix='component_view_', dir=os.environ.get('LOCALAPPDATA')))
    pool = array_storage.Pool(folder / 'pool', minimum_bytes=1)
    report = dict(status='pass', frames=0, kinds={}, seconds=dict(original=0.0, adapted=0.0))
    holders = []

    def variants(movie):
        for dtype in (np.uint16, np.int32, np.int64):
            plain = movie.astype(dtype)
            backed = pool.allocate(plain.shape, plain.dtype, force_disk=True)
            backed[...] = plain
            path = folder / f'mm_{len(holders)}.dat'
            mapped = np.memmap(path, dtype=plain.dtype, mode='w+', shape=plain.shape)
            mapped[...] = plain
            holders.extend([backed, mapped])
            yield 'pool_' + np.dtype(dtype).name, backed
            yield 'memmap_' + np.dtype(dtype).name, mapped
            yield 'plain_' + np.dtype(dtype).name, plain
        yield 'float32', movie.astype(np.float32)

    def same(expected, actual):
        assert type(actual[0]) is type(expected[0]) and actual[0].dtype == expected[0].dtype
        assert np.array_equal(actual[0], expected[0])
        for old, new in zip(expected[1:], actual[1:]):
            assert type(old) is type(new) and old.dtype == new.dtype and np.array_equal(old, new)

    def frame_state(frame):
        return (frame.ids.dtype, frame.ids.tobytes(), frame.owners.tolist(), frame.areas.tolist(),
                np.asarray(frame.x).tobytes(), np.asarray(frame.y).tobytes(),
                {k: [tuple(p) for p in v] for k, v in frame.points.items()})

    with view.install(accounting) as stats:
        assert accounting.label_value_components.__wrapped__ is original
        assert reciprocal.component_accounting.label_value_components is accounting.label_value_components
        adapted = accounting.label_value_components
        for index, movie in enumerate(movies):
            yy, xx = np.indices(movie.shape[1:])
            start = 0 if index == 0 else 200
            by_frame = {int(t): list(g.itertuples(index=False)) for t, g in points.groupby('frame')}
            for kind, stack in variants(movie):
                for t in range(len(stack)):
                    frame = stack[t]
                    t0 = time.perf_counter()
                    expected = original(frame)
                    t1 = time.perf_counter()
                    actual = adapted(frame)
                    t2 = time.perf_counter()
                    report['seconds']['original'] += t1 - t0
                    report['seconds']['adapted'] += t2 - t1
                    same(expected, actual)
                    report['frames'] += 1
                    if t < 8:
                        rows = by_frame.get(start + t, [])
                        accounting.label_value_components = original
                        old = reciprocal.Frame(frame, rows, yy, xx)
                        accounting.label_value_components = adapted
                        new = reciprocal.Frame(frame, rows, yy, xx)
                        assert frame_state(old) == frame_state(new), (kind, t)
                accounting.label_value_components = original
                expected = (accounting.total_component_excess(stack, 3), accounting.duplicate_owner_frames(stack, 3))
                accounting.label_value_components = adapted
                actual = (accounting.total_component_excess(stack, 3), accounting.duplicate_owner_frames(stack, 3))
                assert expected == actual, kind
                report['kinds'][kind] = report['kinds'].get(kind, 0) + len(stack)
    assert accounting.label_value_components is original
    assert reciprocal.component_accounting.label_value_components is original
    assert stats['viewed'] > 0 and stats['direct'] > 0 and stats['versions'] == 1, stats
    report['stats'] = stats
    del holders
    pool.close()
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
