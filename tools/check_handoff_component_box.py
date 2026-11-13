"""Compare owner-handoff component masks with and without the box adapter, exactly.

On real label frames (three windows) every name present, plus an absent name,
is asked for its component at the centre of each outline, at points up to the
search radius beside it, at points between outlines, off the frame edge and at
non-integer coordinates, on plain frames and read-only memmaps. The mask (or
None) must equal the original's, and the boxed area must equal the original
mask's int(sum()). The handoff's _calibrated_area and the dual-contact
_reference_area run on real per-owner point tables with the swap installed and
must return the original's values; the swap must reach both copies of the
handoff module and the three area loops and be undone afterwards.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--starts', default='0,230,470')
    p.add_argument('--frames', type=int, default=3)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import complete_owner_handoff_episodes_r10_a002 as r10
    import complete_owner_handoff_episodes_a003 as a003
    import recover_stable_dual_soma_contacts as dual
    import handoff_component_box as adapter
    stats = {'boxed': 0, 'direct': 0, 'area_boxed': 0, 'area_direct': 0}
    original = r10._at_component
    fast = adapter.boxed(original, stats)
    measure = adapter.area(stats)
    report = dict(status='pass', calls=0, none=0, area_tables=0)
    tables = []
    timings = dict(original=0.0, adapted=0.0)
    scratch = Path(tempfile.mkdtemp(prefix='handoff-'))
    rng = np.random.default_rng(31)
    maps = []
    height, width = movie.shape[1:]
    for start in (int(s) for s in a.starts.split(',')):
        for t in range(max(0, min(start, len(movie) - a.frames)), min(len(movie), start + a.frames)):
            frame = movie[t].copy()
            path = scratch / f'{t}.dat'
            frame.tofile(path)
            mapped = np.memmap(path, dtype=frame.dtype, mode='r', shape=frame.shape)
            maps.append(mapped)
            names = [int(n) for n in np.unique(frame[frame > 0])]
            for owner in names + [int(frame.max()) + 7]:
                yy, xx = np.nonzero(frame == owner)
                points = [(-3.0, 5.0), (width + 2.4, height - 1.0), (width / 2 + .5, height / 2 - .5)]
                if len(yy):
                    k = len(yy) // 2
                    points += [(float(xx[k]), float(yy[k])), (xx[0] - 6.4, yy[0] + .5), (xx[-1] + 4.0, yy[-1] - 5.0),
                               (xx[0] + 7.0, yy[0] - 7.0)]
                    points += [(float(x), float(y)) for x, y in rng.uniform(-8, 8, (3, 2)) + (xx[k], yy[k])]
                for x, y in points:
                    row = next(pd.DataFrame([dict(x=x, y=y)]).itertuples(index=False))
                    for image in (frame, mapped):
                        tick = time.perf_counter()
                        expected = original(image, owner, row)
                        middle = time.perf_counter()
                        actual = fast(image, owner, row)
                        timings['original'] += middle - tick
                        timings['adapted'] += time.perf_counter() - middle
                        assert measure(fast, image, owner, row) == (None if expected is None else int(expected.sum()))
                        if expected is None:
                            assert actual is None, (t, owner, x, y)
                            report['none'] += 1
                        else:
                            assert type(actual) is type(expected) and actual.dtype == expected.dtype
                            assert np.array_equal(actual, expected), (t, owner, x, y)
                        report['calls'] += 1
        # Per-owner point tables over this window, at each outline's middle pixel
        # and at a point beside it (the nearest-component search).
        window = range(max(0, min(start, len(movie) - a.frames)), min(len(movie), start + a.frames))
        for owner in [int(n) for n in np.unique(movie[window[0]])[1:][::4]]:
            rows = []
            for t in window:
                yy, xx = np.nonzero(movie[t] == owner)
                if len(yy):
                    k = len(yy) // 2
                    for dx in (0.0, 3.5):
                        rows.append(dict(frame=t, x=float(xx[k]) + dx, y=float(yy[k]), accepted_owner=owner,
                                         strong=bool(t % 2 or dx), in_encounter=False, state='observed'))
            if rows:
                tables.append((owner, pd.DataFrame(rows), dict(first=window[0], last=window[-1], owner=owner)))
        print(start, report, flush=True)
    expected_areas = [(r10._calibrated_area(movie, g, anchor), a003._calibrated_area(movie, g, anchor),
                       dual._reference_area(movie, g, owner, anchor)) for owner, g, anchor in tables]
    install_stats = {}
    with adapter.install(install_stats):
        a003_original = getattr(a003._at_component, '__wrapped__', None)
        assert r10._at_component.__wrapped__ is original and a003_original is not None
        assert r10._calibrated_area.__wrapped__ and dual._reference_area.__wrapped__
        actual_areas = [(r10._calibrated_area(movie, g, anchor), a003._calibrated_area(movie, g, anchor),
                         dual._reference_area(movie, g, owner, anchor)) for owner, g, anchor in tables]
    assert expected_areas == actual_areas and all(type(v) is float for e in actual_areas for v in e)
    assert any(v > 0 for e in expected_areas for v in e)
    report['area_tables'] = len(tables)
    assert r10._at_component is original and getattr(a003._at_component, '__wrapped__', None) is None
    assert getattr(dual._reference_area, '__wrapped__', None) is None
    assert install_stats['versions'] == 2 and install_stats['area_versions'] == 3
    assert install_stats['area_boxed'] > 0 and install_stats['area_direct'] == 0
    assert stats['boxed'] == report['calls'] and stats['direct'] == 0 and stats['area_direct'] == 0
    assert report['none'] > 0 and report['none'] < report['calls']
    report.update(seconds=timings, stats=stats, install=install_stats)
    for mapped in maps:
        mapped._mmap.close()
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
