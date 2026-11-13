"""Compare bookended owner-flicker recovery with and without per-outline boxes, exactly.

recover_large_bookended_owner_flickers.correct runs on real frames (accepted
labels, registered raw, tracker points) with the large and the smaller
thresholds: as accepted, with injected one-frame foreign names (a large
outline renamed to another present name in every third frame, so candidates
are applied), with no and with scattered protected pixels, and as read-only
disk-backed memmaps. Proposals, audits and applications (values, dtypes,
order) and raised errors must equal the original's.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

from check_reference_area_boxes import same


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--starts', default='0,200,436')
    p.add_argument('--frames', type=int, default=40)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import ndimage
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    cfg = common.Config.load(a.config)
    raw_movie = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    label_movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    all_points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import recover_large_bookended_owner_flickers as large
    import recover_smaller_bookended_owner_flickers as smaller
    import flicker_component_boxes as adapter
    original = large.correct
    rng = np.random.default_rng(5)
    scratch = Path(tempfile.mkdtemp(prefix='flicker-boxes-'))
    report = dict(status='pass', frames=a.frames, cases=[], calls=0)
    timings = dict(original=0.0, adapted=0.0)
    totals = {}

    def mapped(name, array):
        path = scratch / (name + '.dat')
        array.tofile(path)
        return np.memmap(path, dtype=array.dtype, mode='r', shape=array.shape)
    maps = []
    for start in (int(s) for s in a.starts.split(',')):
        start = max(0, min(start, len(label_movie) - a.frames))
        window = slice(start, start + a.frames)
        labels, raw = label_movie[window].copy(), raw_movie[window]
        points = all_points[(all_points.frame >= start) & (all_points.frame < start + a.frames)].copy()
        points['frame'] -= start
        points = points.reset_index(drop=True)
        injected = labels.copy()
        names = np.unique(labels[labels > 0])
        for t in range(1, len(injected) - 1, 3):
            components, count = ndimage.label(injected[t] > 0)
            areas = np.bincount(components.ravel())
            large_ids = np.flatnonzero(areas[1:] >= 150) + 1
            if not len(large_ids) or len(names) < 2:
                continue
            cid = int(rng.choice(large_ids))
            mask = components == cid
            current = int(injected[t][mask][0])
            injected[t][mask & (injected[t] == current)] = int(rng.choice(names[names != current]))
        protected = rng.random(labels.shape) < .002
        for kind, movie in (('accepted', labels), ('injected', injected)):
            variants = [('none', np.zeros(labels.shape, bool)), ('scattered', protected)]
            for guard, mask in variants:
                inputs = [('plain', movie, raw, mask)]
                if guard == 'none':
                    m = (mapped(f'{start}_{kind}_labels', movie), mapped(f'{start}_{kind}_raw', raw),
                         mapped(f'{start}_{kind}_protected', mask))
                    maps.extend(m)
                    inputs.append(('memmap',) + m)
                for storage, l, r, g in inputs:
                    for size, params in (('large', large.Params()), ('smaller', smaller.PARAMS)):
                        tick = time.perf_counter()
                        try:
                            expected = original(l, r, points, g, params)
                        except Exception as error:
                            expected = ('raised', type(error).__name__, str(error))
                        middle = time.perf_counter()
                        stats = {}
                        with adapter.install(stats):
                            assert large.correct.__wrapped__ is original
                            try:
                                if size == 'smaller':
                                    actual = smaller.correct(l, r, points, g)
                                else:
                                    actual = large.correct(l, r, points, g, params)
                            except Exception as error:
                                actual = ('raised', type(error).__name__, str(error))
                        timings['original'] += middle - tick
                        timings['adapted'] += time.perf_counter() - middle
                        assert large.correct is original
                        assert same(list(expected), list(actual)), (start, kind, guard, storage, size)
                        raised = isinstance(expected[0], str)
                        applied = 0 if raised else len(expected[2])
                        report['cases'].append(dict(start=start, labels=kind, protected=guard, input=storage,
                                                    thresholds=size, raised=raised, applied=applied,
                                                    audited=0 if raised else len(expected[1]), stats=stats))
                        for key, value in stats.items():
                            totals[key] = totals.get(key, 0) + value
                        report['calls'] += 1
        print(start, report['calls'], flush=True)
    assert totals.get('versions') == report['calls'] and totals.get('boxed', 0) > 0
    assert sum(case['applied'] for case in report['cases']) > 0
    report['seconds'] = timings
    report['stats'] = totals
    for array in maps:
        array._mmap.close()
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], calls=report['calls'], seconds=timings, stats=totals,
                          applied=sum(case['applied'] for case in report['cases']))))


if __name__ == '__main__':
    main()
