"""Compare whole-movie name counts and the four rewritten corrections with the originals, exactly.

The frame-by-frame count must equal np.bincount(labels.ravel().astype(np.int64))
(values, int64 dtype, length, a fresh writeable array) on the whole accepted
movie, on read-only memmaps, on uint8 and uint32 copies, on movies whose
largest name is in one middle frame, on all-zero movies; signed and empty
movies take the original expression. Then complete short soma-owner
excursions, recurrent soma name conflicts a002, separable bookended soma basins
and soma-free temporal satellites run `correct` on real windows (accepted
labels, registered raw, tracker points), with no and with scattered protected
pixels, plain and memmapped; corrected movies, audits and applications
(values, dtypes, order) and raised errors must equal the original's.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

from check_reference_area_boxes import same

FAMILIES = ('recover_complete_short_soma_owner_excursions', 'recover_recurrent_soma_name_conflicts_a002',
            'recover_separable_bookended_soma_basins', 'recover_soma_free_temporal_satellites')


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--starts', default='0,200,436')
    p.add_argument('--frames', type=int, default=40)
    a = p.parse_args()
    import importlib
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    cfg = common.Config.load(a.config)
    raw_movie = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    label_movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    all_points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    modules = [importlib.import_module(name) for name in FAMILIES]
    import movie_label_counts as adapter
    scratch = Path(tempfile.mkdtemp(prefix='movie-counts-'))
    maps = []

    def mapped(name, array):
        path = scratch / (name + '.dat')
        array.tofile(path)
        view = np.memmap(path, dtype=array.dtype, mode='r', shape=array.shape)
        maps.append(view)
        return view
    report = dict(status='pass', count_cases=0, calls=0, applications=0, errors=0)
    timings = dict(count_original=0.0, count_framed=0.0, original=0.0, adapted=0.0)
    stats = {'framed': 0, 'direct': 0}
    count = adapter.counts(stats)
    middle = np.zeros((5, 7, 9), np.uint16)
    middle[2, 3, 4] = 900
    cases = [label_movie, mapped('whole', label_movie), label_movie[:60].astype(np.uint8),
             label_movie[:60].astype(np.uint32), middle, np.zeros((3, 4, 4), np.uint16),
             label_movie[:5].astype(np.int32), np.zeros((0, 4, 4), np.uint16)]
    for case in cases:
        tick = time.perf_counter()
        expected = np.bincount(case.ravel().astype(np.int64))
        middle_time = time.perf_counter()
        actual = count(case)
        if case is cases[0]:
            timings['count_original'] = middle_time - tick
            timings['count_framed'] = time.perf_counter() - middle_time
        assert actual.dtype == expected.dtype == np.int64 and np.array_equal(actual, expected)
        assert actual.flags.writeable and actual.flags.owndata
        report['count_cases'] += 1
    assert stats['direct'] == 2 and stats['framed'] == len(cases) - 2, stats
    rng = np.random.default_rng(11)
    install_stats = {}
    for start in (int(s) for s in a.starts.split(',')):
        start = max(0, min(start, len(label_movie) - a.frames))
        window = slice(start, start + a.frames)
        labels = np.ascontiguousarray(label_movie[window])
        raw = np.ascontiguousarray(raw_movie[window])
        points = all_points[(all_points.frame >= start) & (all_points.frame < start + a.frames)].copy()
        points['frame'] = points.frame - start
        scattered = rng.random(labels.shape) < .002
        for protected_name, protected in (('none', np.zeros(labels.shape, bool)), ('scattered', scattered)):
            inputs = [(labels, raw, protected, points)]
            if protected_name == 'none':
                inputs.append((mapped(f'labels{start}', labels), mapped(f'raw{start}', raw),
                               mapped(f'protected{start}', protected), points))
                # Rows out of frame order within tracks take the per-track sorts.
                inputs.append((labels, raw, protected, points.sample(frac=1, random_state=start)))
            for movie, raw_input, protected_input, table in inputs:
                for module in modules:
                    def run():
                        try:
                            return ('ok', module.correct(movie, raw_input, table, protected_input))
                        except Exception as error:
                            return ('error', type(error).__name__, str(error))
                    original = module.correct
                    tick = time.perf_counter()
                    expected = run()
                    timings['original'] += time.perf_counter() - tick
                    with adapter.install(install_stats):
                        assert module.correct.__wrapped__ is original
                        tick = time.perf_counter()
                        actual = run()
                        timings['adapted'] += time.perf_counter() - tick
                    assert module.correct is original
                    assert expected[0] == actual[0], (module.__name__, start, expected[:3], actual[:3])
                    if expected[0] == 'ok':
                        assert same(list(expected[1]), list(actual[1])), (module.__name__, start, protected_name)
                        report['applications'] += len(expected[1][2])
                    else:
                        assert expected == actual
                        report['errors'] += 1
                    report['calls'] += 1
        print(start, report, flush=True)
    # Each install swaps all four families; each call runs one of them.
    assert install_stats['versions'] == 4 * report['calls'] and install_stats['framed'] == report['calls']
    assert install_stats['direct'] == 0
    assert install_stats['in_order'] > 0 and install_stats['sorted'] > 0, install_stats
    report.update(seconds=timings, stats=install_stats)
    for view in maps:
        view._mmap.close()
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
