"""Compare the shared-body corrections with their held smoothed frames, exactly.

python tools/check_smoothed_frame_memo.py --engine <A1 engine> --code <engine code> --adapter <identity_candidate> --output <report.json>

On windows of the accepted A1 movie (labels, registered raw, tracker points),
with no and with scattered protected pixels, the overlapping and successive
detector soma-memory corrections and known-soma birth validation run original
and adapted (with track_sort_skips, as in the run). Results (arrays, tables,
dtypes) or raised errors must be identical. _shared_body_basin itself is
called on real frames and on a frame changed in one pixel, a memmap view, a
float copy, an integer sigma and a 3-D raw (run directly), and must match;
a caller writing into a held answer must not change a later one, and the
originals are restored.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile


def same(left, right):
    import numpy as np
    import pandas as pd
    if type(left) is not type(right):
        return False
    if isinstance(left, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError:
            return False
        return True
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(map(same, left, right))
    return bool(left == right)


def run(function, *args):
    try:
        return ('ok', function(*args))
    except Exception as error:
        return ('error', type(error).__name__, str(error))


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--starts', default='0,200,436')
    p.add_argument('--frames', type=int, default=40)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    raw_movie = tifffile.imread(folder / '07_trimmed_evidence/out' / (a.stem + '_registered_raw.tif'))
    label_movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    all_points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import smoothed_frame_memo as adapter
    import track_sort_skips
    names = [spec['module'] for spec in adapter.SPECS]
    modules = [importlib.import_module(name) for name in names]
    known = importlib.import_module('known_soma_birth_validation_a004')
    held = [m._shared_body_basin for m in modules]
    corrections = [(name, m.correct) for name, m in zip(names, modules)] + [('known soma', known.correct)]
    report = dict(status='pass', calls=0, values=0, errors=0)

    def compare(expected, actual, label):
        assert expected[0] == actual[0], (label, expected[:3], actual[:3])
        if expected[0] == 'ok':
            assert same(expected[1], actual[1]), label
            report['values'] += 1
        else:
            assert expected == actual, label
            report['errors'] += 1
        report['calls'] += 1

    stats, sort_stats = {}, {}
    rng = np.random.default_rng(17)
    # _shared_body_basin on one frame's bodies: a mask with tracker points inside.
    t = 120
    frame, labels = raw_movie[t], label_movie[t]
    points = all_points[all_points.frame == t]
    rows = list(points.itertuples(index=False))
    body = labels == np.bincount(labels.ravel())[1:].argmax() + 1
    members, others = rows[:2], rows[2:4]
    changed = frame.copy()
    changed[7, 9] += 1
    store = np.memmap(Path(temporary.name) / 'frame.memmap', dtype=frame.dtype, mode='w+', shape=frame.shape)
    store[:] = frame
    direct = [('frame', frame, 2.0), ('again', frame.copy(), 2.0), ('changed', changed, 2.0),
              ('memmap', store, 2.0), ('float', frame.astype(np.float32), 2.0), ('int sigma', frame, 2),
              ('numpy sigma', frame, np.float64(2.0)), ('3-D', raw_movie[t:t + 2], 2.0),
              ('no others', frame, 2.0)]
    for module, original in zip(modules, held):
        expected = [run(original, raw, body, members, [] if label == 'no others' else others, sigma)
                    for label, raw, sigma in direct]
        with adapter.install(stats):
            assert module._shared_body_basin.__wrapped__ is original
            for (label, raw, sigma), wanted in zip(direct, expected):
                compare(wanted, run(module._shared_body_basin, raw, body, members,
                                    [] if label == 'no others' else others, sigma), ('basin', module.__name__, label))
    for start in (int(s) for s in a.starts.split(',')):
        start = max(0, min(start, len(label_movie) - a.frames))
        window = slice(start, start + a.frames)
        labels = np.ascontiguousarray(label_movie[window])
        raw = np.ascontiguousarray(raw_movie[window])
        points = all_points[(all_points.frame >= start) & (all_points.frame < start + a.frames)].copy()
        points['frame'] = points.frame - start
        for protected_name, protected in (('none', np.zeros(labels.shape, bool)),
                                          ('scattered', rng.random(labels.shape) < .002)):
            args = (labels, raw, points, protected)
            expected = [run(function, *args) for _, function in corrections]
            with track_sort_skips.install(sort_stats), adapter.install(stats):
                assert known._shared_body_basin is modules[0]._shared_body_basin
                assert modules[0]._shared_body_basin.__code__.co_filename == '<smoothed-frame-memo>'
                for (label, function), wanted in zip(corrections, expected):
                    current = getattr(sys.modules[function.__module__], function.__name__)
                    compare(wanted, run(current, *args), (start, protected_name, label))
            for module, original in zip(modules, held):
                assert module._shared_body_basin is original and '_smoothed' not in vars(module)
            assert known._shared_body_basin is held[0]
        print(start, report, stats, flush=True)
    # A caller writing into an answer must not reach a later one.
    with adapter.install(stats):
        smooth = modules[0]._smoothed
        first = smooth(frame, 2.0, lambda: held[0].__globals__['ndi'].gaussian_filter(frame.astype(np.float32), 2.0))
        first[:] = -1
        again = smooth(frame, 2.0, lambda: held[0].__globals__['ndi'].gaussian_filter(frame.astype(np.float32), 2.0))
        assert not np.array_equal(first, again)
    del store
    assert stats['versions'] > 0 and stats['reused'] > 0 and stats['smoothed'] > 0 and stats['direct'] > 0, stats
    report['stats'] = stats
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
