"""Compare restore_spatial_reciprocal_soma_paths_a002 with its named-pixel coordinate sums, exactly.

python tools/check_reciprocal_centroid_sums.py --engine <A1 engine> --code <engine code> --adapter <identity_candidate> --output <report.json>

On windows of the accepted A1 movie (labels, registered raw, tracker points),
with no and with scattered protected pixels, correct() runs original and with
the adapter alone and together with track_sort_skips (whose float grids reach
Frame). Results (arrays, tables, dtypes) or raised errors must be identical.
The helper itself must give the same sums or the same error as the two
whole-frame bincounts for integer, float, negative, unsigned 64-bit and
mismatched inputs; the original Frame.__init__ and module names are restored.
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
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=left.dtype.kind == 'f')
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
    # Memory-mapped inputs live outside the project; Windows frees them at exit.
    temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    import numpy as np
    import pandas as pd
    import tifffile
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    raw_movie = tifffile.imread(folder / '07_trimmed_evidence/out' / (a.stem + '_registered_raw.tif'))
    label_movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    all_points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import reciprocal_centroid_sums as adapter
    import track_sort_skips
    module = importlib.import_module(adapter.MODULE)
    original_init = module.Frame.__init__
    original_correct = module.correct
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
    # The helper against the two whole-frame bincounts it replaces.
    whole = adapter.named_sums({'whole': 0, 'named': 0})
    rng = np.random.default_rng(5)
    frame = label_movie[100].astype(np.int64)
    yy, xx = np.indices(frame.shape)
    count = int(frame.max())
    negative = frame.copy()
    negative[3, 4] = -2
    cases = [('int grids', frame, yy, xx, count), ('float grids', frame, yy.astype(float), xx.astype(float), count),
             ('int32 ids', frame.astype(np.int32), yy, xx, count),
             ('uint16 ids', label_movie[100], yy, xx, count),
             ('random float', frame, rng.random(frame.shape), rng.random(frame.shape) * 1e9, count),
             ('short minlength', frame, yy, xx, 1), ('empty', np.zeros_like(frame), yy, xx, 0),
             ('negative id', negative, yy, xx, count),
             ('uint64 ids', frame.astype(np.uint64), yy, xx, count),
             ('grid shape', frame, yy[:-1], xx[:-1], count), ('list grid', frame, yy.tolist(), xx, count)]
    # Working arrays of a run are np.memmap subclasses (array_storage.WorkingArray).
    scratch = Path(temporary.name) / 'grids.memmap'
    store = np.memmap(scratch, dtype=np.int64, mode='w+', shape=(3,) + frame.shape)
    store[0], store[1], store[2] = frame, yy, xx
    cases.append(('memmap views', store[0], store[1].astype(float).view(np.memmap), store[2], count))
    cases.append(('memmap ids', store[0], yy, xx, count))

    def reference(ids, yy, xx, count):
        # Frame.__init__'s own two statements.
        return (np.bincount(ids.ravel(), weights=xx.ravel(), minlength=count+1)[1:],
                np.bincount(ids.ravel(), weights=yy.ravel(), minlength=count+1)[1:])
    helper_stats = {'whole': 0, 'named': 0}
    helper = adapter.named_sums(helper_stats)
    for label, *args in cases:
        compare(run(reference, *args), run(helper, *args), ('helper', label))
    # Integer, float, int32, uint16, random float, short minlength, negative id
    # (the same error from the named pixels) and both memmap cases.
    assert helper_stats['named'] == 9 and helper_stats['whole'] == 4, helper_stats
    report['helper'] = helper_stats
    for start in (int(s) for s in a.starts.split(',')):
        start = max(0, min(start, len(label_movie) - a.frames))
        window = slice(start, start + a.frames)
        labels = np.ascontiguousarray(label_movie[window])
        raw = np.ascontiguousarray(raw_movie[window])
        points = all_points[(all_points.frame >= start) & (all_points.frame < start + a.frames)].copy()
        points['frame'] = points.frame - start
        stored = np.memmap(Path(temporary.name) / f'labels{start}.memmap', dtype=labels.dtype, mode='w+', shape=labels.shape)
        stored[:] = labels
        for protected_name, labels_in, protected in (
                ('none', labels, np.zeros(labels.shape, bool)),
                ('scattered', labels, rng.random(labels.shape) < .002),
                ('stored labels', stored, np.zeros(labels.shape, bool))):
            args = (labels_in, raw, points, protected)
            expected = run(module.correct, *args)
            with adapter.install(stats):
                assert module.Frame.__init__.__wrapped__ is original_init
                assert module.Frame.__init__.__code__.co_filename == '<reciprocal-centroid-sums>'
                compare(expected, run(module.correct, *args), (start, protected_name, 'alone'))
                with track_sort_skips.install(sort_stats):
                    assert module.correct is not original_correct
                    compare(expected, run(module.correct, *args), (start, protected_name, 'with float grids'))
            assert module.Frame.__init__ is original_init and module.correct is original_correct
            assert '_named_sums' not in vars(module)
        print(start, report, stats, flush=True)
    assert stats['versions'] > 0 and stats['named'] > 0 and stats['whole'] == 0, stats
    report['stats'] = stats
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
