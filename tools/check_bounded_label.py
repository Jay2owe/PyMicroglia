"""Compare the bounded scipy.ndimage.label with the original, exactly.

Every name of real accepted label frames and their foreground (two movies),
synthetic corner pixels, borders, sparse to full noise, with no structure, a
cross and full 3x3 structures as bool and integer arrays. Values, dtype, type
and count must match. Integer inputs, 3-D movies, explicit outputs, empty
masks and scalars keep the original. Under the correction working-array pool
a frame large enough for disk storage must give the same array type. Names
bound by `from scipy.ndimage import label` are swapped and restored.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import types


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=40)
    a = p.parse_args()
    import numpy as np
    import tifffile
    import scipy.ndimage
    from scipy.ndimage import _measurements
    sys.path.insert(0, str(a.adapter.resolve()))
    import bounded_label
    import array_storage
    original = _measurements.label
    history = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    movies = [tifffile.imread(history / '99_well_b3_issue013_anchored_projection_owner_relay.tif', key=range(a.frames)),
              tifffile.imread(history / '08_issue012.tif', key=range(a.frames))]
    cross = scipy.ndimage.generate_binary_structure(2, 1)
    structures = (None, cross, np.ones((3, 3), bool), np.ones((3, 3), np.uint8), cross.astype(int))
    stats = dict(boxed=0, whole=0, fallback=0)
    labeller = bounded_label.bounded(original, stats)
    report = dict(status='pass', masks=0, seconds=dict(original=0.0, bounded=0.0))

    def compare(value, structure, *extra):
        t0 = time.perf_counter()
        expected = original(value, structure, *extra)
        t1 = time.perf_counter()
        actual = labeller(value, structure, *extra)
        t2 = time.perf_counter()
        report['seconds']['original'] += t1 - t0
        report['seconds']['bounded'] += t2 - t1
        if isinstance(expected, tuple):
            assert type(actual[0]) is type(expected[0]) and actual[0].dtype == expected[0].dtype
            assert actual[1] == expected[1] and np.array_equal(actual[0], expected[0])
        else:
            assert actual == expected
        report['masks'] += 1

    for movie in movies:
        for frame in movie:
            for structure in structures[:3]:
                compare(frame > 0, structure)
            for value in np.unique(frame)[1:]:
                mask = frame == value
                for structure in structures:
                    compare(mask, structure)
    rng = np.random.default_rng(70)
    shape = movies[0].shape[1:]
    for y, x in ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1), (9, 3)):
        mask = np.zeros(shape, bool)
        mask[y, x] = True
        for structure in structures:
            compare(mask, structure)
    for density in (1e-4, 0.01, 0.3, 0.9, 1.0):
        mask = rng.random(shape) < density
        for structure in structures:
            compare(mask, structure)
            compare(mask[5:-7:2, 3:], structure)  # strided view
    border = np.zeros(shape, bool)
    border[:, 0] = border[-1, :] = True
    compare(border, None)
    before = stats['fallback']
    compare(np.zeros(shape, bool), None)
    compare(movies[0][0], None)
    compare(movies[0][:3] == movies[0][0, 200, 200], None)
    out = np.empty(shape, np.int32)
    expected_count = original(movies[0][0] == movies[0][0, 200, 200], None, out)
    out2 = np.empty(shape, np.int32)
    assert labeller(movies[0][0] == movies[0][0, 200, 200], None, out2) == expected_count and np.array_equal(out, out2)
    compare(np.array(True), None)
    assert stats['fallback'] - before == 5, stats
    # Same storage type as the original under the correction pool.
    with tempfile.TemporaryDirectory() as folder:
        pool = array_storage.Pool(folder, minimum_bytes=1024**2, temporary_ram_bytes=32 * 1024**2)
        with array_storage.install(pool):
            big = np.zeros((1300, 1300), bool)
            big[400:430, 700:760] = True
            big[1000, 5] = True
            expected = original(big, None)
            actual = labeller(big, None)
            assert type(actual[0]) is type(expected[0]) and actual[1] == expected[1]
            assert np.array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
            report['pool_type'] = type(actual[0]).__name__
            del expected, actual
        pool.close()
    # Alias swap and restoration.
    module = types.ModuleType('bounded_label_alias_check')
    exec('from scipy.ndimage import label', module.__dict__)
    sys.modules[module.__name__] = module
    install_stats = {}
    with bounded_label.install(install_stats):
        assert scipy.ndimage.label is not original and module.label is scipy.ndimage.label
        assert _measurements.label is original
        mask = movies[0][0] == movies[0][0, 200, 200]
        assert np.array_equal(scipy.ndimage.label(mask)[0], original(mask)[0])
    assert scipy.ndimage.label is original and module.label is original
    report['install_stats'] = install_stats
    report['stats'] = stats
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
