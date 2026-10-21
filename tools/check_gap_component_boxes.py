"""Compare the bounded name labeller of native gap owner transport with ndi.label, exactly.

Every name of real accepted label frames (two movies), plus synthetic masks:
single pixels at each corner, full-frame and border-touching masks, sparse and
dense noise, both structures the module family uses. Dense maps must equal the
whole-frame labels in values, dtype and count; compact maps must equal the
compact map of the whole-frame labels in every stored field, its counters and
its reads. Empty masks, non-boolean inputs and explicit outputs keep the
original labeller. The install must engage only for the two audited files.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'frozen', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=40)
    a = p.parse_args()
    import numpy as np
    import tifffile
    from scipy import ndimage
    package = a.adapter.resolve()
    sys.path.insert(0, str(package.parent))
    name = package.name
    spec = importlib.util.spec_from_file_location(name, package / '__init__.py') if (package / '__init__.py').exists() else None
    import types
    if spec is None:
        module = types.ModuleType(name)
        module.__path__ = [str(package)]
        sys.modules[name] = module
    boxes = importlib.import_module(name + '.gap_component_boxes')
    compact = importlib.import_module(name + '.compact_components')
    history = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    movies = [tifffile.imread(history / '99_well_b3_issue013_anchored_projection_owner_relay.tif', key=range(a.frames)),
              tifffile.imread(history / '08_issue012.tif', key=range(a.frames))]
    structures = (np.ones((3, 3), np.uint8), None)
    report = dict(status='pass', masks=0, fallbacks=0, seconds=dict(whole=0.0, boxed=0.0))
    stats = dict(boxed=0, fallback=0)
    counters = lambda: dict(maps=0, live_dense_equivalent_bytes=0, live_crop_bytes=0,
                            peak_dense_equivalent_bytes=0, peak_crop_bytes=0)

    def compare(mask, structure):
        t0 = time.perf_counter()
        expected, count = ndimage.label(mask, structure)
        t1 = time.perf_counter()
        actual, found = boxes.BoxedLabeller(ndimage, stats, False).label(mask, structure)
        t2 = time.perf_counter()
        report['seconds']['whole'] += t1 - t0
        report['seconds']['boxed'] += t2 - t1
        assert type(actual) is np.ndarray and actual.dtype == expected.dtype and count == found
        assert np.array_equal(actual, expected)
        whole_counters, crop_counters = counters(), counters()
        reference = compact.CompactLabels(expected, whole_counters)
        inner = compact.CompactLabeller(ndimage, crop_counters)
        built, again = boxes.BoxedLabeller(inner, stats, True).label(mask, structure)
        assert again == count and type(built) is compact.CompactLabels
        for field in ('shape', 'dtype', 'ndim', 'size', 'nbytes', 'y0', 'y1', 'x0', 'x1', 'region'):
            assert getattr(built, field) == getattr(reference, field), field
        assert built.crop.dtype == reference.crop.dtype and np.array_equal(built.crop, reference.crop)
        assert not built.crop.flags.writeable and built.crop.flags.c_contiguous
        assert crop_counters == whole_counters, (crop_counters, whole_counters)
        assert np.array_equal(np.asarray(built), np.asarray(reference))
        value = int(expected.max())
        assert np.array_equal(built == value, reference == value)
        ys, xs = np.nonzero(mask)
        for y, x in ((ys[0], xs[0]), (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1)):
            assert built[int(y), int(x)] == reference[int(y), int(x)]
        del built, reference
        assert crop_counters['live_crop_bytes'] == whole_counters['live_crop_bytes'] == 0
        report['masks'] += 1

    for movie in movies:
        for frame in movie:
            for value in np.unique(frame)[1:]:
                for structure in structures:
                    compare(frame == value, structure)
    rng = np.random.default_rng(213)
    shape = movies[0].shape[1:]
    for y, x in ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1), (5, 7)):
        mask = np.zeros(shape, bool)
        mask[y, x] = True
        for structure in structures:
            compare(mask, structure)
    for density in (1e-4, 0.01, 0.3, 0.9, 1.0):
        mask = rng.random(shape) < density
        for structure in structures:
            compare(mask, structure)
    border = np.zeros(shape, bool)
    border[:, 0] = border[-1, :] = True
    compare(border, structures[0])
    before = stats['fallback']
    labeller = boxes.BoxedLabeller(ndimage, stats, False)
    empty = np.zeros(shape, bool)
    for call in (lambda: labeller.label(empty, structures[0]),
                 lambda: labeller.label(movies[0][0] > 0, structures[0], output=np.int64),
                 lambda: labeller.label((movies[0][0] > 0).astype(np.uint8), structures[0])):
        result = call()
        report['fallbacks'] += 1
    assert stats['fallback'] - before == 3
    # Install routing on the audited frozen files and an unaudited copy.
    engaged = {}
    for path in sorted(a.frozen.glob('final/*/native_gap_owner_transport_a0*.py')):
        loaded = importlib.util.spec_from_file_location('gap_check_' + path.parent.name, path)
        target = importlib.util.module_from_spec(loaded)
        sys.path.insert(0, str(path.parent))
        loaded.loader.exec_module(target)
        sys.path.remove(str(path.parent))
        install_stats = dict(versions=0, boxed=0, fallback=0)
        with compact.install(target) if boxes.version(target) else __import__('contextlib').nullcontext():
            old = target.ndi
            with boxes.install(target, install_stats):
                engaged[path.parent.name] = type(target.ndi).__name__
            assert target.ndi is old
        assert install_stats['versions'] == 1, path
    report['installed'] = engaged
    report['stats'] = stats
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
