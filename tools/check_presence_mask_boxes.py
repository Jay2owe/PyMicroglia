"""Compare boxed single-mask reads in presence completion with the originals.

Cell masks from real accepted frames and synthetic masks (empty, single pixel,
touching each border), plain and disk-backed, are read with whole-frame
statements and with the boxed helpers: coordinates, image values (integer and
float images with NaN), their means, motion coverage and border contact. The
recompiled functions must install on the engine module.
"""
import argparse, importlib, json, sys, time, types
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def equal(a, b):
    return (type(a) is type(b) or isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) and \
        np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(a, b, equal_nan=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import presence
    import presence_mask_boxes as adapter
    package = types.ModuleType('candidate_adapters')
    package.__path__ = [str(a.adapters.resolve()), str(a.adapters.resolve().parent / 'speed_candidate')]
    sys.modules['candidate_adapters'] = package
    array_storage = importlib.import_module('candidate_adapters.array_storage')
    original_coverage, original_border = presence.motion_coverage, presence._touches_border
    installed = ExitStack()
    supported = installed.enter_context(adapter.install(presence))['supported']
    coverage, border = presence.motion_coverage, presence._touches_border
    stats = dict(boxed=0, whole=0)
    found = adapter.helpers(stats)
    rng = np.random.default_rng(20261020)
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    failures, cases = [], 0
    times = dict(original=0.0, adapted=0.0)
    masks = [np.zeros((60, 70), bool)]
    for y, x in ((0, 0), (59, 69), (0, 35), (30, 0), (30, 69), (59, 10)):
        m = np.zeros((60, 70), bool); m[y, x] = True; masks.append(m)
    for _ in range(300):
        m = rng.random((60, 70)) < rng.choice([0.002, 0.02, 0.3]); masks.append(m)
    for t in rng.choice(len(stack), 30, replace=False):
        frame = stack[int(t)]
        for identity in rng.choice(np.unique(frame)[1:], 20):
            masks.append(frame == identity)
    for storage in ('plain', 'working'):
        with ExitStack() as scope:
            if storage == 'working':
                pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 10, temporary_ram_bytes=0)
                scope.enter_context(array_storage.install(pool))
            for number, mask in enumerate(masks):
                shape = mask.shape
                images = [rng.integers(0, 4000, shape).astype(np.uint16),
                          np.where(rng.random(shape) < .05, np.nan, rng.normal(0, 1, shape)).astype(np.float32)]
                if storage == 'working':
                    mask = np.array(mask); images = [np.array(image) for image in images]
                for image in images:
                    t0 = time.perf_counter()
                    expected = (np.nonzero(mask), image[mask], original_coverage(mask, image, 0.5),
                                original_border(mask, 3))
                    t1 = time.perf_counter()
                    actual = (found['_mask_coordinates'](mask), found['_mask_values'](image, mask),
                              coverage(mask, image, 0.5), border(mask, 3))
                    t2 = time.perf_counter()
                    times['original'] += t1 - t0; times['adapted'] += t2 - t1
                    cases += 1
                    ok = (all(equal(np.asarray(e), np.asarray(f)) for e, f in zip(expected[0], actual[0]))
                          and equal(np.asarray(expected[1]), np.asarray(actual[1]))
                          and (expected[1].size == 0 or expected[1].dtype.kind == 'f'
                               or float(expected[1].mean()) == float(actual[1].mean()))
                          and expected[2] == actual[2] and type(expected[2]) is type(actual[2])
                          and expected[3] == actual[3] and type(expected[3]) is type(actual[3]))
                    if expected[1].size and expected[1].dtype.kind == 'f':
                        ok = ok and np.array_equal(np.nanmean(expected[1]), np.nanmean(actual[1]), equal_nan=True)
                    if not ok:
                        failures.append(f'{storage}-{number}-{image.dtype}')
    installed.close()
    report = dict(status='pass' if supported and not failures else 'fail', supported=supported,
                  cases=cases, boxed=stats['boxed'], whole=stats['whole'], seconds=times,
                  failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
