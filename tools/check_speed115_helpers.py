"""Compare the speed115 raw-support and outline-part changes with the code they replace.

python tools/check_speed115_helpers.py <before folder> <report.json>
<before folder> holds the earlier raw_support_memo.py and spatial_path_views.py.

- _median equals np.median (value, type, NaN) for 1-D integer, float64 and
  other arrays of every small length, plain and memmapped, with ties, zeros of
  both signs, infinities and NaN.
- _measure and _core_support_original give the same parts and support masks as
  before on random int and float patches for every centre and contrast request,
  including NaN, infinite and negative positions and radii.
- raw_supported_points keeps the same rows as before on random point tables,
  first measured and then reused from memory.
- spatial _measure with the standard cross gives the same parts as
  binary_erosion on random outline frames.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

before, report_path = Path(sys.argv[1]), Path(sys.argv[2])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'
sys.path.insert(0, str(here))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load('raw_support_memo_before', before / 'raw_support_memo.py')
new = load('raw_support_memo_after', here / 'raw_support_memo.py')
old_views = load('spatial_path_views_before', before / 'spatial_path_views.py')
new_views = load('spatial_path_views_after', here / 'spatial_path_views.py')
rng = np.random.default_rng(115)
checks = {}


def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return (a == b) or (a != a and b != b)


def outcome(function, *args):
    try:
        return function(*args)
    except Exception as error:
        return type(error)


# _median
median_cases = median_ok = 0
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    dtypes = [np.uint8, np.uint16, np.int16, np.int32, np.uint32, np.float64, np.float32, np.int64, bool]
    for dtype in dtypes:
        for size in list(range(1, 40)) + [61, 64, 200]:
            for trial in range(6):
                if dtype is bool:
                    values = rng.random(size) < .5
                elif np.dtype(dtype).kind == 'f':
                    values = rng.choice([-2.5, -0.0, 0.0, 1.0, 1.0, 3.25, 1e300, -1e300, 7.0], size).astype(dtype)
                    if trial == 1:
                        values[rng.integers(0, size)] = np.inf
                    if trial == 2:
                        values[rng.integers(0, size)] = np.nan
                    if trial == 3:
                        values[:] = -np.inf
                        values[rng.integers(0, size)] = np.inf
                    if trial == 4:
                        values = rng.normal(0, 1e3, size).astype(dtype)
                else:
                    info = np.iinfo(dtype)
                    values = rng.integers(info.min, int(info.max) + 1, size, dtype=dtype)
                    if trial == 1:
                        values = rng.integers(0, 3, size).astype(dtype)
                    if trial == 2:
                        values[:] = info.max
                variants = [values]
                if trial == 0:
                    memmap = np.memmap(Path(folder) / f'{np.dtype(dtype).name}_{size}.dat', values.dtype, 'w+', shape=values.shape)
                    memmap[:] = values
                    variants.append(memmap)
                for variant in variants:
                    median_cases += 1
                    expected = outcome(np.median, variant)
                    got = outcome(new._median, variant)
                    median_ok += same_value(expected, got)
    for empty in (np.array([], np.uint16), np.zeros((2, 3), np.uint16), [3, 1, 2]):
        median_cases += 1
        expected = outcome(np.median, empty)
        got = outcome(new._median, empty)
        median_ok += (expected is got) if isinstance(expected, type) else same_value(expected, got)
checks['median'] = dict(cases=median_cases, equal=int(median_ok))


# _measure and _core_support_original
measure_cases = measure_ok = core_cases = core_ok = 0
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    for kind in ('uint16', 'float', 'memmap_uint16', 'flat', 'int32', 'float32'):
        frames, height, width = 4, 40, 50
        if kind == 'float':
            raw = rng.normal(100, 20, (frames, height, width))
        elif kind == 'float32':
            raw = rng.normal(100, 20, (frames, height, width)).astype(np.float32)
        elif kind == 'flat':
            raw = np.full((frames, height, width), 7, np.uint16)
            raw[:, 10:20, 10:20] = 9
        elif kind == 'int32':
            raw = rng.integers(-50, 60, (frames, height, width)).astype(np.int32)
        else:
            raw = ndi.uniform_filter(rng.integers(0, 60, (frames, height, width)).astype(np.uint16), (1, 3, 3))
        if kind == 'memmap_uint16':
            held = np.memmap(Path(folder) / 'raw.dat', raw.dtype, 'w+', shape=raw.shape)
            held[:] = raw
            raw = held
        for _ in range(500):
            t = int(rng.integers(0, frames))
            r = float(rng.choice([0., .4, 1., 2.5, 3.7, 5., 8.]))
            y, x = float(rng.uniform(-2, height + 2)), float(rng.uniform(-2, width + 2))
            if rng.random() < .1:
                y, x = float(rng.choice([-0.5, 0.5, 1.5, -1.5])) + int(rng.integers(0, height)), float(rng.choice([2.5, -0.5]))
            y0 = max(0, int(np.floor(y-r))); y1 = min(height, int(np.ceil(y+r))+1)
            x0 = max(0, int(np.floor(x-r))); x1 = min(width, int(np.ceil(x+r))+1)
            if y0 >= y1 or x0 >= x1:
                continue
            for centre in (False, True):
                for contrast in (False, True):
                    measure_cases += 1
                    a = outcome(old._measure, raw, t, y0, y1, x0, x1, x, y, r, centre, contrast)
                    b = outcome(new._measure, raw, t, y0, y1, x0, x1, x, y, r, centre, contrast)
                    measure_ok += (a == b and [type(v) for v in (a or [])] == [type(v) for v in (b or [])]
                                   if not isinstance(a, type) else a is b)
            cy, cx = int(np.floor(y+.5))-y0, int(np.floor(x+.5))-x0
            for cross in (new.CROSS, ndi.generate_binary_structure(2, 1), np.ones((3, 3), bool)):
                core_cases += 1
                a = outcome(old._core_support_original, raw, t, y0, y1, x0, x1, x, y, r, cy, cx, cross)
                b = outcome(new._core_support_original, raw, t, y0, y1, x0, x1, x, y, r, cy, cx, cross)
                core_ok += (same_value(a[0], b[0]) and a[1] == b[1] and type(a[1]) is type(b[1])
                            if not isinstance(a, type) else a is b)
    for bad in (np.nan, np.inf):
        measure_cases += 1
        a = outcome(old._measure, raw, 0, 0, 5, 0, 5, bad, 2.0, 2.0, True, True)
        b = outcome(new._measure, raw, 0, 0, 5, 0, 5, bad, 2.0, 2.0, True, True)
        measure_ok += (a is b) if isinstance(a, type) else a == b
checks['measure'] = dict(cases=measure_cases, equal=int(measure_ok))
checks['core_support'] = dict(cases=core_cases, equal=int(core_ok))


# raw_supported_points, measured then reused
class Module:
    @staticmethod
    def recording_intensity_step(raw):
        return 1.0


def original(raw, points, **kwargs):
    raise AssertionError('fallback not expected')


rows_cases = rows_ok = 0
raw = ndi.uniform_filter(rng.integers(0, 60, (6, 60, 60)).astype(np.uint16), (1, 3, 3))
flag_sets = [{}, dict(centre_connected=True), dict(centre_connected=True, internal_peak=True),
             dict(internal_peak=True, peak_quantisation=True), dict(centre_connected=True, internal_peak=True, peak_quantisation=True)]
accepted = ('centre_connected', 'internal_peak', 'peak_quantisation')
for trial in range(30):
    n = int(rng.integers(0, 120))
    points = pd.DataFrame(dict(
        frame=rng.integers(-1, 7, n), x=rng.uniform(-3, 63, n), y=rng.uniform(-3, 63, n),
        radius_px=rng.choice([-1., 0., 1.5, 3., 4.5, 6.], n), strong=rng.random(n) < .85,
        state=rng.choice(['observed', 'observed', 'inferred'], n)))
    if n and trial % 3 == 0:
        points.loc[points.index[0], 'x'] = np.nan
        points.loc[points.index[-1], 'radius_px'] = np.inf
    if trial % 2:
        points['physically_visible'] = rng.random(n) < .9
    old_memo, new_memo = old.Memo(), new.Memo()
    old_call = old.supported_points(Module, original, accepted, old_memo)
    new_call = new.supported_points(Module, original, accepted, new_memo)
    for repeat in range(2):
        for flags in flag_sets:
            rows_cases += 1
            a = old_call(raw.copy() if repeat else raw, points, **flags)
            b = new_call(raw.copy() if repeat else raw, points, **flags)
            rows_ok += a.equals(b) and list(a.index) == list(b.index)
checks['supported_rows'] = dict(cases=rows_cases, equal=int(rows_ok), reused=new_memo.stats['reused'])


# spatial outline parts
parts_cases = parts_ok = 0
eight, cross = np.ones((3, 3), bool), ndi.generate_binary_structure(2, 1)
for trial in range(120):
    h, w = int(rng.integers(3, 40)), int(rng.integers(3, 40))
    frame = ndi.label(rng.random((h, w)) < rng.choice([.3, .6, .9]))[0].astype(rng.choice([np.int32, np.uint16]))
    frame[rng.random((h, w)) < .1] = 0
    observed = np.where(rng.random((h, w)) < .8, frame, 0).astype(frame.dtype)
    a = old_views._measure(ndi, frame, observed, eight, cross)
    b = new_views._measure(ndi, frame, observed, eight, cross, standard=True)
    parts_cases += 1
    parts_ok += (len(a) == len(b) and all(
        ia == ib and ra == rb and len(pa) == len(pb) and all(
            all(same_value(u, v) for u, v in zip(x, y)) for x, y in zip(pa, pb))
        for (ia, ra, pa), (ib, rb, pb) in zip(a, b)))
checks['outline_parts'] = dict(cases=parts_cases, equal=int(parts_ok))

passed = all(checks[k]['cases'] == checks[k]['equal'] for k in checks)
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
