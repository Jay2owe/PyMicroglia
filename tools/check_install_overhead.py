"""Compare the speed110 install and raw-support changes with the code they replace.

python tools/check_install_overhead.py <before folder> <report.json>
<before folder> holds the earlier raw_support_memo.py and module_scan.py.

- _grid equals np.ogrid (values, dtype, shape) for integer bounds.
- _erode equals ndi.binary_erosion for boolean masks of every small shape, as
  plain arrays, memmaps and other subclasses, with the cross, an equal copy
  of it, and other structures (which keep scipy).
- _measure and _core_support_original give the same parts, support masks
  (dtype, shape, values) and flags as before on random int and float patches,
  plain and memmapped, for every centre and contrast request.
- module_scan.file_sha256 gives the same digests inside held() as outside, and
  still raises for a missing file at every call inside it.
- The alias loop skip continues exactly for the namespaces whose loop would
  find no alias, including values that share no id and an empty table.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
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
module_scan = load('module_scan_after', here / 'module_scan.py')
rng = np.random.default_rng(110)
checks = {}


class Sub(np.ndarray):
    pass


def same_array(a, b):
    return type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


# _grid
grid_ok = True
for y0, y1, x0, x1 in [(0, 1, 0, 1), (0, 9, 3, 12), (5, 6, 0, 30), (100, 117, 250, 263), (0, 0, 0, 4)]:
    a, b = np.ogrid[y0:y1, x0:x1], new._grid(y0, y1, x0, x1)
    grid_ok &= all(same_array(p, q) for p, q in zip(a, b))
checks['grid'] = bool(grid_ok)

# _erode
erode_cases = erode_ok = 0
structures = [new.CROSS, new.CROSS.copy(), ndi.generate_binary_structure(2, 1).astype(int),
              np.ones((3, 3), bool), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]], bool)]
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    for h in range(0, 8):
        for w in range(0, 8):
            for density in (.3, .7, .95, 1.0):
                mask = rng.random((h, w)) < density
                memmap = np.memmap(Path(folder) / f'm{h}_{w}_{density}.dat', bool, 'w+', shape=(h, w)) if h and w else None
                if memmap is not None:
                    memmap[:] = mask
                variants = [mask, mask.view(Sub)] + ([memmap] if memmap is not None else [])
                for variant in variants:
                    for structure in structures:
                        erode_cases += 1
                        expected = ndi.binary_erosion(variant, structure)
                        got = new._erode(variant, structure)
                        erode_ok += same_array(expected, got)
    for mask in (rng.integers(0, 2, (6, 6)), rng.random((6, 6)) < .5):
        erode_cases += 1
        erode_ok += same_array(ndi.binary_erosion(mask[None], ndi.generate_binary_structure(3, 1)),
                               new._erode(mask[None], ndi.generate_binary_structure(3, 1)))
        erode_cases += 1
        erode_ok += same_array(ndi.binary_erosion(mask, new.CROSS), new._erode(mask, new.CROSS))
checks['erode'] = dict(cases=erode_cases, equal=int(erode_ok))


# _measure and _core_support_original
def outcome(function, *args):
    try:
        return function(*args)
    except Exception as error:
        return type(error)


measure_cases = measure_ok = core_cases = core_ok = 0
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    for kind in ('int', 'float', 'memmap_int', 'flat'):
        frames, height, width = 4, 40, 50
        if kind == 'float':
            raw = rng.normal(100, 20, (frames, height, width))
        elif kind == 'flat':
            raw = np.full((frames, height, width), 7, np.uint16)
            raw[:, 10:20, 10:20] = 9
        else:
            raw = rng.integers(0, 60, (frames, height, width)).astype(np.uint16)
            raw = ndi.uniform_filter(raw, (1, 3, 3))
        if kind == 'memmap_int':
            held = np.memmap(Path(folder) / 'raw.dat', raw.dtype, 'w+', shape=raw.shape)
            held[:] = raw
            raw = held
        for _ in range(500):
            t = int(rng.integers(0, frames))
            r = float(rng.choice([0., .4, 1., 2.5, 3.7, 5., 8.]))
            y, x = float(rng.uniform(-2, height + 2)), float(rng.uniform(-2, width + 2))
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
            partial = [True, None, None]
            measure_cases += 1
            a = outcome(old._measure, raw, t, y0, y1, x0, x1, x, y, r, True, True, list(partial))
            b = outcome(new._measure, raw, t, y0, y1, x0, x1, x, y, r, True, True, list(partial))
            measure_ok += a == b if not isinstance(a, type) else a is b
            cy, cx = int(np.floor(y+.5))-y0, int(np.floor(x+.5))-x0
            for cross in (new.CROSS, ndi.generate_binary_structure(2, 1), np.ones((3, 3), bool)):
                core_cases += 1
                a = outcome(old._core_support_original, raw, t, y0, y1, x0, x1, x, y, r, cy, cx, cross)
                b = outcome(new._core_support_original, raw, t, y0, y1, x0, x1, x, y, r, cy, cx, cross)
                core_ok += (same_array(a[0], b[0]) and a[1] == b[1] and type(a[1]) is type(b[1])
                            if not isinstance(a, type) else a is b)
checks['measure'] = dict(cases=measure_cases, equal=int(measure_ok))
checks['core_support'] = dict(cases=core_cases, equal=int(core_ok))

# module_scan.held
files = sorted(here.glob('*.py'))[:20]
outside = [module_scan.file_sha256(str(p)) for p in files]
with module_scan.held():
    inside = [module_scan.file_sha256(str(p)) for p in files] + [module_scan.file_sha256(str(p)) for p in files]
    raised = 0
    for _ in range(2):
        try:
            module_scan.file_sha256(str(here / 'no_such_module.py'))
        except OSError:
            raised += 1
    try:
        module_scan.file_sha256(None)
    except TypeError:
        raised += 1
held_cleared = module_scan._HELD is None
checks['held'] = bool(inside == outside + outside and raised == 3 and held_cleared)


# alias loop skip
def aliases_original(names, swap):
    found = []
    for name, value in list(names.items()):
        entry = swap.get(id(value))
        if entry is not None and value is entry[0]:
            found.append(name)
    return found


skip_cases = skip_ok = 0
objects = [object() for _ in range(50)]
for trial in range(400):
    names = {f'n{i}': objects[int(rng.integers(0, 50))] for i in range(int(rng.integers(0, 30)))}
    chosen = [objects[int(i)] for i in rng.integers(0, 50, int(rng.integers(0, 4)))]
    swap = {id(o): (o, object()) for o in chosen}
    skip = not swap or swap.keys().isdisjoint(map(id, names.values()))
    skip_cases += 1
    skip_ok += (skip == (not aliases_original(names, swap)))
    found = (chosen[0], object()) if chosen else (object(), object())
    skip = id(found[0]) not in map(id, names.values())
    skip_cases += 1
    skip_ok += (skip == (not [n for n, v in names.items() if v is found[0]]))
checks['alias_skip'] = dict(cases=skip_cases, equal=int(skip_ok))

passed = (checks['grid'] and checks['held']
          and all(checks[k]['cases'] == checks[k]['equal'] for k in ('erode', 'measure', 'core_support', 'alias_skip')))
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
