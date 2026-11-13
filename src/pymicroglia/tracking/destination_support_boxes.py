"""Destination soma support: identity boxes from one find_objects pass per frame.

constrain_destination_soma's support_at and adjacent_at found each identity's
bounding box with a whole-frame np.nonzero(frame == identity), once per
identity per frame. ndi.find_objects gives the same box (first to last row
and column holding the identity) for every identity in one pass over the
frame; the box is only used to slice, so the rest of each test is unchanged.
Frames that are not non-negative integer images keep the original statements.

The same rewrite also (where a version has the statement):
- builds a frame's footprint array once, not a fresh zero frame per point
  that setdefault then discards;
- takes floor, ceil and the finite test of each point's Python floats with
  math, which gives the same integers and verdicts as the numpy scalars;
- lists a frame's names from its histogram (ascending, as np.unique) for
  non-negative integer frames;
- keeps the last 16 frames' `observed[t] > 0`, which callers only read and
  observed is never written, instead of recomputing it for every record;
- builds each disk's coordinates, erodes two-dimensional boolean masks by the
  cross and labels bright components with raw_support_memo's measuring
  helpers (np.ogrid's int64 ranges; four shifted ANDs with a zero border;
  scipy's own label, which the bounded label adapter only re-boxes). Every
  erosion in the verified versions is a two-argument call with the cross;
  a version with any other erosion call keeps them all.
- finds a recovery source's box from its occupied rows and columns (the same
  first and last indices as np.nonzero, without listing every pixel);
- passes raw_support_memo's cross to the cross erosion instead of making an
  equal one for every point (the structure is only read).
"""
import math
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys

import re

import numpy as np
from scipy.ndimage import _measurements as _ndi_measurements
try:
    from .module_scan import file_sha256 as _file_sha256
    from .raw_support_memo import CROSS, _erode, _grid
    from .point_disks import footprints as _point_disks
except ImportError:
    from module_scan import file_sha256 as _file_sha256
    from raw_support_memo import CROSS, _erode, _grid
    from point_disks import footprints as _point_disks

VERSIONS = {
    'a733590c56b09bc211d59e3219e0d5a40f0f2d3b5a68479fc9bdf39275da16e0': 2,
    'fbbf22c20518b6d0434d1052a9128025fe71881c3824cf69c69692849519ecd7': 1,
    'ed604f7753ee09f7ca8fd760860b17b83dc59ad600ea898aa7e4648c81db465d': 1,
    '2bb68c69e75a96c2ee739255b7dd3f2f2e18442f42c3836b797fe9eca441fb57': 2,
    # Earlier versions without measured-path support: only the extra statements apply.
    'e432c71d9c38d60a1b2407e658d474c9b823cc4e9d9a5de37990410de1ebbaaa': 0,
    'e6315a5243c4344926c684bebbb8fe02318341fc2a09d54642f71e9db1e0700b': 0,
}
OLD = '''                yy, xx = np.nonzero(frame == identity)
                roi = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
'''
NEW = '''                roi = _identity_box(frame, identity)
'''
EXTRA = (
    ('        coverage = footprints.setdefault(t, np.zeros(raw.shape[1:], bool))\n',
     '        coverage = footprints.get(t)\n'
     '        if coverage is None:\n'
     '            coverage = footprints[t] = np.zeros(raw.shape[1:], bool)\n'),
    ('not np.isfinite([x, y, radius]).all()',
     'not (_isfinite(x) and _isfinite(y) and _isfinite(radius))'),
    ('y0=max(0,int(np.floor(y-radius))); y1=min(raw.shape[1],int(np.ceil(y+radius))+1)',
     'y0=max(0,_floor(y-radius)); y1=min(raw.shape[1],_ceil(y+radius)+1)'),
    ('x0=max(0,int(np.floor(x-radius))); x1=min(raw.shape[2],int(np.ceil(x+radius))+1)',
     'x0=max(0,_floor(x-radius)); x1=min(raw.shape[2],_ceil(x+radius)+1)'),
    ('iy, ix = int(np.floor(y+.5)), int(np.floor(x+.5))',
     'iy, ix = _floor(y+.5), _floor(x+.5)'),
    ('        for name in np.unique(frame):\n            if name:\n',
     '        for name in _destination_names(frame):\n            if name:\n'),
)
HELD = ('    footprints = {}\n', '    footprints = {}\n    _positive_held = {}\n')
# The measured footprint loop (centre kept) as one call after `footprints = {}`,
# which HELD extends; see point_disks.
DISKS = ('    for q in selected.itertuples(index=False):\n'
         '        t = int(q.frame); x = float(q.x); y = float(q.y)\n'
         "        radius = float(getattr(q, 'radius_px', 0.))\n"
         '        if not 0 <= t < len(raw) or not np.isfinite([x, y, radius]).all() or radius < 0:\n'
         '            continue\n'
         '        y0=max(0,int(np.floor(y-radius))); y1=min(raw.shape[1],int(np.ceil(y+radius))+1)\n'
         '        x0=max(0,int(np.floor(x-radius))); x1=min(raw.shape[2],int(np.ceil(x+radius))+1)\n'
         '        if y0 >= y1 or x0 >= x1:\n'
         '            continue\n'
         '        yy, xx = np.ogrid[y0:y1, x0:x1]\n'
         '        coverage = footprints.setdefault(t, np.zeros(raw.shape[1:], bool))\n'
         '        coverage[y0:y1, x0:x1] |= (yy-y)**2 + (xx-x)**2 <= radius**2\n'
         '        # Preserve the independently measured centre even for a subpixel disk.\n'
         '        iy, ix = int(np.floor(y+.5)), int(np.floor(x+.5))\n'
         '        if 0 <= iy < raw.shape[1] and 0 <= ix < raw.shape[2]:\n'
         '            coverage[iy, ix] = True\n',
         '    footprints = _destination_disks(selected, raw, True)\n')
MEASURE = (('yy, xx = np.ogrid[y0:y1, x0:x1]', 'yy, xx = _destination_grid(y0, y1, x0, x1)'),
           ('components, _ = ndi.label(bright, np.ones((3, 3), bool))',
            'components, _ = _destination_label(bright, _destination_eight)'),
           # The labels are read only at the centre, which is 0 unless that pixel
           # is bright, and only a nonzero centre reads them again.
           ('                components, _ = _destination_label(bright, _destination_eight)\n'
            '                component = components[cy, cx] if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1] else 0\n',
            '                component = 0\n'
            '                if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1] and bright[cy, cx]:\n'
            '                    components, _ = _destination_label(bright, _destination_eight)\n'
            '                    component = components[cy, cx]\n'))
EROSION = re.compile(r'ndi\.binary_erosion\(((?:[^()]|\([^()]*\))*), (cross|lattice)\)')
POSITIVE = (('return observed[t] > 0\n', 'return _destination_positive(observed, t, _positive_held)\n'),
            ("else observed[f] > 0)", "else _destination_positive(observed, f, _positive_held))"))
# The source's box from its occupied rows and columns; sy and sx are not read again.
SOURCE_BOX = ('            sy, sx = np.nonzero(source)\n'
              '            source_roi = np.s_[sy.min():sy.max()+1, sx.min():sx.max()+1]\n',
              '            source_roi = _destination_source_box(source)\n')
# The cross made once per call; it is only passed to the cross erosion.
LATTICE = ('        lattice = ndi.generate_binary_structure(2, 1)\n',
           '        lattice = _destination_lattice\n')
# Count tables and find_objects lists are this long; larger values keep the original.
MAXIMUM_LABEL = 1 << 20


def names(frame):
    # Read only as `if name: int(name)`, so Python ints in ascending order
    # serve as the unique values; a uint32 frame counts only small labels.
    plain = np.asarray(frame)
    if (plain.ndim == 2 and plain.dtype.kind == 'u' and plain.dtype.itemsize <= 4 and plain.size
            and (plain.dtype.itemsize <= 2 or int(plain.max()) <= MAXIMUM_LABEL)):
        return np.flatnonzero(np.bincount(plain.ravel())).tolist()
    return np.unique(frame)


def positive(stats):
    def _destination_positive(observed, t, held):
        entry = held.get(t)
        if entry is not None and entry[0] is observed:
            stats['positive_reused'] += 1
            return entry[1]
        value = observed[t] > 0
        if len(held) >= 16:
            held.pop(next(iter(held)))
        held[t] = (observed, value)
        stats['positive_measured'] += 1
        return value
    return _destination_positive


def rewrite(source, expected):
    """The verified version's source with the box lookup and the extra statements, or None."""
    if source.count(OLD) != expected:
        return None
    source = source.replace(OLD, NEW)
    if source.count(DISKS[0]) == 1:
        source = source.replace(*DISKS)
    for old, new in EXTRA:
        source = source.replace(old, new)
    if source.count(HELD[0]) == 1:
        source = source.replace(HELD[0], HELD[1])
        for old, new in POSITIVE:
            source = source.replace(old, new)
    for old, new in MEASURE:
        source = source.replace(old, new)
    if len(EROSION.findall(source)) == source.count('ndi.binary_erosion('):
        source = EROSION.sub(r'_destination_erode(\1, \2)', source)
    if source.count(SOURCE_BOX[0]) == 1 and len(re.findall(r'\bs[yx]\b', source)) == 6:
        source = source.replace(*SOURCE_BOX)
    if (source.count(LATTICE[0]) == 1 and len(re.findall(r'\blattice\b', source))
            == 1 + len(re.findall(r'_destination_erode\([^()]*, lattice\)', source))):
        source = source.replace(*LATTICE)
    return source


def source_box(mask):
    """np.s_ box of a mask's first to last occupied row and column, as from np.nonzero."""
    if type(mask) is np.ndarray and mask.ndim == 2 and mask.dtype == np.bool_:
        rows = np.flatnonzero(mask.any(axis=1))
        if len(rows):
            columns = np.flatnonzero(mask[rows[0]:rows[-1]+1].any(axis=0))
            return np.s_[rows[0]:rows[-1]+1, columns[0]:columns[-1]+1]
    sy, sx = np.nonzero(mask)
    return np.s_[sy.min():sy.max()+1, sx.min():sx.max()+1]


def identity_box(stats):
    held = {}

    def _identity_box(frame, identity):
        if held.get('frame') is not frame:
            plain = np.asarray(frame)
            boxes = None
            if plain.ndim == 2 and plain.dtype.kind in 'iu' and plain.size:
                low, high = int(plain.min()), int(plain.max())
                if low >= 0 and high < MAXIMUM_LABEL:
                    from scipy import ndimage
                    boxes = ndimage.find_objects(plain)
                    stats['frames'] += 1
            held.update(frame=frame, boxes=boxes)
        boxes = held['boxes']
        value = int(identity)
        if boxes is not None and 0 < value <= len(boxes) and boxes[value - 1] is not None:
            stats['boxed'] += 1
            return boxes[value - 1]
        stats['fallback'] += 1
        yy, xx = np.nonzero(frame == identity)
        return np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
    return _identity_box


def adapt(module, stats):
    """The rewritten constrain_destination_soma of a verified version, else None."""
    original = getattr(module, 'constrain_destination_soma', None)
    path = getattr(module, '__file__', None)
    if (original is None or not path or getattr(original, '__module__', None) != module.__name__
            or getattr(original, '__wrapped__', None) is not None):
        return None
    try:
        expected = VERSIONS.get(_file_sha256(path))
        source = inspect.getsource(original)
    except (OSError, TypeError):
        return None
    source = rewrite(source, expected) if expected is not None else None
    if source is None:
        return None
    for key in ('positive_reused', 'positive_measured'):
        stats.setdefault(key, 0)
    namespace = dict(vars(module))
    namespace['_identity_box'] = identity_box(stats)
    namespace.update(_isfinite=math.isfinite, _floor=math.floor, _ceil=math.ceil,
                     _destination_names=names, _destination_positive=positive(stats),
                     _destination_grid=_grid, _destination_erode=_erode,
                     _destination_label=_ndi_measurements.label, _destination_eight=np.ones((3, 3), bool),
                     _destination_disks=_point_disks, _destination_source_box=source_box,
                     _destination_lattice=CROSS)
    exec(compile(source, '<destination-support-boxes>', 'exec'), namespace)
    replacement = namespace['constrain_destination_soma']
    replacement.__wrapped__ = original
    return original, replacement


@contextmanager
def install(stats=None):
    """Replace every loaded verified version and its imported aliases; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'frames', 'boxed', 'fallback'):
        stats.setdefault(key, 0)
    swap = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        found = adapt(module, stats) if hasattr(module, 'constrain_destination_soma') else None
        if found is not None:
            swap[id(found[0])] = found
    aliases = []
    if swap:
        stats['versions'] += len(swap)
        for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
            names = getattr(child, '__dict__', None)
            if not names:
                continue
            if not swap or swap.keys().isdisjoint(map(id, names.values())):  # alias loop skip
                continue
            for name, value in list(names.items()):
                found = swap.get(id(value))
                if found is not None and value is found[0]:
                    aliases.append((child, name, value))
                    setattr(child, name, found[1])
    try:
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
