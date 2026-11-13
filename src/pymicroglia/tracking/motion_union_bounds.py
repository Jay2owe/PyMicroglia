"""Unbounded motion support finds the union's box from the two masks' boxes.

Without supplied bounds, motion support formed the whole-frame union of the two
masks and listed its pixels only to take their extreme rows and columns. The
union occupies exactly the rows and columns either mask occupies, so its box
is the outer box of the two masks' boxes; an empty union still returns the same
empty result. Disk-backed masks and transition frames are read through plain
views of the same memory; the rest of the calculation is unchanged. Masks that are not
two-dimensional Boolean arrays of one shape keep the whole-frame statements;
lazy cell masks are first read as the Boolean frame the union would use.
"""
import __future__
import hashlib
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
try:
    from . import resolved_paths as _paths
except ImportError:
    import resolved_paths as _paths

import numpy as np

SOURCE = 'adc5a36b7fc8ca1072dab86ff2649424bc454834d84a439535b5670db1eefeab'
OLD = ("        union = earlier | later\n"
       "        yy, xx = np.nonzero(union)\n"
       "        if not len(yy):\n")
NEW = ("        yy, xx = _union_extremes(earlier, later)\n"
       "        if not len(yy):\n")


PLAIN = (("    old = earlier[y0:y1, x0:x1]\n    new = later[y0:y1, x0:x1]\n",
          "    old = _plain_view(earlier)[y0:y1, x0:x1]\n    new = _plain_view(later)[y0:y1, x0:x1]\n"),
         ("    local = lag[y0:y1, x0:x1]\n", "    local = _plain_view(lag)[y0:y1, x0:x1]\n"))


def _rows_columns(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    columns = np.flatnonzero(mask[rows[0]:rows[-1] + 1].any(axis=0))
    return int(rows[0]), int(rows[-1]), int(columns[0]), int(columns[-1])


def _mask_array(value):
    # A lazy cell mask (frozen_io.LabelMask: a label frame and one identity)
    # becomes its Boolean frame, the same array the union would convert it to.
    if type(value).__name__ == 'LabelMask' and isinstance(getattr(value, 'labels', None), np.ndarray):
        return np.asarray(value)
    return value


def helpers(stats):
    def union_extremes(earlier, later):
        earlier, later = _mask_array(earlier), _mask_array(later)
        if not (isinstance(earlier, np.ndarray) and isinstance(later, np.ndarray)
                and earlier.ndim == later.ndim == 2 and earlier.shape == later.shape
                and earlier.dtype == later.dtype == np.bool_):
            stats['whole'] += 1
            return np.nonzero(earlier | later)
        boxes = [box for box in (_rows_columns(earlier.view(np.ndarray)),
                                 _rows_columns(later.view(np.ndarray))) if box is not None]
        stats['boxed'] += 1
        if not boxes:
            return np.empty(0, np.intp), np.empty(0, np.intp)
        # The caller reads only the extreme rows and columns.
        return (np.array([min(b[0] for b in boxes), max(b[1] for b in boxes)]),
                np.array([min(b[2] for b in boxes), max(b[3] for b in boxes)]))
    def plain_view(array):
        # Disk-backed frames are read through a plain view of the same memory.
        return array.view(np.ndarray) if isinstance(array, np.memmap) else array

    return dict(_union_extremes=union_extremes, _plain_view=plain_view)


@contextmanager
def install(tracking, code_root):
    stats = dict(supported=False, boxed=0, whole=0)
    current = tracking.motion_support
    cells = dict(zip(current.__code__.co_freevars, current.__closure__ or ()))
    original, wrapper_cell = current, None
    if 'original' in cells:  # the lazy-mask dispatcher keeps the original for unbounded calls
        wrapper_cell = cells['original']
        original = wrapper_cell.cell_contents
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        yield stats
        return
    if (hashlib.sha256(source.encode()).hexdigest() != SOURCE or source.count(OLD) != 1
            or any(source.count(old) != 1 for old, _ in PLAIN)):
        yield stats
        return
    source = source.replace(OLD, NEW)
    for old, new in PLAIN:
        source = source.replace(old, new)
    globals_ = vars(sys.modules[original.__module__])
    added = helpers(stats)
    globals_.update(added)
    namespace = {}
    exec(compile(source, '<motion-union-bounds>', 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), globals_, namespace)
    replacement = namespace['motion_support']
    aliases = []
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        location = getattr(module, '__file__', None)
        if location is None:
            continue
        names = getattr(module, '__dict__', None)
        if names is not None and '__getattr__' not in names and names.get('motion_support') is not original:
            continue                # not an alias; skip resolving its path
        try:
            inside = _paths.resolve(location).is_relative_to(_paths.resolve(code_root))
        except (ValueError, OSError):
            inside = False
        if inside and getattr(module, 'motion_support', None) is original:
            aliases.append(module)
            module.motion_support = replacement
    if wrapper_cell is not None:
        wrapper_cell.cell_contents = replacement
    stats['supported'] = True
    try:
        yield stats
    finally:
        if wrapper_cell is not None:
            wrapper_cell.cell_contents = original
        for module in aliases:
            module.motion_support = original
        for name in added:
            globals_.pop(name, None)
