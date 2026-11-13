"""Component coordinate sums of restore_spatial_reciprocal_soma_paths_a002 over named pixels only.

Frame.__init__ summed every frame's x and y coordinates per component with
np.bincount over the whole frame and dropped bin 0, the unnamed background.
bincount adds each bin's weights in pixel order, so selecting the nonzero
pixels (np.flatnonzero keeps that order) hands every named bin the same
weights in the same order: the sums are the same bits, length and dtype.
Named pixels are a few percent of a frame. Negative names stay selected and
raise the same error. Array subclasses (a stored movie's memmap frames) are
read through plain views, as bincount reads them. A frame with no named pixel
(bincount of nothing returns integers whatever the weights), ids that are not
an integer ndarray, or grids that are not ndarrays of the same shape keep the
two whole-frame calls.
"""
import __future__
from contextlib import contextmanager
import inspect
import textwrap

import numpy as np
try:
    from .track_sort_skips import _digest, _loaded
except ImportError:
    from track_sort_skips import _digest, _loaded

KEYS = ('versions', 'named', 'whole')
MODULE = 'restore_spatial_reciprocal_soma_paths_a002'
DIGESTS = {'7d5c2be48bde80f567f244a428c8b6d538f7076b79ad05315b4b4e50cfe9f6c4'}
OLD = ("        sx = np.bincount(self.ids.ravel(), weights=xx.ravel(), minlength=count+1)[1:]\n"
       "        sy = np.bincount(self.ids.ravel(), weights=yy.ravel(), minlength=count+1)[1:]\n")
NEW = "        sx, sy = _named_sums(self.ids, yy, xx, count)\n"


def named_sums(stats):
    def _named_sums(ids, yy, xx, count):
        named = None
        if (isinstance(ids, np.ndarray) and ids.dtype.kind in 'iu' and ids.dtype != np.uint64
                and isinstance(yy, np.ndarray) and isinstance(xx, np.ndarray)
                and yy.shape == ids.shape and xx.shape == ids.shape):
            # Frames of a stored movie are WorkingArray (np.memmap) views;
            # bincount reads every array as a plain one, so read them so too.
            flat = ids.view(np.ndarray).ravel()
            # The same positions; NumPy finds them five times faster in a Boolean array.
            named = np.flatnonzero(flat != 0)
        if named is None or not len(named):
            stats['whole'] += 1
            return (np.bincount(ids.ravel(), weights=xx.ravel(), minlength=count+1)[1:],
                    np.bincount(ids.ravel(), weights=yy.ravel(), minlength=count+1)[1:])
        values = flat[named]
        stats['named'] += 1
        return (np.bincount(values, weights=xx.view(np.ndarray).ravel()[named], minlength=count+1)[1:],
                np.bincount(values, weights=yy.view(np.ndarray).ravel()[named], minlength=count+1)[1:])
    return _named_sums


@contextmanager
def install(stats=None):
    """Replace each verified Frame.__init__ for the duration; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in KEYS:
        stats.setdefault(key, 0)
    swapped, added = [], []
    try:
        for module in {id(m): m for m in _loaded(MODULE)}.values():
            frame = getattr(module, 'Frame', None)
            original = vars(frame).get('__init__') if isinstance(frame, type) else None
            if (original is None or getattr(original, '__wrapped__', None) is not None
                    or _digest(module) not in DIGESTS):
                continue
            try:
                source = textwrap.dedent(inspect.getsource(original))
            except (OSError, TypeError):
                continue
            if source.count(OLD) != 1:
                continue
            namespace = {}
            vars(module)['_named_sums'] = named_sums(stats)
            added.append(module)
            flags = original.__code__.co_flags & __future__.annotations.compiler_flag
            exec(compile(source.replace(OLD, NEW), '<reciprocal-centroid-sums>', 'exec',
                         flags=flags, dont_inherit=True), vars(module), namespace)
            replacement = namespace['__init__']
            replacement.__defaults__ = original.__defaults__
            replacement.__kwdefaults__ = original.__kwdefaults__
            replacement.__qualname__ = original.__qualname__
            replacement.__wrapped__ = original
            frame.__init__ = replacement
            swapped.append((frame, original))
        stats['versions'] += len(swapped)
        yield stats
    finally:
        for frame, original in reversed(swapped):
            frame.__init__ = original
        for module in added:
            vars(module).pop('_named_sums', None)
