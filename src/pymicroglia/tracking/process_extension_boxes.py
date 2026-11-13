"""Approved process extension gaps measured inside each component's widened box.

extend_approved_process_paths.correct computed a whole-frame distance
transform from the large cell for every group and frame, then read it only at
each candidate component, keeping the component only when its nearest
distance was at most maximum_edge_distance_px. A large-cell pixel outside the
component's box widened by ceil(that limit) is farther than the limit, so a
kept gap always comes from a pixel inside the widened box and has the same
value there; a component with no large-cell pixel in its widened box was
always beyond the limit. Non-finite or negative limits keep the original.

Each component's coordinates, and the largest part's in _largest_centre, come
from np.nonzero inside the mask's bounding box (sparse_nonzero): the same
coordinates. The rewritten correct() reads its own copy of the module's names,
so its _largest_centre is rewritten into that copy; the module's is untouched.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
from pathlib import Path
import sys

import numpy as np
try:
    from .sparse_nonzero import sparse_nonzero
except ImportError:
    from sparse_nonzero import sparse_nonzero

SOURCE = 'aa48d6bd4a30058c853e5e9406d9b764f0d3d336731bc8a57f568ecb6ccb67de'
REPLACEMENTS = [
    ("""            large = labels[t] == large_name
            distance = ndi.distance_transform_edt(~large)
""", """            large = labels[t] == large_name
            _limit = float(params.maximum_edge_distance_px)
            _boxed = math.isfinite(_limit) and _limit >= 0
            distance = None if _boxed else ndi.distance_transform_edt(~large)
"""),
    ("""                gap = float(distance[yy, xx].min())
                if gap > params.maximum_edge_distance_px:
                    continue
""", """                if _boxed:
                    gap = _boxed_gap(large, small, yy, xx, _limit)
                    if gap is None:
                        continue
                else:
                    gap = float(distance[yy, xx].min())
                if gap > params.maximum_edge_distance_px:
                    continue
"""),
    ("""                yy, xx = np.where(small)
""", """                yy, xx = _sparse_nonzero(small, np.where)
"""),
]
CENTRE = ('    yy, xx = np.where(parts == int(np.argmax(sizes[1:])+1))\n',
          '    yy, xx = _sparse_nonzero(parts == int(np.argmax(sizes[1:])+1), np.where)\n')


def boxed_gap(stats):
    def _boxed_gap(large, small, yy, xx, limit):
        """Nearest large-cell distance over the component, or None when beyond the limit."""
        pad = math.ceil(limit)
        y0, y1 = max(int(yy.min()) - pad, 0), min(int(yy.max()) + pad + 1, large.shape[0])
        x0, x1 = max(int(xx.min()) - pad, 0), min(int(xx.max()) + pad + 1, large.shape[1])
        held = large[y0:y1, x0:x1]
        if not held.any():
            stats['beyond'] += 1
            return None
        from scipy import ndimage
        stats['boxed'] += 1
        return float(ndimage.distance_transform_edt(~held)[small[y0:y1, x0:x1]].min())
    return _boxed_gap


def adapt(module, stats):
    original = getattr(module, 'correct', None)
    path = getattr(module, '__file__', None)
    if (original is None or not path or getattr(original, '__module__', None) != module.__name__
            or getattr(original, '__wrapped__', None) is not None):
        return None
    try:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE:
            return None
        source = inspect.getsource(original)
    except (OSError, TypeError):
        return None
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            return None
        source = source.replace(old, new)
    namespace = vars(module)
    coordinates = stats.setdefault('coordinates', dict(boxed=0, whole=0))
    added = dict(_boxed_gap=boxed_gap(stats), math=math, _sparse_nonzero=sparse_nonzero(coordinates))
    scope = {**namespace, **added}
    centre = namespace.get('_largest_centre')
    if (callable(centre) and getattr(centre, '__module__', None) == module.__name__
            and getattr(centre, '__wrapped__', None) is None):
        try:
            centre_source = inspect.getsource(centre)
        except (OSError, TypeError):
            centre_source = ''
        if centre_source.count(CENTRE[0]) == 1:
            exec(compile(centre_source.replace(*CENTRE), '<process-extension-boxes>', 'exec'), scope)
            scope['_largest_centre'].__wrapped__ = centre
    local = {}
    exec(compile(source, '<process-extension-boxes>', 'exec'), scope, local)
    replacement = local['correct']
    replacement.__wrapped__ = original
    return original, replacement


@contextmanager
def install(stats=None):
    """Replace the loaded verified correct() and its imported aliases; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'boxed', 'beyond'):
        stats.setdefault(key, 0)
    module = sys.modules.get('extend_approved_process_paths')
    found = adapt(module, stats) if module is not None else None
    aliases = []
    if found is not None:
        stats['versions'] += 1
        for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
            names = getattr(child, '__dict__', None)
            if not names:
                continue
            if id(found[0]) not in map(id, names.values()):  # alias loop skip
                continue
            for name, value in list(names.items()):
                if value is found[0]:
                    aliases.append((child, name, value))
                    setattr(child, name, found[1])
    try:
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
