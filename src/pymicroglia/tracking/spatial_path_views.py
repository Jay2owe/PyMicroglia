"""Spatial soma-path validation reads its movies through plain views of the same memory.

constrain_spatial_soma_paths (nine frozen versions) reads `observed[t][roi]`
and `raw[t][roi]` once per connected part and iterates the proposal movie
frame by frame. The movies are disk-backed array subclasses, so every index
and comparison went through the subclass's Python wrappers. A plain ndarray
view of the same memory gives the same values, dtypes and shapes to the same
comparisons, labels, medians and tests; nothing in the function writes to the
three inputs, and the output copy, the tracker call and every other statement
keep the original arrays.

The tracker footprints were placed in the frame with np.pad (constant zeros).
The same frame is a zero array of the mask's dtype with the mask assigned at
(y0, x0); a mask whose shape is not the crop's keeps np.pad.

The ten calls of one replay validate mostly the same frames under different
flags. Each outline part's body, interior, measured pixels and measured
interior depend only on the proposal frame, the observed frame and the two
fixed structuring elements, so they are computed once per frame content
(found by the SHA-256 of the proposal frame and confirmed by equal bytes,
dtypes and shapes of the frames, kept as copies within PartsMemo.copy_budget;
beyond it by the SHA-256 of all their bytes) and handed to every later call
as fresh copies, in the original identity and part order.

With the raw frame's bytes added to that key, each part also carries the
values its raw tests produced: the measured-rim median and the two coherence
tests on it, the convex hull of the measured pixels, its erosion, the
enclosure median and core test, and the inferred-path raw soma test. Each is
the same expression on the same part, raw patch and structuring element, so
later calls (other versions and flags) take the stored value; arrays come
back as copies. A statement is only rewritten when the statements it reads
from are, in every version, the nearest assignments before it.

Overlap pairs between consecutive frames were found with np.unique(axis=0)
over (previous, current) node columns. Node ids are positive, so the pair
a*(largest b + 1) + b orders exactly as the rows do; a one-dimensional
unique on that key gives the same sorted pairs, dtype and counts.

When the structuring elements are the standard ones, each part's body and
measured pixels (plain two-dimensional boolean arrays) are eroded by the cross
with raw_support_memo's direct erosion, which equals binary_erosion's.

The opening fixed-proposal test compares the movies frame by frame through
plain views instead of building three whole-movie temporaries; the same
elements are compared, so it raises exactly when the original does. Whether a
part touches the support extent's edge is read from its extreme pixels:
np.nonzero lists rows in ascending order, so the first and last row and the
smallest and largest column decide the four tests the whole arrays did.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import re
import sys

import numpy as np
from scipy.ndimage import binary_erosion as _scipy_erosion
try:
    from . import narrow_labels as _narrow_labels
except ImportError:
    import narrow_labels as _narrow_labels
try:
    from .module_scan import file_sha256 as _file_sha256
    from .raw_support_memo import _erode
except ImportError:
    from module_scan import file_sha256 as _file_sha256
    from raw_support_memo import _erode

VERSIONS = {
    '640a0d220649c1e7', 'ce57e98b35ddbbe6', '87c9144db28af324', '4a51c2a7cda614bf',
    '53c11dbb37dd86da', '621bcecf4fda8f39', 'd9b3e3139872778b', '5c4f3f290de79428',
    'be805a5aade28f9b',
}
LOOP = ('    for t, frame in enumerate(proposals):\n',
        '    for t, frame in enumerate(_spatial_plain(proposals)):\n')
VIEWS = ('    clipped = set()\n',
         '    _observed_plain, _raw_plain = _spatial_plain(observed), _spatial_plain(raw)\n'
         '    clipped = set()\n')
READS = (('observed[t][roi]', '_observed_plain[t][roi]'), ('raw[t][roi]', '_raw_plain[t][roi]'))
PAD = ("np.pad(mask, ((y0,raw.shape[1]-y1),(x0,raw.shape[2]-x1)))",
       "_spatial_place(mask, raw.shape, y0, y1, x0, x1)")
# Applied last; field, padded, neighbour and hit are not read after this block in any version.
TOUCH = ('        field = nodes[t]; padded = np.pad(field, 1)\n'
         '        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):\n'
         '            neighbour = padded[1+dy:1+dy+field.shape[0], 1+dx:1+dx+field.shape[1]]\n'
         '            hit = (field > 0) & (neighbour > 0) & (field != neighbour)\n'
         '            touching.update(map(int, np.unique(field[hit])))\n'
         '            touching.update(map(int, np.unique(neighbour[hit])))\n',
         '        touching.update(_spatial_touching(nodes[t]))\n')
# Applied after READS; `components`, `count` and `part` are not read elsewhere in any version.
PARTS = ('        for identity, roi in enumerate(ndi.find_objects(frame), start=1):\n'
         '            if roi is None:\n'
         '                continue\n'
         '            components, count = ndi.label(frame[roi] == identity, eight)\n'
         '            for part in range(1, count+1):\n'
         '                body = components == part\n'
         '                interior = ndi.binary_erosion(body, cross)\n'
         '                measured = body & (_observed_plain[t][roi] == identity)\n'
         '                measured_interior = ndi.binary_erosion(measured, cross)\n',
         '        for identity, roi, _parts in _spatial_parts(frame, _observed_plain[t], eight, cross, _raw_plain[t]):\n'
         '            for body, interior, measured, measured_interior, _pc in _parts:\n')
# Per-part raw tests: (statement text, replacement); applied after PARTS.
ONCE = (
    ('level = np.median(patch[measured & ~measured_interior])',
     "level = _spatial_once(_pc, 'L0', lambda: np.median(patch[measured & ~measured_interior]))"),
    ('coherent = bool(ndi.binary_erosion(measured & (patch > level), cross).any())',
     "coherent = _spatial_once(_pc, 'CE', lambda: bool(ndi.binary_erosion(measured & (patch > level), cross).any()))"),
    ('coherent = bool(np.any(measured_interior & (patch > level)))',
     "coherent = _spatial_once(_pc, 'CA', lambda: bool(np.any(measured_interior & (patch > level))))"),
    ('enclosure = convex_hull_image(measured)',
     "enclosure = _spatial_array(_pc, 'H', lambda: convex_hull_image(measured))"),
    ('enclosed_inner = ndi.binary_erosion(enclosure, cross)',
     "enclosed_inner = _spatial_array(_pc, 'I', lambda: ndi.binary_erosion(enclosure, cross))"),
    ('level = np.median(patch[enclosure & ~enclosed_inner])',
     "level = _spatial_once(_pc, 'L1', lambda: np.median(patch[enclosure & ~enclosed_inner]))"),
    ('bool(ndi.binary_erosion(enclosure & (patch > level), cross).any())',
     "_spatial_once(_pc, 'E1', lambda: bool(ndi.binary_erosion(enclosure & (patch > level), cross).any()))"),
    ('enclosure_resolved = bool(ndi.binary_erosion(enclosed_inner, cross).any())',
     "enclosure_resolved = _spatial_once(_pc, 'R', lambda: bool(ndi.binary_erosion(enclosed_inner, cross).any()))"),
    ('has_raw_soma(body, _raw_plain[t][roi])',
     "_spatial_soma(_pc, has_raw_soma, body, _raw_plain[t][roi])"),
)
# For each rewritten statement: the variable assignments that must be the
# nearest before it (variable -> the assignment's right-hand side).
NEAREST = {
    'L0': {'patch': ('_raw_plain[t][roi]',)},
    'CE': {'level': ('np.median(patch[measured & ~measured_interior])',)},
    'CA': {'level': ('np.median(patch[measured & ~measured_interior])',)},
    'I': {'enclosure': ('convex_hull_image(measured)',)},
    'L1': {'enclosure': ('convex_hull_image(measured)',), 'enclosed_inner': ('ndi.binary_erosion(enclosure, cross)',),
           'patch': ('_raw_plain[t][roi]',)},
    'E1': {'level': ('np.median(patch[enclosure & ~enclosed_inner])',), 'enclosure': ('convex_hull_image(measured)',),
           'patch': ('_raw_plain[t][roi]',)},
    'R': {'enclosed_inner': ('ndi.binary_erosion(enclosure, cross)',)},
}
# Each part's node is written through a plain view of the same memory: the
# movie-sized working array would otherwise build two storage views per write.
NODES = (('    nodes = np.zeros(proposals.shape, np.int32)\n',
          '    nodes = np.zeros(proposals.shape, np.int32)\n    _nodes_plain = _spatial_plain(nodes)\n'),
         ('nodes[t][roi][body] = node', '_nodes_plain[t][roi][body] = node'))
# The fixed-proposal test, frame by frame on plain views (no whole-movie
# temporaries); the same elements are compared, so the same verdict.
FIXED = ('    if np.any((observed > 0) & (observed != proposals)):\n',
         '    if _spatial_fixed_violated(observed, proposals):\n')
# A part touches the extent edge when its extreme pixels do: np.nonzero gives
# rows ascending, so yy[0] and yy[-1] are the row bounds. gy and gx have no
# other reader in any version.
CLIP = ('                yy, xx = np.nonzero(body)\n'
        '                gy, gx = yy+roi[0].start, xx+roi[1].start\n'
        '                if np.any((gy <= y0) | (gy >= y1-1) | (gx <= x0) | (gx >= x1-1)):\n',
        '                yy, xx = np.nonzero(body)\n'
        '                if _spatial_clipped(yy, xx, roi, y0, y1, x0, x1):\n')
PAIRS = (("np.unique(np.column_stack((nodes[t-1][shared], nodes[t][shared])), axis=0, return_counts=True)",
          "_spatial_pairs(nodes[t-1][shared], nodes[t][shared], True)"),
         ("np.unique(np.column_stack((nodes[t-1][shared], nodes[t][shared])), axis=0)",
          "_spatial_pairs(nodes[t-1][shared], nodes[t][shared], False)"))


class PartsMemo:
    """Outline parts per frame content, shared by every call inside one install.

    Entries are found by a digest of the proposal frame alone and confirmed by
    comparing all three frames' bytes with copies kept in the entry; once the
    copies would pass copy_budget, entries are kept by the digest of all three
    frames instead, as before.
    """

    def __init__(self, budget=192 * 1024**2, copy_budget=96 * 1024**2):
        self.frames = {}
        self.bytes = 0
        self.budget = budget
        self.copied = {}
        self.copy_bytes = 0
        self.copy_budget = copy_budget


def _measure(ndi, frame, observed, eight, cross, standard=False):
    erode = _erode if standard else ndi.binary_erosion
    result = []
    for identity, roi in enumerate(ndi.find_objects(frame), start=1):
        if roi is None:
            continue
        components, count = ndi.label(frame[roi] == identity, eight)
        pieces = []
        for part in range(1, count+1):
            body = components == part
            interior = erode(body, cross)
            measured = body & (observed[roi] == identity)
            measured_interior = erode(measured, cross)
            pieces.append((body, interior, measured, measured_interior))
        result.append((identity, roi, pieces))
    return result


def _fresh(result):
    """Parts with an empty per-part value store that is not kept."""
    return [(identity, roi, [piece + ({None: False},) for piece in pieces]) for identity, roi, pieces in result]


def once(stats, memo):
    def value(store, key, compute):
        if key in store:
            stats['values_reused'] += 1
            return store[key]
        result = compute()
        stats['values_measured'] += 1
        if store.get(None) is not False:
            store[key] = result
        return result
    return value


def array(stats, memo):
    def value(store, key, compute):
        held = store.get(key)
        if held is not None:
            stats['values_reused'] += 1
            return held.copy()
        result = compute()
        stats['values_measured'] += 1
        if (store.get(None) is not False and type(result) is np.ndarray and memo is not None
                and memo.bytes + result.nbytes + 112 <= memo.budget):
            store[key] = result.copy()
            memo.bytes += result.nbytes + 112
        return result
    return value


def soma(stats):
    def value(store, has_raw_soma, body, patch):
        held = store.get('S')
        if held is not None and held[0] is has_raw_soma:
            stats['values_reused'] += 1
            return held[1]
        result = has_raw_soma(body, patch)
        stats['values_measured'] += 1
        if store.get(None) is not False and type(result) in (bool, np.bool_):
            store['S'] = (has_raw_soma, result)
        return result
    return value


def touching(stats):
    def _spatial_touching(field):
        """The names the four forward-neighbour comparisons of a frame add.

        A comparison hits a pixel holding a positive name whose neighbour (one of
        the four forward offsets, zero beyond the edge) holds another positive
        name, and adds both names. Only positive pixels can hit, so their
        neighbours are gathered from the same zero-padded frame and the same
        tests pick the same pairs; the names form the same set."""
        plain = field.view(np.ndarray) if isinstance(field, np.ndarray) else None
        if plain is None or plain.ndim != 2 or plain.dtype.kind not in 'iu' or not plain.size:
            stats['touching_direct'] += 1
            names = set()
            padded = np.pad(field, 1)
            for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
                neighbour = padded[1+dy:1+dy+field.shape[0], 1+dx:1+dx+field.shape[1]]
                hit = (field > 0) & (neighbour > 0) & (field != neighbour)
                names.update(map(int, np.unique(field[hit])))
                names.update(map(int, np.unique(neighbour[hit])))
            return names
        stats['touching_gathered'] += 1
        width = plain.shape[1]
        flat = np.pad(plain, 1).ravel()
        ys, xs = np.divmod(np.flatnonzero((plain > 0).ravel()), width)
        base = (ys + 1) * (width + 2) + (xs + 1)
        values = flat[base]
        names = set()
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = flat[base + (dy * (width + 2) + dx)]
            hit = (neighbour > 0) & (values != neighbour)
            names.update(map(int, np.unique(values[hit])))
            names.update(map(int, np.unique(neighbour[hit])))
        return names
    return _spatial_touching


def pairs(stats):
    def unique(before, after, counts):
        if (type(before) is np.ndarray and type(after) is np.ndarray and before.ndim == 1 and after.ndim == 1
                and before.dtype == after.dtype and before.dtype.kind in 'iu' and before.dtype.itemsize <= 4
                and len(before) == len(after) and len(before) and before.min() > 0 and after.min() > 0):
            stats['pairs_keyed'] += 1
            width = int(after.max()) + 1
            key = before.astype(np.int64) * width + after.astype(np.int64)
            found = np.unique(key, return_counts=counts)
            values = found[0] if counts else found
            result = np.empty((len(values), 2), before.dtype)
            result[:, 0] = values // width
            result[:, 1] = values % width
            return (result, found[1]) if counts else result
        stats['pairs_direct'] += 1
        return np.unique(np.column_stack((before, after)), axis=0, return_counts=counts) if counts else \
            np.unique(np.column_stack((before, after)), axis=0)
    return unique


def _nearest_ok(lines, index, required):
    """Each required variable's nearest assignment before line `index` has an allowed right-hand side."""
    for name, allowed in required.items():
        for line in reversed(lines[:index]):
            text = line.strip()
            if text.startswith('def '):
                return False
            if text.startswith(name + ' = '):
                if text[len(name) + 3:] not in allowed:
                    return False
                break
        else:
            return False
    return True


def _once_rewrite(source):
    """Rewrite the per-part raw tests whose inputs are proven; None if any occurrence is unproven."""
    lines = source.split('\n')
    for old, new in ONCE:
        key = new.split("'")[1] if "'" in new else 'S'
        required = NEAREST.get(key, {})
        for index, line in enumerate(lines):
            if old in line:
                if not _nearest_ok(lines, index, required):
                    return None
    for old, new in ONCE:
        source = source.replace(old, new)
    return source


def parts(stats, memo, ndi):
    eight_ = np.ones((3, 3), bool)
    cross_ = ndi.generate_binary_structure(2, 1)

    def standard(eight, cross):
        return (type(eight) is np.ndarray and type(cross) is np.ndarray
                and eight.dtype == bool and cross.dtype == bool
                and np.array_equal(eight, eight_) and np.array_equal(cross, cross_))

    def frame_parts(frame, observed, eight, cross, raw):
        if (memo is None or type(frame) is not np.ndarray or type(observed) is not np.ndarray
                or type(raw) is not np.ndarray or frame.ndim != 2 or observed.shape != frame.shape
                or raw.shape != frame.shape or not standard(eight, cross)):
            stats['parts_direct'] += 1
            return _fresh(_measure(ndi, frame, observed, eight, cross))
        # Label frames are hashed narrowed without loss; the key names each narrowed type,
        # so the hashed byte stream still splits into the three arrays one way only.
        frame_bytes, frame_code = _narrow_labels.narrow(np.ascontiguousarray(frame))
        observed_bytes, observed_code = _narrow_labels.narrow(np.ascontiguousarray(observed))
        arrays = tuple(np.ascontiguousarray(value).reshape(-1).view(np.uint8) for value in (frame, observed, raw))
        # The full key continues the proposal frame's digest and is taken at most once per call.
        frame_digest = hashlib.sha256(np.ascontiguousarray(frame_bytes).view(np.uint8))
        keys = []

        def full_key():
            if keys:
                return keys[0]
            digest = frame_digest.copy()
            digest.update(np.ascontiguousarray(observed_bytes).view(np.uint8))
            digest.update(np.ascontiguousarray(raw).view(np.uint8))
            key = (frame.dtype.str, observed.dtype.str, raw.dtype.str, frame.shape, digest.digest())
            if frame_code is not None or observed_code is not None:
                key += (frame_code, observed_code)
            keys.append(key)
            return key
        # The proposal frame's digest finds candidates; equal bytes of all three frames confirm one.
        selector = (frame.dtype.str, observed.dtype.str, raw.dtype.str, frame.shape, frame_code, observed_code,
                    frame_digest.digest())
        held = None
        for copies, stored in memo.copied.get(selector, ()):
            if all(np.array_equal(kept, now) for kept, now in zip(copies, arrays)):
                held = stored
                break
        if held is None and memo.frames:
            held = memo.frames.get(full_key())
        if held is not None:
            stats['parts_reused'] += 1
            return [(identity, roi, [tuple(a.copy() for a in piece[:4]) + (piece[4],) for piece in pieces])
                    for identity, roi, pieces in held]
        result = _measure(ndi, frame, observed, eight, cross, standard=ndi.binary_erosion is _scipy_erosion)
        stats['parts_measured'] += 1
        size = 200 + sum(240 + sum(a.nbytes + 112 for piece in pieces for a in piece) + 400 * len(pieces)
                         for _, _, pieces in result)
        if memo.bytes + size > memo.budget:
            return _fresh(result)
        # The stored parts and this call share each part's value store.
        stored = [(identity, roi, [tuple(a.copy() for a in piece) + ({},) for piece in pieces])
                  for identity, roi, pieces in result]
        extra = sum(value.nbytes for value in arrays)
        if memo.copy_bytes + extra <= memo.copy_budget:
            memo.copied.setdefault(selector, []).append((tuple(value.copy() for value in arrays), stored))
            memo.copy_bytes += extra
        else:
            memo.frames[full_key()] = stored
        memo.bytes += size
        return [(identity, roi, [piece + (held_piece[4],) for piece, held_piece in zip(pieces, held_pieces)])
                for (identity, roi, pieces), (_, _, held_pieces) in zip(result, stored)]
    return frame_parts


def plain(stats):
    def view(value):
        if isinstance(value, np.ndarray) and type(value) is not np.ndarray:
            stats['views'] += 1
            return value.view(np.ndarray)
        return value
    return view


def fixed(stats):
    def _spatial_fixed_violated(observed, proposals):
        # Line 3 of every version has already required equal 3-D shapes.
        if (isinstance(observed, np.ndarray) and isinstance(proposals, np.ndarray)
                and observed.ndim == 3 and observed.shape == proposals.shape):
            stats['fixed_frames'] += 1
            seen, made = observed.view(np.ndarray), proposals.view(np.ndarray)
            return any(np.any((seen[t] > 0) & (seen[t] != made[t])) for t in range(len(seen)))
        stats['fixed_direct'] += 1
        return np.any((observed > 0) & (observed != proposals))
    return _spatial_fixed_violated


def clipped(stats):
    def _spatial_clipped(yy, xx, roi, y0, y1, x0, x1):
        sy, sx = roi[0].start, roi[1].start
        if (type(sy) is int and type(sx) is int and type(yy) is np.ndarray and type(xx) is np.ndarray
                and yy.dtype == np.intp and xx.dtype == np.intp and len(yy) and len(yy) == len(xx)):
            stats['clip_bounds'] += 1
            return (int(yy[0]) + sy <= y0 or int(yy[-1]) + sy >= y1 - 1
                    or int(xx.min()) + sx <= x0 or int(xx.max()) + sx >= x1 - 1)
        stats['clip_direct'] += 1
        gy, gx = yy + sy, xx + sx
        return np.any((gy <= y0) | (gy >= y1 - 1) | (gx <= x0) | (gx >= x1 - 1))
    return _spatial_clipped


def place(stats):
    def placed(mask, shape, y0, y1, x0, x1):
        if type(mask) is not np.ndarray or mask.ndim != 2 or mask.shape != (y1 - y0, x1 - x0):
            stats['padded'] += 1
            return np.pad(mask, ((y0, shape[1] - y1), (x0, shape[2] - x1)))
        result = np.zeros((shape[1], shape[2]), mask.dtype)
        result[y0:y1, x0:x1] = mask
        stats['placed'] += 1
        return result
    return placed


def _digest(module):
    try:
        return _file_sha256(module.__file__)[:16]
    except (AttributeError, OSError, TypeError):
        return None


def rewrite(source, parts=False):
    if source.count(LOOP[0]) != 1 or source.count(VIEWS[0]) != 1:
        return None
    for old, _ in READS:
        if not source.count(old):
            return None
    if source.count(PAD[0]) not in (1, 2):
        return None
    source = source.replace(*LOOP).replace(*VIEWS).replace(*PAD)
    for old, new in READS:
        source = source.replace(old, new)
    if parts and source.count(PARTS[0]) == 1:
        source = source.replace(*PARTS)
        rewritten = _once_rewrite(source)
        if rewritten is not None:
            source = rewritten
    for old, new in PAIRS:
        source = source.replace(old, new)
    if source.count(TOUCH[0]) == 1:
        source = source.replace(*TOUCH)
    if all(source.count(old) == 1 for old, _ in NODES):
        for old, new in NODES:
            source = source.replace(old, new)
    if source.count(FIXED[0]) == 1:
        source = source.replace(*FIXED)
    if (source.count(CLIP[0]) == 1 and len(re.findall(r'\bgy\b', source)) == 3
            and len(re.findall(r'\bgx\b', source)) == 3):
        source = source.replace(*CLIP)
    return source


@contextmanager
def install(stats=None, memo=None):
    """Replace every loaded verified version's constrain_spatial_soma_paths and its aliases."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'views', 'placed', 'padded', 'parts_reused', 'parts_measured', 'parts_direct',
                'values_reused', 'values_measured', 'pairs_keyed', 'pairs_direct', 'touching_gathered',
                'touching_direct', 'fixed_frames', 'fixed_direct', 'clip_bounds', 'clip_direct'):
        stats.setdefault(key, 0)
    replaced = []
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, 'constrain_spatial_soma_paths', None)
        if (original is None or getattr(original, '__module__', None) != module.__name__
                or getattr(original, '__wrapped__', None) is not None or _digest(module) not in VERSIONS):
            continue
        globals_ = original.__globals__
        source = rewrite(inspect.getsource(original), parts=globals_.get('ndi') is _ndimage())
        if source is None:
            continue
        globals_['_spatial_plain'] = plain(stats)
        globals_['_spatial_place'] = place(stats)
        globals_['_spatial_parts'] = parts(stats, memo, globals_.get('ndi'))
        globals_['_spatial_once'] = once(stats, memo)
        globals_['_spatial_array'] = array(stats, memo)
        globals_['_spatial_soma'] = soma(stats)
        globals_['_spatial_pairs'] = pairs(stats)
        globals_['_spatial_touching'] = touching(stats)
        globals_['_spatial_fixed_violated'] = fixed(stats)
        globals_['_spatial_clipped'] = clipped(stats)
        namespace = {}
        exec(compile(source, '<spatial-path-views>', 'exec', flags=__future__.annotations.compiler_flag
                     if original.__code__.co_flags & __future__.annotations.compiler_flag else 0,
                     dont_inherit=True), globals_, namespace)
        replacement = namespace['constrain_spatial_soma_paths']
        replacement.__wrapped__ = original
        replaced.append((module, original, replacement))
    stats['versions'] += len(replaced)
    swap = {id(original): (original, new) for _, original, new in replaced}
    aliases = []
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        names = getattr(child, '__dict__', None)
        if not names:
            continue
        if not swap or swap.keys().isdisjoint(map(id, names.values())):  # alias loop skip
            continue
        for name, value in list(names.items()):
            entry = swap.get(id(value))
            if entry is not None and value is entry[0]:
                aliases.append((child, name, value))
                setattr(child, name, entry[1])
    try:
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
        for module, original, _ in replaced:
            original.__globals__.pop('_spatial_plain', None)
            original.__globals__.pop('_spatial_place', None)
            for name in ('_spatial_parts', '_spatial_once', '_spatial_array', '_spatial_soma', '_spatial_pairs', '_spatial_touching',
                         '_spatial_fixed_violated', '_spatial_clipped'):
                original.__globals__.pop(name, None)


def _ndimage():
    from scipy import ndimage
    return ndimage
