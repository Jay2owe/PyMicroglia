"""Raw-supported tracker points: each point's decision remembered across correction stages.

raw_supported_points decides every strong observed point from the raw pixels
of its disk's bounding patch, the point's position and radius, the frame's
bounds and the method flags; about twenty accepted corrections ask again for
the same points of the same movie. The decision is kept with a copy of the
patch it was made from and reused only when the patch now read has the same
dtype and values, the same position, radius, clipped bounds, flags and
recording step. The three frozen versions add flags whose defaults reproduce
the previous version, so their decisions are shared. The flags only choose
which of three measured parts decide a point (coherent bright core, the
centre's component holding it, interior-over-boundary contrast), so one
measurement per point serves every combination: each part is measured the
first time a combination needs it and kept with the point. Point filtering,
the kept rows and their order are unchanged.

A repeated whole call is answered from its kept-row mask when the movie's
buffer has the same SHA-256 content and the view the same place in it, the
flags are the same and the points table is equal (columns, dtypes, index,
values and element types).

spatial_soma_paths.tracker_support_frames without ownership labels is a
function of the same movie, points and peak flags (its later versions only add
ownership options), so a repeated call returns copies of the remembered
frame masks in their original order.

Every other call runs the version's own function text with two statements
answered from memory. A point's core support (its disk, the bright component
holding its centre and whether that component has an interior) depends only on
its clipped bounds, position, radius and the raw patch, so it is reused when
those and the patch's dtype and values are the same. component_owners depends
only on one ownership frame, so it is reused when that frame's shape, dtype and
SHA-256 content are the same. Remembered arrays are read-only; the function only
reads them. The same text also takes each point's finiteness, floor and ceil
with math, and makes each frame's coverage array once rather than a fresh
zero frame per point that setdefault discards. Before its point loop, the
core support of every point not yet remembered is measured in one batch and
remembered (see _core_prepare), so the loop's own reuse test answers it.

temporally_supported_points runs its own text with each track's stable frame
sort skipped when the track's integer frames never decrease (the sort would
keep every row in place; the track is only read).

The measurements build each disk's coordinates as np.ogrid does for integer
bounds (int64 ranges shaped (h, 1) and (1, w)), erode two-dimensional boolean
masks by the four-neighbour cross directly (a pixel stays when it and its four
neighbours are set; neighbours outside the patch count as unset, as
binary_erosion's zero border does) and label with scipy's own label, which the
bounded label adapter only re-boxes.

Per point, floor, ceil and finiteness of the Python float position and radius
are taken with math (the same integers and tests as NumPy's on one double),
and the rim median of an integer patch of at most 32 bits or a float64 patch is
read from its sorted values: the middle value, or the float64 sum of the two
middle values halved, as np.median's mean of them gives (NaN keeps np.median).

For an integer movie of at most 32 bits, the points of one call that are not
yet remembered are measured together, all three parts at once, grouped by
patch shape (see _measure_many); the per-point loop, its reuse tests, the
remembered entries and the kept rows are otherwise unchanged. A part measured
before a combination needs it is the value that combination would measure.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi
try:
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from module_scan import file_sha256 as _file_sha256

# Frozen recovery_qualified_soma versions and the keywords each accepts.
VERSIONS = {
    'a15cda3f7e4b92968c50be0815b4dfce1ce07481e206fa9297aa9824a81ae378': (),
    '535a87d3760f02cf7ad804b832f15d1451c8491f27e4542a7c71f4dcf6e5d000': ('centre_connected',),
    '7ae897cd7478b5fa227ee8f31eb656a2b58e010a4efd677a7eda1bdf212474a6':
        ('centre_connected', 'internal_peak', 'peak_quantisation'),
}
# tracker_support_frames sources (function text) and the frozen
# recovery_temporal_soma_support files they may call.
TRACKER_SUPPORT = {
    '334771a1f02977fd266ad2f1cc30c54bc1f997944011d811a3064c428497015a': (),
    '34ede163662d81c521a963bc0ebf1e5f7786736b370533e8ce1a900b9011a4cd': ('internal_peak', 'peak_quantisation'),
    'd9214c0cf43b6949c911392d11a1fa7d1cc7552c885bdc0b88e4d5e9d051a528':
        ('internal_peak', 'peak_quantisation', 'ownership_labels'),
    'c586dba202f0625ada499651cc7a2e6be814de30046073586f3c6c384c198e75':
        ('internal_peak', 'peak_quantisation', 'ownership_labels', 'require_foreground_majority', 'retain_named_centre'),
    '94381f73e4ae640ef40eea475d376aae0bd6f65ce740ed3eeb5257be40e3dcd8':
        ('internal_peak', 'peak_quantisation', 'ownership_labels', 'require_foreground_majority',
         'retain_named_centre', 'prefer_centred_owner'),
}
TEMPORAL = {'13b006b732e13c4500904a927dcf4f66843c2d4f82bed1ca952dbb6b216f1aad',
            '4c009c6dfac15e18bf2a086345aca1a210dcec574bb0d21862b87f11b56dce72',
            # r185-r189: the same function without the centre_connected keyword.
            '3984e8984c349b90cad0f51eac95c3136ba031f4d49a77cfa7c1ef461c824d15'}
CROSS = ndi.generate_binary_structure(2, 1)
EIGHT = np.ones((3, 3), bool)
from scipy.ndimage import _measurements as _ndi_measurements


def _grid(y0, y1, x0, x1):
    """np.ogrid[y0:y1, x0:x1] for integer bounds."""
    return np.arange(y0, y1, dtype=np.int64)[:, None], np.arange(x0, x1, dtype=np.int64)[None, :]


def _erode(mask, structure):
    """ndi.binary_erosion(mask, structure) for the 2-D cross on a 2-D boolean mask.

    binary_erosion reads its input as a plain array and returns a new plain
    boolean array, so an array subclass is read through a plain view."""
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2
            or not (structure is CROSS or (np.shape(structure) == (3, 3) and np.array_equal(structure, CROSS)
                                           and np.asarray(structure).dtype == np.bool_))):
        return ndi.binary_erosion(mask, structure)
    if type(mask) is not np.ndarray:
        mask = mask.view(np.ndarray)
    out = np.zeros(mask.shape, bool)
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        middle = mask[1:-1, 1:-1]
        out[1:-1, 1:-1] = middle & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return out


def _median(values):
    """np.median(values) for a non-empty 1-D integer (at most 32-bit) or float64 array."""
    if (not isinstance(values, np.ndarray) or values.ndim != 1 or not values.size
            or not ((values.dtype.kind in 'iu' and values.dtype.itemsize <= 4) or values.dtype == np.float64)):
        return np.median(values)
    ordered = np.sort(values)
    if ordered.dtype.kind == 'f' and ordered[-1] != ordered[-1]:
        return np.median(values)
    half = ordered.size // 2
    if ordered.size % 2:
        return np.float64(ordered[half])
    return (np.float64(ordered[half-1]) + np.float64(ordered[half])) / 2.0


class Memo:
    def __init__(self, budget=512 * 1024**2, table_budget=1024**3, core_budget=128 * 1024**2,
                 owner_budget=128 * 1024**2):
        self.decisions = {}
        self.bytes = 0
        self.budget = budget
        # Core support per point and component owners per ownership frame;
        # entries are kept while their budgets last and never replaced.
        self.cores = {}
        self.core_bytes = 0
        self.core_budget = core_budget
        self.owners = {}
        self.owner_sources = {}
        self.owner_bytes = 0
        self.owner_budget = owner_budget
        self.fast = {}
        self.tables = []  # (call key, points copy, keep mask, bytes), oldest first
        self.table_bytes = 0
        self.table_budget = table_budget
        self.stats = dict(calls=0, points=0, reused=0, measured=0, fallback=0, versions=0,
                          calls_reused=0, calls_hashed=0, parts_measured=0, frames_calls=0, frames_reused=0, frames_fallback=0,
                          frames_ownership=0, cores_reused=0, cores_measured=0, owners_reused=0,
                          owners_measured=0)

    def find(self, key, points):
        for held_key, table, keep, _ in self.tables:
            if held_key == key and same_table(table, points):
                return keep
        return None

    def store(self, key, points, keep, size=None):
        if size is None:
            size = int(points.memory_usage(index=True, deep=True).sum()) + keep.nbytes
        if size > self.table_budget:
            return
        while self.tables and self.table_bytes + size > self.table_budget:
            self.table_bytes -= self.tables.pop(0)[3]
        self.tables.append((key, points.copy(deep=True), keep.copy(), size))
        self.table_bytes += size


def same_table(held, points):
    """Equal columns, dtypes, index, values (NaN where NaN) and element types."""
    if (list(held.columns) != list(points.columns) or not held.dtypes.equals(points.dtypes)
            or not held.index.equals(points.index) or not held.equals(points)):
        return False
    for column in range(points.shape[1]):
        if points.dtypes.iloc[column] == object:
            if list(map(type, held.iloc[:, column])) != list(map(type, points.iloc[:, column])):
                return False
    return True


def movie_token(raw):
    """SHA-256 of the movie's whole contiguous buffer and the view's place in it, or None."""
    root = raw
    while isinstance(root.base, np.ndarray):
        root = root.base
    if not root.flags.c_contiguous:
        return None
    start = raw.__array_interface__['data'][0] - root.__array_interface__['data'][0]
    if not 0 <= start < max(root.nbytes, 1):
        return None
    try:
        digest = hashlib.sha256(memoryview(root).cast('B')).hexdigest()
    except (TypeError, ValueError, BufferError):
        return None
    return (digest, root.shape, root.dtype.str, start, raw.shape, raw.strides, raw.dtype.str)


def _measure(raw, t, y0, y1, x0, x1, x, y, r, centre, contrast, parts=None):
    """The frozen statements as parts [coherent, centre component, contrast]; None when `continue`.

    Only the parts asked for (or missing from parts) are measured."""
    yy, xx = _grid(y0, y1, x0, x1)
    disk = (yy-y)**2+(xx-x)**2 <= r*r
    iy, ix = math.floor(y+.5), math.floor(x+.5)
    if 0 <= iy-y0 < disk.shape[0] and 0 <= ix-x0 < disk.shape[1]:
        disk[iy-y0, ix-x0] = True
    interior = _erode(disk, CROSS)
    if not interior.any():
        return None
    boundary = disk & ~interior
    patch = raw[t, y0:y1, x0:x1]
    if parts is None:
        parts = [None, None, None]
    if parts[0] is None or (centre and parts[1] is None):
        level = _median(patch[boundary])
        bright = disk & (patch > level)
        coherent = _erode(bright, CROSS)
        parts[0] = bool(coherent.any())
        if centre and parts[1] is None and parts[0]:
            components, _ = _ndi_measurements.label(bright, EIGHT)
            cy, cx = iy-y0, ix-x0
            component = components[cy, cx] if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1] else 0
            parts[1] = bool(component and np.any(coherent & (components == component)))
    if contrast and parts[2] is None:
        parts[2] = float(patch[interior].max()) - float(patch[boundary].max())
    return parts


def _erode_many(masks):
    """_erode of each (h, w) mask of an (n, h, w) boolean stack."""
    out = np.zeros(masks.shape, bool)
    if masks.shape[1] > 2 and masks.shape[2] > 2:
        middle = masks[:, 1:-1, 1:-1]
        out[:, 1:-1, 1:-1] = (middle & masks[:, :-2, 1:-1] & masks[:, 2:, 1:-1]
                              & masks[:, 1:-1, :-2] & masks[:, 1:-1, 2:])
    return out


def _grow_many(seed, within):
    """Each seed's eight-connected component inside `within` (geodesic dilation to a fixed point)."""
    grown = seed & within
    while True:
        padded = np.zeros((grown.shape[0], grown.shape[1] + 2, grown.shape[2] + 2), bool)
        padded[:, 1:-1, 1:-1] = grown
        h, w = grown.shape[1], grown.shape[2]
        spread = grown.copy()
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                spread |= padded[:, dy:dy + h, dx:dx + w]
        spread &= within
        if np.array_equal(spread, grown):
            return grown
        grown = spread


def _measure_many(raw, items):
    """{item: _measure(raw, *item, True, True)} for an integer movie of at most 32 bits, else {}.

    Items of one patch shape are measured together with the same elementwise
    statements: the disk from int64 coordinates and the float64 position and
    radius, the centre pixel, the cross erosion, the rim median read from the
    sorted rim values (the middle value, or the float64 sum of the two middle
    values halved), the bright and coherent masks, and the contrast of the
    interior and rim maxima. The centre's component is the eight-connected
    bright region grown from the centre pixel, which holds the same pixels as
    the label of the centre in scipy's eight-connected labelling. The centre
    part is set only when the coherent part is, as _measure leaves it.
    """
    out = {}
    plain = np.asarray(raw)
    if plain.dtype.kind not in 'iu' or plain.dtype.itemsize > 4 or plain.ndim != 3:
        return out
    shapes = {}
    for item in items:
        shapes.setdefault((item[2] - item[1], item[4] - item[3]), []).append(item)
    for (h, w), group in shapes.items():
        n = len(group)
        T = np.fromiter((g[0] for g in group), np.int64, n)
        Y0 = np.fromiter((g[1] for g in group), np.int64, n)
        X0 = np.fromiter((g[3] for g in group), np.int64, n)
        X = np.fromiter((g[5] for g in group), np.float64, n)
        Y = np.fromiter((g[6] for g in group), np.float64, n)
        R = np.fromiter((g[7] for g in group), np.float64, n)
        yy = Y0[:, None, None] + np.arange(h, dtype=np.int64)[None, :, None]
        xx = X0[:, None, None] + np.arange(w, dtype=np.int64)[None, None, :]
        disk = (yy-Y[:, None, None])**2 + (xx-X[:, None, None])**2 <= (R*R)[:, None, None]
        cy = np.fromiter((math.floor(g[6]+.5) - g[1] for g in group), np.int64, n)
        cx = np.fromiter((math.floor(g[5]+.5) - g[3] for g in group), np.int64, n)
        inside = (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
        rows = np.flatnonzero(inside)
        disk[rows, cy[rows], cx[rows]] = True
        interior = _erode_many(disk)
        valid = interior.any(axis=(1, 2))
        boundary = disk & ~interior
        patch = plain[T[:, None, None], yy, xx].astype(np.int64)
        counts = boundary.sum(axis=(1, 2))
        rim = np.sort(np.where(boundary, patch, np.iinfo(np.int64).max).reshape(n, h * w), axis=1)
        half = counts // 2
        upper = rim[np.arange(n), half].astype(np.float64)
        lower = rim[np.arange(n), np.maximum(half - 1, 0)].astype(np.float64)
        level = np.where(counts % 2 == 1, upper, (lower + upper) / 2.0)
        bright = disk & (patch > level[:, None, None])
        coherent = _erode_many(bright)
        coherent_any = coherent.any(axis=(1, 2))
        seed = np.zeros(disk.shape, bool)
        seed[rows, cy[rows], cx[rows]] = True
        connected = (_grow_many(seed, bright) & coherent).any(axis=(1, 2))
        low = np.iinfo(np.int64).min
        inner = np.where(interior, patch, low).max(axis=(1, 2))
        outer = np.where(boundary, patch, low).max(axis=(1, 2))
        for j, item in enumerate(group):
            if not valid[j]:
                out[item] = None
                continue
            first = bool(coherent_any[j])
            out[item] = [first, bool(connected[j]) if first else None,
                         float(inner[j]) - float(outer[j])]
    return out


def _derive(parts, centre_connected, internal_peak, peak_quantisation, quantum):
    """_decide's keep from measured parts."""
    keep = parts[0]
    if centre_connected and keep:
        keep = parts[1]
    if internal_peak and keep:
        contrast = parts[2]
        keep = bool(contrast >= -quantum if peak_quantisation else contrast > 0)
    return keep


def supported_points(module, original, accepted, memo):
    def raw_supported_points(raw, points, *args, **kwargs):
        if (args or set(kwargs) - set(accepted) or not isinstance(raw, np.ndarray) or raw.ndim != 3
                or raw.dtype.kind not in 'iuf'):
            memo.stats['fallback'] += 1
            return original(raw, points, *args, **kwargs)
        centre_connected = bool(kwargs.get('centre_connected', False))
        internal_peak = kwargs.get('internal_peak', False)
        peak_quantisation = kwargs.get('peak_quantisation', False)
        memo.stats['calls'] += 1
        token = movie_token(raw) if points.columns.is_unique else None
        if token is not None:
            memo.stats['calls_hashed'] += 1
            call_key = (token, centre_connected, bool(internal_peak), bool(peak_quantisation))
            held = memo.find(call_key, points)
            if held is not None:
                memo.stats['calls_reused'] += 1
                return points.iloc[np.flatnonzero(held)].copy()
        keep = np.ones(len(points), dtype=bool)
        quantum = module.recording_intensity_step(raw) if internal_peak and peak_quantisation else 0.0
        frames, height, width = raw.shape
        dtype = raw.dtype.str
        decisions = memo.decisions
        eligible = []
        for i, q in enumerate(points.itertuples(index=False)):
            if not bool(q.strong) or str(q.state) != 'observed':
                continue
            if hasattr(q, 'physically_visible') and not bool(q.physically_visible):
                continue
            if not hasattr(q, 'physically_visible') and hasattr(q, 'visibility') and float(q.visibility) <= 0:
                continue
            t = int(q.frame); x = float(q.x); y = float(q.y); r = float(getattr(q, 'radius_px', 0.))
            if not 0 <= t < frames or not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)) or r < 0:
                continue
            y0 = max(0, math.floor(y-r)); y1 = min(height, math.ceil(y+r)+1)
            x0 = max(0, math.floor(x-r)); x1 = min(width, math.ceil(x+r)+1)
            if y0 >= y1 or x0 >= x1:
                continue
            eligible.append((i, t, x, y, r, y0, y1, x0, x1))
        # Points not yet remembered are measured together, every part at once;
        # a point whose remembered patch differs is measured alone below.
        batch = _measure_many(raw, list(dict.fromkeys(
            (t, y0, y1, x0, x1, x, y, r) for _, t, x, y, r, y0, y1, x0, x1 in eligible
            if (t, y0, y1, x0, x1, x, y, r, dtype) not in decisions)))
        memo.stats['batched'] = memo.stats.get('batched', 0) + len(batch)
        for i, t, x, y, r, y0, y1, x0, x1 in eligible:
            memo.stats['points'] += 1
            key = (t, y0, y1, x0, x1, x, y, r, dtype)
            patch = np.asarray(raw[t, y0:y1, x0:x1])
            held = decisions.get(key)
            if held is not None and held[0].shape == patch.shape and bool((held[0] == patch).all()):
                memo.stats['reused'] += 1
                parts = held[1]
                if parts is not None:
                    need_centre = centre_connected and parts[0] and parts[1] is None
                    need_contrast = (bool(internal_peak) and parts[0] and parts[2] is None
                                     and (not centre_connected or need_centre or parts[1]))
                    if need_centre or need_contrast:
                        memo.stats['parts_measured'] += 1
                        _measure(raw, t, y0, y1, x0, x1, x, y, r, need_centre, need_contrast, parts)
                        if parts[2] is None and internal_peak and _derive(parts, centre_connected, False, False, 0.0):
                            _measure(raw, t, y0, y1, x0, x1, x, y, r, False, True, parts)
            else:
                measured = batch.get((t, y0, y1, x0, x1, x, y, r), False)
                if measured is False:
                    parts = _measure(raw, t, y0, y1, x0, x1, x, y, r, centre_connected, bool(internal_peak))
                else:
                    parts = None if measured is None else list(measured)
                memo.stats['measured'] += 1
                if held is None and memo.bytes + patch.nbytes <= memo.budget:
                    decisions[key] = (patch.copy(), parts)
                    memo.bytes += patch.nbytes
            decision = None if parts is None else _derive(parts, centre_connected, internal_peak,
                                                          peak_quantisation, quantum)
            if decision is not None:
                keep[i] = decision
        if token is not None:
            memo.store(call_key, points, keep)
        return points.iloc[np.flatnonzero(keep)].copy()
    raw_supported_points.__wrapped__ = original
    return raw_supported_points


# The per-point statements of every tracker_support_frames version, from the
# disk through the resolved component; versions before ownership labels have no
# resolved_core flag and ignore it.
CORE_BLOCK = '''        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy-y)**2 + (xx-x)**2 <= r*r
        cy, cx = int(np.floor(y+.5))-y0, int(np.floor(x+.5))-x0
        if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1]:
            disk[cy, cx] = True
        support = disk
RESOLVED        interior = ndi.binary_erosion(disk, cross)
        if interior.any():
            patch = raw[t, y0:y1, x0:x1]
            bright = disk & (patch > np.median(patch[disk & ~interior]))
            components, _ = ndi.label(bright, np.ones((3, 3), bool))
            seed = int(components[cy, cx]) if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1] else 0
            if seed and ndi.binary_erosion(components == seed, cross).any():
                support = components == seed
FLAG'''
CORE_CALL = '''        cy, cx = _floor(y+.5)-y0, _floor(x+.5)-x0
        support, resolved_core = _core_support(raw, t, y0, y1, x0, x1, x, y, r, cy, cx, cross)
'''
# Per-point statements answered without NumPy scalars (math gives the same
# integers and finiteness for Python floats), and each frame's coverage made
# once instead of a fresh zero frame per point that setdefault discards. Each
# applies only where its text appears once.
FRAME_EDITS = (
    ('        if not 0 <= t < len(raw) or not np.isfinite([x, y, r]).all() or r < 0:\n',
     '        if not 0 <= t < len(raw) or not (_isfinite(x) and _isfinite(y) and _isfinite(r)) or r < 0:\n'),
    ('y0 = max(0, int(np.floor(y-r))); y1 = min(raw.shape[1], int(np.ceil(y+r))+1)',
     'y0 = max(0, _floor(y-r)); y1 = min(raw.shape[1], _ceil(y+r)+1)'),
    ('x0 = max(0, int(np.floor(x-r))); x1 = min(raw.shape[2], int(np.ceil(x+r))+1)',
     'x0 = max(0, _floor(x-r)); x1 = min(raw.shape[2], _ceil(x+r)+1)'),
    ('        result.setdefault(t, np.zeros(raw.shape[1:], bool))[y0:y1, x0:x1] |= support\n',
     '        _cover = result.get(t)\n'
     '        if _cover is None:\n'
     '            _cover = result[t] = np.zeros(raw.shape[1:], bool)\n'
     '        _cover[y0:y1, x0:x1] |= support\n'),
    # Before the point loop, measure every point's core support not yet
    # remembered in one batch (see _core_prepare).
    ('    for q in selected.itertuples(index=False):\n',
     '    _core_prepare(raw, selected, cross)\n'
     '    for q in selected.itertuples(index=False):\n'),
)
OWNERS = ('ownership_cache[t] = component_owners(ownership_labels[t])',
          'ownership_cache[t] = _component_owners(component_owners, ownership_labels[t])')


def _core_support_original(raw, t, y0, y1, x0, x1, x, y, r, cy, cx, cross):
    """The frozen per-point statements; returns (support, resolved_core)."""
    yy, xx = _grid(y0, y1, x0, x1)
    disk = (yy-y)**2 + (xx-x)**2 <= r*r
    if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1]:
        disk[cy, cx] = True
    support = disk
    resolved_core = False
    interior = _erode(disk, cross)
    if interior.any():
        patch = raw[t, y0:y1, x0:x1]
        bright = disk & (patch > _median(patch[disk & ~interior]))
        components, _ = _ndi_measurements.label(bright, EIGHT)
        seed = int(components[cy, cx]) if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1] else 0
        if seed and _erode(components == seed, cross).any():
            support = components == seed
            resolved_core = True
    return support, resolved_core


def fast_frames(module, original, memo):
    """The version's own function text with core support and component owners remembered, or None."""
    import inspect
    import textwrap
    held = memo.fast.get(id(original))
    if held is not None and held[0] is original:
        return held[1]
    function = None
    try:
        text = inspect.getsource(original).rstrip(chr(10))
    except (OSError, TypeError):
        text = None
    if text is not None:
        resolved = '        resolved_core = False\n' in text
        block = (CORE_BLOCK.replace('RESOLVED', '        resolved_core = False\n' if resolved else '')
                 .replace('FLAG', '                resolved_core = True\n' if resolved else ''))
        owners = text.count(OWNERS[0])
        if text.count(block) == 1 and owners == int(resolved):
            text = text.replace(block, CORE_CALL).replace(OWNERS[0], OWNERS[1])
            for old, new in FRAME_EDITS:
                if text.count(old) == 1:
                    text = text.replace(old, new)
            factory = ('def _factory(_core_support, _component_owners, _isfinite, _floor, _ceil, _core_prepare):\n'
                       + textwrap.indent(text, '    ') + '\n    return tracker_support_frames\n')
            scope = {}
            exec(compile(factory, getattr(module, '__file__', '<tracker_support_frames>'), 'exec'),
                 module.__dict__, scope)
            function = scope['_factory'](lambda *args: _remembered_core(memo, *args),
                                         lambda owner, field: _remembered_owners(memo, owner, field),
                                         math.isfinite, math.floor, math.ceil,
                                         lambda raw, selected, cross: _core_prepare(memo, raw, selected, cross))
    memo.fast[id(original)] = (original, function)
    return function


def _cores_many(raw, items):
    """{item: _core_support_original(raw, *item, cy, cx, CROSS)} for an integer movie of at most 32 bits, else {}.

    The same statements as _measure_many for the disk, centre, interior and
    bright mask; the centre's bright component is grown from the centre pixel
    (scipy's eight-connected label of the centre holds the same pixels). The
    support is that component when it has a cross-eroded pixel, else the disk.
    """
    out = {}
    plain = np.asarray(raw)
    if plain.dtype.kind not in 'iu' or plain.dtype.itemsize > 4 or plain.ndim != 3:
        return out
    shapes = {}
    for item in items:
        shapes.setdefault((item[2] - item[1], item[4] - item[3]), []).append(item)
    for (h, w), group in shapes.items():
        n = len(group)
        T = np.fromiter((g[0] for g in group), np.int64, n)
        Y0 = np.fromiter((g[1] for g in group), np.int64, n)
        X0 = np.fromiter((g[3] for g in group), np.int64, n)
        X = np.fromiter((g[5] for g in group), np.float64, n)
        Y = np.fromiter((g[6] for g in group), np.float64, n)
        R = np.fromiter((g[7] for g in group), np.float64, n)
        yy = Y0[:, None, None] + np.arange(h, dtype=np.int64)[None, :, None]
        xx = X0[:, None, None] + np.arange(w, dtype=np.int64)[None, None, :]
        disk = (yy-Y[:, None, None])**2 + (xx-X[:, None, None])**2 <= (R*R)[:, None, None]
        cy = np.fromiter((math.floor(g[6]+.5) - g[1] for g in group), np.int64, n)
        cx = np.fromiter((math.floor(g[5]+.5) - g[3] for g in group), np.int64, n)
        rows = np.flatnonzero((cy >= 0) & (cy < h) & (cx >= 0) & (cx < w))
        disk[rows, cy[rows], cx[rows]] = True
        interior = _erode_many(disk)
        valid = interior.any(axis=(1, 2))
        boundary = disk & ~interior
        patch = plain[T[:, None, None], yy, xx].astype(np.int64)
        counts = boundary.sum(axis=(1, 2))
        rim = np.sort(np.where(boundary, patch, np.iinfo(np.int64).max).reshape(n, h * w), axis=1)
        half = counts // 2
        upper = rim[np.arange(n), half].astype(np.float64)
        lower = rim[np.arange(n), np.maximum(half - 1, 0)].astype(np.float64)
        level = np.where(counts % 2 == 1, upper, (lower + upper) / 2.0)
        bright = disk & (patch > level[:, None, None])
        seed = np.zeros(disk.shape, bool)
        seed[rows, cy[rows], cx[rows]] = True
        core = _grow_many(seed, bright)
        resolved = valid & core.any(axis=(1, 2)) & _erode_many(core).any(axis=(1, 2))
        for j, item in enumerate(group):
            if resolved[j]:
                out[item] = (core[j].copy(), True)
            else:
                out[item] = (disk[j].copy(), False)
    return out


def _core_prepare(memo, raw, selected, cross):
    """Remember, in one batch, the core support of every selected point not yet remembered.

    The points, bounds and keys are those of the version's loop; entries are
    kept while the core budget lasts, as _remembered_core keeps them. Any
    failure leaves the loop to measure each point itself."""
    try:
        if not (isinstance(cross, np.ndarray) and cross.dtype == np.bool_ and cross.shape == (3, 3)
                and np.array_equal(cross, CROSS)):
            return
        frames, height, width = raw.shape
        dtype = raw.dtype.str
        wanted = {}
        for q in selected.itertuples(index=False):
            t = int(q.frame); x = float(q.x); y = float(q.y); r = float(q.radius_px)
            if not 0 <= t < frames or not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)) or r < 0:
                continue
            y0 = max(0, math.floor(y-r)); y1 = min(height, math.ceil(y+r)+1)
            x0 = max(0, math.floor(x-r)); x1 = min(width, math.ceil(x+r)+1)
            if y0 >= y1 or x0 >= x1:
                continue
            key = (t, y0, y1, x0, x1, x, y, r, dtype)
            if key not in memo.cores:
                wanted[key] = None
        if not wanted:
            return
        found = _cores_many(raw, [key[:8] for key in wanted])
        for key in wanted:
            entry = found.get(key[:8])
            if entry is None:
                continue
            support, resolved = entry
            patch = np.asarray(raw[key[0], key[1]:key[2], key[3]:key[4]])
            support.flags.writeable = False
            size = patch.nbytes + support.nbytes + 160
            if memo.core_bytes + size > memo.core_budget:
                break
            memo.cores[key] = (patch.copy(), support, resolved)
            memo.core_bytes += size
            memo.stats['cores_batched'] = memo.stats.get('cores_batched', 0) + 1
    except Exception:
        memo.stats['cores_batch_skipped'] = memo.stats.get('cores_batch_skipped', 0) + 1


def _remembered_core(memo, raw, t, y0, y1, x0, x1, x, y, r, cy, cx, cross):
    key = (t, y0, y1, x0, x1, x, y, r, raw.dtype.str)
    patch = np.asarray(raw[t, y0:y1, x0:x1])
    held = memo.cores.get(key)
    if (held is not None and held[0].dtype == patch.dtype and held[0].shape == patch.shape
            and bool((held[0] == patch).all())):
        memo.stats['cores_reused'] += 1
        return held[1], held[2]
    support, resolved = _core_support_original(raw, t, y0, y1, x0, x1, x, y, r, cy, cx, cross)
    memo.stats['cores_measured'] += 1
    if type(support) is np.ndarray and support.dtype == bool:
        support.flags.writeable = False
        size = patch.nbytes + support.nbytes + 160
        if held is None and memo.core_bytes + size <= memo.core_budget:
            memo.cores[key] = (patch.copy(), support, resolved)
            memo.core_bytes += size
    return support, resolved


def _remembered_owners(memo, component_owners, field):
    plain = np.ascontiguousarray(field)
    if type(field) is not np.ndarray and not isinstance(field, np.ndarray):
        return component_owners(field)
    # Frozen versions load their own copies of the same function text.
    source = memo.owner_sources.get(id(component_owners))
    if source is None or source[0] is not component_owners:
        import inspect
        try:
            text = inspect.getsource(component_owners)
        except (OSError, TypeError):
            text = None
        names = getattr(component_owners, '__globals__', {})
        same = names.get('np') is np and names.get('ndi') is ndi
        source = (component_owners, hashlib.sha256(text.encode()).digest() if text and same else None)
        memo.owner_sources[id(component_owners)] = source
    identity = source[1] if source[1] is not None else component_owners
    key = (identity, plain.shape, plain.dtype.str,
           hashlib.sha256(memoryview(plain).cast('B')).digest())
    held = memo.owners.get(key)
    if held is not None:
        memo.stats['owners_reused'] += 1
        shape, dtype, where, values = held
        owners = np.zeros(shape, dtype)
        owners.reshape(-1)[where] = values
        return owners
    owners = component_owners(field)
    memo.stats['owners_measured'] += 1
    # Owner frames are about 3% foreground: keep only the nonzero pixels, so the
    # whole movie fits the budget, and rebuild a fresh array on each reuse.
    if type(owners) is np.ndarray and owners.dtype.kind in 'iu' and owners.flags.c_contiguous:
        # Positions found in a Boolean array: the same, five times faster than in labels.
        where = np.flatnonzero(owners != 0)
        values = owners.reshape(-1)[where]
        if len(values) and values.min() >= 0 and values.max() <= np.iinfo(np.uint16).max:
            values = values.astype(np.uint16)
        where = where.astype(np.int32) if owners.size <= np.iinfo(np.int32).max else where
        size = where.nbytes + values.nbytes
        if memo.owner_bytes + size <= memo.owner_budget:
            memo.owners[key] = (owners.shape, owners.dtype, where, values)
            memo.owner_bytes += size
    return owners


def support_frames(module, original, accepted, memo):
    def tracker_support_frames(raw, points, *args, **kwargs):
        if (args or set(kwargs) - set(accepted)
                or not isinstance(raw, np.ndarray) or raw.ndim != 3 or not points.columns.is_unique):
            memo.stats['frames_fallback'] += 1
            return original(raw, points, *args, **kwargs)
        if kwargs.get('ownership_labels') is not None:
            # Ownership frames are corrected between stages; only per-point
            # core support and per-frame owners are remembered.
            fast = fast_frames(module, original, memo)
            if fast is None:
                memo.stats['frames_fallback'] += 1
                return original(raw, points, *args, **kwargs)
            memo.stats['frames_ownership'] += 1
            return fast(raw, points, *args, **kwargs)
        memo.stats['frames_calls'] += 1
        token = movie_token(raw)
        if token is None:
            memo.stats['frames_fallback'] += 1
            return original(raw, points, *args, **kwargs)
        key = ('tracker_support_frames', token, bool(kwargs.get('internal_peak', False)),
               bool(kwargs.get('peak_quantisation', False)))
        held = memo.find(key, points)
        if held is not None:
            memo.stats['frames_reused'] += 1
            shape = raw.shape[1:]
            size = shape[0] * shape[1]
            return {t: np.unpackbits(bits, count=size).reshape(shape).astype(bool) for t, bits in held}
        fast = fast_frames(module, original, memo)
        result = (original if fast is None else fast)(raw, points, *args, **kwargs)
        packed = [(t, np.packbits(mask.ravel())) for t, mask in result.items()]
        memo.store(key, points, packed, int(points.memory_usage(index=True, deep=True).sum())
                   + sum(bits.nbytes + 64 for _, bits in packed))
        return result
    tracker_support_frames.__wrapped__ = original
    return tracker_support_frames


def _file_digest(path):
    try:
        return _file_sha256(path)
    except (OSError, TypeError):
        return None


def _frames_version(module, function):
    """Accepted keywords of a known tracker_support_frames whose helpers are known versions, else None."""
    import inspect
    try:
        # Function text as it stands in the file, without the trailing newline.
        digest = hashlib.sha256(inspect.getsource(function).rstrip(chr(10)).encode()).hexdigest()
    except (OSError, TypeError):
        return None
    temporal = getattr(module, 'temporally_supported_points', None)
    if digest not in TRACKER_SUPPORT or temporal is None:
        return None
    temporal_module = sys.modules.get(getattr(temporal, '__module__', None))
    if temporal_module is None or _file_digest(getattr(temporal_module, '__file__', None)) not in TEMPORAL:
        return None
    qualified = getattr(temporal_module, 'raw_supported_points', None)
    qualified = getattr(qualified, '__wrapped__', qualified)
    qualified_module = sys.modules.get(getattr(qualified, '__module__', None))
    if qualified_module is None or _file_digest(getattr(qualified_module, '__file__', None)) not in VERSIONS:
        return None
    if getattr(module, 'np', None) is not np or getattr(module, 'ndi', None) is not ndi:
        return None
    return TRACKER_SUPPORT[digest]


TEMPORAL_SORT = ("        track=track.sort_values('frame',kind='stable')\n",
                 "        track=_temporal_ordered(track)\n")


def _temporal_ordered(memo):
    def ordered(track):
        # A stable sort of frames that never decrease keeps every row in place;
        # the track is only read afterwards.
        frame = track['frame'] if track.columns.is_unique and 'frame' in track.columns else None
        if frame is not None and isinstance(frame.dtype, np.dtype) and frame.dtype.kind in 'iu':
            values = frame.to_numpy()
            if len(values) < 2 or bool(np.all(values[1:] >= values[:-1])):
                memo.stats['temporal_in_order'] = memo.stats.get('temporal_in_order', 0) + 1
                return track
        memo.stats['temporal_sorted'] = memo.stats.get('temporal_sorted', 0) + 1
        return track.sort_values('frame', kind='stable')
    return ordered


def _temporal_text(module, original, memo):
    """The version's temporally_supported_points with the per-track sort skipped when in order, else None."""
    import inspect
    import textwrap
    try:
        text = inspect.getsource(original).rstrip(chr(10))
    except (OSError, TypeError):
        return None
    if text.count(TEMPORAL_SORT[0]) != 1:
        return None
    text = text.replace(*TEMPORAL_SORT)
    factory = ('def _factory(_temporal_ordered):\n' + textwrap.indent(text, '    ')
               + '\n    return temporally_supported_points\n')
    scope = {}
    exec(compile(factory, getattr(module, '__file__', '<temporally_supported_points>'), 'exec'),
         module.__dict__, scope)
    function = scope['_factory'](_temporal_ordered(memo))
    function.__defaults__ = original.__defaults__
    function.__kwdefaults__ = original.__kwdefaults__
    return function


def temporal_points(module, original, memo):
    """temporally_supported_points answered from its kept-row mask for a repeated movie, table and flags."""
    import inspect
    try:
        accepted = set(inspect.signature(original).parameters) - {'raw', 'points'}
        source = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    except (OSError, TypeError, ValueError):
        return None
    measure = _temporal_text(module, original, memo) or original

    def temporally_supported_points(raw, points, *args, **kwargs):
        token = None
        if (not args and not set(kwargs) - accepted and all(type(v) is bool for v in kwargs.values())
                and isinstance(raw, np.ndarray) and raw.ndim == 3 and hasattr(points, 'columns')
                and points.columns.is_unique and points.index.is_unique):
            token = movie_token(raw)
        if token is None:
            memo.stats['temporal_direct'] = memo.stats.get('temporal_direct', 0) + 1
            return measure(raw, points, *args, **kwargs)
        key = ('temporally_supported_points', source, token, tuple(sorted(kwargs.items())))
        held = memo.find(key, points)
        if held is not None:
            memo.stats['temporal_reused'] = memo.stats.get('temporal_reused', 0) + 1
            return points.iloc[np.flatnonzero(held)].copy()
        result = measure(raw, points, **kwargs)
        memo.stats['temporal_measured'] = memo.stats.get('temporal_measured', 0) + 1
        # The result is points.iloc[rows].copy() for increasing rows; keep them
        # only when the rows are recovered exactly from its unique index.
        positions = points.index.get_indexer(result.index)
        if (len(positions) == len(result) and (positions >= 0).all() and (np.diff(positions) > 0).all()
                and same_table(points.iloc[positions], result)):
            keep = np.zeros(len(points), bool)
            keep[positions] = True
            memo.store(key, points, keep)
        return result
    temporally_supported_points.__wrapped__ = original
    return temporally_supported_points


@contextmanager
def install(memo):
    """Replace every loaded frozen version's raw_supported_points, including imported aliases."""
    replaced = []
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        path = getattr(module, '__file__', None)
        original = getattr(module, 'raw_supported_points', None)
        if not path or original is None or getattr(original, '__module__', None) != module.__name__:
            continue
        if getattr(original, '__wrapped__', None) is not None:
            continue
        try:
            digest = _file_sha256(path)
        except OSError:
            continue
        if digest not in VERSIONS:
            continue
        replaced.append((module, original, supported_points(module, original, VERSIONS[digest], memo)))
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, 'tracker_support_frames', None)
        if (original is None or getattr(original, '__module__', None) != module.__name__
                or getattr(original, '__wrapped__', None) is not None):
            continue
        accepted = _frames_version(module, original)
        if accepted is not None:
            replaced.append((module, original, support_frames(module, original, accepted, memo)))
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, 'temporally_supported_points', None)
        if (original is None or getattr(original, '__module__', None) != module.__name__
                or getattr(original, '__wrapped__', None) is not None
                or _file_digest(getattr(module, '__file__', None)) not in TEMPORAL):
            continue
        qualified = getattr(module, 'raw_supported_points', None)
        qualified = getattr(qualified, '__wrapped__', qualified)
        qualified_module = sys.modules.get(getattr(qualified, '__module__', None))
        if qualified_module is None or _file_digest(getattr(qualified_module, '__file__', None)) not in VERSIONS:
            continue
        wrapper = temporal_points(module, original, memo)
        if wrapper is not None:
            replaced.append((module, original, wrapper))
    aliases = []
    if replaced:
        memo.stats['versions'] += len(replaced)
        swap = {id(original): new for _, original, new in replaced}
        for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
            names = getattr(child, '__dict__', None)
            if not names:
                continue
            if not swap or swap.keys().isdisjoint(map(id, names.values())):  # alias loop skip
                continue
            for name, value in list(names.items()):
                new = swap.get(id(value))
                if new is not None and any(value is original for _, original, _ in replaced):
                    aliases.append((child, name, value))
                    setattr(child, name, new)
    try:
        yield memo.stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
