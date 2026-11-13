"""Per-frame cell measurements reused for frames with identical content.

Frame observations depend only on the label frame, the raw frame, whether the
shape is measured, and three optional per-cell lookups that fill the last
three fields. Stages read the same frames many times; a frame whose label and
raw bytes match an earlier call gets that call's measurements again. Label
frames are kept as the positions and values of their nonzero pixels (or
compressed, when the snapshot's dtype or shape differs from the frame's) and
compared in full (found by an Adler-32 checksum); raw frames by a SHA-256
digest of dtype, shape and bytes.
The rebuilt frame is a read-only snapshot like the reader's own. Every call receives new row
dictionaries, in the same key order, with their own position and shape arrays
and equality masks over the shared snapshot, so callers that edit rows cannot
reach each other or the cache. Eviction only repeats the original work.
"""
import hashlib
import zlib
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
try:
    from . import resolved_paths as _paths
    from . import narrow_labels as _narrow_labels
except ImportError:
    import resolved_paths as _paths
    import narrow_labels as _narrow_labels
import sys

import numpy as np

FIELDS = ('local_id', 'mask', 'position', 'bounds', 'area', 'mean', 'shape',
          'green_track_ids', 'preferred_identity', 'preferred_identity_protected')


def _plain(value):
    return value.view(np.ndarray) if isinstance(value, np.ndarray) else None


def _digest(array):
    # SHA-256 runs on the processor's SHA instructions, about three times
    # faster here than BLAKE2b; the digest only selects an entry.
    data = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str((data.dtype.str, data.shape)).encode())
    hasher.update(memoryview(data).cast('B'))
    return hasher.digest()


def _label_digest(array):
    # Only selects an entry: a hit still compares the whole label frame, so an
    # Adler-32 checksum of the bytes (dtype and shape are in the key) is enough
    # and costs less than narrowing the frame for SHA-256.
    return zlib.adler32(memoryview(np.ascontiguousarray(array)).cast('B'))


def _pack(snapshot, dtype, shape):
    plain = np.ascontiguousarray(snapshot.view(np.ndarray))
    if plain.dtype != dtype or plain.shape != shape:
        packed = zlib.compress(plain.tobytes(), 1)
        return ('bytes', packed), len(packed)
    flat = plain.ravel()
    # A number is nonzero exactly when it compares unequal to 0; NumPy finds
    # those positions about five times faster in the Boolean array.
    where = np.flatnonzero(flat != 0) if flat.dtype.kind in 'iufc' else np.flatnonzero(flat)
    index = where.astype(np.int32) if flat.size < (1 << 31) else where
    values = flat[where]
    return ('sparse', index, values), index.nbytes + values.nbytes


def _matches(packed, labels):
    """Whether the frame equals the stored snapshot, and that snapshot."""
    if packed[0] == 'bytes':
        snapshot = np.frombuffer(zlib.decompress(packed[1]), dtype=labels.dtype).reshape(labels.shape)
        return np.array_equal(snapshot, labels), snapshot
    _, index, values = packed
    flat = labels.ravel()
    # Every stored value is nonzero: the frame equals the snapshot exactly when
    # it has as many nonzero pixels and those positions hold those values.
    if np.count_nonzero(flat) != len(index) or not np.array_equal(flat[index], values):
        return False, None
    snapshot = np.zeros(flat.size, labels.dtype)
    snapshot[index] = values
    snapshot = snapshot.reshape(labels.shape)
    snapshot.flags.writeable = False
    return True, snapshot


class ObservationCache:
    def __init__(self, reader, label_mask, *, entries=4096, ram_bytes=256 << 20):
        self.reader, self.label_mask = reader, label_mask
        self.entries, self.ram_bytes = int(entries), int(ram_bytes)
        self.cache = OrderedDict()
        self.ram = 0
        self.stats = dict(hits=0, misses=0, uncached=0, evicted=0)

    def __call__(self, labels, raw, green_tracks=None, preferred_identities=None,
                 protected_preferred=None, include_shape=True, *rest, **kwargs):
        plain_labels, plain_raw = _plain(labels), _plain(raw)
        if (rest or kwargs or plain_labels is None or plain_raw is None or plain_labels.ndim != 2
                or plain_labels.dtype.kind not in 'iub'):
            self.stats['uncached'] += 1
            return self.reader(labels, raw, green_tracks, preferred_identities,
                               protected_preferred, include_shape, *rest, **kwargs)
        key = (plain_labels.dtype.str, plain_labels.shape, _label_digest(plain_labels),
               _digest(plain_raw), bool(include_shape))
        entry = self.cache.get(key)
        if entry is not None:
            if entry[0] is None:
                same, snapshot = not plain_labels.any(), None
            else:
                same, snapshot = _matches(entry[0], plain_labels)
            if same:
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return self._rows(snapshot, entry[1], green_tracks, preferred_identities,
                                  protected_preferred)
        rows = self.reader(labels, raw, green_tracks, preferred_identities,
                           protected_preferred, include_shape)
        self.stats['misses'] += 1
        snapshot = None
        cores = []
        for local_id, row in rows.items():
            if list(row) != list(FIELDS) or not isinstance(row['mask'], self.label_mask):
                return rows
            snapshot = row['mask'].labels
            cores.append((local_id, row['position'].copy(), row['bounds'], row['area'], row['mean'],
                          None if row['shape'] is None else row['shape'].copy()))
        if snapshot is None and plain_labels.any():
            return rows
        packed, cost = (None, 0) if snapshot is None else _pack(
            snapshot, plain_labels.dtype, plain_labels.shape)
        self.cache[key] = (packed, cores, cost)
        self.ram += cost
        while self.cache and (len(self.cache) > self.entries or self.ram > self.ram_bytes):
            _, (_, _, old) = self.cache.popitem(last=False)
            self.ram -= old
            self.stats['evicted'] += 1
        return rows

    def _rows(self, snapshot, cores, green_tracks, preferred_identities, protected_preferred):
        result = {}
        for local_id, position, bounds, area, mean, shape in cores:
            result[local_id] = {
                "local_id": local_id,
                "mask": self.label_mask(snapshot, local_id),
                "position": position.copy(),
                "bounds": bounds,
                "area": area,
                "mean": mean,
                "shape": None if shape is None else shape.copy(),
                "green_track_ids": set() if green_tracks is None
                                   else set(green_tracks.get(local_id, set())),
                "preferred_identity": 0 if preferred_identities is None
                                      else int(preferred_identities.get(local_id, 0)),
                "preferred_identity_protected": (
                    False if protected_preferred is None
                    else bool(protected_preferred.get(local_id, False))),
            }
        return result


@contextmanager
def install(code_root, label_mask, expected_source, **limits):
    """Wrap the installed observation reader in every engine alias."""
    import importlib, inspect
    tracking = importlib.import_module('tracking')
    reader = tracking.frame_observations
    cells = dict(zip(reader.__code__.co_freevars, reader.__closure__ or ()))
    original = cells.get('original')
    original = None if original is None else original.cell_contents
    stats = dict(supported=False)
    if (original is None or hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
            != expected_source):
        yield stats
        return
    cache = ObservationCache(reader, label_mask, **limits)
    aliases = []
    try:
        root = _paths.resolve(code_root)
    except (ValueError, OSError):
        root = None
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        location = getattr(module, '__file__', None)
        # Only a module holding the reader can be wrapped, so the path is
        # resolved for those alone.
        if location is None or root is None or getattr(module, 'frame_observations', None) is not reader:
            continue
        try:
            inside = _paths.resolve(location).is_relative_to(root)
        except (ValueError, OSError):
            inside = False
        if inside:
            aliases.append(module)
            module.frame_observations = cache
    stats.update(supported=True, cache=cache.stats)
    try:
        yield stats
    finally:
        for module in aliases:
            module.frame_observations = reader
        cache.cache.clear()
        print('observation_cache', cache.stats, file=sys.stderr)
