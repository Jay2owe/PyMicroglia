"""The smoothed raw frame of _shared_body_basin, measured once per frame and sigma.

overlapping_detector_soma_memory_a003 (also called from
known_soma_birth_validation_a004) and successive_detector_soma_memory_a001
split shared bodies with _shared_body_basin, which smoothed the whole raw frame
with ndi.gaussian_filter(raw.astype(np.float32), sigma) for every component it
tested; components of one frame share the frame and sigma. The statement runs
unchanged in the module's names; its result is held under the frame's
SHA-256 (with dtype and shape), sigma's type and value and the statement's
compiled code, and a later call with the same key gets a fresh copy of the
same array. Frames that are not 2-D numeric ndarrays run the statement
directly. A few frames are held at a time.
"""
import hashlib

import numpy as np
try:
    from .track_sort_skips import rewrite
except ImportError:
    from track_sort_skips import rewrite

KEYS = ('versions', 'smoothed', 'reused', 'direct')
HELD = 4
OLD = "    smooth=ndi.gaussian_filter(raw.astype(np.float32),sigma)\n"
NEW = "    smooth=_smoothed(raw,sigma,lambda:ndi.gaussian_filter(raw.astype(np.float32),sigma))\n"

# One store for the process: content keys make a held frame valid in any stage.
_HELD = {}


def smoothed(stats):
    held = _HELD

    def _smoothed(raw, sigma, block):
        if not (isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.dtype.kind in 'iuf'):
            stats['direct'] += 1
            return block()
        plain = np.ascontiguousarray(raw.view(np.ndarray))
        code = block.__code__
        key = (raw.dtype.str, raw.shape, hashlib.sha256(memoryview(plain).cast('B')).digest(),
               type(sigma).__name__ + repr(sigma), code.co_code, repr(code.co_consts), code.co_names)
        value = held.get(key)
        if value is not None:
            stats['reused'] += 1
            return value.copy()
        value = block()
        stats['smoothed'] += 1
        if type(value) is np.ndarray:
            if len(held) >= HELD:
                held.pop(next(iter(held)))
            held[key] = value.copy()
        return value
    return _smoothed


def _spec(module, digest):
    return dict(module=module, function='_shared_body_basin', digests={digest}, edits=((OLD, NEW),),
                helper='_smoothed', factory=None)


SPECS = (
    _spec('overlapping_detector_soma_memory_a003',
          '5a4c49e32a9109e6925c594da2868c32d844b700947913901ab8d9d76abd1259'),
    _spec('successive_detector_soma_memory_a001',
          '58de79bab104f0127770ab80c4c49073a2278f03488b3f94922f939136fca157'),
)


def install(stats=None):
    """Rewrite the verified _shared_body_basin versions to share held smoothed frames; restore afterwards."""
    stats = stats if stats is not None else {}
    shared = smoothed(stats)
    specs = tuple(dict(spec, factory=lambda _stats: shared) for spec in SPECS)
    return rewrite(specs, KEYS, '<smoothed-frame-memo>', stats)
