"""Stable overbright cores measured once per raw movie for the three overbright-source corrections.

remove_persistent_overbright_sources, remove_compact_stable_overbright_sources
and remove_extreme_compact_overbright_sources each begin _eligible_cores with
the same six statements: every frame's median positive signal (background),
the count of frames in which each pixel is at least CORE_SIGNAL_FACTOR times
it, and the pixels counted in at least CORE_TIME_FRACTION of the frames
(stable). All three use the same two constants, and each function runs twice
on the same raw movie (its own correction and the optical-core union of
remove_optical_core_adjacent_pseudosomata). The six statements are kept,
unchanged, in a local function run in the module's own names; their result is
held under the raw movie's SHA-256 (with its dtype and shape), the two
constants and the statements' compiled code, and a later call with the same
key gets fresh copies of the same background and stable arrays. Raw movies
that are not 3-D numeric ndarrays run the statements directly.
"""
from contextlib import contextmanager
import hashlib

import numpy as np
try:
    from .track_sort_skips import rewrite
except ImportError:
    from track_sort_skips import rewrite

KEYS = ('versions', 'measured', 'reused', 'direct')
OLD = ("    background = np.array([np.median(frame[frame > 0]) if np.any(frame > 0) else 1.\n"
       "                           for frame in raw], dtype=float)\n"
       "    counts = np.zeros(raw.shape[1:], np.uint16)\n"
       "    for t, frame in enumerate(raw):\n"
       "        counts += frame >= CORE_SIGNAL_FACTOR * background[t]\n"
       "    stable = counts >= int(np.ceil(CORE_TIME_FRACTION * len(raw)))\n")
NEW = ("    def _block():\n"
       + ''.join('    ' + line + '\n' for line in OLD.splitlines())
       + "        return background, stable\n"
       "    background, stable = _core_stable(raw, (CORE_SIGNAL_FACTOR, CORE_TIME_FRACTION), _block)\n")


# One slot for the process: the six calls fall in different stages, each with
# its own install. Content keys make a held answer valid in any of them.
_HELD = {}


def core_stable(stats):
    held = _HELD

    def _core_stable(raw, constants, block):
        if not (isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.dtype.kind in 'iuf'):
            stats['direct'] += 1
            return block()
        plain = np.ascontiguousarray(raw)
        digest = hashlib.sha256(memoryview(plain).cast('B')).digest()
        code = block.__code__
        key = (raw.dtype.str, raw.shape, digest, tuple(type(c).__name__ + repr(c) for c in constants),
               code.co_code, repr(code.co_consts), code.co_names, code.co_freevars)
        if held.get('key') == key:
            stats['reused'] += 1
            background, stable = held['value']
            return background.copy(), stable.copy()
        background, stable = block()
        stats['measured'] += 1
        held.clear()
        if type(background) is np.ndarray and type(stable) is np.ndarray:
            held.update(key=key, value=(background.copy(), stable.copy()))
        return background, stable
    return _core_stable


def _spec(module, digest):
    return dict(module=module, function='_eligible_cores', digests={digest}, edits=((OLD, NEW),),
                helper='_core_stable', factory=None)


SPECS = (
    _spec('remove_persistent_overbright_sources',
          '646f7944ba8df5dfafb1fa6498b36ed7b9f50f5baee712565d7e5678399ccd59'),
    _spec('remove_compact_stable_overbright_sources',
          '61401bc2d4339b607445680d54b90191d0e0b1824e9cef94234421eed8e28048'),
    _spec('remove_extreme_compact_overbright_sources',
          '78788fc50cdd06821e7834c250eceb8c35a8e320908dcd0a46a1e9c13f71180b'),
)


@contextmanager
def install(stats=None):
    """Rewrite the three verified _eligible_cores to share one held measurement; restore afterwards."""
    stats = stats if stats is not None else {}
    shared = core_stable(stats)
    specs = tuple(dict(spec, factory=lambda _stats: shared) for spec in SPECS)
    with rewrite(specs, KEYS, '<core-stable-memo>', stats) as found:
        yield found
