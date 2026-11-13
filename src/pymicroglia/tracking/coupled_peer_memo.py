"""Coupled observed name return asks once whether each observation's own outline holds a peer.

coupled_observed_name_return_a002.correct finds bookends for every strong
observation by walking its track: each observation within the gap window has
its own outline (component, cached per frame and track) tested for another
track's point (peers), so one observation is tested again for every
neighbour whose window contains it. The overlap calibration repeats the same
test. That outline is a function of (frame, track) and peers reads only that
frame's observations and the outline, and labels are never written (the
correction edits a copy), so the answer is fixed per (frame, track). Peer
tests against other masks (the borrower's external outline) are unchanged.

Each track's row list is made after a frame sort; when no track repeats or
reverses a frame in table order the sort is skipped
(track_sort_skips.tracks_in_order) and the same rows come out in the same order.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys
try:
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from module_scan import file_sha256 as _file_sha256
try:
    from .track_sort_skips import tracks_in_order as _tracks_in_order_helper
except ImportError:
    from track_sort_skips import tracks_in_order as _tracks_in_order_helper

VERSIONS = {'3c4d7bc6d6ec9ed3'}
MEMO = ('                and mask[round(float(p.y)), round(float(p.x))]]\n',
        '                and mask[round(float(p.y)), round(float(p.x))]]\n'
        '\n'
        '    _own_peer_cache = {}\n'
        '\n'
        '    def _own_peers(point, mask):\n'
        '        # mask is component(point), itself cached by this key.\n'
        '        key = int(point.frame), int(point.track_id)\n'
        '        if key not in _own_peer_cache:\n'
        '            _own_peer_cache[key] = bool(peers(point, mask))\n'
        '        else:\n'
        '            _coupled_peer_stats["reused"] += 1\n'
        '        return _own_peer_cache[key]\n')
CALLS = (('            if mask is None or peers(p, mask):\n',
          '            if mask is None or _own_peers(p, mask):\n'),
         ('                         and not peers(ref, refmask))\n',
          '                         and not _own_peers(ref, refmask))\n'))
SORT = ("    by_track = {int(tr): list(g.sort_values('frame').itertuples(index=False))\n"
        "                for tr, g in observed.groupby('track_id')}\n",
        "    by_track = {tr: list(g.itertuples(index=False))\n"
        "                for tr, g in _coupled_tracks(observed, 'track_id').items()}\n")


def _digest(module):
    try:
        return _file_sha256(module.__file__)[:16]
    except (AttributeError, OSError, TypeError):
        return None


def rewrite(source):
    for old, _ in (MEMO,) + CALLS:
        if source.count(old) != 1:
            return None
    for old, new in (MEMO,) + CALLS:
        source = source.replace(old, new)
    if source.count(SORT[0]) == 1:
        source = source.replace(*SORT)
    return source


@contextmanager
def install(stats=None):
    """Replace every loaded verified correct (and its aliases) for the duration; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'reused', 'groups_in_order', 'groups_sorted'):
        stats.setdefault(key, 0)
    swap = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, 'correct', None)
        if (original is None or getattr(original, '__module__', None) != module.__name__
                or getattr(original, '__wrapped__', None) is not None or id(original) in swap
                or _digest(module) not in VERSIONS):
            continue
        source = rewrite(inspect.getsource(original))
        if source is None:
            continue
        globals_ = original.__globals__
        globals_['_coupled_peer_stats'] = stats
        globals_['_coupled_tracks'] = _tracks_in_order_helper(stats)
        namespace = {}
        exec(compile(source, '<coupled-peer-memo>', 'exec',
                     flags=__future__.annotations.compiler_flag
                     if original.__code__.co_flags & __future__.annotations.compiler_flag else 0,
                     dont_inherit=True), globals_, namespace)
        replacement = namespace['correct']
        replacement.__wrapped__ = original
        swap[id(original)] = (original, replacement)
    stats['versions'] += len(swap)
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
        for original, _ in swap.values():
            original.__globals__.pop('_coupled_peer_stats', None)
            original.__globals__.pop('_coupled_tracks', None)
