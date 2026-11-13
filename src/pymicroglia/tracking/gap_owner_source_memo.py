"""Native gap owner transport reuses each pass's accepted-source answers and counts support once.

carry_measured_gap_owners (r215 a003 and r213 a010) asks accepted_source(after,
identity, t) for every root row's source frame and every earlier frame of its
chain; roots of one identity repeat the same frames. Within one pass `after`,
`parent`, `cores` and `offset` are never written (only the proposal, claimed
and conflict arrays are), so the integer answer for one (identity, frame) is
the same every time it is asked.

The canonical measured support counted, per hit name, the pixels where the
identity's body and parent == name overlap, as two full-frame masks per name.
The same count is the number of the body's parent values equal to name.

parent and the measured cores are never written during a call (each pass
writes its own proposal copy), so each (frame, name) component labelling and
each frame's measured pixels are kept for all passes instead of one; callers
only read the returned arrays.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import re
try:
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from module_scan import file_sha256 as _file_sha256

VERSIONS = {
    '81a7217a8ef0b56fc768522f1f797c0fd597e25235138f267526ef50ab212fa5',  # r215 a003
    '208766a7f376fae2870592f657583c01c3014206e6def06af99afa45a1870bbb',  # r213 a010
}
MEMO = ('        canonical_support_cache = {}\n',
        '        canonical_support_cache = {}\n'
        '        _accepted_cache = {}\n'
        '\n'
        '        def _accepted(native_owner, t):\n'
        '            key = (native_owner, t)\n'
        '            if key not in _accepted_cache:\n'
        '                _accepted_cache[key] = accepted_source(after, native_owner, t)\n'
        '            _gap_owner_stats["asked"] += 1\n'
        '            return _accepted_cache[key]\n')
CALLS = (('            owner = accepted_source(after, identity, source)\n',
          '            owner = _accepted(identity, source)\n'),
         ('            witnessed = [accepted_source(after, identity, t) for t in previous]\n',
          '            witnessed = [_accepted(identity, t) for t in previous]\n'))
SUPPORT = ('                        body = after[f + offset] == identity\n'
           '                        for name in names:\n'
           '                            support[name] = support.get(name, 0) + int(np.count_nonzero(\n'
           '                                body & (parent[f] == name)))\n',
           '                        body = after[f + offset] == identity\n'
           '                        _inside = parent[f][body]\n'
           '                        for name in names:\n'
           '                            support[name] = support.get(name, 0) + int(np.count_nonzero(\n'
           '                                _inside == name))\n')
# parent and cores are never written during a call, so a (frame, name)
# labelling and a frame's measured pixels hold for every pass, not just one.
PASSES = ('    for pass_index, trace in enumerate(traces):\n',
          '    _components_held = {}\n'
          '    _pixels_held = {}\n'
          '    for pass_index, trace in enumerate(traces):\n')
COMPONENTS = ('        component_cache = {}\n', '        component_cache = _components_held\n')
PIXELS = ('        def measured_pixels(t):\n'
          '            observed = cores.get(t, np.empty((0, 3)))\n',
          '        def measured_pixels(t):\n'
          '            if t in _pixels_held:\n'
          '                _gap_owner_stats["pixels_reused"] += 1\n'
          '                return _pixels_held[t]\n'
          '            _pixels_held[t] = _measured_pixels(t)\n'
          '            return _pixels_held[t]\n'
          '\n'
          '        def _measured_pixels(t):\n'
          '            observed = cores.get(t, np.empty((0, 3)))\n')
# Names other than these that write parent or cores keep the per-pass caches.
WRITES = re.compile(r'\b(parent|cores)\b(\[[^\]]*\])?\s*([-+*/|&^]?=(?!=))|\b(parent|cores)\.(update|pop|setdefault|clear|fill|put|sort|resize)\b'
                    r'|\b(observed|y|x|cy|cx|sy|sx|wy|wx|all_cores)\b(\[[^\]]*\])?\s*[-+*/|&^]=|\b(observed|y|x|cy|cx|sy|sx|wy|wx|all_cores)\[[^\]]*\]\s*=(?!=)')
CORES = '    cores = observed_core_geometry(native_points)\n'


def _digest(module):
    try:
        return _file_sha256(module.__file__)
    except (AttributeError, OSError, TypeError):
        return None


def rewrite(source):
    for old, _ in (MEMO, SUPPORT) + CALLS:
        if source.count(old) != 1:
            return None
    for old, new in (MEMO, SUPPORT) + CALLS:
        source = source.replace(old, new)
    if (source.count(CORES) == 1 and not WRITES.search(source.replace(CORES, '')) and all(source.count(old) == 1 for old, _ in (PASSES, COMPONENTS, PIXELS))
            and source.count('component_cache') == 4):
        for old, new in (PASSES, COMPONENTS, PIXELS):
            source = source.replace(old, new)
    return source


@contextmanager
def install(module, stats=None):
    """Replace an audited module's carry_measured_gap_owners for the duration; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'asked', 'pixels_reused'):
        stats.setdefault(key, 0)
    original = getattr(module, 'carry_measured_gap_owners', None)
    source = None
    if (original is not None and getattr(original, '__wrapped__', None) is None
            and _digest(module) in VERSIONS):
        source = rewrite(inspect.getsource(original))
    if source is None:
        yield stats
        return
    globals_ = original.__globals__
    globals_['_gap_owner_stats'] = stats
    namespace = {}
    exec(compile(source, '<gap-owner-source-memo>', 'exec',
                 flags=__future__.annotations.compiler_flag
                 if original.__code__.co_flags & __future__.annotations.compiler_flag else 0,
                 dont_inherit=True), globals_, namespace)
    replacement = namespace['carry_measured_gap_owners']
    replacement.__wrapped__ = original
    module.carry_measured_gap_owners = replacement
    stats['versions'] += 1
    try:
        yield stats
    finally:
        module.carry_measured_gap_owners = original
        globals_.pop('_gap_owner_stats', None)
