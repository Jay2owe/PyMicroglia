"""Correction gap transport walks measured cores with the evidence stage's exact walk.

The native gap transport stages import trace_measured_core_path by name and
walk from every source core, target core and reciprocal end. The walk from a
point depends only on the point, the frame of its last move, the direction and
the geometry, so measured_gap_path_reuse.walker remembers each step and later
walks follow it; the arrays, arithmetic, tolerance and tie stop are the frozen
original's. Both audited transport versions build their core geometry once
per call and never write to it (they only rebind names and take boolean
selections), which the walker's per-geometry table relies on. Other callers or
walk versions keep the original.
"""
from contextlib import contextmanager
import inspect

try:
    from .measured_gap_path_reuse import SOURCES, walker
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from measured_gap_path_reuse import SOURCES, walker
    from module_scan import file_sha256 as _file_sha256

CALLERS = {
    '81a7217a8ef0b56fc768522f1f797c0fd597e25235138f267526ef50ab212fa5',  # native_gap_owner_transport_a003
    '208766a7f376fae2870592f657583c01c3014206e6def06af99afa45a1870bbb',  # native_gap_owner_transport_a010
}


@contextmanager
def install(module, stats=None):
    """Replace an audited caller's walk for the duration; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'trace_calls', 'requested_frames', 'extended_frames',
                'cache_hits', 'peak_cache_bytes', 'linked_steps'):
        stats.setdefault(key, 0)
    original = getattr(module, 'trace_measured_core_path', None)
    path = getattr(module, '__file__', None)
    try:
        supported = (original is not None and path is not None
                     and getattr(original, '__wrapped__', None) is None
                     and _file_sha256(path) in CALLERS
                     and _file_sha256(inspect.getsourcefile(original)) in SOURCES)
    except (OSError, TypeError):
        supported = False
    if not supported:
        yield stats
        return
    trace, states = walker(original, stats)
    trace.__wrapped__ = original
    module.trace_measured_core_path = trace
    stats['versions'] += 1
    try:
        yield stats
    finally:
        module.trace_measured_core_path = original
        states.clear()
