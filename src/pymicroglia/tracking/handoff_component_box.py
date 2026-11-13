"""Owner handoffs label the owner's outlines inside the owner's bounding box.

complete_owner_handoff_episodes' _at_component (one text, in r10_a002 and
a003; recover_stable_dual_soma_contacts calls the r10_a002 copy) labelled
`frame == owner` over the whole frame, picked the component at or nearest the
soma point (separable_merge_recovery._component_at) and returned that
component's full-frame mask. Every owner pixel lies inside the owner's
bounding box, and connectivity inside the box is connectivity in the frame,
so labelling the box and placing it in a zero frame gives the same
components; the chosen pixel depends only on which pixels are non-zero, so
the same component and the same mask come back.

The handoff's _calibrated_area and the dual-contact _reference_area only took
int(mask.sum()) of that mask; they now count the component's pixels inside
the box, which holds all of them.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
import sys

import numpy as np

VERSIONS = {'1242c49d748f1490'}
AREA_VERSIONS = {
    'e636a8ce113c148d': ('_calibrated_area', (
        ('        mask = _at_component(labels[int(row.frame)], int(anchor["owner"]), row)\n'
         '        if mask is not None:\n'
         '            areas.append(int(mask.sum()))\n',
         '        _area = _handoff_area(_at_component, labels[int(row.frame)], int(anchor["owner"]), row)\n'
         '        if _area is not None:\n'
         '            areas.append(_area)\n'),)),
    'eb565160711517e3': ('_reference_area', (
        ('        mask = body._at_component(labels[int(row.frame)], owner, row)\n'
         '        if mask is not None:\n'
         '            areas.append(int(mask.sum()))\n',
         '        _area = _handoff_area(body._at_component, labels[int(row.frame)], owner, row)\n'
         '        if _area is not None:\n'
         '            areas.append(_area)\n'),)),
}


def _source_digest(function):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return None


def _components(names, plain, owner, row):
    """The original's chosen component label, the component image and the owner's box."""
    ndi, physical, structure = names['ndi'], names['physical'], names['STRUCTURE']
    found = plain == owner
    components = np.zeros(plain.shape, np.int32)
    rows = np.flatnonzero(found.any(axis=1))
    box = None
    if len(rows):
        cols = np.flatnonzero(found.any(axis=0))
        box = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
        components[box], _ = ndi.label(found[box], structure)
    component = physical._component_at(
        components, float(row.x), float(row.y), search_radius=6)
    return component, components, box


def _plain(frame):
    plain = frame.view(np.ndarray) if isinstance(frame, np.ndarray) else None
    return plain if plain is not None and plain.ndim == 2 and plain.dtype.kind in 'iub' else None


def boxed(original, stats):
    names = original.__globals__

    def _at_component(frame, owner, row):
        plain = _plain(frame)
        if plain is None:
            stats['direct'] += 1
            return original(frame, owner, row)
        component, components, _ = _components(names, plain, owner, row)
        stats['boxed'] += 1
        return components == component if component else None
    _at_component.__wrapped__ = original
    _at_component._handoff_names = names
    return _at_component


def area(stats):
    def _handoff_area(at_component, frame, owner, row):
        names = getattr(at_component, '_handoff_names', None)
        plain = _plain(frame)
        if names is None or plain is None:
            stats['area_direct'] += 1
            mask = at_component(frame, owner, row)
            return None if mask is None else int(mask.sum())
        component, components, box = _components(names, plain, owner, row)
        stats['area_boxed'] += 1
        # A chosen component (non-zero) exists only where the box was labelled.
        return int(np.count_nonzero(components[box] == component)) if component else None
    return _handoff_area


@contextmanager
def install(stats=None):
    """Replace every loaded verified _at_component and area loop (and their aliases) for the duration."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'boxed', 'direct', 'area_versions', 'area_boxed', 'area_direct'):
        stats.setdefault(key, 0)
    swap = {}
    rewritten = []
    helper = area(stats)
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        original = getattr(module, '_at_component', None)
        if (original is not None and callable(original)
                and getattr(original, '__module__', None) == getattr(module, '__name__', None)
                and getattr(original, '__wrapped__', None) is None and id(original) not in swap
                and _source_digest(original) in VERSIONS):
            swap[id(original)] = (original, boxed(original, stats))
            stats['versions'] += 1
        for name in ('_calibrated_area', '_reference_area'):
            original = getattr(module, name, None)
            if (original is None or not callable(original)
                    or getattr(original, '__module__', None) != getattr(module, '__name__', None)
                    or getattr(original, '__wrapped__', None) is not None or id(original) in swap):
                continue
            entry = AREA_VERSIONS.get(_source_digest(original))
            if entry is None or entry[0] != name:
                continue
            source = inspect.getsource(original)
            if not all(source.count(old) == 1 for old, _ in entry[1]):
                continue
            for old, new in entry[1]:
                source = source.replace(old, new)
            globals_ = original.__globals__
            globals_['_handoff_area'] = helper
            rewritten.append(globals_)
            namespace = {}
            exec(compile(source, '<handoff-component-area>', 'exec',
                         flags=__future__.annotations.compiler_flag
                         if original.__code__.co_flags & __future__.annotations.compiler_flag else 0,
                         dont_inherit=True), globals_, namespace)
            replacement = namespace[name]
            replacement.__wrapped__ = original
            swap[id(original)] = (original, replacement)
            stats['area_versions'] += 1
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
        for globals_ in rewritten:
            globals_.pop('_handoff_area', None)
