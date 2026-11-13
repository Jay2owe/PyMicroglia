"""Substantial-soma separation with each identity labelled inside its box.

For every identity on every frame the separation labelled the identity's mask
over the whole frame. One bounding-box pass per frame gives each positive
identity's box; every component lies inside it, and scanning the box in raster
order meets the components in the same order as scanning the frame, so they
receive the same numbers. Each component is returned as the same whole-frame
mask, and its area is the count of its pixels. Identities without a box
(non-positive values) keep the whole-frame labelling.
"""
import __future__
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SOURCE = 'b56dcf690a131bb3526ee6e6d2f1e54cfbd0ebfadab7ec45f6b5fb1f1734531d'
REPLACEMENTS = (
    ("        local_object = 1\n",
     "        local_object = 1\n        _boxes = {}\n"),
    ("            components, count = ndi.label(\n"
     "                frame == identity, structure=CONNECTIVITY)\n",
     "            _parts = _identity_parts(frame, identity, _boxes)\n"
     "            count = len(_parts)\n"),
    ("                mask = components == component\n",
     "                mask, _area = _parts[component - 1]\n"),
    ("                    \"area\": int(mask.sum()),\n",
     "                    \"area\": _area,\n"),
)


def helpers(module, stats):
    structure = module.CONNECTIVITY

    def identity_parts(frame, identity, boxes):
        plain = frame.view(np.ndarray) if isinstance(frame, np.ndarray) else np.asarray(frame)
        if 'boxes' not in boxes:
            boxes['boxes'] = (ndi.find_objects(plain)
                              if plain.ndim == 2 and plain.dtype.kind in 'iu' else None)
        found = boxes['boxes']
        if identity <= 0 or found is None or identity > len(found) or found[identity - 1] is None:
            components, count = ndi.label(frame == identity, structure=structure)
            stats['whole'] += 1
            return [(mask, int(mask.sum())) for mask in
                    (components == component for component in range(1, count + 1))]
        box = found[identity - 1]
        local, count = ndi.label(plain[box] == identity, structure=structure)
        parts = []
        for component in range(1, count + 1):
            part = local == component
            mask = np.zeros(plain.shape, dtype=bool)
            mask[box] = part
            parts.append((mask, int(np.count_nonzero(part))))
        stats['boxed'] += 1
        return parts

    return dict(_identity_parts=identity_parts)


@contextmanager
def install(module):
    stats = dict(supported=False, boxed=0, whole=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    original = module.separate_substantial_soma_objects
    source = inspect.getsource(original)
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            yield stats
            return
        source = source.replace(old, new)
    globals_ = vars(module)
    added = helpers(module, stats)
    globals_.update(added)
    namespace = {}
    exec(compile(source, '<soma-separation-boxes>', 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), globals_, namespace)
    replacement = namespace['separate_substantial_soma_objects']
    aliases = [(module, 'separate_substantial_soma_objects')]
    import sys
    for other in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        if other is not module and getattr(other, 'separate_substantial_soma_objects', None) is original:
            aliases.append((other, 'separate_substantial_soma_objects'))
    for owner, name in aliases:
        setattr(owner, name, replacement)
    stats['supported'] = True
    try:
        yield stats
    finally:
        for owner, name in aliases:
            setattr(owner, name, original)
        for name in added:
            globals_.pop(name, None)
