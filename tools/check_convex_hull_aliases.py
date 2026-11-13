"""Check convex_hull_image_fast.aliases against the full walk of every module's names.

python tools/check_convex_hull_aliases.py <output.json>
Planted modules hold two names bound to skimage's original hull, none, a NumPy
array (whose comparison is ambiguous) and an object whose comparison raises.
The swapped (module, name) list must equal the full walk's, in the same order;
only those names point at the fast hull inside the block, and every name is
restored afterwards.
"""
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'))
import convex_hull_image_fast as hull
from skimage.morphology import convex_hull as skimage_hull


class Raises:
    def __eq__(self, other):
        raise RuntimeError('compared')
    __hash__ = object.__hash__


# Held in a list: a module-level name bound to the hull is itself an alias the block swaps.
HELD = [skimage_hull.convex_hull_image]
planted = {}
for index, names in enumerate(({'a': HELD[0], 'b': HELD[0], 'c': 1},
                               {'x': np.arange(3), 'y': Raises()},
                               {},
                               {'z': np.zeros((2, 2)), 'hull': HELD[0]})):
    module = types.ModuleType(f'_hull_alias_check_{index}')
    module.__dict__.update(names)
    sys.modules[module.__name__] = module
    planted[module.__name__] = module


def full_walk(fast):
    found = []
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        names = getattr(child, '__dict__', None)
        if not names or child is skimage_hull or child is sys.modules['__main__']:
            continue
        for name, value in list(names.items()):
            if value is HELD[0]:
                found.append((child.__name__, name))
    return found


before = {name: dict(module.__dict__) for name, module in planted.items()}
with hull.install():
    fast = skimage_hull.convex_hull_image
    assert fast is not HELD[0] and fast.__wrapped__ is HELD[0]
    expected = full_walk(fast)
    start = time.perf_counter()
    with hull.aliases() as count:
        seconds = time.perf_counter() - start
        swapped = [(child.__name__, name) for child in list(sys.modules.values())
                   if type(child).__module__ != 'Cython.Shadow' and getattr(child, '__dict__', None)
                   and child is not skimage_hull for name, value in list(child.__dict__.items()) if value is fast]
        assert count == len(expected)
        assert [s for s in swapped if s in expected] == expected, (swapped, expected)
        for name, module in planted.items():
            for key, value in before[name].items():
                assert (module.__dict__[key] is fast) == (value is HELD[0]), (name, key)
    after = full_walk(fast)
assert after == expected, (after, expected)
for name, module in planted.items():
    assert all(module.__dict__[k] is v for k, v in before[name].items())
    del sys.modules[name]
report = dict(status='pass', aliases=expected, planted=len(planted), seconds=seconds)
Path(sys.argv[1]).write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
print(json.dumps(report))
