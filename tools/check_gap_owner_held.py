"""Check that the gap owner rewrite keeps labellings and measured pixels for the whole call, and only that.

python tools/check_gap_owner_held.py <frozen> <output.json>
Both audited transport versions must take the cross-pass edits, compile, and
contain no other write to parent, cores or the measured pixel arrays. A copy
of each version with one write added must keep the per-pass caches.
measured_pixels with the held lookup must return the same arrays as the
original for present, absent, numpy-integer and edge frames.
"""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

frozen, output = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'))
import gap_owner_source_memo as memo

report = dict(status='pass', versions={})
for path in sorted(frozen.glob('final/*/native_gap_owner_transport_a0*.py')):
    if hashlib.sha256(path.read_bytes()).hexdigest() not in memo.VERSIONS:
        continue
    sys.path.insert(0, str(path.parent))
    sys.modules.pop(path.stem, None)
    module = importlib.import_module(path.stem)
    sys.path.pop(0)
    original = inspect.getsource(module.carry_measured_gap_owners)
    text = memo.rewrite(original)
    assert 'component_cache = _components_held' in text and '_pixels_held[t] = _measured_pixels(t)' in text, path
    compile(text, '<gap-owner-source-memo>', 'exec')
    blocked = {}
    for name, write in (('parent', '        parent[0] = 0\n'), ('cores', '        cores.pop(0, None)\n'),
                        ('pixels', '            y += 0\n')):
        anchor = '        roots = table[' if name != 'pixels' else '            observed, y, x = measured_pixels(t)\n'
        changed = original.replace(anchor, (write + anchor) if name != 'pixels' else anchor + write, 1)
        assert changed != original
        again = memo.rewrite(changed)
        blocked[name] = again is not None and '_components_held' not in again
        assert blocked[name], name
    # The held lookup returns the original's arrays for every kind of frame key.
    namespace = {'np': np, '_gap_owner_stats': {'pixels_reused': 0}}
    cores = {3: np.array([[1.4, 2.6, 3.], [-1., 5., 2.], [9.9, 0., 1.]]), 4: np.empty((0, 3))}
    parent = np.zeros((2, 10, 10), np.uint16)
    start = text.index('        def measured_pixels(t):\n')
    end = text.index('        def displaced_fragment_owner')
    body = 'def make(cores, parent, _pixels_held):\n' + text[start:end] + '        return measured_pixels\n'
    exec(compile(body.replace('\n        ', '\n    '), '<held>', 'exec'), namespace)
    held = namespace['make'](cores, parent, {})

    def plain(t):
        observed = cores.get(t, np.empty((0, 3)))
        y = np.floor(observed[:, 0] + .5).astype(int)
        x = np.floor(observed[:, 1] + .5).astype(int)
        good = (y >= 0) & (y < parent.shape[1]) & (x >= 0) & (x < parent.shape[2])
        return observed[good], y[good], x[good]
    for t in (3, 4, 7, np.int64(3), 3.0, 3, 7):
        for a, b in zip(plain(t), held(t)):
            assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    assert namespace['_gap_owner_stats']['pixels_reused'] == 4
    report['versions'][path.parent.name] = dict(edits=True, blocked=blocked)
assert len(report['versions']) == len(memo.VERSIONS)
output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
print(json.dumps(report))
