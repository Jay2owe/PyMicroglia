"""Give audited history modules the engine's exact bounded ndimage.

These modules dilate cell-sized masks across whole frames. The engine's
merge_partition_speed.BoundedNdimage already crops dilation, erosion and
labelling to the occupied box plus the reach of the structure, and falls back
to scipy for every other argument. An erosion asking for `iterations=1` by
keyword is scipy's default, so it takes the bounded call made without it.
Install it before other loader adapters copy module globals, so their compiled
copies see the same object.
"""
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import sys

from scipy import ndimage

SUPPORTED = {
    'reconcile_dormant_releases': 'a57c13f3f51f1b893366bbcc655194c7fb3dc2d34be169ab6544f293d2e46411',
    'reconcile_domain_ownership': 'c2343dee1275675c35597dcfed84babd2596652f3f33a95dc63cdbbca475a2dc',
    'reclaim_gaps': 'b7caa01f951ff4d6805e8244f160b635b8738ee96a58682d638fdead29ccceba',
}
BOUNDED = '60676569ff34d0e63aa1f419bc65412d0f892bdbef29732183222bbc0c6aa9cf'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def install_loader(history):
    stats = dict(supported=False, modules=0, default_iterations=0)
    speed = importlib.import_module('merge_partition_speed')
    if _digest(speed.__file__) != BOUNDED:
        yield stats
        return
    class Bounded(speed.BoundedNdimage):
        def binary_erosion(self, mask, structure=None, **kwargs):
            if (len(kwargs) == 1 and type(kwargs.get('iterations')) is int
                    and kwargs['iterations'] == 1):
                stats['default_iterations'] += 1
                return super().binary_erosion(mask, structure=structure)
            return super().binary_erosion(mask, structure=structure, **kwargs)

    bounded = Bounded()
    original = history._issue_module
    patched = []
    digests = {}

    def load(code_dir, name, *args, **kwargs):
        module = original(code_dir, name, *args, **kwargs)
        for local in SUPPORTED:
            loaded = sys.modules.get(local)
            if loaded is None or getattr(loaded, 'ndi', None) is not ndimage:
                continue
            path = getattr(loaded, '__file__', None)
            if path is None:
                continue
            if path not in digests:
                digests[path] = _digest(path)
            if digests[path] == SUPPORTED[local]:
                loaded.ndi = bounded
                patched.append(loaded)
                stats['modules'] += 1
        return module

    history._issue_module = load
    stats['supported'] = True
    try:
        yield stats
    finally:
        history._issue_module = original
        for module in patched:
            module.ndi = ndimage
