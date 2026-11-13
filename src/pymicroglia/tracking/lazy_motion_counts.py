"""Use the existing exact bounded motion counts for immutable lazy masks."""
from contextlib import contextmanager
import hashlib
import inspect
import numpy as np


@contextmanager
def install(module, mask_type):
    original = module.motion_support
    stats = dict(supported=False, bounded_calls=0, fallback_calls=0)
    # Pin the original count implementation as well as its dispatcher.
    expected = {
        'motion_support': 'adc5a36b7fc8ca1072dab86ff2649424bc454834d84a439535b5670db1eefeab',
        '_bounded_motion_counts': '9d461b53bba0e154a5490d832eac845bfef47e01e072733f4d2fd88a8546a0ac',
        '_motion_mask_counts': 'b5d473bc30daf916794ff47698a90265e77edb15c254383ec893a6688a9abbbd',
    }
    if any(hashlib.sha256(inspect.getsource(getattr(module, name)).encode()).hexdigest() != digest
           for name, digest in expected.items()):
        yield stats
        return

    def support(earlier, later, lag, earlier_bounds=None, later_bounds=None, _count_cache=None):
        def valid(mask, bounds):
            return (type(mask) in (np.ndarray, mask_type) and mask.ndim == 2
                    and mask.dtype == np.bool_ and bounds is not None
                    and 0 <= bounds[0] < bounds[1] <= mask.shape[0]
                    and 0 <= bounds[2] < bounds[3] <= mask.shape[1])
        if valid(earlier, earlier_bounds) and valid(later, later_bounds):
            stats['bounded_calls'] += 1
            # A disk-backed transition image is read through a plain view of
            # the same memory: identical values without the per-index memmap
            # subclass and working-array ufunc dispatch.
            plain = np.asarray(lag)
            if isinstance(plain, np.memmap):
                plain = plain.view(np.ndarray)
            return module._bounded_motion_counts(
                earlier, later, plain, earlier_bounds, later_bounds, _count_cache)
        stats['fallback_calls'] += 1
        return original(earlier, later, lag, earlier_bounds, later_bounds, _count_cache)

    stats['supported'] = True
    module.motion_support = support
    try:
        yield stats
    finally:
        module.motion_support = original
