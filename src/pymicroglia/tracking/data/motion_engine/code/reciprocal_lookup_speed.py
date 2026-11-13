"""Reuse immutable observation lookups for one reciprocal discovery call.

Original functions and arithmetic remain authoritative. Each call has a private
namespace/cache, so changing labels or points between calls cannot reuse evidence.
"""
from contextlib import contextmanager
from functools import wraps
from types import FunctionType


@contextmanager
def cached_reciprocal_discovery(module):
    original = module.discover
    statistics = []

    @wraps(original)
    def discover(*args, **kwargs):
        indices, retained, validity = {}, {}, {}
        counts = dict(point_calls=0, point_misses=0, valid_calls=0, valid_misses=0)
        original_point, original_valid = module._point, module._valid

        def point(index, track, frame, *rest):
            counts['point_calls'] += 1
            identifier = id(index)
            if identifier not in indices:
                indices[identifier] = (index, {})
            saved, cache = indices[identifier]
            assert saved is index
            key = (int(track), int(frame))
            if key not in cache:
                counts['point_misses'] += 1
                # rest is the discovery's row positions, the same for every lookup of one index.
                cache[key] = original_point(index, track, frame, *rest)
            return cache[key]

        def valid(value, minimum_purity, minimum_coverage):
            counts['valid_calls'] += 1
            identifier = id(value)
            retained[identifier] = value
            key = (identifier, minimum_purity, minimum_coverage)
            if key not in validity:
                counts['valid_misses'] += 1
                validity[key] = original_valid(value, minimum_purity, minimum_coverage)
            return validity[key]

        namespace = dict(original.__globals__)
        namespace.update(_point=point, _valid=valid)
        # These helpers resolve _point/_valid in the same private namespace.
        for name in ('_bracket', '_owner_support', 'discover'):
            source = original if name == 'discover' else namespace[name]
            clone = FunctionType(source.__code__, namespace, source.__name__,
                                 source.__defaults__, source.__closure__)
            clone.__kwdefaults__ = source.__kwdefaults__
            namespace[name] = clone
        try:
            return namespace['discover'](*args, **kwargs)
        finally:
            indices.clear(); retained.clear(); validity.clear()
            counts['retained_entries_after_call'] = 0
            statistics.append(counts)

    module.discover = discover
    try:
        yield statistics
    finally:
        module.discover = original
