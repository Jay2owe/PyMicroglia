"""Capture complete original tracking passes without unused later operations.

Both passes still process every source frame, in both directions. The accepted
parent supplies later established processing; gap-owner transport consumes only
these exact captured tracker outputs. The stop is after the second tracker
returns, before later reconciliation can run.
"""
from contextlib import nullcontext
import hashlib
from pathlib import Path
import time

import numpy as np


def _array_digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode()); h.update(str(a.shape).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def prepared_digest(value):
    """Fingerprint every cached mask, measurement and metadata value."""
    h = hashlib.sha256()
    def visit(item):
        h.update(type(item).__name__.encode() + b':')
        if isinstance(item, np.ndarray):
            h.update(_array_digest(item).encode())
        elif isinstance(item, dict):
            for key, element in item.items():visit(key);visit(element)
        elif isinstance(item, (list, tuple)):
            for element in item:visit(element)
        elif isinstance(item, set):
            for element in sorted(item):visit(element)
        else:h.update(repr(item).encode())
        h.update(b';')
    visit(value)
    return h.hexdigest()


def collect_tracking_prefix(config, physical, reservation, raw, lag,
                            observations, observation_table, anchor_t,
                            motion_reserved, *, points=None, rule_context=None,
                            prepared_cache=None):
    import accepted_history as ah
    import tracking
    original_track = ah.track_physical_somas
    original_prepare = ah.prepare_tracking_observations
    captured = []; events = []; timings = []
    if prepared_cache is None:
        prepared_cache = {}
    key = (_array_digest(physical), _array_digest(raw),
           hashlib.sha256(Path(tracking.__file__).read_bytes()).hexdigest())

    class PrefixComplete(Exception):
        pass

    def prepare(labels, intensity, **kwargs):
        # The source arrays are the same complete inputs already fingerprinted
        # above. No geometry is reused across a different input or code hash.
        if labels is not physical or intensity is not raw:
            return original_prepare(labels, intensity, **kwargs)
        started = time.perf_counter()
        call_key = (key, tuple(sorted(kwargs.items())))
        reused = prepared_cache.get('key') == call_key
        if not reused:
            value = original_prepare(labels, intensity, **kwargs)
            prepared_cache.clear()
            prepared_cache.update(key=call_key, value=value, digest=prepared_digest(value))
        else:
            if prepared_digest(prepared_cache['value']) != prepared_cache['digest']:
                raise AssertionError('Prepared scientific evidence was mutated')
        timings.append(dict(operation='prepare_geometry', reused=reused,
                            seconds=time.perf_counter()-started))
        return prepared_cache['value']

    def track(*args, **kwargs):
        started = time.perf_counter();event_start=len(events)
        result = original_track(*args, **kwargs)
        captured.append((result[0].copy(), result[1].copy(deep=True)))
        timings.append(dict(operation='complete_tracking_pass', index=len(captured),
                            seconds=time.perf_counter()-started,
                            modified_gap_evaluations=len(events)-event_start))
        if len(captured) == 2:
            raise PrefixComplete()
        return result

    ah.track_physical_somas = track
    ah.prepare_tracking_observations = prepare
    scope = rule_context(points, events) if rule_context is not None else nullcontext()
    completed = False
    try:
        with scope:
            try:
                ah._confidence_layered_retracking(config, physical, reservation, raw, lag, observations,
                             observation_table, anchor_t, motion_reserved)
            except PrefixComplete:
                completed = True
    finally:
        ah.track_physical_somas = original_track
        ah.prepare_tracking_observations = original_prepare
    if not completed or len(captured) != 2:
        raise AssertionError('Original producer did not complete its two tracking passes')
    if prepared_digest(prepared_cache['value']) != prepared_cache['digest']:
        raise AssertionError('Tracking changed the cached scientific evidence')
    return captured, events, timings
