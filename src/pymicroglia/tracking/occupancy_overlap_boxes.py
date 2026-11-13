"""Occupancy reservation visits only the observations whose box meets the state's box.

_occupancy_reservations ran its box intersection in Python for every one-step
state and every observation of the frame, and skipped a pair whose boxes do not
meet (`y0 >= y1 or x0 >= x1`) before counting a pixel. When the state and every
observation carry bounds, one vectorised test of the same integer comparisons
lists the observations whose boxes meet, in ascending order, and the unchanged
loop body runs for those; the skipped pairs are exactly the ones the original
skipped, and their matrices keep their zero entries. A state or observation
without bounds keeps the full loop.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect

import numpy as np

DIGEST = '01c35c5d217bca14bdb244842f5faa800625b8d5ed47bb005661484019fdf0fa'
LOOP = '''        for col, observation in enumerate(observations):
            state_bounds = state.bounds
'''
FAST = '''        for col in _occupancy_columns(state, _observation_boxes, len(observations)):
            observation = observations[col]
            state_bounds = state.bounds
'''
SETUP = '''    for row, state in enumerate(states):
        if int(features[row, 0]["gap_steps"]) != 1:
'''
FAST_SETUP = '''    _observation_boxes = _occupancy_boxes(observations)
    for row, state in enumerate(states):
        if int(features[row, 0]["gap_steps"]) != 1:
'''


def boxes(observations):
    """(y0, y1, x0, x1) integer columns of every observation's bounds, or None."""
    values = []
    for observation in observations:
        bounds = observation.get("bounds")
        if bounds is None:
            return None
        values.append((int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])))
    return np.array(values, np.int64).reshape(len(values), 4)


def columns(stats):
    def occupancy_columns(state, table, count):
        bounds = state.bounds
        if table is None or bounds is None:
            stats['full'] += 1
            return range(count)
        y0 = np.maximum(int(bounds[0]), table[:, 0])
        y1 = np.minimum(int(bounds[1]), table[:, 1])
        x0 = np.maximum(int(bounds[2]), table[:, 2])
        x1 = np.minimum(int(bounds[3]), table[:, 3])
        stats['boxed'] += 1
        return np.flatnonzero((y0 < y1) & (x0 < x1)).tolist()
    return occupancy_columns


@contextmanager
def install(module):
    stats = dict(supported=False, boxed=0, full=0)
    original = getattr(module, '_occupancy_reservations', None)
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        yield stats
        return
    if (hashlib.sha256(source.encode()).hexdigest() != DIGEST
            or source.count(LOOP) != 1 or source.count(SETUP) != 1):
        yield stats
        return
    source = source.replace(SETUP, FAST_SETUP).replace(LOOP, FAST)
    globals_ = original.__globals__
    sentinel = object()
    names = ('_occupancy_boxes', '_occupancy_columns')
    previous = {name: globals_.get(name, sentinel) for name in names}
    globals_.update(_occupancy_boxes=boxes, _occupancy_columns=columns(stats))
    namespace = {}
    exec(compile(source, '<occupancy-overlap-boxes>', 'exec', flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), globals_, namespace)
    replacement = namespace[original.__name__]
    replacement.__wrapped__ = original
    module._occupancy_reservations = replacement
    stats['supported'] = True
    try:
        yield stats
    finally:
        module._occupancy_reservations = original
        for name, value in previous.items():
            if value is sentinel:
                globals_.pop(name, None)
            else:
                globals_[name] = value
