"""DataFrame.itertuples without rebuilding its row class and column views on every call.

pandas 2.2 builds a new namedtuple class (about 0.3 ms) and one Series per
column through the positional indexer on every call. About 15 correction
families call it once per track or per group, so this costs about 90 s.
The class depends only on its name and the text of its field names, so one
class per (name, field texts) is kept, with the module pandas gives it. Each
column's values are read from the block manager; iterating them repeats the
Series iterator: Python scalars from NumPy arrays, the array's own iterator
otherwise. The rows are the same values in the same order.
"""
from contextlib import contextmanager
import collections
import hashlib
import inspect

import numpy as np

_classes = {}


def _source_digest(function):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return None


def _values(values):
    if not isinstance(values, np.ndarray):
        return iter(values)
    return map(values.item, range(values.size))


def _row_class(name, fields):
    key = (name, tuple(map(str, fields)))
    held = _classes.get(key)
    if held is None:
        held = collections.namedtuple(name, fields, rename=True, module='pandas.core.frame')
        _classes[key] = held
    return held


def make(original, stats):
    def itertuples(self, index=True, name='Pandas'):
        stats['calls'] += 1
        if type(name) is not str and name is not None:
            stats['direct'] += 1
            return original(self, index=index, name=name)
        arrays = []
        fields = list(self.columns)
        if index:
            arrays.append(self.index)
            fields.insert(0, 'Index')
        arrays.extend(_values(self._get_column_array(k)) for k in range(len(self.columns)))
        if name is not None:
            return map(_row_class(name, fields)._make, zip(*arrays))
        return zip(*arrays)
    itertuples.__wrapped__ = original
    itertuples.__doc__ = original.__doc__
    return itertuples


# Source hashes of the pandas 2.2.2 methods whose behaviour is repeated here.
EXPECTED = {'itertuples': '7d00d9b76eab7d36', 'iter': 'f5c96d2504fdf4f2', 'column': 'fdc9bed3956efede'}


@contextmanager
def install(stats=None):
    import pandas as pd
    from pandas.core.base import IndexOpsMixin
    stats = {} if stats is None else stats
    for key in ('calls', 'direct'):
        stats.setdefault(key, 0)
    original = pd.DataFrame.__dict__.get('itertuples')
    digests = dict(itertuples=_source_digest(original), iter=_source_digest(IndexOpsMixin.__iter__),
                   column=_source_digest(getattr(pd.DataFrame, '_get_column_array', None)))
    if original is None or digests != EXPECTED:
        stats['versions_unmatched'] = digests
        yield stats
        return
    pd.DataFrame.itertuples = make(original, stats)
    try:
        yield stats
    finally:
        pd.DataFrame.itertuples = original
