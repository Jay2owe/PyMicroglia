"""Resident-takeover runs from column lists instead of DataFrame.itertuples.

resident_takeover._runs walks each visible track's points in frame order with
group.sort_values("frame").itertuples(index=False). For every track that built a
new namedtuple class and a Series per column, then took each value with
ndarray.item. The adapter keeps the sort and the run logic and makes the same
rows: a namedtuple class named Pandas over the same fields (rename=True, made
once per column set) holding each column's ndarray.tolist(), which gives the
values ndarray.item gives. Tables with a non-NumPy column keep itertuples.
Rows whose integer frames already strictly increase are read in place of the
sort, which would keep them where they are (track_sort_skips.frame_ordered).

discover_proposals built a table of every field of a takeover run's points
only to read the medians of two columns. A DataFrame from a list of dicts
infers each column's dtype from that column's values alone, so the same two
keys per point give the same two columns; points of mixed types or without
both fields keep the whole table (and its error).
"""
import collections
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from .track_sort_skips import frame_ordered as _frame_ordered_helper
except ImportError:
    from track_sort_skips import frame_ordered as _frame_ordered_helper

SOURCE = '29e786ddd157429831438bc0d62ce6d43c1e95f1d0ab3458da040206551ba42f'
OLD = '    for row in group.sort_values("frame").itertuples(index=False):\n'
NEW = '    for row in _run_rows(_frame_ordered(group)):\n'
VALUES_OLD = ('            values = pd.DataFrame([\n'
              '                point._asdict() for point in source["points"]])\n')
VALUES_NEW = '            values = _median_columns(source["points"])\n'
MEDIAN = ('candidate_owner_purity', 'candidate_coverage_fraction')


def median_columns(stats):
    def _median_columns(points):
        kind = type(points[0]) if len(points) else None
        fields = getattr(kind, '_fields', ())
        if kind is not None and all(name in fields for name in MEDIAN) and all(type(p) is kind for p in points):
            stats['median_columns'] += 1
            return pd.DataFrame([{name: getattr(p, name) for name in MEDIAN} for p in points])
        stats['median_tables'] += 1
        return pd.DataFrame([point._asdict() for point in points])
    return _median_columns


def rows(stats):
    classes = {}

    def _run_rows(frame):
        arrays = list(frame._iter_column_arrays())
        if not all(isinstance(values, np.ndarray) for values in arrays):
            stats['fallback'] += 1
            return frame.itertuples(index=False)
        fields = tuple(frame.columns)
        row = classes.get(fields)
        if row is None:
            row = classes[fields] = collections.namedtuple('Pandas', fields, rename=True)
        stats['tracks'] += 1
        return map(row._make, zip(*[values.tolist() for values in arrays]))
    return _run_rows


@contextmanager
def install(module, stats=None):
    stats = stats if stats is not None else {}
    for key in ('supported', 'tracks', 'fallback', 'median_columns', 'median_tables', 'in_order', 'sorted'):
        stats.setdefault(key, 0)
    original = getattr(module, '_runs', None)
    if (original is None or getattr(original, '__wrapped__', None) is not None
            or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE):
        yield stats
        return
    source = inspect.getsource(original)
    if source.count(OLD) != 1:
        yield stats
        return
    namespace = dict(original.__globals__, _run_rows=rows(stats), _frame_ordered=_frame_ordered_helper(stats))
    exec(compile(source.replace(OLD, NEW), '<resident-run-rows>', 'exec'), namespace)
    adapted = namespace[original.__name__]
    adapted.__wrapped__ = original
    module._runs = adapted
    stats['supported'] = 1
    # discover_proposals runs in the module's own names, so it reads _runs
    # (the adapted one while installed) at call time, as the original does.
    proposals = getattr(module, 'discover_proposals', None)
    proposal_source = None
    if proposals is not None and getattr(proposals, '__wrapped__', None) is None:
        try:
            proposal_source = inspect.getsource(proposals)
        except (OSError, TypeError):
            proposal_source = None
    if proposal_source is not None and proposal_source.count(VALUES_OLD) == 1:
        vars(module)['_median_columns'] = median_columns(stats)
        found = {}
        exec(compile(proposal_source.replace(VALUES_OLD, VALUES_NEW), '<resident-median-columns>', 'exec'),
             vars(module), found)
        found['discover_proposals'].__wrapped__ = proposals
        module.discover_proposals = found['discover_proposals']
    else:
        proposals = None
    try:
        yield stats
    finally:
        module._runs = original
        if proposals is not None:
            module.discover_proposals = proposals
            vars(module).pop('_median_columns', None)
