"""Recompute the accepted fragment and two-body contact prefix.

The three functions preserve the recorded methods, defaults and protected-pixel
boundaries. They accept scientific arrays/tables, never a recording or feedback
selector. The protection mask is rebuilt from this execution's own changes.
"""
from io import StringIO

import numpy as np
import pandas as pd
import local_fragment_reunion
import episode_consistent_fusion
import bookended_remnant_partition


def table_boundary(table):
    """Preserve the original CSV write/read boundary, including float parsing."""
    stream = StringIO()
    table.to_csv(stream, index=False)
    stream.seek(0)
    try:
        return pd.read_csv(stream)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def fragments(labels, points, tables):
    """Input lineage tables have already crossed their original CSV boundary."""
    result, audit, applications = local_fragment_reunion.correct(
        labels, points, tables['lineage_union_audit'],
        applied_unions=tables['lineage_union_applications'])
    return result, dict(fragment_reunion_audit=audit,
                       fragment_reunion_applications=applications), result != labels


def contacts(labels, raw, points, protected):
    result, audit, applications = episode_consistent_fusion.correct(
        labels, raw, points, protected)
    _guards(labels, result, protected, identities=False)
    return result, dict(episode_audit=audit, episode_applications=applications), (
        protected | (result != labels))


def remnants(labels, tables, protected):
    result, audit, applications = bookended_remnant_partition.correct(
        labels, table_boundary(tables['episode_audit']),
        table_boundary(tables['episode_applications']), protected)
    _guards(labels, result, protected, identities=True)
    return result, dict(remnant_interval_audit=audit, remnant_applications=applications), (
        protected | (result != labels))


def _guards(parent, result, protected, identities):
    if not np.array_equal(parent > 0, result > 0):
        raise AssertionError('Accepted contact method changed foreground')
    if np.any((parent != result) & protected):
        raise AssertionError('Accepted contact method changed protected pixels')
    if identities and not np.array_equal(np.unique(parent), np.unique(result)):
        raise AssertionError('Accepted remnant method changed the active identity set')
