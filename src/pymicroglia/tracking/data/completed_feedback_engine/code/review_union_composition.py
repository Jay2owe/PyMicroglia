"""Uniform composition of the two reviewed handoff branches and continuation.

Both branches run on the same parent for every recording. Their scientific
results determine the union; recording names and saved feedback are not inputs.
"""
from io import StringIO

import numpy as np
import pandas as pd

import recover_emergent_reciprocal_late_handoff as emergent
import recover_short_anchor_soma_name as short_anchor
import recover_fragmented_holder_terminal_name as fragmented
import recover_temporal_gap_owner_bridge as gap_bridge


def combine_branches(parent, first, second):
    if parent.shape != first.shape or parent.shape != second.shape:
        raise ValueError('Branch arrays must align')
    first_changed, second_changed = first != parent, second != parent
    conflict = first_changed & second_changed & (first != second)
    if np.any(conflict):
        raise ValueError('Independent branches disagree at '+str(int(conflict.sum()))+' pixels; union unproven')
    result = parent.copy()
    result[first_changed] = first[first_changed]
    result[second_changed] = second[second_changed]
    return result


def _guard(parent, candidate, applications):
    if not np.array_equal(parent > 0, candidate > 0):
        raise AssertionError('Accepted handoff changed foreground')
    if set(map(int, np.unique(candidate))) - set(map(int, np.unique(parent))):
        raise AssertionError('Accepted handoff created an identity')
    applied_pixels = int(applications.changed_pixels.sum()) if len(applications) else 0
    if np.count_nonzero(candidate != parent) != applied_pixels:
        raise AssertionError('Application pixels do not match handoff output')


def _table_boundary(table):
    stream = StringIO()
    table.to_csv(stream, index=False)
    stream.seek(0)
    try:
        return pd.read_csv(stream)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def apply(labels, ledger, raw, points, restored_baseline):
    first, audit, applications = emergent.correct(labels, points)
    _guard(labels, first, applications)
    yield 'emergent_branch', first, ledger.copy(), {
        'emergent_late_handoff_audit': audit, 'emergent_late_handoff_applications': applications}

    second, audit, applications = short_anchor.correct(labels, raw, points, labels != restored_baseline)
    _guard(labels, second, applications)
    yield 'short_anchor_branch', second, ledger.copy(), {
        'short_anchor_name_audit': audit, 'short_anchor_name_applications': applications}

    # The historical package has no scientific CSV tables of its own. Both
    # branch tables are retained and checked at their separate boundaries.
    combined = combine_branches(labels, first, second)
    yield 'reviewed_union', combined, ledger.copy(), {}

    corrected, audit, applications = fragmented.correct(
        combined, points, combined != restored_baseline)
    _guard(combined, corrected, applications)
    yield 'fragmented_holder', corrected, ledger.copy(), {
        'fragmented_holder_terminal_audit': audit, 'fragmented_holder_terminal_applications': applications}

    final, audit, final_applications = gap_bridge.correct(
        corrected, points, _table_boundary(applications), corrected != restored_baseline)
    _guard(corrected, final, final_applications)
    yield 'temporal_gap', final, ledger.copy(), {
        'temporal_gap_bridge_audit': audit, 'temporal_gap_bridge_applications': final_applications}
