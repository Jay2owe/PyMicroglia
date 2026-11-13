"""Fresh general branch discovery followed by the recorded accepted rule."""
from io import StringIO

import numpy as np
import pandas as pd

import recurrent_branch_inventory
import reunite_recurrent_elongated_branches


def _table_boundary(table):
    stream = StringIO()
    table.to_csv(stream, index=False)
    stream.seek(0)
    try:
        return pd.read_csv(stream)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def apply(labels, ledger, raw, points, restored_baseline):
    pair_frames, pairs = recurrent_branch_inventory.scan(labels, raw, points)
    yield 'fresh_discovery', labels, ledger.copy(), {'pair_frames': pair_frames, 'pairs': pairs}
    pair_frames, pairs = _table_boundary(pair_frames), _table_boundary(pairs)
    protected = labels != restored_baseline
    if len(pairs) and pairs.eligible.astype(bool).any():
        candidate, audit, applications = reunite_recurrent_elongated_branches.correct(
            labels, raw, points, protected, pair_frames, pairs)
    else:
        candidate, audit, applications = labels.copy(), pd.DataFrame(), pd.DataFrame()
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError('Recurrent branch rule changed foreground')
    if np.any((candidate != labels) & protected):
        raise AssertionError('Recurrent branch rule changed a protected assignment')
    if set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels))):
        raise AssertionError('Recurrent branch rule created an identity')
    applied_pixels = int(applications.changed_pixels.sum()) if len(applications) else 0
    if np.count_nonzero(candidate != labels) != applied_pixels:
        raise AssertionError('Branch applications do not match the resulting labels')
    yield 'recurrent_branches', candidate, ledger.copy(), {
        'recurrent_branch_episode_audit': audit, 'recurrent_branch_applications': applications}
