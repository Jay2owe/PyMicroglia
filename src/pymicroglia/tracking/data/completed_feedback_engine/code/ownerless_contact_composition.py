"""Accepted ownerless-recovery and later contact sequence, with fresh ledgers.

Methods and defaults are unchanged. The restored baseline is the caller's
current execution of the initial accepted stage, never a recording selector.
"""
from io import StringIO
import importlib

import numpy as np
import pandas as pd

STAGES = (
    ('ownerless_runs', 'recover_ownerless_soma_runs', 'ownerless_soma'),
    ('short_ownerless_runs', 'recover_short_ownerless_soma_runs', 'short_ownerless_soma'),
    ('expanded_dropout_cores', 'expand_short_dropout_soma_cores', 'expanded_core'),
    ('leading_soma', 'recover_leading_soma_from_future_run', 'leading_soma'),
    ('flanked_contacts', 'recover_flanked_large_contact_interruptions', 'flanked_contact'),
    ('sustained_contacts', 'recover_sustained_weak_contact_splits', 'contact'),
)


def _table_boundary(table):
    stream = StringIO()
    table.to_csv(stream, index=False)
    stream.seek(0)
    try:
        return pd.read_csv(stream)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def apply(labels, ledger, raw, points, restored_baseline):
    """Yield six consecutive fresh label, ledger and evidence boundaries."""
    short_applications = None
    for stage, module_name, prefix in STAGES:
        method = importlib.import_module(module_name).correct
        if stage in ('flanked_contacts', 'sustained_contacts'):
            # Recorded protection is a net difference, not a cumulative union.
            protected = labels != restored_baseline
            candidate, audit, applications = method(labels, raw, points, protected)
            next_ledger = ledger.copy()
            if np.any((candidate != labels) & protected):
                raise AssertionError(stage + ': protected assignment changed')
            if not np.array_equal(candidate > 0, labels > 0):
                raise AssertionError(stage + ': foreground changed')
            if not np.array_equal(next_ledger, ledger):
                raise AssertionError(stage + ': remembered identities changed')
        else:
            args = [labels, ledger, raw, points]
            if stage == 'expanded_dropout_cores':
                if short_applications is None:
                    raise AssertionError('Missing preceding short-recovery evidence')
                args.append(short_applications)
            candidate, next_ledger, audit, applications = method(*args)
            if np.any((candidate != labels) & (labels > 0)):
                raise AssertionError(stage + ': positive parent assignment changed')
            if np.any((next_ledger != ledger) & (candidate == labels)):
                raise AssertionError(stage + ': ledger changed outside label changes')
        if not np.array_equal(np.unique(candidate), np.unique(labels)):
            raise AssertionError(stage + ': active identity set changed')
        if stage == 'short_ownerless_runs':
            short_applications = _table_boundary(applications)
        tables = {prefix+'_audit': audit, prefix+'_applications': applications}
        labels, ledger = candidate, next_ledger
        yield stage, labels, ledger, tables
