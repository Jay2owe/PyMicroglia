"""Recorded owner-episode sequence following the accepted remnant partition.

Stage names identify algorithms, not biological identities. All operations run
field-wide; protected pixels and contact history come from this execution.
"""
import importlib

import numpy as np
import pandas as pd

from fragment_contact_composition import table_boundary


STAGES = (
    ('owner_handoffs', 'complete_owner_handoff_episodes_a003'),
    ('protected_owner_closure', 'complete_owner_handoff_episodes_r10_a002'),
    ('weak_soma_bridge', 'bridge_weak_soma_gaps'),
    ('reciprocal_owner_bridge', 'bridge_strong_owner_flickers'),
    ('dormant_raw_soma_bridge', 'bridge_dormant_raw_soma_gaps'),
    ('stable_large_contacts', 'recover_stable_large_soma_contacts'),
    ('persistent_resident_contacts', 'recover_persistent_resident_contacts'),
    ('contact_edge_continuation', 'recover_contact_edge_continuation'),
)


def apply(labels, raw, points, protected):
    """Yield each fresh accepted boundary, its tables and cumulative protection."""
    contact_history = []
    for stage, module_name in STAGES:
        method = importlib.import_module(module_name).correct
        if stage == 'contact_edge_continuation':
            history = pd.concat(contact_history, ignore_index=True)
            result = method(labels, raw, points, history, protected)
            names = ('contact_edge_audit', 'contact_applications')
        elif stage in ('stable_large_contacts', 'persistent_resident_contacts'):
            result = method(labels, raw, points, protected)
            names = ('contact_audit', 'contact_applications')
        else:
            result = method(labels, raw, points, protected)
            names = ('owner_episode_audit', 'owner_episode_applications',
                     'owner_episode_frame_audit')
        candidate, *values = result
        if len(values) != len(names):
            raise AssertionError('Accepted operation changed its table contract')
        if not np.array_equal(labels > 0, candidate > 0):
            raise AssertionError(stage + ': foreground changed')
        if np.any((candidate != labels) & protected):
            raise AssertionError(stage + ': protected assignment changed')
        if not np.array_equal(np.unique(labels), np.unique(candidate)):
            raise AssertionError(stage + ': active identity set changed')
        tables = dict(zip(names, values))
        if stage in ('stable_large_contacts', 'persistent_resident_contacts'):
            table = table_boundary(tables['contact_applications'])
            if len(table.columns) == 0:
                table = pd.DataFrame(columns=['track_id', 'reference_owner', 'frame'])
            contact_history.append(table)
        protected = protected | (candidate != labels)
        labels = candidate
        yield stage, labels, tables, protected
