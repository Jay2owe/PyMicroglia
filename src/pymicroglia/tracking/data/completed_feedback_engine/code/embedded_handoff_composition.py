"""Recorded embedded-soma and reciprocal-handoff continuation.

This preserves the previously accepted identity-creation and persistence
boundaries. It introduces no new gate exception or recording-specific selector.
"""
import importlib
import numpy as np

import recover_terminal_name_switch_contacts_a002 as terminal_contact

STAGES = (
    ('embedded_soma', 'recover_bounded_embedded_somata', 'embedded_soma'),
    ('coupled_exchange', 'recover_reciprocal_retired_name_a002', 'reciprocal_exchange'),
    ('terminal_handoff', 'extend_terminal_raw_soma_handoff_v2', 'handoff'),
    ('multistage_handoff', 'recover_multistage_reciprocal_handoff', 'multistage_handoff'),
)


def apply(labels, ledger, raw, points, restored_baseline):
    for stage, module_name, prefix in STAGES:
        method = importlib.import_module(module_name).correct
        old_names = set(map(int, np.unique(labels)))
        if stage == 'embedded_soma':
            protected = labels != restored_baseline
            candidate, audit, applications = method(labels, raw, points, protected)
            next_ledger = ledger.copy()
            if np.any((candidate != labels) & protected):
                raise AssertionError(stage + ': protected assignment changed')
            new_names = set(map(int, np.unique(candidate)))
            if not old_names.issubset(new_names):
                raise AssertionError(stage + ': an existing identity disappeared')
            if bool(np.any(candidate != labels)) != bool(new_names - old_names):
                raise AssertionError(stage + ': creation and changed-pixel evidence disagree')
            tables = {prefix+'_audit': audit, prefix+'_applications': applications}
        elif stage == 'coupled_exchange':
            protected = labels != restored_baseline
            exchanged, next_ledger, audit, applications = method(labels, ledger, raw, points, protected)
            candidate, terminal_audit, terminal_applications = terminal_contact.correct(
                exchanged, raw, points, protected)
            if np.any((candidate != labels) & protected):
                raise AssertionError(stage + ': protected assignment changed')
            if not np.array_equal(np.unique(candidate), np.unique(labels)):
                raise AssertionError(stage + ': active identities changed')
            if not np.array_equal(next_ledger, ledger):
                raise AssertionError(stage + ': remembered identities changed')
            tables = {prefix+'_audit': audit, prefix+'_applications': applications,
                      'terminal_contact_audit': terminal_audit,
                      'terminal_contact_applications': terminal_applications}
        else:
            if stage == 'terminal_handoff':
                candidate, audit, applications, terminal = method(labels, raw, points)
                applied_pixels = int(applications.changed_pixels.sum()) if len(applications) else 0
                applied_pixels += int(terminal.target_pixels.sum()) if len(terminal) else 0
                tables = {prefix+'_audit': audit, prefix+'_applications': applications,
                          'terminal_soma_applications': terminal}
            else:
                candidate, audit, applications = method(labels, points)
                applied_pixels = int(applications.changed_pixels.sum()) if len(applications) else 0
                tables = {prefix+'_audit': audit, prefix+'_applications': applications}
            next_ledger = ledger.copy()
            if set(map(int, np.unique(candidate))) - old_names:
                raise AssertionError(stage + ': new identity introduced')
            if np.count_nonzero(candidate != labels) != applied_pixels:
                raise AssertionError(stage + ': application pixels do not match output')
        if not np.array_equal(candidate > 0, labels > 0):
            raise AssertionError(stage + ': foreground changed')
        labels, ledger = candidate, next_ledger
        yield stage, labels, ledger, tables
