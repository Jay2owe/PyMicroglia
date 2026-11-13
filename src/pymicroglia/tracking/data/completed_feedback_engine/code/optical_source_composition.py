"""Recorded contact-name restoration and five optical-source removals.

The original protection, removal and application-accounting boundaries are
retained. The original methods copy remembered identities unchanged here.
"""
import importlib
import numpy as np

STAGES = (
    ('contact_name', 'recover_contact_bounded_soma_name',
     'contact_bounded_name_audit', 'contact_bounded_name_applications'),
    ('persistent_optical', 'remove_persistent_overbright_sources',
     'persistent_optical_core_audit', 'persistent_optical_source_applications'),
    ('compact_optical', 'remove_compact_stable_overbright_sources',
     'compact_optical_core_audit', 'compact_optical_source_applications'),
    ('extreme_optical', 'remove_extreme_compact_overbright_sources',
     'extreme_compact_optical_core_audit', 'extreme_compact_optical_source_applications'),
    ('adjacent_pseudosoma', 'remove_optical_core_adjacent_pseudosomata',
     'optical_adjacent_track_audit', 'optical_adjacent_pseudosoma_applications'),
    ('transient_weak_seat', 'remove_transient_weak_seats',
     'transient_weak_seat_track_audit', 'transient_weak_seat_applications'),
)


def apply(labels, ledger, raw, points, restored_baseline):
    for stage, module_name, audit_name, application_name in STAGES:
        protected = labels != restored_baseline
        candidate, audit, applications = importlib.import_module(module_name).correct(
            labels, raw, points, protected)
        difference = candidate != labels
        if stage == 'contact_name':
            if not np.array_equal(candidate > 0, labels > 0):
                raise AssertionError(stage + ': foreground changed')
        else:
            survivor = candidate > 0
            if np.any(survivor & (labels == 0)) or np.any(candidate[survivor] != labels[survivor]):
                raise AssertionError(stage + ': changed a surviving assignment or added foreground')
            if np.any(difference & protected):
                raise AssertionError(stage + ': protected assignment changed')
        if set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels))):
            raise AssertionError(stage + ': introduced an identity')
        applied_pixels = int(applications.changed_pixels.sum()) if len(applications) else 0
        if np.count_nonzero(difference) != applied_pixels:
            raise AssertionError(stage + ': output differs from application accounting')
        labels = candidate
        ledger = ledger.copy()
        yield stage, labels, ledger, {audit_name: audit, application_name: applications}
