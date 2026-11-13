"""Original detector-gap memory operations with their established boundaries."""
import numpy as np
import pandas as pd
import stationary_reconciliation as stationary
import successive_detector_soma_memory_a004 as successive
import dual_soma_ancestry_memory_a001 as dual
import observed_soma_gap_memory_a003 as gap
import recording_start_soma_memory_a001 as resident
import bookended_multi_owner_soma_a002 as bookended


def apply(labels, ledger, raw, points, restored, source_frame_offset):
    aligned = stationary.align_stack(raw, len(labels), source_frame_offset)
    for stage, rule in [('successive', successive), ('dual', dual), ('gap', gap),
                        ('resident', resident), ('bookended', bookended)]:
        protected = labels != restored
        candidate, audit, applications = rule.correct(labels, aligned, points, protected)
        if (np.any((candidate != labels) & protected)
                or not np.array_equal(candidate > 0, labels > 0)
                or len(np.setdiff1d(np.unique(candidate), np.unique(labels)))
                or int(applications.changed_pixels.sum()) != int((candidate != labels).sum())):
            raise AssertionError('Original detector-gap foreground/name/protection/accounting boundary failed')
        labels = candidate
        yield stage, labels, ledger.copy(), dict(
            birth_decision_audit=pd.DataFrame(columns=['decision']),
            affected_body_audit=audit, birth_applications=applications,
            detector_handoff_audit=audit.copy())
