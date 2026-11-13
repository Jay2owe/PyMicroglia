"""Original duplicate-site, remnant, detached-soma and short-contact operations."""
import numpy as np
import pandas as pd
import recover_persistent_duplicate_soma_sites as duplicates
import recover_bookended_owner_with_small_remnants as remnants
import recover_detached_soma_shared_outline as detached
import recover_short_bookended_soma_beside_transient_holders as contact

def apply(labels, ledger, raw, points, restored):
    for stage, rule, prefix in [('duplicate_sites',duplicates,'persistent_duplicate'),
                                ('small_remnants',remnants,'bookended_remnant')]:
        protected = labels != restored
        proposals = rule.discover(labels, points)
        if proposals:
            candidate, audit, applications = rule.correct(labels, raw, points, protected)
        else:
            candidate, audit, applications = labels.copy(), pd.DataFrame(), pd.DataFrame()
        applied = int(applications.changed_pixels.sum()) if len(applications) else 0
        if int((labels != candidate).sum()) != applied or not np.array_equal(labels > 0, candidate > 0) or np.any((labels != candidate) & protected):
            raise AssertionError('Original foreground/protection/application boundary')
        labels = candidate
        yield stage, labels, ledger.copy(), {prefix+'_audit':audit,prefix+'_applications':applications}
    protected = labels != restored
    candidate, new_ledger, audit, applications = detached.correct(labels, ledger, raw, points, protected)
    gained = (candidate > 0) & (labels == 0)
    if (np.any((candidate != labels) & protected) or np.any((labels > 0) & (candidate == 0))
        or not np.array_equal(new_ledger[~gained],ledger[~gained]) or np.any(new_ledger[gained] != 0)
        or set(map(int,np.unique(candidate))) - set(map(int,np.unique(labels)))):
        raise AssertionError('Original detached-soma label/ledger boundary')
    labels, ledger = candidate, new_ledger
    yield 'detached_soma', labels, ledger, dict(detached_soma_audit=audit,detached_soma_applications=applications)
    protected = labels != restored
    candidate, audit, applications = contact.correct(labels, raw, points, protected)
    if (np.any((candidate != labels) & protected) or not np.array_equal(candidate > 0,labels > 0)
        or set(map(int,np.unique(candidate))) - set(map(int,np.unique(labels)))):
        raise AssertionError('Original bookended-contact boundary')
    yield 'short_contact',candidate,ledger.copy(),dict(bookended_contact_audit=audit,bookended_contact_applications=applications)
