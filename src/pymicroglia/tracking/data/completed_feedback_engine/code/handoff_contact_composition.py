"""Original branch handoffs, displaced continuation and separable contacts."""
from io import StringIO
import numpy as np
import pandas as pd
import recover_stable_soma_branch_handoffs as branch
import recover_stable_branch_after_absent_contact as absent
import continue_displaced_soma_after_tracker_loss as displaced
import rejoin_raw_soma_after_tracker_handoff as rejoin
import restore_brief_name_in_raw_separable_contact as brief
import restore_terminal_exclusive_soma_from_shared_name as exclusive
import restore_terminal_soma_after_recent_precontact as precontact

def _csv(table):
    try:
        return pd.read_csv(StringIO(table.to_csv(index=False)))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def _guard(labels,candidate,protected):
    if (np.any((candidate != labels) & protected) or not np.array_equal(candidate > 0,labels > 0)
        or not np.array_equal(np.unique(candidate),np.unique(labels))):
        raise AssertionError('Original foreground/protection/name-set boundary failed')

def apply(labels,ledger,raw,points,restored):
    protected=labels != restored
    candidate,audit,applications=branch.correct(labels,raw,points,protected)
    _guard(labels,candidate,protected)
    labels=candidate
    after_branch=labels.copy()
    branch_audit=_csv(audit)
    yield 'stable_branch',labels,ledger.copy(),dict(stable_soma_branch_handoff_audit=audit,stable_soma_branch_handoff_applications=applications)
    protected=labels != restored
    candidate,audit,applications=absent.correct(labels,raw,points,protected,branch_audit)
    _guard(labels,candidate,protected)
    labels=candidate
    absent_applications=_csv(applications)
    yield 'absent_return',labels,ledger.copy(),dict(stable_branch_after_absent_contact_audit=audit,stable_branch_after_absent_contact_applications=applications)
    protected=labels != restored
    candidate,audit,applications=displaced.correct(labels,after_branch,protected,absent_applications)
    _guard(labels,candidate,protected)
    labels=candidate
    yield 'displaced_continuation',labels,ledger.copy(),dict(displaced_soma_continuation_audit=audit,displaced_soma_continuation_applications=applications)
    protected=labels != restored
    candidate,audit,applications=rejoin.correct(labels,raw,points,protected,absent_applications)
    _guard(labels,candidate,protected)
    labels=candidate
    yield 'raw_rejoin',labels,ledger.copy(),dict(raw_soma_rejoin_audit=audit,raw_soma_rejoin_applications=applications)
    protected=labels != restored
    candidate,audit,applications=brief.correct(labels,raw,points,restored,protected)
    reversed_prior=(candidate != labels) & protected
    expected_reversal=int(applications.reversed_prior_pixels.sum()) if len(applications) else 0
    if (int(reversed_prior.sum()) != expected_reversal or np.any(candidate[reversed_prior] != restored[reversed_prior])
        or not np.array_equal(candidate > 0,labels > 0) or not np.array_equal(np.unique(candidate),np.unique(labels))):
        raise AssertionError('Original declared prior-reversal boundary failed')
    labels=candidate
    yield 'brief_contact',labels,ledger.copy(),dict(brief_name_raw_contact_audit=audit,brief_name_raw_contact_applications=applications)
    for stage,rule,prefix in [('exclusive_terminal',exclusive,'terminal_exclusive_soma'),
                               ('precontact_terminal',precontact,'terminal_precontact_soma')]:
        protected=labels != restored
        candidate,audit,applications=rule.correct(labels,raw,points,protected)
        _guard(labels,candidate,protected)
        audited=int(applications.changed_pixels.sum()) if len(applications) else 0
        if audited != int((labels != candidate).sum()):
            raise AssertionError('Original application-pixel accounting failed')
        labels=candidate
        yield stage,labels,ledger.copy(),{prefix+'_audit':audit,prefix+'_applications':applications}
