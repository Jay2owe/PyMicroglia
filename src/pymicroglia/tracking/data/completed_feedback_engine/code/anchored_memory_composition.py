"""Original aligned-raw anchored, recurrent and observed-memory corrections."""
import numpy as np
import stationary_reconciliation as stationary
import anchored_soma_owner_recovery_a002_snapshot as anchored
import recover_recurrent_soma_name_conflicts_a002 as recurrent
import coupled_observed_name_return_a002 as coupled
import observed_lineage_name_memory_a009 as observed
import recurrent_core_memory_a002 as memory

def apply(labels,ledger,raw,points,restored):
    # Original producers each align the same raw movie once before calling the rule.
    # Sharing that immutable view preserves every input pixel and frame offset.
    aligned=stationary.align_stack(raw,len(labels),2)
    protected=labels != restored
    candidate,audit,applications=anchored.correct(labels,aligned,protected,points)
    _guard(labels,candidate,protected,applications)
    labels=candidate
    yield 'anchored_owner',labels,ledger.copy(),dict(anchored_soma_owner_recovery_audit=audit,anchored_soma_owner_recovery_applications=applications)
    for stage,rule,prefix,has_links in [
        ('recurrent_borrower',recurrent,'recurrent_contact',False),
        ('coupled_return',coupled,'coupled_return',False),
        ('observed_memory',observed,'predecessor_memory',True),
        ('recurrent_memory',memory,'predecessor_memory',True)]:
        protected=labels != restored
        candidate,audit,applications=rule.correct(labels,aligned,points,protected)
        _guard(labels,candidate,protected,applications)
        labels=candidate
        outputs={prefix+'_audit':audit,prefix+'_applications':applications}
        if has_links:
            outputs['predecessor_memory_links']=dict(links=audit.attrs.get('links',[]),memory=audit.attrs.get('memory',[]))
        yield stage,labels,ledger.copy(),outputs

def _guard(labels,candidate,protected,applications):
    changed=candidate != labels
    audited=int(applications.changed_pixels.sum()) if len(applications) else 0
    if (np.any(changed & protected) or not np.array_equal(labels > 0,candidate > 0)
        or len(np.setdiff1d(np.unique(candidate),np.unique(labels))) or audited != int(changed.sum())):
        raise AssertionError('Original memory foreground/protection/name/accounting boundary failed')
