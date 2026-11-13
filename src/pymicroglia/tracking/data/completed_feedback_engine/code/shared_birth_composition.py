"""Original shared-parent, detector-memory and birth-validation operations.

Historical human-authority exceptions are evaluated outside this numerical
composition. No recording, identity, coordinate or feedback selector enters.
"""
import numpy as np
import stationary_reconciliation as stationary
import shared_parent_soma_ownership_a003 as shared
import overlapping_detector_soma_memory_a003 as overlap
import coobserved_bookended_soma_memory_a001 as coobserved
import known_soma_birth_validation_a004 as birth

def _guard(labels,candidate,applications):
    audited=int(applications.changed_pixels.sum()) if len(applications) else 0
    if (not np.array_equal(labels > 0,candidate > 0) or len(np.setdiff1d(np.unique(candidate),np.unique(labels)))
        or audited != int((candidate != labels).sum())):
        raise AssertionError('Original memory foreground/name/accounting boundary failed')

def apply(labels,ledger,raw,points,restored,birth_labels,births,source_frame_offset,revise_isolated_prior_reassignment):
    aligned=stationary.align_stack(raw,len(labels),source_frame_offset)
    for stage,rule,prefix,links,strict in [
        ('shared_parent',shared,'predecessor_memory',True,False),
        ('overlapping_detector',overlap,'predecessor_memory',True,True),
        ('coobserved_memory',coobserved,'coobserved_soma',False,True)]:
        protected=labels != restored
        candidate,audit,applications=rule.correct(labels,aligned,points,protected)
        _guard(labels,candidate,applications)
        if strict and np.any((candidate != labels) & protected):
            raise AssertionError('Original strict protection boundary failed')
        # shared.correct itself computes and enforces reinterpretable masks.
        labels=candidate
        outputs={prefix+'_audit':audit,prefix+'_applications':applications}
        if links:
            outputs['predecessor_memory_links']=dict(links=audit.attrs.get('links',[]),memory=audit.attrs.get('memory',[]))
        yield stage,labels,ledger.copy(),outputs
    stage=stationary.align_stack(birth_labels,len(labels),source_frame_offset)
    events=births.rename(columns={'pass':'pass_direction'}).copy()
    events['source_to_t']=events.to_t
    events['to_t']-=source_frame_offset
    candidate,new_ledger,birth_audit,body_audit,applications,handoffs=birth.correct(
        labels,aligned,points,labels != restored,original_labels=restored,birth_labels=stage,births=events,
        unclaimed=ledger,revise_isolated_prior_reassignment=revise_isolated_prior_reassignment)
    if (not np.array_equal((candidate > 0) | (new_ledger > 0),(labels > 0) | (ledger > 0))
        or len(np.setdiff1d(np.unique(candidate),np.unique(labels)))
        or int(applications.changed_pixels.sum()) != int((candidate != labels).sum())):
        raise AssertionError('Original birth-validation combined foreground/name/accounting boundary failed')
    yield 'birth_validation',candidate,new_ledger,dict(birth_decision_audit=birth_audit,affected_body_audit=body_audit,
        birth_applications=applications,detector_handoff_audit=handoffs)
