"""An independently retained source cannot donate its name to another soma."""
from bracketed_consensus_memory import restore_bracketed_consensus_memory

def restore_retained_source_memory(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels):
    return restore_bracketed_consensus_memory(observed,proposals,raw,points,params,
        proposal_history=proposal_history,current_labels=current_labels,original_labels=original_labels,
        retained_source_only=True,independent_core_history=True)
