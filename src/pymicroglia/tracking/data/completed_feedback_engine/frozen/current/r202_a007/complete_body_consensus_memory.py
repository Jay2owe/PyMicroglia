"""Retain measured identity across a complete pre-existing body partition."""
from bracketed_consensus_memory import restore_bracketed_consensus_memory

def restore_complete_body_memory(observed,proposals,raw,points,params=None,*,proposal_history=None,current_labels,original_labels):
    return restore_bracketed_consensus_memory(observed,proposals,raw,points,params,proposal_history=proposal_history,current_labels=current_labels,original_labels=original_labels,retained_source_only=True,independent_core_history=True,returned_same_name_fragments=True,preserve_label_partition=True,nearby_same_name_fragments=True)
